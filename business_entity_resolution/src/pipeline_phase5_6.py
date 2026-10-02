"""Phase 5A-6B: Pairwise features + LightGBM classifier + threshold tuning.

Reuses the existing feature engineering in features.py (which already uses rapidfuzz)
and the candidate pairs from blocking.py. Creates cache/ground truth splits,
trains a clean classifier, and reports detailed threshold sweep results.

Usage:
    python pipeline_phase5_6.py
"""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

# ── project imports ──────────────────────────────────────────────────────
from config import WORK
from features import FEATURES as _ALL_FEATURES, build_idf, context_features
from metrics import macro_f05

# ── constants ────────────────────────────────────────────────────────────
PY = sys.executable
CACHE = WORK / "cache"
CACHE.mkdir(exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Auto-detect available features from the actual parquet files
# (features.py may list columns added after the features were computed)
def _detect_features(folder="train_feats"):
    """Return the subset of FEATURES actually present in the feature parquets."""
    parts = sorted((WORK / folder).glob("*.parquet"))
    if not parts:
        return _ALL_FEATURES
    cols = set(pl.read_parquet(parts[0], n_rows=0).columns)
    available = [f for f in _ALL_FEATURES if f in cols]
    missing = [f for f in _ALL_FEATURES if f not in cols]
    if missing:
        print(f"  Note: {len(missing)} features in FEATURES list not in data: {missing}")
        print(f"  Using {len(available)} features that exist in the parquet files.")
    return available

FEATURES = _detect_features()

# The split scheme matches the existing pipeline exactly:
#   bucket = hash(entity_id, seed=7) % 10
#   train: bucket < 3    (30%)
#   val:   bucket == 9   (10%)
TRAIN_BUCKETS = lambda col="s1": pl.col(col).hash(7) % 10 < 3
VAL_BUCKET    = lambda col="s1": pl.col(col).hash(7) % 10 == 9

# Known reference numbers from the conversation:
EXPECTED_FLOOR = 0.05585       # predict-empty-for-everyone baseline
EXPECTED_SINGLETON_RATE = 5.5  # approximate %

# LightGBM hyperparameters (matching existing model.py)
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=255,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=127,
    num_threads=11,
    verbose=-1,
)
N_ROUNDS = 600


# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: Create cache/ground truth files
# ═══════════════════════════════════════════════════════════════════════

def create_cache():
    """Create cache/train_split_ground_truth.tsv and cache/val_ground_truth.tsv."""
    print("=" * 70)
    print("PHASE 0: Creating cache/ground truth splits")
    print("=" * 70)

    s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet", columns=["entity_id"])
    pairs = pl.read_parquet(WORK / "train_pairs.parquet")

    # Split S1 IDs
    bucket = s1_all["entity_id"].hash(7) % 10
    train_s1 = s1_all.filter(bucket < 3)["entity_id"]
    val_s1 = s1_all.filter(bucket == 9)["entity_id"]

    # Split ground truth pairs
    train_gt = pairs.filter(pl.col("s1").is_in(train_s1.implode()))
    val_gt = pairs.filter(pl.col("s1").is_in(val_s1.implode()))

    # Write TSVs
    train_gt.write_csv(CACHE / "train_split_ground_truth.tsv", separator="\t")
    val_gt.write_csv(CACHE / "val_ground_truth.tsv", separator="\t")

    # Compute singletons
    val_gt_s1 = val_gt["s1"].unique()
    val_singletons = val_s1.filter(~val_s1.is_in(val_gt_s1.implode()))

    print(f"  Total S1 entities:         {len(s1_all)}")
    print(f"  Train S1 (bucket<3):       {len(train_s1)}")
    print(f"  Val S1 (bucket==9):        {len(val_s1)}")
    print(f"  Train ground truth pairs:  {len(train_gt)}")
    print(f"  Val ground truth pairs:    {len(val_gt)}")
    print(f"  Val singletons:            {len(val_singletons)} ({len(val_singletons)/len(val_s1)*100:.4f}%)")
    if abs(len(val_singletons)/len(val_s1)*100 - EXPECTED_SINGLETON_RATE) > 1.0:
        print(f"  ⚠ DISCREPANCY: expected ~{EXPECTED_SINGLETON_RATE}% singleton rate")
    print(f"  Saved: {CACHE / 'train_split_ground_truth.tsv'}")
    print(f"  Saved: {CACHE / 'val_ground_truth.tsv'}")

    return train_s1, val_s1, train_gt, val_gt, val_singletons


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5A: Load pre-computed features + report
# ═══════════════════════════════════════════════════════════════════════

def load_feature_parts(folder):
    """Stack feature parts into one float32 matrix."""
    parts = sorted((WORK / folder).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No feature parts in {WORK / folder}")
    frames = []
    for p in parts:
        frames.append(pl.read_parquet(p))
    return pl.concat(frames)


def phase5a_features(train_gt):
    """Load pre-computed features and attach labels."""
    print("\n" + "=" * 70)
    print("PHASE 5A: PAIRWISE FEATURE ENGINEERING")
    print("=" * 70)
    t0 = time.time()

    # Load pre-computed features
    print("  Loading train features from train_feats/ ...")
    train_df = load_feature_parts("train_feats")

    print("  Loading val features from val_feats/ ...")
    val_df = load_feature_parts("val_feats")

    # Attach labels to training features
    lab = train_gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
    train_labeled = train_df.join(lab.select("s1", "other", "y"), on=["s1", "other"], how="left")
    train_labeled = train_labeled.with_columns(pl.col("y").fill_null(0))

    n_pos = int(train_labeled["y"].sum())
    n_total = len(train_labeled)
    n_neg = n_total - n_pos
    pct_pos = n_pos / n_total * 100

    elapsed = time.time() - t0

    print(f"\n  Feature set: {len(FEATURES)} features")
    print(f"  Features: {FEATURES}")
    print(f"\n  ── Training set ──")
    print(f"  Total feature rows:        {n_total}")
    print(f"  Positive pairs (label=1):  {n_pos}")
    print(f"  Negative pairs (label=0):  {n_neg}")
    print(f"  Class balance:             {pct_pos:.4f}% positive")
    print(f"\n  ── Validation set ──")
    print(f"  Total feature rows:        {len(val_df)}")
    print(f"\n  Runtime: {elapsed:.1f}s")

    # Feature mapping to user's requested names:
    print(f"\n  ── Feature mapping to requested features ──")
    mapping = {
        "n_ratio":    "rapidfuzz.fuzz.ratio on core",
        "n_tsort":    "rapidfuzz.fuzz.token_sort_ratio on core",
        "n_tset":     "rapidfuzz.fuzz.token_set_ratio on core",
        "n_partial":  "rapidfuzz.fuzz.partial_ratio on core",
        "alias_tset": "rapidfuzz.fuzz.token_set_ratio on alias_core",
        "n_jacc":     "token-Jaccard (reuses blocking.py jaccard logic)",
        "n_len1":     "length token count side 1 (related to length ratio)",
        "n_len2":     "length token count side 2 (related to length ratio)",
        "legal_eq":   "has_legal_suffix match flag (both match)",
        "legal_both": "has_legal_suffix overlap flag",
        "a_tset":     "rapidfuzz.fuzz.token_set_ratio on address",
        "state_eq":   "exact state/postal region match flag",
        "a_jacc":     "token-Jaccard on address/blocking tokens",
        "is_s3":      "is_source_3 flag",
        "sim_name_blk": "blocking rank/score (name view)",
        "sim_addr_blk": "blocking rank/score (address view)",
    }
    for feat, desc in mapping.items():
        if feat in FEATURES:
            print(f"    {feat:20s} → {desc}")

    # Check blocking scores are present
    if "sim_name_blk" not in FEATURES or "sim_addr_blk" not in FEATURES:
        print("  ⚠ BLOCKING SCORES MISSING FROM FEATURES — this is a bug!")

    return train_labeled, val_df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5B: CLASSIFIER TRAINING
# ═══════════════════════════════════════════════════════════════════════

def phase5b_classifier(train_labeled):
    """Train LightGBM binary classifier."""
    print("\n" + "=" * 70)
    print("PHASE 5B: CLASSIFIER TRAINING")
    print("=" * 70)
    t0 = time.time()

    X = train_labeled.select(FEATURES).to_numpy().astype(np.float32, copy=False)
    y = train_labeled["y"].to_numpy().astype(np.int8, copy=False)

    print(f"  Training rows:    {len(y)}")
    print(f"  Features:         {len(FEATURES)}")
    print(f"  Positive rate:    {y.mean():.6f}")
    print(f"  LightGBM params:  lr={LGB_PARAMS['learning_rate']}, leaves={LGB_PARAMS['num_leaves']}, "
          f"rounds={N_ROUNDS}")

    dtrain = lgb.Dataset(X, y, feature_name=FEATURES, params=LGB_PARAMS, free_raw_data=True).construct()
    del X  # free memory

    booster = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=N_ROUNDS,
        callbacks=[lgb.log_evaluation(100)]
    )

    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")

    # Save model
    model_path = MODEL_DIR / "classifier.txt"
    booster.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # Feature importance (gain-based)
    importance = sorted(
        zip(FEATURES, booster.feature_importance("gain")),
        key=lambda x: -x[1]
    )

    print(f"\n  ── Top 10 features by importance (gain) ──")
    for i, (feat, gain) in enumerate(importance[:10], 1):
        print(f"    {i:2d}. {feat:20s}  gain={int(gain)}")

    # Check if blocking scores are in top 10
    top10_names = {f for f, _ in importance[:10]}
    blocking_feats = {"sim_name_blk", "sim_addr_blk"}
    if not blocking_feats & top10_names:
        print(f"\n  ⚠ FLAG: Neither sim_name_blk nor sim_addr_blk is in the top 10 features.")
        print(f"    This is unusual — blocking scores are typically strong priors.")
        print(f"    Positions: ", end="")
        for feat, gain in importance:
            if feat in blocking_feats:
                rank = [f for f, _ in importance].index(feat) + 1
                print(f"{feat}=#{rank} (gain={int(gain)}), ", end="")
        print("\n    Worth investigating for a feature-building bug.")
    else:
        print(f"\n  ✓ Blocking score(s) in top 10: {blocking_feats & top10_names}")

    return booster


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: THRESHOLD TUNING ON VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def phase6_threshold_sweep(booster, val_df, val_s1, val_gt):
    """Score validation pairs and sweep tau for optimal macro F0.5."""
    print("\n" + "=" * 70)
    print("PHASE 6: THRESHOLD TUNING ON VALIDATION")
    print("=" * 70)
    t0 = time.time()

    # Score all validation pairs
    print("  Scoring validation pairs...")
    X_val = val_df.select(FEATURES).to_numpy().astype(np.float32, copy=False)
    probs = booster.predict(X_val, num_threads=11)
    scored = val_df.select("s1", "other").with_columns(
        pl.Series("p", probs).cast(pl.Float32)
    )
    del X_val, probs

    print(f"  Scored {len(scored)} pairs in {time.time() - t0:.1f}s")

    # One-owner assignment: each S2/S3 record goes to its highest-scoring S1
    owner = scored.sort("p", descending=True).unique("other", keep="first")
    # Keep only pairs involving validation S1 entities
    owner = owner.filter(pl.col("s1").is_in(val_s1.implode()))

    print(f"  After one-owner assignment + val filter: {len(owner)} pairs")

    # Ground truth for validation
    truth = val_gt.select("s1", "other")

    # Blocking recall on validation
    cand_hit = truth.join(scored, on=["s1", "other"], how="semi")
    blocking_recall = len(cand_hit) / len(truth) if len(truth) > 0 else 0
    print(f"  Blocking recall on val: {blocking_recall:.4f} ({len(cand_hit)}/{len(truth)})")

    # Oracle score (perfect matcher on candidates)
    oracle = macro_f05(cand_hit, truth, val_s1)
    print(f"  Oracle F0.5 (perfect matcher on candidates): {oracle:.6f}")

    # Sweep tau from 0.30 to 0.90 in steps of 0.02
    taus = [round(0.30 + i * 0.02, 2) for i in range(31)]  # 0.30 to 0.90
    # Also ensure the specific taus requested are included
    report_taus = {0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9}

    results = []
    best_f05 = -1
    best_tau = None

    print("\n  Sweeping tau...")
    for tau in taus:
        pred = owner.filter(pl.col("p") >= tau)
        f05 = macro_f05(pred, truth, val_s1)

        # Detailed breakdown
        s1_frame = pl.DataFrame({"s1": val_s1.to_list()})
        p_set = pred.select("s1", "other").unique()
        t_set = truth.select("s1", "other").unique()

        # Per-entity analysis
        tp_per = p_set.join(t_set, on=["s1", "other"]).group_by("s1").len("tp")
        np_per = p_set.group_by("s1").len("np")
        nt_per = t_set.group_by("s1").len("nt")
        d = (s1_frame
             .join(np_per, on="s1", how="left")
             .join(nt_per, on="s1", how="left")
             .join(tp_per, on="s1", how="left")
             .fill_null(0))

        # Breakdown categories:
        # perfect_singleton: nt==0 and np==0 (true singleton, correctly predicted empty)
        # false_merge_on_singleton: nt==0 and np>0 (true singleton but we predicted matches)
        # total_miss: nt>0 and np==0 (had matches but we predicted empty)
        # partial_or_full_match: nt>0 and np>0 (we predicted something for a non-singleton)
        perfect_singleton = d.filter((pl.col("nt") == 0) & (pl.col("np") == 0)).height
        false_merge_singleton = d.filter((pl.col("nt") == 0) & (pl.col("np") > 0)).height
        total_miss = d.filter((pl.col("nt") > 0) & (pl.col("np") == 0)).height
        partial_or_full = d.filter((pl.col("nt") > 0) & (pl.col("np") > 0)).height

        # Compute macro precision and recall
        prec_col = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
        rec_col = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
        d2 = d.with_columns(prec_col.alias("prec"), rec_col.alias("rec"))
        # For singletons with np==0, precision is undefined but F0.5=1.0, handle separately
        non_singleton = d2.filter(pl.col("nt") > 0)
        singleton_correct = d2.filter((pl.col("nt") == 0) & (pl.col("np") == 0))
        singleton_wrong = d2.filter((pl.col("nt") == 0) & (pl.col("np") > 0))

        macro_prec = (non_singleton["prec"].sum() + len(singleton_correct) * 1.0 + len(singleton_wrong) * 0.0) / len(d2) if len(d2) > 0 else 0
        macro_rec = (non_singleton["rec"].sum() + len(singleton_correct) * 1.0 + len(singleton_wrong) * 0.0) / len(d2) if len(d2) > 0 else 0

        results.append({
            "tau": tau, "f05": f05, "precision": macro_prec, "recall": macro_rec,
            "perfect_singleton": perfect_singleton,
            "false_merge_singleton": false_merge_singleton,
            "total_miss": total_miss,
            "partial_or_full": partial_or_full,
        })

        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau

    # Print table
    print(f"\n  {'tau':>6s}  {'macro_F0.5':>10s}  {'precision':>10s}  {'recall':>8s}  "
          f"{'perf_sing':>9s}  {'false_mrg':>9s}  {'tot_miss':>8s}  {'part/full':>9s}")
    print("  " + "-" * 90)
    for r in results:
        marker = " *" if r["tau"] == best_tau else ""
        if r["tau"] in report_taus or r["tau"] == best_tau:
            print(f"  {r['tau']:6.2f}  {r['f05']:10.6f}  {r['precision']:10.6f}  {r['recall']:8.6f}  "
                  f"{r['perfect_singleton']:9d}  {r['false_merge_singleton']:9d}  "
                  f"{r['total_miss']:8d}  {r['partial_or_full']:9d}{marker}")

    print(f"\n  ── Best result ──")
    print(f"  Best tau:       {best_tau}")
    print(f"  Best macro F0.5: {best_f05:.6f}")
    print(f"  Baseline floor:  {EXPECTED_FLOOR:.5f} (predict-empty-for-everyone)")
    print(f"  Lift over floor: {best_f05 - EXPECTED_FLOOR:.6f} ({best_f05/EXPECTED_FLOOR:.1f}x)")
    print(f"  Oracle ceiling:  {oracle:.6f}")

    # Consistency check
    if best_f05 < EXPECTED_FLOOR:
        print(f"  ⚠ DISCREPANCY: best F0.5 ({best_f05:.6f}) is BELOW the floor ({EXPECTED_FLOOR})!")
        print(f"    This should never happen — investigate feature or label construction.")

    return best_tau, best_f05, results, owner, truth, scored


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6B: SINGLETON SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════

def phase6b_singleton_check(best_tau, val_s1, val_gt, owner):
    """At optimal tau, check singleton prediction accuracy."""
    print("\n" + "=" * 70)
    print("PHASE 6B: SINGLETON SANITY CHECK")
    print("=" * 70)

    truth = val_gt.select("s1", "other")
    true_singletons = val_s1.filter(~val_s1.is_in(truth["s1"].unique().implode()))

    pred_at_tau = owner.filter(pl.col("p") >= best_tau)
    pred_s1_with_matches = pred_at_tau["s1"].unique()

    # Of the true singletons, how many got a correctly empty prediction?
    correctly_empty = true_singletons.filter(~true_singletons.is_in(pred_s1_with_matches.implode()))
    false_merges = true_singletons.filter(true_singletons.is_in(pred_s1_with_matches.implode()))

    n_true_sing = len(true_singletons)
    n_correct = len(correctly_empty)
    n_false_merge = len(false_merges)

    print(f"  True singletons in val:         {n_true_sing}")
    print(f"  Correctly empty prediction:     {n_correct} ({n_correct/n_true_sing*100:.2f}%)")
    print(f"  False merges on singletons:     {n_false_merge} ({n_false_merge/n_true_sing*100:.2f}%)")
    print(f"  At tau = {best_tau}")
    print()
    print(f"  Impact: Each false merge on a singleton scores 0.0 instead of 1.0.")
    print(f"  Total F0.5 penalty from singleton false merges: "
          f"~{n_false_merge/len(val_s1):.6f} per-entity average loss")

    if n_false_merge / n_true_sing > 0.05:
        print(f"\n  ⚠ WARNING: {n_false_merge/n_true_sing*100:.1f}% false merge rate on singletons is HIGH.")
        print(f"    Consider raising tau or adding singleton-detection features.")
    else:
        print(f"\n  ✓ False merge rate on singletons is {n_false_merge/n_true_sing*100:.2f}% — acceptable.")

    # Additional detail: what P(match) values do the false merges have?
    if n_false_merge > 0:
        fm_preds = pred_at_tau.filter(pl.col("s1").is_in(false_merges.implode()))
        max_p = fm_preds["p"].max()
        min_p = fm_preds["p"].min()
        median_p = fm_preds["p"].median()
        print(f"\n  False-merge P(match) distribution:")
        print(f"    min={min_p:.4f}  median={median_p:.4f}  max={max_p:.4f}")
        print(f"    Total false-merge pairs: {len(fm_preds)}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    overall_t0 = time.time()

    # Phase 0: Create cache
    train_s1, val_s1, train_gt, val_gt, val_singletons = create_cache()

    # Phase 5A: Load features + report
    train_labeled, val_df = phase5a_features(train_gt)

    # Phase 5B: Train classifier
    booster = phase5b_classifier(train_labeled)
    del train_labeled  # free memory

    # Phase 6: Threshold sweep
    best_tau, best_f05, results, owner, truth, scored = phase6_threshold_sweep(
        booster, val_df, val_s1, val_gt
    )
    del val_df  # free memory

    # Phase 6B: Singleton check
    phase6b_singleton_check(best_tau, val_s1, val_gt, owner)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total runtime:      {time.time() - overall_t0:.0f}s")
    print(f"  Best tau:           {best_tau}")
    print(f"  Best val macro F0.5: {best_f05:.6f}")
    print(f"  Baseline floor:     {EXPECTED_FLOOR}")
    print(f"  Lift:               {best_f05/EXPECTED_FLOOR:.1f}x")
    print()
    print("  *** STOP: Do not generate final test-set submission yet. ***")
    print("  *** Review threshold table and selected tau before committing. ***")

    # Save results
    results_path = CACHE / "threshold_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_tau": best_tau,
            "best_f05": best_f05,
            "baseline_floor": EXPECTED_FLOOR,
            "results": results,
        }, f, indent=2)
    print(f"\n  Threshold sweep saved: {results_path}")


if __name__ == "__main__":
    main()
