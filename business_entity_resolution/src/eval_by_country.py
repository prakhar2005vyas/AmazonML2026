"""Evaluate threshold sweep and blocking recall separately for US vs India.

Streams val_feats in parts through the trained LightGBM model, performs one-owner
assignment, caches the scored owner pairs, and evaluates macro F0.5, micro P/R,
and singleton breakdowns across tau for:
  1. All Validation entities
  2. US Validation entities
  3. India Validation entities
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK
from features import FEATURES as _ALL_FEATURES
from metrics import macro_f05

# ── Setup ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "models" / "classifier.txt"
CACHE = WORK / "cache"
OWNER_CACHE = CACHE / "val_owner_scored.parquet"


def detect_features():
    parts = sorted((WORK / "val_feats").glob("*.parquet"))
    cols = set(pl.read_parquet(parts[0], n_rows=0).columns)
    return [f for f in _ALL_FEATURES if f in cols]


FEATURES = detect_features()


def get_owner_predictions(booster):
    """Score val_feats in streaming chunks and retain highest p per other."""
    if OWNER_CACHE.exists():
        print(f"Loading cached owner predictions from {OWNER_CACHE} ...")
        return pl.read_parquet(OWNER_CACHE)

    print("Scoring val_feats parts ...")
    t0 = time.time()
    parts = sorted((WORK / "val_feats").glob("*.parquet"))

    # Load validation S1 entities to filter
    s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet", columns=["entity_id"])
    val_s1 = s1_all.filter(pl.col("entity_id").hash(7) % 10 == 9)["entity_id"]
    val_s1_set = set(val_s1.to_list())

    scored_chunks = []
    total_rows = 0

    for i, p in enumerate(parts):
        pt0 = time.time()
        df = pl.read_parquet(p)
        total_rows += len(df)
        X = df.select(FEATURES).to_numpy().astype(np.float32, copy=False)
        probs = booster.predict(X, num_threads=11)
        scored = df.select("s1", "other").with_columns(
            pl.Series("p", probs).cast(pl.Float32)
        )
        # Keep best per chunk to save memory
        best_chunk = scored.sort("p", descending=True).unique("other", keep="first")
        scored_chunks.append(best_chunk)
        print(f"  Part {i+1:2d}/{len(parts)} ({len(df):,d} rows) scored in {time.time()-pt0:.1f}s", flush=True)

    print(f"Total {total_rows:,d} rows scored in {time.time()-t0:.1f}s. Merging owners ...")
    all_scored = pl.concat(scored_chunks)
    owner = all_scored.sort("p", descending=True).unique("other", keep="first")
    owner = owner.filter(pl.col("s1").is_in(val_s1.implode()))
    print(f"One-owner pairs touching val_s1: {len(owner):,d}")

    owner.write_parquet(OWNER_CACHE)
    print(f"Cached owner predictions to {OWNER_CACHE}")
    return owner


def evaluate_split(owner, truth, s1_ids, country_name, taus):
    print("\n" + "=" * 80)
    print(f"THRESHOLD SWEEP: {country_name.upper()} ({len(s1_ids):,d} S1 entities, {len(truth):,d} GT pairs)")
    print("=" * 80)

    # Filter owner and truth to this country's S1 IDs
    s1_set = set(s1_ids.to_list())
    split_owner = owner.filter(pl.col("s1").is_in(s1_ids.implode()))
    split_truth = truth.filter(pl.col("s1").is_in(s1_ids.implode()))

    # True singletons
    gt_s1 = split_truth["s1"].unique()
    true_singletons = s1_ids.filter(~s1_ids.is_in(gt_s1.implode()))
    n_singletons = len(true_singletons)

    # Candidate blocking recall on this split
    cand_hit = split_truth.join(split_owner, on=["s1", "other"], how="semi")
    blk_recall = len(cand_hit) / len(split_truth) if len(split_truth) > 0 else 0
    oracle = macro_f05(cand_hit, split_truth, s1_ids)

    print(f"  True Singletons:        {n_singletons:,d} ({n_singletons/len(s1_ids)*100:.2f}%)")
    print(f"  Blocking recall:        {blk_recall:.4f} ({len(cand_hit):,d}/{len(split_truth):,d})")
    print(f"  Oracle candidate F0.5:  {oracle:.6f}")
    print()

    print(f"  {'tau':>5s}  {'macro_F0.5':>10s}  {'micro_P':>8s}  {'micro_R':>8s}  "
          f"{'macro_P':>8s}  {'macro_R':>8s}  {'perf_sing':>9s}  {'false_mrg':>9s}  {'miss':>6s}  {'part/full':>9s}")
    print("  " + "-" * 102)

    results = []
    best_f05 = -1
    best_tau = None

    t_set = split_truth.select("s1", "other").unique()
    s1_frame = pl.DataFrame({"s1": s1_ids.to_list()})
    nt_per = t_set.group_by("s1").len("nt")

    for tau in taus:
        pred = split_owner.filter(pl.col("p") >= tau)
        f05 = macro_f05(pred, split_truth, s1_ids)

        p_set = pred.select("s1", "other").unique()
        tp_frame = p_set.join(t_set, on=["s1", "other"], how="inner")
        n_tp = len(tp_frame)
        n_pred = len(p_set)
        n_true = len(t_set)

        # Micro (pooled) metrics
        micro_prec = n_tp / n_pred if n_pred > 0 else 1.0
        micro_rec = n_tp / n_true if n_true > 0 else 0.0

        # Per-entity breakdown
        tp_per = tp_frame.group_by("s1").len("tp")
        np_per = p_set.group_by("s1").len("np")
        d = (s1_frame
             .join(np_per, on="s1", how="left")
             .join(nt_per, on="s1", how="left")
             .join(tp_per, on="s1", how="left")
             .fill_null(0))

        perfect_singleton = d.filter((pl.col("nt") == 0) & (pl.col("np") == 0)).height
        false_merge_singleton = d.filter((pl.col("nt") == 0) & (pl.col("np") > 0)).height
        total_miss = d.filter((pl.col("nt") > 0) & (pl.col("np") == 0)).height
        partial_or_full = d.filter((pl.col("nt") > 0) & (pl.col("np") > 0)).height

        prec_col = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
        rec_col = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
        d2 = d.with_columns(prec_col.alias("prec"), rec_col.alias("rec"))
        non_sing = d2.filter(pl.col("nt") > 0)
        sing_ok = d2.filter((pl.col("nt") == 0) & (pl.col("np") == 0))

        macro_prec = (non_sing["prec"].sum() + len(sing_ok) * 1.0) / len(d2)
        macro_rec = (non_sing["rec"].sum() + len(sing_ok) * 1.0) / len(d2)

        results.append({
            "tau": tau, "f05": f05,
            "micro_prec": micro_prec, "micro_rec": micro_rec,
            "macro_prec": macro_prec, "macro_rec": macro_rec,
            "perfect_singleton": perfect_singleton,
            "false_merge_singleton": false_merge_singleton,
            "total_miss": total_miss,
            "partial_or_full": partial_or_full,
        })

        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau

        marker = " *" if f05 == best_f05 else ""
        if tau in {0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9} or tau == best_tau:
            print(f"  {tau:5.2f}  {f05:10.6f}  {micro_prec:8.4f}  {micro_rec:8.4f}  "
                  f"{macro_prec:8.4f}  {macro_rec:8.4f}  {perfect_singleton:9d}  {false_merge_singleton:9d}  "
                  f"{total_miss:6d}  {partial_or_full:9d}{marker}")

    print(f"\n  Best tau for {country_name}: {best_tau} (macro F0.5 = {best_f05:.6f})")
    return results, best_tau, best_f05, blk_recall


def main():
    print("Loading LightGBM model from", MODEL_PATH)
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    # Load S1 entities with country
    s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet", columns=["entity_id", "country"])
    b9 = s1_all.filter(pl.col("entity_id").hash(7) % 10 == 9)
    us_s1 = b9.filter(pl.col("country") == "us")["entity_id"]
    in_s1 = b9.filter(pl.col("country") == "india")["entity_id"]
    all_s1 = b9["entity_id"]

    # Load ground truth
    truth = pl.read_csv(CACHE / "val_ground_truth.tsv", separator="\t")

    # Get owner predictions
    owner = get_owner_predictions(booster)

    taus = [round(0.30 + i * 0.02, 2) for i in range(31)]

    # Run for All, US, India
    res_all, best_all, f05_all, blk_all = evaluate_split(owner, truth, all_s1, "overall", taus)
    res_us, best_us, f05_us, blk_us = evaluate_split(owner, truth, us_s1, "us", taus)
    res_in, best_in, f05_in, blk_in = evaluate_split(owner, truth, in_s1, "india", taus)

    summary = {
        "overall": {"best_tau": best_all, "best_f05": f05_all, "blk_recall": blk_all, "results": res_all},
        "us": {"best_tau": best_us, "best_f05": f05_us, "blk_recall": blk_us, "results": res_us},
        "india": {"best_tau": best_in, "best_f05": f05_in, "blk_recall": blk_in, "results": res_in},
    }

    out_file = CACHE / "country_threshold_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCountry-stratified threshold sweep saved to {out_file}")


if __name__ == "__main__":
    main()
