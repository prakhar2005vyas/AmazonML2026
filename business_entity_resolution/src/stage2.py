"""Second-stage re-scorer.

Stage 1 scores every candidate pair in isolation. Stage 2 adds how that score compares with the competing
candidates of the same S2/S3 record and of the same S1 record (max, runner-up, rank, mass), then re-scores
the pairs that stage 1 did not already rule out.

usage:
  stage2.py val  [tag] [--x | --x2]              cross-validate stage 2 on held-out S1, fit final stage 2
  stage2.py test [tag] [--x | --x2] [--out DIR]  stage-1 score test parts, apply stage 2, write outputs

--x adds the distractor-aware extras (FEATURES_X); --x2 adds those plus FEATURES_X2 (plan v2, stage 1).
Stage-1-scored base frames and every extra feature group are cached in WORK, so a later run with the same
stage-1 model only computes what is new. --out DIR writes the submission to output/DIR instead of output/.
"""
import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import hard_keep
from config import WORK
from features import FEATURES
from features2 import FEATURES_X, FEATURES_X2, iter_extra
from metrics import macro_f05

PRUNE = 0.002  # pairs below this stage-1 probability are dropped before stage 2 (and never predicted)
PARAMS2 = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               max_bin=255, num_threads=11, verbose=-1)
N_ROUNDS2 = 800
TEST_SFX = {"_hard2": "_v2", "_hard3": "_v3"}  # train tag -> test candidate/feature tag (the "_hard" pipeline uses untagged test files)
PRED_BATCH = 1_000_000  # rows per predict call: bounds the dense matrix copies LightGBM makes


# integer keys keep the global window functions over ~20M pairs cheap
ID_INTS = [pl.col("s1").str.slice(3).cast(pl.Int64).alias("s1i"),
           (pl.col("other").str.slice(1, 1).cast(pl.Int64) * 10_000_000_000
            + pl.col("other").str.slice(3).cast(pl.Int64)).alias("oi")]


def stage1_score(booster, folder):
    """All pairs as (s1, other, p); feature rows only for pairs that survive PRUNE."""
    allp, kept = [], []
    for part in sorted((WORK / folder).glob("*.parquet")):
        df = pl.read_parquet(part)
        x = df.select(FEATURES).to_numpy()
        p = booster.predict(x, num_threads=11).astype(np.float32)
        del x
        df = df.with_columns(pl.Series("p", p)).with_columns(ID_INTS)
        del p
        allp.append(df.select("s1i", "oi", "p"))
        kept.append(df.filter(pl.col("p") >= PRUNE))
        del df
    gc.collect()
    return pl.concat(allp), pl.concat(kept)


def predict_batched(booster, df, cols):
    return np.concatenate([booster.predict(df.slice(i, PRED_BATCH).select(cols).to_numpy(), num_threads=11)
                           for i in range(0, len(df), PRED_BATCH)]).astype(np.float32)


def p_context(allp):
    lg = (pl.col("p").clip(1e-7, 1 - 1e-7) / (1 - pl.col("p").clip(1e-7, 1 - 1e-7))).log()
    def side(key, tag):
        return [
            pl.col("p").max().over(key).alias(f"pmax_{tag}"),
            pl.col("p").rank("ordinal", descending=True).over(key).cast(pl.Float32).alias(f"prk_{tag}"),
            pl.col("p").sum().over(key).alias(f"psum_{tag}"),
            (pl.col("p") >= 0.5).sum().over(key).cast(pl.Float32).alias(f"n05_{tag}"),
            (pl.col("p") >= 0.1).sum().over(key).cast(pl.Float32).alias(f"n01_{tag}"),
        ]
    out = allp.with_columns(lg.alias("logit"), *side("oi", "o"), *side("s1i", "s"))
    # runner-up probability: the best score among the other candidates of the same key
    for key, tag in (("oi", "o"), ("s1i", "s")):
        top2 = (out.select(key, "p").sort("p", descending=True).group_by(key, maintain_order=True)
                   .agg(pl.col("p").head(2)).with_columns(pl.col("p").list.get(1, null_on_oob=True).fill_null(0.0)
                   .alias(f"p2_{tag}")).select(key, f"p2_{tag}"))
        out = out.join(top2, on=key, how="left")
    return out.with_columns(
        (pl.col("p") - pl.col("pmax_o")).alias("gap_o"),
        (pl.col("p") - pl.col("pmax_s")).alias("gap_s"),
        # score of the best rival: runner-up if this pair is the top, else the top
        pl.when(pl.col("prk_o") == 1).then(pl.col("p2_o")).otherwise(pl.col("pmax_o")).alias("rival_o"),
        pl.when(pl.col("prk_s") == 1).then(pl.col("p2_s")).otherwise(pl.col("pmax_s")).alias("rival_s"),
    )


CTX = ["p", "logit", "pmax_o", "p2_o", "prk_o", "psum_o", "n05_o", "n01_o", "gap_o", "rival_o",
       "pmax_s", "p2_s", "prk_s", "psum_s", "n05_s", "n01_s", "gap_s", "rival_s"]
FEATURES2 = FEATURES + CTX
GROUPS2 = []
if "--x2" in sys.argv:
    GROUPS2 = [("x", FEATURES_X), ("x2", FEATURES_X2)]
elif "--x" in sys.argv:
    GROUPS2 = [("x", FEATURES_X)]
for _g, _names in GROUPS2:
    FEATURES2 = FEATURES2 + _names
TAG2 = GROUPS2[-1][0] if GROUPS2 else ""


def build(booster, folder):
    """Stage-1 scores plus context features for the pairs that survive PRUNE (no extra feature groups)."""
    allp, kept = stage1_score(booster, folder)
    ctx = p_context(allp).filter(pl.col("p") >= PRUNE).select(["s1i", "oi"] + [c for c in CTX if c != "p"])
    del allp
    df = kept.join(ctx, on=["s1i", "oi"], how="inner").drop("s1i", "oi")
    del kept, ctx
    return df


def with_groups(df, split, stem):
    """Append every requested extra feature group. Each group is cached as WORK/{stem}_{group}.parquet
    (keys + features, same row order as df) and reused when its keys match df exactly."""
    for group, names in GROUPS2:
        path = WORK / f"{stem}_{group}.parquet"
        x = None
        if path.exists():
            x = pl.read_parquet(path)
            if not (len(x) == len(df) and x["s1"].equals(df["s1"]) and x["other"].equals(df["other"])):
                print("cache", path.name, "does not match the rows, recomputing", flush=True)
                x = None
        if x is None:
            x = pl.concat([f for _, f in iter_extra(df.select("s1", "other"), split, group=group)])
            x = pl.concat([df.select("s1", "other"), x], how="horizontal")
            tmp = path.with_suffix(".tmp")
            x.write_parquet(tmp)
            tmp.replace(path)
        df = pl.concat([df, x.select(names)], how="horizontal")
        del x
        gc.collect()
    return df


def val_ids(sfx):
    s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    m = pl.col("s1").hash(7) % 10 == 9
    if sfx.startswith("_hard"):
        m = m & hard_keep("s1")
    return s1_all.filter(m)["s1"]


def sweep(scored, truth, ids, tag):
    owner = scored.sort("q", descending=True).unique("other", keep="first").filter(pl.col("s1").is_in(ids.implode()))
    best = (0, None)
    for thr in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        f = macro_f05(owner.filter(pl.col("q") >= thr), truth, ids)
        print(f"{tag} thr {thr:.2f} F0.5 {f:.4f}", flush=True)
        if f > best[0]:
            best = (f, thr)
    return best


def run_val(sfx):
    lab = pl.read_parquet(WORK / "train_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    base = WORK / f"val{sfx}_s2.parquet"
    if base.exists():  # reuse the stage-1 scored, pruned pairs of a previous run
        df = pl.read_parquet(base)
    else:
        b1 = lgb.Booster(model_file=str(WORK / f"lgb{sfx}.txt"))
        df = build(b1, f"val{sfx}_feats")
        df = df.join(lab, on=["s1", "other"], how="left").with_columns(pl.col("y").fill_null(0))
        df.write_parquet(base)
    df = with_groups(df, "train", f"val{sfx}_s2")
    ids = val_ids(sfx)
    truth = lab.filter(pl.col("s1").is_in(ids.implode())).select("s1", "other")
    print("stage2 rows", len(df), "pos", int(df["y"].sum()), "truth pairs", len(truth),
          "recall after prune", round(truth.join(df, on=["s1", "other"], how="semi").height / len(truth), 4), flush=True)
    print("stage1 alone:", sweep(df.with_columns(pl.col("p").alias("q")), truth, ids, "s1"), flush=True)
    # 2-fold CV grouped by the S2/S3 record so that competing pairs of one record are scored together
    fold = (df["other"].hash(13) % 2).to_numpy()
    x, y = df.select(FEATURES2).to_numpy().astype(np.float32), df["y"].to_numpy()
    q = np.zeros(len(df), dtype=np.float32)
    for k in (0, 1):
        tr = fold != k
        b = lgb.train(PARAMS2, lgb.Dataset(x[tr], y[tr], feature_name=FEATURES2), num_boost_round=N_ROUNDS2)
        q[~tr] = b.predict(x[~tr], num_threads=11)
    oof = df.select("s1", "other", "y").with_columns(pl.Series("q", q))
    oof.write_parquet(WORK / f"val{sfx}_oof{TAG2}.parquet")
    best = sweep(oof, truth, ids, "s2")
    print("stage2 CV:", best, flush=True)
    b = lgb.train(PARAMS2, lgb.Dataset(x, y, feature_name=FEATURES2), num_boost_round=N_ROUNDS2)
    b.save_model(str(WORK / f"lgb2{sfx}{TAG2}.txt"))
    with open(WORK / f"decision2{sfx}{TAG2}.json", "w") as fh:
        json.dump({"thr": best[1], "cv_f05": best[0]}, fh)
    imp = sorted(zip(FEATURES2, b.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", [(k, int(v)) for k, v in imp[:20]], flush=True)


def run_test(sfx):
    from predict import write_lists
    from config import OUTPUT
    b1 = lgb.Booster(model_file=str(WORK / f"lgb{sfx}.txt"))
    b2 = lgb.Booster(model_file=str(WORK / f"lgb2{sfx}{TAG2}.txt"))
    with open(WORK / f"decision2{sfx}{TAG2}.json") as fh:
        thr = json.load(fh)["thr"]
    tsfx = TEST_SFX.get(sfx, "")
    base = WORK / f"test{tsfx}{sfx}_s2.parquet"
    if base.exists():  # stage-1 scored, pruned test pairs of a previous run with the same stage-1 model
        df = pl.read_parquet(base)
    else:
        df = build(b1, f"test{tsfx}_feats")
        tmp = base.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.replace(base)
    del b1
    df = with_groups(df, "test", f"test{tsfx}{sfx}_s2")
    q = predict_batched(b2, df, FEATURES2)
    df = df.select("s1", "other", "p").with_columns(pl.Series("q", q))  # drop the wide feature frame
    gc.collect()
    df.write_parquet(WORK / f"test{sfx}_s2{TAG2}_scored.parquet")
    pred = df.sort("q", descending=True).unique("other", keep="first").filter(pl.col("q") >= thr)
    print("test pairs after prune", len(df), "predicted", len(pred), "thr", thr, flush=True)
    s1_ids = pl.read_parquet(WORK / "test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    cand = pl.read_parquet(WORK / f"test{tsfx}_cand.parquet", columns=["s1", "other"])
    if "--out" in sys.argv:
        # one numbered folder per iteration, files named after it: output/03_stage2_norm/03_stage2_norm_matching_results.tsv
        name = iteration_name(sys.argv[sys.argv.index("--out") + 1])
        out, prefix = OUTPUT / name, f"{name}_"
    else:  # a new pipeline never overwrites the current submission
        out, prefix = (OUTPUT / tsfx.lstrip("_") if tsfx else OUTPUT), ""
    out.mkdir(parents=True, exist_ok=True)
    write_lists(cand, s1_ids, "candidate_entity_ids", out / f"{prefix}candidate_pairs.tsv")
    write_lists(pred, s1_ids, "matched_entity_ids", out / f"{prefix}matching_results.tsv")


def iteration_name(stage):
    """'stage2_norm' -> 'NN_stage2_norm', NN = next free iteration number in OUTPUT (an existing folder is reused)."""
    import re
    from config import OUTPUT
    existing = {p.name: p.name for p in OUTPUT.iterdir() if p.is_dir() and re.match(r"\d\d_", p.name)}
    for n in existing:
        if n[3:] == stage:
            return n
    nums = [int(n[:2]) for n in existing]
    return f"{max(nums, default=0) + 1:02d}_{stage}"


if __name__ == "__main__":
    split = sys.argv[1]
    skip = {sys.argv.index("--out") + 1} if "--out" in sys.argv else set()
    args = [a for i, a in enumerate(sys.argv) if i >= 2 and i not in skip and not a.startswith("--")]
    sfx = f"_{args[0]}" if args else ""
    if split == "val":
        run_val(sfx)
    else:
        run_test(sfx)
