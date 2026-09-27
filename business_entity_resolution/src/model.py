"""Train the pair classifier on train features and tune the decision rule for macro F0.5 on validation."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK
from features import FEATURES
from blocking import hard_keep
from metrics import macro_f05

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=127, num_threads=11, verbose=-1)
N_ROUNDS = 600


def labels():
    return pl.read_parquet(WORK / "train_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))


def load_xy(folder):
    """Stack feature parts into one float32 matrix part by part, so peak memory stays near the matrix size."""
    parts = sorted((WORK / folder).glob("*.parquet"))
    n = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in parts)
    x = np.empty((n, len(FEATURES)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    lab = labels()
    i = 0
    for p in parts:
        df = pl.read_parquet(p).join(lab, on=["s1", "other"], how="left")
        x[i:i + len(df)] = df.select(FEATURES).to_numpy().astype(np.float32, copy=False)
        y[i:i + len(df)] = df["y"].fill_null(0).to_numpy()
        i += len(df)
    return x, y


def train(sfx=""):
    x, y = load_xy(f"train{sfx}_feats")
    print("train rows", len(y), "pos rate", round(float(y.mean()), 4), flush=True)
    dtrain = lgb.Dataset(x, y, feature_name=FEATURES, params=PARAMS, free_raw_data=True).construct()
    del x
    booster = lgb.train(PARAMS, dtrain, num_boost_round=N_ROUNDS, callbacks=[lgb.log_evaluation(100)])
    booster.save_model(str(WORK / f"lgb{sfx}.txt"))
    return booster


def score_parts(booster, folder):
    out = []
    for part in sorted((WORK / folder).glob("*.parquet")):
        df = pl.read_parquet(part)
        out.append(df.select("s1", "other").with_columns(
            pl.Series("p", booster.predict(df.select(FEATURES).to_numpy(), num_threads=11)).cast(pl.Float32)))
    return pl.concat(out)


def decide(df, thr, rel=0.0):
    """One owner per S2/S3 record (argmax p), then keep pairs above thr and within rel of the S1's best."""
    best = (df.select("s1", "other", "p").sort("p", descending=True).unique("other", keep="first")
              .filter(pl.col("p") >= thr))
    if rel > 0:
        best = best.filter(pl.col("p") >= rel * pl.col("p").max().over("s1"))
    return best


def evaluate(booster, sfx=""):
    va = score_parts(booster, f"val{sfx}_feats")
    va.write_parquet(WORK / f"val{sfx}_scored.parquet")
    s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    val_mask = pl.col("s1").hash(7) % 10 == 9
    if sfx.startswith("_hard"):
        val_mask = val_mask & hard_keep("s1")
    val_ids = s1_all.filter(val_mask)["s1"]
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("s1").is_in(val_ids.implode()))
    cand_hit = truth.join(va, on=["s1", "other"], how="semi")
    print(f"val S1 {len(val_ids)}  blocking recall on val {len(cand_hit) / len(truth):.4f}", flush=True)
    print("oracle (perfect matcher on candidates):", round(macro_f05(cand_hit, truth, val_ids), 4), flush=True)
    # argmax assignment is threshold independent: compute once, then sweep
    owner = va.sort("p", descending=True).unique("other", keep="first")
    owner = owner.filter(pl.col("s1").is_in(val_ids.implode()))
    best = (0, None)
    for thr in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        for rel in [0.0, 0.5, 0.7]:
            pred = owner.filter(pl.col("p") >= thr)
            if rel > 0:
                pred = pred.filter(pl.col("p") >= rel * pl.col("p").max().over("s1"))
            f = macro_f05(pred, truth, val_ids)
            print(f"thr {thr:.2f} rel {rel:.1f}  F0.5 {f:.4f}", flush=True)
            if f > best[0]:
                best = (f, {"thr": thr, "rel": rel})
    print("best", best, flush=True)
    with open(WORK / f"decision{sfx}.json", "w") as fh:
        json.dump(best[1], fh)
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(k, int(v)) for k, v in imp[:15]], flush=True)


if __name__ == "__main__":
    # usage: model.py [tag] [--eval-with MODEL_FILE]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sfx = f"_{args[0]}" if args else ""
    if "--eval-with" in sys.argv:
        evaluate(lgb.Booster(model_file=str(WORK / sys.argv[sys.argv.index("--eval-with") + 1])), sfx)
    else:
        evaluate(train(sfx), sfx)
