"""Out-of-fold stage 1 over every train pair, so stage 2 can learn from all of train without in-sample scores.

S1 records are hashed into buckets 0-9 (same hash as features.py). Model A trains on buckets {0,2,4} and scores
the odd buckets; model B trains on {1,3,5} and scores the even ones. Every train pair then carries a stage-1
probability from a model that never saw its S1.

usage:
  oof.py fit     train the two fold models
  oof.py score   score all train pairs, build stage-2 rows (context + prune + extra features)
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK
from features import FEATURES
from features2 import extra_features
from model import PARAMS, N_ROUNDS, labels
from stage2 import CTX, ID_INTS, PRUNE, p_context

SFX = "_hard"
BUCKET = pl.col("s1").hash(7) % 10
FOLDS = {"a": (0, 2, 4), "b": (1, 3, 5)}
SCORED_BY = lambda b: "a" if b % 2 else "b"  # odd buckets are scored by model a


def parts():
    """(path, row filter) covering each train pair exactly once."""
    out = [(p, None) for p in sorted((WORK / f"train{SFX}_feats").glob("*.parquet"))]
    out += [(p, BUCKET >= 3) for p in sorted((WORK / f"val{SFX}_feats").glob("*.parquet"))]
    out += [(p, None) for p in sorted((WORK / f"rest{SFX}_feats").glob("*.parquet"))]
    return out


def read(path, flt):
    df = pl.read_parquet(path)
    return df.filter(flt) if flt is not None else df


def fit():
    lab = labels()
    for name, buckets in FOLDS.items():
        sel = lambda path, flt: (flt & BUCKET.is_in(buckets)) if flt is not None else BUCKET.is_in(buckets)
        n = sum(pl.scan_parquet(p).filter(sel(p, f)).select(pl.len()).collect().item() for p, f in parts())
        x = np.empty((n, len(FEATURES)), dtype=np.float32)  # preallocated: one copy of the matrix at a time
        y = np.empty(n, dtype=np.int8)
        i = 0
        for path, flt in parts():
            df = pl.read_parquet(path).filter(sel(path, flt)).join(lab, on=["s1", "other"], how="left")
            x[i:i + len(df)] = df.select(FEATURES).to_numpy()
            y[i:i + len(df)] = df["y"].fill_null(0).to_numpy()
            i += len(df)
        print("fold", name, "rows", len(y), "pos", round(float(y.mean()), 4), flush=True)
        ds = lgb.Dataset(x, y, feature_name=FEATURES, params=PARAMS, free_raw_data=True).construct()
        del x
        b = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
        b.save_model(str(WORK / f"lgb{SFX}_fold{name}.txt"))
        del ds, b


def score():
    boosters = {k: lgb.Booster(model_file=str(WORK / f"lgb{SFX}_fold{k}.txt")) for k in FOLDS}
    allp, kept = [], []
    for path, flt in parts():
        df = read(path, flt).with_columns(BUCKET.alias("bucket"))
        p = np.zeros(len(df), dtype=np.float32)
        odd = (df["bucket"] % 2 == 1).to_numpy()
        x = df.select(FEATURES).to_numpy()
        for k, m in (("a", odd), ("b", ~odd)):
            if m.any():
                p[m] = boosters[k].predict(x[m], num_threads=11)
        df = df.with_columns(pl.Series("p", p), *ID_INTS)
        allp.append(df.select("s1i", "oi", "p"))
        kept.append(df.filter(pl.col("p") >= PRUNE))
        print(path.parent.name, path.name, len(df), flush=True)
    allp = pl.concat(allp)
    print("all train pairs", len(allp), flush=True)
    ctx = p_context(allp).filter(pl.col("p") >= PRUNE).select(["s1i", "oi"] + [c for c in CTX if c != "p"])
    del allp
    df = pl.concat(kept).join(ctx, on=["s1i", "oi"], how="inner").drop("s1i", "oi")
    del kept, ctx
    df = df.join(labels(), on=["s1", "other"], how="left").with_columns(pl.col("y").fill_null(0))
    df = extra_features(df, "train")
    df.write_parquet(WORK / f"all{SFX}_s2x.parquet")
    print("stage-2 rows", len(df), "pos", int(df["y"].sum()), flush=True)


if __name__ == "__main__":
    {"fit": fit, "score": score}[sys.argv[1]]()
