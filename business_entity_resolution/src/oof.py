"""Out-of-fold stage 1 over every train pair, so stage 2 can learn from all of train without in-sample scores.

S1 records are hashed into buckets 0-9 (same hash as features.py). Model A trains on buckets {0,2,4} and scores
the odd buckets; model B trains on {1,3,5} and scores the even ones. Every train pair then carries a stage-1
probability from a model that never saw its S1.

Every step checkpoints to WORK and skips work already on disk, so an interrupted run is simply re-launched.
Chunk boundaries come from a fixed row order (the score1 checkpoint), so resuming never drops or repeats a pair.

usage:
  oof.py fit [a|b]   train the fold models (both by default; an existing model is kept)
  oof.py score1      stage-1 score all train pairs, context + prune + labels -> all_hard_s2.parquet
  oof.py score2      extra features for those rows, in resumable parts -> all_hard_s2x/part*.parquet
  oof.py score       score1 then score2
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK
from features import FEATURES
from features2 import iter_extra
from model import PARAMS, N_ROUNDS, labels
from stage2 import CTX, ID_INTS, PRUNE, p_context

SFX = "_hard"
BUCKET = pl.col("s1").hash(7) % 10
FOLDS = {"a": (0, 2, 4), "b": (1, 3, 5)}  # model a scores odd buckets, model b even ones
S2_BASE = WORK / f"all{SFX}_s2.parquet"
S2X_DIR = WORK / f"all{SFX}_s2x"
X_PART = 1_000_000  # rows per score2 part


def parts():
    """(path, row filter) covering each train pair exactly once: train = buckets 0-2; val parts restricted to
    buckets >= 3 (their bucket 0-2 pairs are already in train); rest = the remaining buckets >= 3 pairs."""
    out = []
    for name, flt in ((f"train{SFX}_feats", None), (f"val{SFX}_feats", BUCKET >= 3), (f"rest{SFX}_feats", None)):
        files = sorted((WORK / name).glob("*.parquet"))
        assert files, f"{WORK / name} is empty or missing (rest: run `features.py train hard --rest`)"
        out += [(p, flt) for p in files]
    return out


def fit(names):
    lab = labels()
    for name in names:
        dest = WORK / f"lgb{SFX}_fold{name}.txt"
        if dest.exists():
            print("fold", name, "exists, skipped", flush=True)
            continue
        buckets = FOLDS[name]
        sel = lambda flt: (flt & BUCKET.is_in(buckets)) if flt is not None else BUCKET.is_in(buckets)
        n = sum(pl.scan_parquet(p).filter(sel(f)).select(pl.len()).collect().item() for p, f in parts())
        x = np.empty((n, len(FEATURES)), dtype=np.float32)  # preallocated: one copy of the matrix at a time
        y = np.empty(n, dtype=np.int8)
        i = 0
        for path, flt in parts():
            df = pl.read_parquet(path).filter(sel(flt)).join(lab, on=["s1", "other"], how="left")
            x[i:i + len(df)] = df.select(FEATURES).to_numpy()
            y[i:i + len(df)] = df["y"].fill_null(0).to_numpy()
            i += len(df)
        assert i == n
        print("fold", name, "rows", n, "pos", round(float(y.mean()), 4), flush=True)
        ds = lgb.Dataset(x, y, feature_name=FEATURES, params=PARAMS, free_raw_data=True).construct()
        del x, y
        b = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
        tmp = dest.with_suffix(".tmp")
        b.save_model(str(tmp))
        tmp.replace(dest)
        del ds, b


def score1():
    if S2_BASE.exists():
        print(S2_BASE.name, "exists, skipped", flush=True)
        return
    boosters = {k: lgb.Booster(model_file=str(WORK / f"lgb{SFX}_fold{k}.txt")) for k in FOLDS}
    allp, kept = [], []
    for path, flt in parts():
        df = pl.read_parquet(path)
        if flt is not None:
            df = df.filter(flt)
        odd = (df.select(BUCKET % 2 == 1).to_series()).to_numpy()
        x = df.select(FEATURES).to_numpy()
        p = np.zeros(len(df), dtype=np.float32)
        for k, m in (("a", odd), ("b", ~odd)):
            if m.any():
                p[m] = boosters[k].predict(x[m], num_threads=11)
        del x
        df = df.with_columns(pl.Series("p", p), *ID_INTS)
        allp.append(df.select("s1i", "oi", "p"))
        kept.append(df.filter(pl.col("p") >= PRUNE))
        del df
        print(path.parent.name, path.name, flush=True)
    allp = pl.concat(allp)
    n_all = len(allp)
    # the three sources are disjoint by construction, so equal counts mean every pair is covered exactly once
    n_cand = pl.scan_parquet(WORK / f"train{SFX}_cand.parquet").select(pl.len()).collect().item()
    assert n_all == n_cand, f"{n_all} scored pairs vs {n_cand} candidates"
    ctx = p_context(allp).filter(pl.col("p") >= PRUNE).select(["s1i", "oi"] + [c for c in CTX if c != "p"])
    del allp
    df = pl.concat(kept).join(ctx, on=["s1i", "oi"], how="inner").drop("s1i", "oi")
    del kept, ctx
    df = (df.join(labels(), on=["s1", "other"], how="left").with_columns(pl.col("y").fill_null(0))
            .sort("s1", "other"))  # fixed order: score2 part boundaries depend on it
    tmp = S2_BASE.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(S2_BASE)
    print("all train pairs", n_all, "stage-2 rows", len(df), "pos", int(df["y"].sum()), flush=True)


def score2():
    df = pl.read_parquet(S2_BASE)
    S2X_DIR.mkdir(exist_ok=True)
    dest = lambda start: S2X_DIR / f"part{start // X_PART:04d}.parquet"
    done = {s for s in range(0, len(df), X_PART) if dest(s).exists()}
    print("stage-2 rows", len(df), "parts done", len(done), "of", -(-len(df) // X_PART), flush=True)
    for start, x in iter_extra(df, "train", batch_size=X_PART, skip=done):
        tmp = dest(start).with_suffix(".tmp")
        pl.concat([df.slice(start, X_PART), x], how="horizontal").write_parquet(tmp)
        tmp.replace(dest(start))
    n = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in S2X_DIR.glob("part*.parquet"))
    assert n == len(df), f"{n} rows in parts vs {len(df)} expected"
    print("score2 complete", n, "rows in", S2X_DIR, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2:] or list(FOLDS))
    elif cmd == "score":
        score1()
        score2()
    else:
        {"score1": score1, "score2": score2}[cmd]()
