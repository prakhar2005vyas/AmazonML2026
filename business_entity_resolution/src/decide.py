"""Per-S1 decision rule that maximises expected F0.5 instead of applying one global threshold.

Each S2/S3 record goes to its highest-scoring S1. For every S1 the owned candidates are sorted by q, and the
chosen set is the prefix (possibly empty) with the highest plug-in expected F0.5:
    empty set : P(no match)  ~ prod(1 - q_i) * (1 - m)   (m: chance a match exists outside the candidates)
    top-k     : 1.25 TP / (1.25 TP + 0.25 FN + FP) with TP = sum of the k q's, FN = remaining q's + m
"""
import sys

import polars as pl

from config import WORK
from metrics import macro_f05


def expected_f_decide(scored, m=0.02, power=1.0, floor=0.02):
    q = pl.col("q") ** power
    own = (scored.sort("q", descending=True).unique("other", keep="first")
                 .filter(pl.col("q") >= floor).with_columns(q.alias("qq"))
                 .sort(["s1", "qq"], descending=[False, True]))
    own = own.with_columns(
        pl.col("qq").cum_sum().over("s1").alias("ctp"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
        pl.col("qq").sum().over("s1").alias("tot"),
        (1 - pl.col("qq")).clip(1e-9, 1).log().sum().over("s1").exp().alias("p0"),
    ).with_columns(
        (1.25 * pl.col("ctp") / (1.25 * pl.col("ctp") + 0.25 * (pl.col("tot") - pl.col("ctp") + m)
                                 + (pl.col("k") - pl.col("ctp")))).alias("ef"))
    best = own.group_by("s1").agg(pl.col("ef").max().alias("ef_best"), pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"),
                                  pl.col("p0").first())
    keep = best.filter(pl.col("ef_best") > pl.col("p0") * (1 - m))
    return own.join(keep, on="s1").filter(pl.col("k") <= pl.col("k_best")).select("s1", "other", "q")


if __name__ == "__main__":
    from stage2 import val_ids
    sfx = f"_{sys.argv[1]}" if len(sys.argv) > 1 else "_hard"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    oof = pl.read_parquet(WORK / f"val{sfx}_oof{tag}.parquet")
    ids = val_ids(sfx)
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("s1").is_in(ids.implode()))
    owner = oof.sort("q", descending=True).unique("other", keep="first").filter(pl.col("s1").is_in(ids.implode()))
    for thr in (0.6, 0.65, 0.7, 0.75):
        print("global thr", thr, round(macro_f05(owner.filter(pl.col("q") >= thr), truth, ids), 5))
    sub = oof.join(oof.filter(pl.col("s1").is_in(ids.implode())).select("other").unique(), on="other", how="semi")
    for power in (1.0, 1.5, 2.0):
        for m in (0.0, 0.02, 0.05, 0.1):
            pred = expected_f_decide(sub, m=m, power=power).filter(pl.col("s1").is_in(ids.implode()))
            print(f"expF power {power} m {m}  F0.5 {macro_f05(pred, truth, ids):.5f}", flush=True)
