"""plan v2 B1: how many true pairs of state-less S2/S3 records does the country-wide name search recover at each K?

Only the no-state search is run (char view + word view, as in the wide blocking tags), on the train split with
the hard S1 subset. Prints pair recall and candidate counts per K, overall and for the validation S1 bucket.
usage: sweep_nostate_k.py [K ...]
"""
import sys
import time

import polars as pl

from blocking import hard_keep, infer_missing_states, search_name, search_name_word, with_blocks
from config import WORK

ks = [int(k) for k in sys.argv[1:]] or [5, 20, 50, 100]
kmax = max(ks)
s1 = pl.read_parquet(WORK / "train_s1_norm.parquet").filter(hard_keep())
other = pl.concat([pl.read_parquet(WORK / f"train_s{i}_norm.parquet") for i in (2, 3)])
other = infer_missing_states("train", s1, other)
truth = pl.read_parquet(WORK / "train_pairs.parquet").join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
parts = []
for country in s1["country"].unique().sort().to_list():
    t = time.time()
    s1c = s1.filter(pl.col("country") == country).unique("entity_id", keep="first")
    ns = other.filter((pl.col("country") == country) & (pl.col("state") == "")).unique("entity_id")
    for fn in (search_name, search_name_word):
        r = fn(s1c, ns, kmax)
        # rank within each query record so every smaller K is a prefix of the K=kmax result
        parts.append(r.with_columns(pl.col("sim_name_blk").rank("ordinal", descending=True).over("other").alias("rk")))
    print(country, "S1", len(s1c), "no-state records", len(ns), f"{time.time() - t:.0f}s", flush=True)
res = pl.concat(parts)
ns_ids = other.filter(pl.col("state") == "").select(pl.col("entity_id").alias("other"))
t_ns = truth.join(ns_ids, on="other", how="semi")
val = pl.col("s1").hash(7) % 10 == 9
print(f"true pairs whose S2/S3 record has no state: {len(t_ns)} (validation bucket {t_ns.filter(val).height})")
for k in sorted(ks):
    c = res.filter(pl.col("rk") <= k).select("s1", "other").unique()
    hit = t_ns.join(c, on=["s1", "other"], how="semi")
    print(f"K={k:>3}  candidates {len(c):>9}  per record {len(c) / max(len(ns_ids), 1):.1f}  "
          f"pair recall {len(hit) / len(t_ns):.4f}  validation {hit.filter(val).height / max(t_ns.filter(val).height, 1):.4f}",
          flush=True)
