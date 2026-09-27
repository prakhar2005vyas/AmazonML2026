"""Break validation F0.5 loss down by error type and dump examples. usage: analyze_errors.py [tag] [thr]"""
import sys

import polars as pl

from blocking import hard_keep
from config import WORK

sfx = f"_{sys.argv[1]}" if len(sys.argv) > 1 and sys.argv[1] else ""
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6

src = sys.argv[3] if len(sys.argv) > 3 else f"val{sfx}_scored.parquet"
va = pl.read_parquet(WORK / src)
if "q" in va.columns:
    va = va.select("s1", "other", pl.col("q").alias("p"))
s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
mask = pl.col("s1").hash(7) % 10 == 9
if sfx.startswith("_hard"):
    mask = mask & hard_keep("s1")
val_ids = s1_all.filter(mask)
truth = pl.read_parquet(WORK / "train_pairs.parquet").join(val_ids, on="s1", how="semi")
# the true owner of every S2/S3 record that has one (any train S1, since competition spans all S1s)
all_truth = pl.read_parquet(WORK / "train_pairs.parquet")

owner = va.sort("p", descending=True).unique("other", keep="first")
pred = owner.filter(pl.col("p") >= thr).join(val_ids, on="s1", how="semi")

tp = pred.join(truth, on=["s1", "other"], how="semi")
fp = pred.join(truth, on=["s1", "other"], how="anti")
fn = truth.join(pred, on=["s1", "other"], how="anti")

n = val_ids.height
per = (val_ids.join(pred.group_by("s1").len("np"), on="s1", how="left")
       .join(truth.group_by("s1").len("nt"), on="s1", how="left")
       .join(tp.group_by("s1").len("tp"), on="s1", how="left").fill_null(0))
per = per.with_columns(
    pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0).alias("P"),
    pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0).alias("R"))
per = per.with_columns(
    pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
    .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))).alias("F"))
print(f"thr {thr}  val S1 {n}  macro F0.5 {per['F'].mean():.4f}")
per = per.with_columns((1 - pl.col("F")).alias("loss"))
cat = (per.with_columns(
    pl.when(pl.col("loss") == 0).then(pl.lit("ok"))
    .when(pl.col("nt") == 0).then(pl.lit("singleton_fp"))
    .when(pl.col("np") == 0).then(pl.lit("missed_all"))
    .when((pl.col("np") > pl.col("tp")) & (pl.col("tp") < pl.col("nt"))).then(pl.lit("fp_and_fn"))
    .when(pl.col("np") > pl.col("tp")).then(pl.lit("fp_only"))
    .otherwise(pl.lit("fn_only")).alias("cat"))
    .group_by("cat").agg(pl.len().alias("n"), (pl.col("loss").sum() / n).alias("F_lost")).sort("F_lost", descending=True))
print(cat)

# why were true pairs missed?
cand_hit = fn.join(va, on=["s1", "other"], how="left")
own = owner.select("other", pl.col("s1").alias("owner_s1"), pl.col("p").alias("owner_p"))
fn_why = cand_hit.join(own, on="other", how="left").with_columns(
    pl.when(pl.col("p").is_null()).then(pl.lit("not_in_candidates"))
    .when(pl.col("owner_s1") != pl.col("s1")).then(pl.lit("stolen_by_other_s1"))
    .otherwise(pl.lit("below_thr")).alias("why"))
print("FN pairs", len(fn), fn_why.group_by("why").len().sort("len", descending=True))

fp_why = fp.join(all_truth.rename({"s1": "true_s1"}), on="other", how="left").with_columns(
    pl.when(pl.col("true_s1").is_null()).then(pl.lit("other_is_distractor"))
    .otherwise(pl.lit("other_belongs_elsewhere")).alias("why"))
print("FP pairs", len(fp), fp_why.group_by("why").len().sort("len", descending=True))

# plan v2 E1: the same breakdown restricted to S2/S3 records with a given trait, so each plan item has a
# measured target size before and after it lands
oth = pl.concat([pl.read_parquet(WORK / f"train_s{i}.parquet", columns=["entity_id", "business_name", "business_address"])
                 for i in (2, 3)])
st = pl.read_parquet(WORK / "train_other_state.parquet")
addr = pl.col("business_address").fill_null("")
traits = (oth.join(st, on="entity_id", how="left").select(
    pl.col("entity_id").alias("other"),
    pl.col("business_name").fill_null("").str.contains("[ऀ-෿]").alias("native_name"),
    (addr.str.strip_chars() == "").alias("addr_empty"),
    addr.str.contains(r"(?i)\b(?:\d+[a-z]{4,}|[a-z]{3,}\d+)").alias("glued_token"),
    ((pl.col("state").fill_null("") == "") & (addr.str.strip_chars() != "")).alias("addr_but_no_state")))
del oth
rows = []
for trait in ["all", "native_name", "addr_empty", "glued_token", "addr_but_no_state"]:
    keep = traits.select("other") if trait == "all" else traits.filter(pl.col(trait)).select("other")
    t_n = truth.join(keep, on="other", how="semi").height
    f = fn_why.join(keep, on="other", how="semi")
    by = dict(f.group_by("why").len().iter_rows())
    fp_n = fp.join(keep, on="other", how="semi").height
    nc = by.get("not_in_candidates", 0)
    rows.append((trait, t_n, len(f), nc, by.get("below_thr", 0), by.get("stolen_by_other_s1", 0), fp_n,
                 round(1 - nc / t_n, 4) if t_n else None, round(len(f) / t_n, 4) if t_n else None))
with pl.Config(tbl_rows=10, tbl_width_chars=200):
    print("--- E1 slices (pairs whose S2/S3 record has the trait)")
    print(pl.DataFrame(rows, orient="row", schema=["trait", "truth", "fn", "fn_not_cand", "fn_below_thr", "fn_stolen",
                                                    "fp", "block_recall", "miss_rate"]))

raw = pl.concat([pl.read_parquet(WORK / f"train_s{i}.parquet", columns=["entity_id", "business_name", "business_address"])
                 for i in (1, 2, 3)])
nm = lambda c: raw.rename({"entity_id": c, "business_name": c + "_name", "business_address": c + "_addr"})
with pl.Config(tbl_rows=40, fmt_str_lengths=70, tbl_width_chars=250):
    print("--- sample FPs")
    print(fp_why.sample(min(25, len(fp_why)), seed=1).join(nm("s1"), on="s1").join(nm("other"), on="other")
          .select("p", "why", "s1_name", "other_name", "s1_addr", "other_addr"))
    print("--- sample FNs below thr / stolen")
    print(fn_why.filter(pl.col("why") != "not_in_candidates").sample(25, seed=1, with_replacement=True).unique()
          .join(nm("s1"), on="s1").join(nm("other"), on="other")
          .select("p", "owner_p", "why", "s1_name", "other_name", "s1_addr", "other_addr"))
    print("--- sample FNs not in candidates")
    print(fn_why.filter(pl.col("why") == "not_in_candidates").sample(20, seed=1, with_replacement=True).unique()
          .join(nm("s1"), on="s1").join(nm("other"), on="other")
          .select("s1_name", "other_name", "s1_addr", "other_addr"))
