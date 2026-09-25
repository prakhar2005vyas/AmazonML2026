"""Score test candidates and write output/matching_results.tsv and output/candidate_pairs.tsv."""
import json

import lightgbm as lgb
import polars as pl

from config import OUTPUT, WORK
from features import FEATURES
from model import decide


def write_lists(pairs, s1_ids, col, path):
    lists = pairs.group_by("s1").agg(pl.col("other").unique().sort().str.join(",").alias(col))
    out = (s1_ids.join(lists, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
                 .rename({"s1": "source1_entity_id"}))
    out.write_csv(path, separator="\t", quote_style="never")
    print(path, "rows", len(out), "non-empty", (out[col] != "").sum())


def main():
    booster = lgb.Booster(model_file=str(WORK / "lgb.txt"))
    with open(WORK / "decision.json") as fh:
        rule = json.load(fh)
    scored = []
    for part in sorted((WORK / "test_feats").glob("*.parquet")):
        df = pl.read_parquet(part)
        scored.append(df.select("s1", "other").with_columns(
            pl.Series("p", booster.predict(df.select(FEATURES).to_numpy()))))
    scored = pl.concat(scored)
    s1_ids = pl.read_parquet(WORK / "test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    pred = decide(scored, rule["thr"], rule["rel"])
    write_lists(scored, s1_ids, "candidate_entity_ids", OUTPUT / "candidate_pairs.tsv")
    write_lists(pred, s1_ids, "matched_entity_ids", OUTPUT / "matching_results.tsv")


if __name__ == "__main__":
    main()
