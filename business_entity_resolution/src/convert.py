import polars as pl
from config import DATA, WORK


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def main():
    for split in ["train", "test"]:
        for s in [1, 2, 3]:
            df = read_tsv(DATA / split / f"{split}_source{s}.tsv").fill_null("")
            df.write_parquet(WORK / f"{split}_s{s}.parquet")
            print(split, s, df.shape)
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv").fill_null("")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "")
               .rename({"source1_entity_id": "s1", "matched_entity_ids": "other"}))
    pairs.write_parquet(WORK / "train_pairs.parquet")
    print("gt pairs", pairs.shape)


if __name__ == "__main__":
    main()
