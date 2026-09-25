import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids) -> float:
    """pred/truth: long frames (s1, other). Macro F0.5 over s1_ids, singletons included."""
    s1 = pl.DataFrame({"s1": list(s1_ids)})
    p = pred.select("s1", "other").unique()
    t = truth.select("s1", "other").unique()
    tp = p.join(t, on=["s1", "other"]).group_by("s1").len("tp")
    np_ = p.group_by("s1").len("np")
    nt = t.group_by("s1").len("nt")
    d = (s1.join(np_, on="s1", how="left").join(nt, on="s1", how="left").join(tp, on="s1", how="left")
           .fill_null(0))
    prec = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
    rec = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
    d = d.with_columns(prec.alias("p"), rec.alias("r"))
    f = (pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r"))))
    return d.select(f.mean()).item()
