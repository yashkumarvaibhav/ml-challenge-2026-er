"""Replace the France rows of output/matching_results.tsv with those of another matching file (sub-2026-09-27-5:
US/India from the stage-3 decisions, France exactly as sub-2026-09-27-2). Row order and all other rows are kept.

  python -m ber.final_s3 --france-from cache/v6/final_2702_matching.tsv
"""
import argparse

import polars as pl

from . import config as C


def others(src: str):
    """Keep the France rows of output/, take every non-France row from another matching file."""
    out = C.OUTPUT / "matching_results.tsv"
    new = pl.read_csv(out, separator="\t", quote_char=None, infer_schema_length=0)
    old = pl.read_csv(C.ROOT / src, separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = new.columns
    cty = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias(c1), "country")
    fr = new.join(cty, on=c1).filter(pl.col("country") == "France").select(c1, pl.col(c2).fill_null("").alias("_fr"), pl.lit(True).alias("_isfr"))
    res = (old.join(fr, on=c1, how="left")
           .with_columns(pl.when(pl.col("_isfr")).then(pl.col("_fr")).otherwise(pl.col(c2)).fill_null("").alias(c2)).select(c1, c2))
    assert res[c1].to_list() == old[c1].to_list()
    res.write_csv(out, separator="\t", quote_style="never")
    print(f"non-France rows taken from {src}")


def main(src: str):
    out = C.OUTPUT / "matching_results.tsv"
    new = pl.read_csv(out, separator="\t", quote_char=None, infer_schema_length=0)
    old = pl.read_csv(C.ROOT / src, separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = new.columns
    cty = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias(c1), "country")
    fr = old.join(cty, on=c1).filter(pl.col("country") == "France").select(c1, pl.col(c2).fill_null("").alias("_fr"), pl.lit(True).alias("_isfr"))
    res = (new.join(fr, on=c1, how="left")
           .with_columns(pl.when(pl.col("_isfr")).then(pl.col("_fr")).otherwise(pl.col(c2)).fill_null("").alias(c2)).select(c1, c2))
    assert res[c1].to_list() == new[c1].to_list()
    res.write_csv(out, separator="\t", quote_style="never")
    print(f"France rows replaced from {src}: {fr.height:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--france-from")
    ap.add_argument("--others-from")
    a = ap.parse_args()
    main(a.france_from) if a.france_from else others(a.others_from)
