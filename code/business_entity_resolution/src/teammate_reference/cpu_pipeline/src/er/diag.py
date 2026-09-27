"""Label-free per-country profile of a matching_results.tsv (compare France with US/India).

The generator gives every country the same match-count law (train: 5.58% singletons, 3.46 matches per S1), so a
well-calibrated France output should look like US/India: empty share ~5.6%, mean matches ~3.4.

  python -m er.diag output/run2/matching_results.tsv [other.tsv ...]
"""
import sys

import polars as pl

from er.common import CACHE


def profile(path) -> pl.DataFrame:
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": c1})
    m = m.join(cty, on=c1).with_columns(pl.col(c2).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
    return m.group_by("country").agg(pl.len().alias("n_s1"), pl.col("k").mean().round(3).alias("mean_k"),
                                     (pl.col("k") == 0).mean().round(4).alias("empty"),
                                     (pl.col("k") >= 6).mean().round(4).alias("k6plus")).sort("country")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        print(profile(p))
