"""Diagnostic public probe: an archived matching file with one country's predictions removed (rows kept, empty).

With the official per-S1 macro F0.5, an empty prediction scores 1 on a singleton S1 and 0 otherwise, so
  public(base) - public(probe) = w_c * (F_c - s_c)
(w_c: the country's share of public S1s; F_c: our F0.5 on that country; s_c: its singleton rate). With
w_France = 259,452 / 1,732,544 = 0.150 and s ≈ 0.0557 (US/India truth; one generator), the drop measures France's
actual F0.5, the quantity every France decision so far has had to assume (85/15 scenario).

  python -m ber.experiments.probe_country_empty --base sub-2026-09-26-5 --country France
writes output/matching_results.tsv (+ copies candidate_pairs.tsv unchanged) for `ber.package`.
"""
import argparse
import shutil

import polars as pl

from .. import config as C


def main(base: str, country: str):
    arch = C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    col_s1, col_m = m.columns
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"])
    drop = t1.filter(pl.col("country") == country).select(pl.col("entity_id").alias(col_s1)).with_columns(pl.lit(True).alias("_d"))
    out = m.join(drop, on=col_s1, how="left").with_columns(
        pl.when(pl.col("_d")).then(pl.lit("")).otherwise(pl.col(col_m).fill_null("")).alias(col_m)).drop("_d")
    assert out.height == m.height and out[col_s1].to_list() == m[col_s1].to_list()
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    n = out.filter(pl.col(col_m) != "").height
    print(f"{country} rows blanked: {drop.height:,}; non-empty rows {m.filter(pl.col(col_m).fill_null('') != '').height:,} -> {n:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--country", required=True)
    a = ap.parse_args()
    main(a.base, a.country)
