"""Read-only structural audit of the archived #27-16 TSVs."""
import json
from pathlib import Path

import polars as pl

R = Path(__file__).resolve().parents[1]
A = R / "submissions/archive/sub-2026-09-27-16"


def pairs(path):
    rows = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    key, values = rows.columns
    pairs_ = (
        rows.select(pl.col(key).alias("s1"), pl.col(values).fill_null("").str.split(",").alias("m"))
        .explode("m")
        .filter(pl.col("m") != "")
    )
    return rows, pairs_


def sample(frame, n=5):
    return frame.head(n).to_dicts()


def main():
    match_rows, match = pairs(A / "matching_results.tsv")
    cand_rows, cand = pairs(A / "candidate_pairs.tsv")
    out = {"matching_rows": match_rows.height, "matching_pairs": match.height,
           "candidate_rows": cand_rows.height, "candidate_pairs": cand.height}
    for name, frame in [("matching", match), ("candidate", cand)]:
        dups = frame.group_by("s1", "m").len().filter(pl.col("len") > 1)
        out[f"{name}_duplicate_pairs"] = dups.height
        out[f"{name}_duplicate_pair_samples"] = sample(dups)
    absent = match.join(cand, on=["s1", "m"], how="anti")
    out["matches_absent_from_candidates"] = absent.height
    out["matches_absent_from_candidates_samples"] = sample(absent)

    owners = match.group_by("m").agg(pl.len().alias("n"), pl.col("s1").head(3).alias("owners"))
    mult = owners.filter(pl.col("n") > 1)
    out["multiply_owned_records"] = mult.height
    out["extra_ownership_claims"] = int((mult["n"] - 1).sum()) if mult.height else 0
    out["multiply_owned_samples"] = sample(mult)

    caps = (
        match.with_columns(pl.col("m").str.starts_with("S2-").alias("is_s2"))
        .group_by("s1")
        .agg(pl.col("is_s2").sum().alias("s2"), (~pl.col("is_s2")).sum().alias("s3"))
    )
    bad_caps = caps.filter((pl.col("s2") > 5) | (pl.col("s3") > 6))
    out["s1_over_source_caps"] = bad_caps.height
    out["source_cap_samples"] = sample(bad_caps)
    out["maximum_s2_per_s1"] = caps["s2"].max()
    out["maximum_s3_per_s1"] = caps["s3"].max()

    s1 = pl.read_parquet(R / "cache/test_source1.parquet", columns=["entity_id", "country"]).rename(
        {"entity_id": "s1", "country": "country_s1"}
    )
    target = pl.concat([
        pl.read_parquet(R / "cache/test_source2.parquet", columns=["entity_id", "country"]),
        pl.read_parquet(R / "cache/test_source3.parquet", columns=["entity_id", "country"]),
    ]).rename({"entity_id": "m", "country": "country_m"})
    joined = match.join(s1, on="s1", how="left").join(target, on="m", how="left")
    bad_country = joined.filter(pl.col("country_s1") != pl.col("country_m"))
    missing_country = joined.filter(pl.col("country_s1").is_null() | pl.col("country_m").is_null())
    out["country_mismatch_pairs"] = bad_country.height
    out["country_mismatch_samples"] = sample(bad_country)
    out["missing_country_join_pairs"] = missing_country.height
    out["missing_country_samples"] = sample(missing_country)
    out["matching_country_counts"] = joined.group_by("country_s1").len().sort("country_s1").to_dicts()
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
