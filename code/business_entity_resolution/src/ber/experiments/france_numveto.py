"""Reproduce the scored #27-11 France rows from the team's number veto and filler rescue.

The small exact pair lists are derived solely from supplied test records by the team
implementation preserved in ``teammate_reference``.  They are included in the final
archive.  The standard stage-2 score caps the veto at p <= 0.999, as used for #27-11.
"""
import argparse
import os
from pathlib import Path

import polars as pl

from ber import config as C


def pairs(path: Path) -> pl.DataFrame:
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    a, b = d.columns
    return (d.with_columns(pl.col(b).fill_null("").str.split(",")).explode(b)
            .filter(pl.col(b) != "").rename({a: "s1", b: "m"}).select("s1", "m"))


def extract_team_lists(france_base: Path, teammate_output: Path, teammate_drops: Path, scores: Path,
                       drop_dest: Path, add_dest: Path) -> None:
    """Freeze the exact team-generated pair lists used for the public #27-11 result."""
    old, team = pairs(france_base), pairs(teammate_output)
    raw_drop = pl.read_parquet(teammate_drops).select("s1", "m").unique()
    actual_drop = old.join(team, on=["s1", "m"], how="anti")
    assert actual_drop.height == raw_drop.height == 9547
    assert actual_drop.join(raw_drop, on=["s1", "m"], how="anti").is_empty()
    scored = (raw_drop.lazy().join(pl.scan_parquet(scores).select("s1", "m", "p"),
                                   on=["s1", "m"], how="left").collect()
              .with_columns(pl.col("p").fill_null(0.0)))
    drop = scored.filter(pl.col("p") <= .999).select("s1", "m").sort("s1", "m")
    add = team.join(old, on=["s1", "m"], how="anti").sort("s1", "m")
    assert drop.height == 7367 and add.height == 339
    for path, table in ((drop_dest, drop), (add_dest, add)):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(path)
    print(f"#27-11 exact France lists: {drop.height} drops, {add.height} adds")


def write(us_base: Path, france_base: Path, drop_path: Path, add_path: Path, out: Path) -> None:
    """Replace France rows of US/India stage-3 v4 output; preserve all row and ID order."""
    m9 = pl.read_csv(us_base, separator="\t", quote_char=None, infer_schema_length=0)
    m7 = pl.read_csv(france_base, separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m9.columns
    assert m7.columns == m9.columns
    fr = (pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"])
          .filter(pl.col("country") == "France").select(pl.col("entity_id").alias(c1)))
    drop = pl.read_parquet(drop_path).select("s1", "m")
    add = pl.read_parquet(add_path).select("s1", "m")
    assert drop.height == 7367 and add.height == 339
    base_pairs = pairs(france_base).join(fr.rename({c1: "s1"}), on="s1")
    assert base_pairs.join(drop, on=["s1", "m"], how="inner").height == drop.height
    assert base_pairs.join(add, on=["s1", "m"], how="inner").is_empty()
    kept = pl.concat([base_pairs.join(drop, on=["s1", "m"], how="anti"), add])
    g = kept.group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias("_fr"))
    new_fr = (m7.select(c1).join(fr, on=c1).join(g.rename({"s1": c1}), on=c1, how="left")
              .with_columns(pl.col("_fr").fill_null("")))
    result = (m9.join(new_fr.select(c1, "_fr").with_columns(pl.lit(True).alias("_is_fr")), on=c1, how="left")
              .with_columns(pl.when(pl.col("_is_fr")).then(pl.col("_fr")).otherwise(pl.col(c2))
                            .fill_null("").alias(c2)).select(c1, c2))
    assert result[c1].to_list() == m9[c1].to_list()
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(out, separator="\t", quote_style="never")
    print(f"#27-11 output: {out}; France −{drop.height}/+{add.height}; US/India rows preserved")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-team", action="store_true")
    ap.add_argument("--france-from", default="cache/v6/final_2707_matching.tsv")
    ap.add_argument("--us-base", default="output/matching_results.tsv")
    ap.add_argument("--teammate-output", default="pinaksh/sub7_fr/matching_results.tsv")
    ap.add_argument("--teammate-drops", default="pinaksh/sub7_fr/france_numveto_dropped.parquet")
    ap.add_argument("--scores", default="cache/v6/s2/test_pred_e5base2.parquet")
    ap.add_argument("--drops", default=os.environ.get("BER_FR_DROPS", "cache/v6/s2/france_numveto_2711_drops.parquet"))
    ap.add_argument("--adds", default=os.environ.get("BER_FR_ADDS", "cache/v6/s2/france_filler_2711_adds.parquet"))
    ap.add_argument("--out", default="output/matching_results.tsv")
    a = ap.parse_args()
    if a.extract_team:
        extract_team_lists(Path(a.france_from), Path(a.teammate_output), Path(a.teammate_drops), Path(a.scores),
                           Path(a.drops), Path(a.adds))
    else:
        write(Path(a.us_base), Path(a.france_from), Path(a.drops), Path(a.adds), Path(a.out))


if __name__ == "__main__":
    main()
