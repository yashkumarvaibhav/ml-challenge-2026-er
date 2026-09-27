"""France filler normalization: build a France-only test split ("testfr") whose names drop the generator's French filler
words (fils, freres, associes, developpement, groupe, distribution, france, services, service) on both sides.

The US/India-trained model reads an added filler word as a name mismatch: France has 26.5k rejected candidate pairs
whose names differ only by fillers (18% of French records carry one). No two French S1s at the same address differ
only by fillers, so stripping them merges no real businesses. Candidates are the France rows of the test candidates;
feature/score steps then run with --split testfr and their France scores replace the test ones.

  python -m er.frfill
"""
import shutil

import polars as pl

from er.common import CACHE, log
from er.france_post import FILLER

SPLIT = "testfr"


def main():
    fr = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France")
    for s in (1, 2, 3):
        d = pl.read_parquet(CACHE / "norm_final" / f"test_s{s}.parquet").filter(pl.col("country") == "France")
        t = pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(FILLER)))
        d = d.with_columns(pl.when(t.list.len() > 0).then(t).otherwise(pl.col("n_tok")).alias("n_tok"))
        d = d.with_columns(pl.col("n_tok").list.join(" ").alias("n_str"))
        d.write_parquet(CACHE / "norm_final" / f"{SPLIT}_s{s}.parquet")
        log(f"frfill: {SPLIT}_s{s} {d.height:,} France rows")
    c = pl.read_parquet(CACHE / "cand" / "b2_test_w20.parquet").join(fr.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
    c.write_parquet(CACHE / "cand" / f"b2_{SPLIT}_w20.parquet")
    (CACHE / "addon" / SPLIT).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CACHE / "addon" / "test" / "idf.npz", CACHE / "addon" / SPLIT / "idf.npz")
    log(f"frfill: {c.height:,} France candidate pairs")


if __name__ == "__main__":
    main()
