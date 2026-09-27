"""Exact #27-11 -> #27-19 final assembly from the shipped pair lists (the audited, uploaded best submission).

The steps s12a ... s19 in this package produced #27-19 from #27-11; they read large intermediate score files from the
pipeline cache (stage-2 p2, stage-3 v4 p3, LLM band scores) and the teammate's output. To make the final file
reproducible without re-running those steps, the net effect of all of them is frozen as two small pair lists:

  artifacts/final2719/final2719_drops.parquet   pairs in #27-11 that are not in #27-19
  artifacts/final2719/final2719_adds.parquet    pairs in #27-19 that are not in #27-11

  # apply (default): #27-11 matching_results.tsv -> #27-19 matching_results.tsv, then SHA-256 check
  python -m ber.final2719.apply_2719 --base output/matching_results.tsv --out output/matching_results.tsv
  # how the lists were extracted (audit): diff of the two uploaded files
  python -m ber.final2719.apply_2719 --extract --base <#27-11 tsv> --final <#27-19 tsv>

Row order (all S1 ids of test_source1, in the base file's order) is preserved; each row's ids are sorted and
comma-joined, exactly as every step script writes them.
"""
import argparse
import hashlib
from pathlib import Path

import polars as pl

ART = Path(__file__).resolve().parents[3] / "artifacts" / "final2719"
SHA_2711 = "186f4d1f85deba7ab6b5bbffdc2e535cd2cf26318cc2dabfda6e4a2d9d95e8de"
SHA_2719 = "720dce1652fc585ed3569dcddd6013e5d016f8b59dd725e13ebab8319ee301e0"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p: Path):
    m = pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    pairs = (m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "")
             .rename({c1: "s1", c2: "m"}).select("s1", "m"))
    return m, pairs


def write(m: pl.DataFrame, pairs: pl.DataFrame, out: Path) -> None:
    c1, c2 = m.columns
    g = pairs.group_by("s1").agg(pl.col("m").sort().str.join(",").alias(c2)).rename({"s1": c1})
    o = m.select(c1).join(g, on=c1, how="left").with_columns(pl.col(c2).fill_null(""))
    assert o[c1].to_list() == m[c1].to_list()
    out.parent.mkdir(parents=True, exist_ok=True)
    o.write_csv(out, separator="\t", quote_style="never")


def extract(base: Path, final: Path) -> None:
    assert sha256(base) == SHA_2711, "base is not the scored #27-11 file"
    assert sha256(final) == SHA_2719, "final is not the uploaded #27-19 file"
    _, b = read(base)
    _, f = read(final)
    drops = b.join(f, on=["s1", "m"], how="anti").sort("s1", "m")
    adds = f.join(b, on=["s1", "m"], how="anti").sort("s1", "m")
    ART.mkdir(parents=True, exist_ok=True)
    drops.write_parquet(ART / "final2719_drops.parquet")
    adds.write_parquet(ART / "final2719_adds.parquet")
    print(f"#27-11 -> #27-19: {drops.height} drops, {adds.height} adds")


def apply(base: Path, out: Path) -> None:
    m, b = read(base)
    drops = pl.read_parquet(ART / "final2719_drops.parquet").select("s1", "m")
    adds = pl.read_parquet(ART / "final2719_adds.parquet").select("s1", "m")
    assert b.join(drops, on=["s1", "m"], how="inner").height == drops.height, "a listed drop is not in the base file"
    assert b.join(adds, on=["s1", "m"], how="inner").is_empty(), "a listed add is already in the base file"
    new = pl.concat([b.join(drops, on=["s1", "m"], how="anti"), adds])
    assert new.group_by("m").len()["len"].max() == 1, "a record would be matched to two S1s"
    write(m, new, out)
    got = sha256(out)
    print(f"{out}: {new.height} pairs ({drops.height} dropped, {adds.height} added), sha256 {got}")
    assert got == SHA_2719, "output differs from the uploaded #27-19 file"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="#27-11 matching_results.tsv (reproduce_final_2711.sh)")
    ap.add_argument("--out", type=Path, default=Path("output/matching_results.tsv"))
    ap.add_argument("--extract", action="store_true", help="derive the lists from the two uploaded files")
    ap.add_argument("--final", type=Path, help="#27-19 matching_results.tsv (with --extract)")
    a = ap.parse_args()
    extract(a.base, a.final) if a.extract else apply(a.base, a.out)
