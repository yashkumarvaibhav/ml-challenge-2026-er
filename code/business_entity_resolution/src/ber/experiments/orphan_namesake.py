"""Orphan-namesake drop rule (subagent known-country hunt, experiments/v6/subagent_known_country_hunt.md).

An 'orphan' is a record with an address whose best stage-1 score at any world S1 is < 0.5. Two or more orphans
sharing a core name (per country) mean a same-name entity outside S1 (ghost/distractor) probably exists. Drop an
accepted ADDRESS-NULL pair if its stage-2 p < P_MAX and >= MIN_ORPH orphans share the S1's or the record's core name.
Holdout (US/India): w20 +0.000161 (A->B +0.000169, B->A +0.000153), test-like ghost world +0.000214, w0 -0.000047.

  python -m ber.experiments.orphan_namesake --size             # sizes on test by country (+ France k1-share check)
  python -m ber.experiments.orphan_namesake --write BASE [--countries US,India]
"""
import argparse
import shutil

import polars as pl

from .. import config as C

LEG = ["llc", "inc", "ltd", "corp", "co", "pvt", "private", "limited", "incorporated", "corporation", "company", "llp", "pc", "plc", "lp",
       "sarl", "sas", "sasu", "sa", "eurl", "ei", "sci", "snc"]
P_MAX, MIN_ORPH = 0.90, 2


def core(c):
    return (pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}0-9 ]", " ").str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "")).list.set_difference(pl.lit(LEG)).list.sort().list.join(" "))


def drops(acc: pl.DataFrame) -> pl.DataFrame:
    """acc: accepted (s1, m) on test. Returns the dropped (s1, m)."""
    s1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), "country", core("business_name").alias("k1"))
    recs = pl.concat([pl.read_parquet(C.CACHE / f"test_source{k}.parquet") for k in (2, 3)]).select(
        pl.col("entity_id").alias("m"), "country", core("business_name").alias("km"), pl.col("business_address").is_null().alias("anull"))
    st1 = pl.read_parquet(C.WORK / "pred" / "test.parquet")
    pm = st1.group_by("m").agg(pl.col("p").max().alias("pm"))
    r = recs.join(pm, on="m", how="left").with_columns(pl.col("pm").fill_null(0))
    orph = r.filter(~pl.col("anull")).group_by("country", "km").agg((pl.col("pm") < 0.5).sum().alias("o5"))
    p2 = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2.parquet")
    a = acc.join(p2, on=["s1", "m"], how="left").join(s1, on="s1").join(recs.select("m", "km", "anull"), on="m")
    a = (a.join(orph.rename({"km": "k1", "o5": "s_o5"}), on=["country", "k1"], how="left")
          .join(orph.rename({"o5": "m_o5"}), on=["country", "km"], how="left")
          .with_columns(pl.max_horizontal(pl.col("s_o5").fill_null(0), pl.col("m_o5").fill_null(0)).alias("o")))
    return a.filter(pl.col("anull") & (pl.col("o") >= MIN_ORPH) & (pl.col("p").fill_null(0) < P_MAX)).select("s1", "m", "country", "p")


def _pairs(base):
    arch = C.OUTPUT if base == "output" else C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    return arch, m, (m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "").rename({c1: "s1", c2: "m"}))


def size(base="sub-2026-09-27-2"):
    _, _, pairs = _pairs(base)
    d = drops(pairs)
    print(d.group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("S1")).sort("country"))
    fr = d.filter(pl.col("country") == "France")
    k = pairs.group_by("s1").len().rename({"len": "k"})
    h = fr.join(k, on="s1").filter(pl.col("p") >= 0.95)
    print("France drops with p>=0.95:", h.height, " (rule requires p<0.9, so k1-share is n/a); p quantiles:",
          [round(fr["p"].quantile(q), 3) for q in (0.1, 0.5, 0.9)] if fr.height else None)
    kk = fr.join(k, on="s1")
    print("France drops whose S1 has only this record:", (kk["k"] == 1).mean() if kk.height else None)


def write(base, countries):
    arch, m, pairs = _pairs(base)
    c1, c2 = m.columns
    d = drops(pairs).filter(pl.col("country").is_in(countries)).select("s1", "m")
    kept = pairs.join(d, on=["s1", "m"], how="anti")
    g = kept.group_by("s1").agg(pl.col("m").sort().str.join(",").alias(c2)).rename({"s1": c1})
    out = m.select(c1).join(g, on=c1, how="left").with_columns(pl.col(c2).fill_null(""))
    assert out[c1].to_list() == m[c1].to_list()
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    if arch != C.OUTPUT:
        shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    print(f"{base}: orphan-namesake drops in {countries}: {d.height:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", action="store_true")
    ap.add_argument("--write")
    ap.add_argument("--countries", default="US,India")
    a = ap.parse_args()
    if a.write:
        write(a.write, a.countries.split(","))
    else:
        size()
