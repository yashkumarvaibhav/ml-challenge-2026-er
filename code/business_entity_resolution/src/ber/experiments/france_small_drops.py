"""Small France removal sets on top of the two public-confirmed rules (26-27 Sep):
  (a) extended category swaps WITHOUT 'federation' (dotted legal forms 'S.A.S.U.' collapsed, function words dropped):
      ber.experiments.france_street_diff.swap2 with the original 54 mined words; k1-share(p>=0.95) 7.1%.
  (b) consensus: standard CE (e5base2) < 0.1 AND the synthetic-trained France model (simulator 2) p < 0.3, excluding
      name-less (initials/pseudo), filler-substituted and address-null records; k1-share 4.2%.
  python -m ber.experiments.france_small_drops --write BASE
"""
import argparse
import json
import shutil

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from .france_category_swap import toks
from .france_street_diff import swap2

FILL = {"services", "service", "france", "fils", "associes", "developpement", "groupe"}


def drops(pairs: pl.DataFrame) -> pl.DataFrame:
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{k}.parquet") for k in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    cats = set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"])
    acc = pairs.join(t1, on="s1").join(rec, on="m")
    a = acc.filter(pl.Series([swap2(x, y, cats) for x, y in zip(acc["n1"], acc["n2"])])).select("s1", "m")
    ce = pl.read_parquet(C.WORK / "s2" / "ce_france_accepted_e5base2.parquet").select("s1", "m", "ce")
    fm = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2_france_model_s2.parquet").rename({"p": "pf"})
    x = acc.join(ce, on=["s1", "m"], how="left").join(fm, on=["s1", "m"], how="left").filter((pl.col("ce") < 0.1) & (pl.col("pf") < 0.3) & pl.col("a2").is_not_null())
    nameless = lambda u, v: not any(max((fuzz.ratio(p, q) for q in toks(v)), default=0) >= 75 for p in toks(u))
    fillsub = lambda u, v: bool(toks(u) and toks(v) and any(w in FILL and w not in toks(u) for w in toks(v)))
    b = x.filter(pl.Series([not nameless(u, v) and not fillsub(u, v) for u, v in zip(x["n1"], x["n2"])])).select("s1", "m")
    return pl.concat([a, b]).unique()


def write(base):
    arch = C.OUTPUT if base == "output" else C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    pairs = m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "").rename({c1: "s1", c2: "m"})
    d = drops(pairs)
    kept = pairs.join(d, on=["s1", "m"], how="anti")
    g = kept.group_by("s1").agg(pl.col("m").sort().str.join(",").alias(c2)).rename({"s1": c1})
    out = m.select(c1).join(g, on=c1, how="left").with_columns(pl.col(c2).fill_null(""))
    assert out[c1].to_list() == m[c1].to_list()
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    if arch != C.OUTPUT:
        shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    print(f"{base}: France small drops {d.height:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", required=True)
    write(ap.parse_args().write)
