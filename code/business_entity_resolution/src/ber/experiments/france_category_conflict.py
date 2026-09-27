"""Broad France category conflict (27 Sep): an accepted France pair whose record carries a category word the S1 name
does not have, while the S1's own category word is absent from the record -- regardless of other noise (typos, dotted
legal forms 'S.A.S.U.', extra fillers). This catches siblings the one-word-substitution rule (france_category_swap)
misses. Filler words (services, france, fils, associes, developpement, groupe, cie, et) never count as categories;
word families (sport/sportive/sportif, culture/culturel/culturelle, ...) share a 5-letter prefix.
Label-free check: share of pairs whose S1 has no other accepted record = 6.9% (1,188 pairs) vs 2.3% for other
accepted France pairs and 7.05% for the public-confirmed category-swap drops. Applied together with
france_small_drops (dotted-form swaps + consensus) as one France drop set: 1,263 pairs, one-record share 7.2%.

  python -m ber.experiments.france_category_conflict --write BASE   # output/ = BASE minus these France pairs
"""
import argparse
import json
import re
import shutil
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from .france_small_drops import drops as small_drops

EXTRA = {"federation", "association", "fondation", "institution", "syndicat", "cooperative", "entente", "ligue", "cercle", "groupement"}
FILL = {"services", "france", "fils", "associes", "developpement", "groupe", "cie", "et"}


def _cats():
    return set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"]) | EXTRA


def toks(s):
    if s is None:
        return []
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b\.?)", "", s).replace(".", "")
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def _catset(t, cats):
    out = set()
    for x in t:
        if x in FILL:
            continue
        for c in cats:
            if x == c or (len(x) >= 5 and fuzz.ratio(x, c) >= 88):
                out.add(c[:5])
    return out


def conflict(n1, n2, cats) -> bool:
    t1, t2 = toks(n1), toks(n2)
    c1 = _catset(t1, cats)
    if not c1:
        return False
    c2, comp2 = _catset(t2, cats), "".join(t2)
    return bool(c2 - c1) and not any(c in c2 or c in comp2 for c in c1)


def drops(pairs: pl.DataFrame) -> pl.DataFrame:
    cats = _cats()
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(
        pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{k}.parquet") for k in (2, 3)]).select(
        pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"))
    a = pairs.join(t1, on="s1").join(rec, on="m")
    cc = a.filter(pl.Series([conflict(x, y, cats) for x, y in zip(a["n1"], a["n2"])])).select("s1", "m")
    return pl.concat([cc, small_drops(pairs)]).unique()


def write(base: str):
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
    print(f"{base}: France category-conflict + small drops {d.height:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", required=True)
    write(ap.parse_args().write)
