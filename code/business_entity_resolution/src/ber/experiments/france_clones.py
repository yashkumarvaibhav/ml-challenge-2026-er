"""H1 sibling-clone diagnostic (cloud-agent hypothesis, experiments/v6/cloud_france_hypotheses.md).

A France S1 E is a 'clone' candidate when it has sibling evidence and NONE of its accepted records carries E's own
category word (E's name has one). Evidence: (a) E lost pairs to the category-swap/street rules (#5 -> #27-2), or
(c) a category-swap candidate of E exists (france_veto_pairs_test, p1 >= 0.005). Null model: S1s with a category word
and no sibling evidence, at the same accepted count k; clone precision ~ 1 - P(own_cat=0 | k, no sib) / P(... | sib).
"""
import json
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from .. import config as C

LEG = {"sarl", "sas", "sasu", "sa", "eurl", "ei", "sci", "snc"}


def norm_toks(s):
    if s is None:
        return []
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b\.?)", "", s).replace(".", "")
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in LEG]


def carries(cat_words, n2):
    t = norm_toks(n2)
    if not t:
        return False
    if any(fuzz.ratio(w, x) >= 80 for w in cat_words for x in t):
        return True
    comp = "".join(t)
    return any(w in comp for w in cat_words if len(w) >= 4)


def pairs(tag):
    d = pl.read_csv(C.ROOT / "submissions" / "archive" / tag / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c = d.columns
    return d.with_columns(pl.col(c[1]).fill_null("").str.split(",")).explode(c[1]).rename({c[0]: "s1", c[1]: "m"}).filter(pl.col("m") != "")


def main():
    cats = set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"])
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{k}.parquet") for k in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    best = pairs("sub-2026-09-27-2").join(t1.select("s1"), on="s1")
    old = pairs("sub-2026-09-26-5").join(t1.select("s1"), on="s1")
    lost = old.join(best, on=["s1", "m"], how="anti").select("s1").unique().with_columns(pl.lit(True).alias("sib_a"))
    veto = pl.read_parquet(C.WORK / "s2" / "france_veto_pairs_test.parquet").select("s1").unique().with_columns(pl.lit(True).alias("sib_c"))
    ecat = {s: [w for w in norm_toks(n) if w in cats] for s, n in zip(t1["s1"], t1["n1"])}
    a = best.join(rec, on="m")
    a = a.with_columns(pl.Series("own", [carries(ecat.get(s) or [], n) if ecat.get(s) else None for s, n in zip(a["s1"], a["n2"])]))
    g = a.group_by("s1").agg(pl.len().alias("k"), pl.col("own").sum().alias("own_cat"), pl.col("a2").is_null().sum().alias("n_null"))
    g = g.join(t1, on="s1")
    g = g.with_columns(pl.col("s1").map_elements(lambda s: bool(ecat.get(s)), return_dtype=pl.Boolean).alias("has_cat"))
    g = g.join(lost, on="s1", how="left").join(veto, on="s1", how="left").with_columns(
        pl.col("sib_a").fill_null(False), pl.col("sib_c").fill_null(False)).with_columns((pl.col("sib_a") | pl.col("sib_c")).alias("sib"))
    g = g.filter(pl.col("has_cat"))
    rep = {"S1_with_cat_and_accepted": g.height, "sib_S1": int(g["sib"].sum())}
    rows = []
    for k in (1, 2, 3, 4, 5):
        gk = g.filter(pl.col("k").clip(1, 5) == k)
        ns, s = gk.filter(~pl.col("sib")), gk.filter(pl.col("sib"))
        p0 = (ns["own_cat"] == 0).mean() if ns.height else None
        p1 = (s["own_cat"] == 0).mean() if s.height else None
        prec = (1 - p0 / p1) if (p0 is not None and p1) else None
        rows.append({"k": k if k < 5 else "5+", "no_sib_n": ns.height, "no_sib_owncat0": round(p0, 4) if p0 is not None else None,
                     "sib_n": s.height, "sib_owncat0": round(p1, 4) if p1 is not None else None,
                     "flagged": int((s["own_cat"] == 0).sum()), "implied_precision": round(prec, 3) if prec is not None else None})
        print(rows[-1], flush=True)
    rep["by_k"] = rows
    flagged = g.filter(pl.col("sib") & (pl.col("own_cat") == 0))
    rep["flagged_S1"] = flagged.height
    rep["flagged_pairs"] = int(flagged["k"].sum())
    print("flagged S1s", flagged.height, "pairs", int(flagged["k"].sum()))
    flagged.select("s1").write_parquet(C.WORK / "s2" / "france_clone_s1.parquet")
    (C.EXP_RUN / "france_clones.json").write_text(json.dumps(rep, indent=1))
    ex = a.join(flagged.select("s1"), on="s1").join(t1, on="s1")
    for s in flagged["s1"].sample(min(10, flagged.height), seed=1):
        e = ex.filter(pl.col("s1") == s)
        print(f"\n  S1: {e['n1'][0]}")
        for r in e.iter_rows(named=True):
            print(f"     acc: {r['n2']} | {r['a2']}")


if __name__ == "__main__":
    main()
