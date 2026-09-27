"""Post-processing of accepted test pairs.

Category-word-swap veto (from the team's sub-2026-09-26-8, +0.003063 public): per country, from that country's own
accepted pairs, a *category word* is one that records substitute for an S1 name word often (>= 5 times) but almost
never add as a pure extra word (< 5% of that). Such a substitution marks a sibling business ("Socio Pharmacie" vs
"Socio Amis"), not generator noise, so the pair is dropped. Label-free; on the US/India holdout it removes ~0 pairs.
"""
import polars as pl

from ber.experiments.france_category_swap import category_words, flag
from er.common import CACHE, log


def raw_names(split):
    t1 = pl.read_parquet(CACHE / f"{split}_source1.parquet").select(
        pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), "country")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_source{s}.parquet") for s in (2, 3)]).select(
        pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"))
    return t1, rec


def category_veto(pred: pl.DataFrame, split: str = "test", countries=("France",)) -> tuple:
    """pred: accepted (s1, m). Returns (kept pairs, dropped pairs, report)."""
    t1, rec = raw_names(split)
    acc = pred.join(t1, on="s1").join(rec, on="m")
    drops, rep = [], {}
    for cty in countries:
        a = acc.filter(pl.col("country") == cty)
        if a.height == 0:
            continue
        cats = category_words(a, t1.filter(pl.col("country") == cty)["n1"].to_list())
        a = a.with_columns(flag(a, cats))
        d = a.filter(pl.col("swap")).select("s1", "m")
        drops.append(d)
        rep[cty] = {"n_category_words": len(cats), "accepted": a.height, "dropped": d.height,
                    "words": sorted(cats)[:80]}
        log(f"category veto {cty}: {len(cats)} words, dropped {d.height:,} of {a.height:,} accepted pairs")
    dropped = pl.concat(drops) if drops else pred.head(0)
    return pred.join(dropped, on=["s1", "m"], how="anti"), dropped, rep
