"""France house-number veto.

Label-free evidence (run 2 test scores): among accepted pairs whose house numbers are fully disjoint, only 6.9% of the
French ones reach p >= 0.999, against 45-47% in US/India, while pairs whose numbers agree reach it at similar rates
(France 0.84, India 0.86, US 0.92). US/India records carry house-number noise that the model learned to forgive (98%
of US/India holdout disjoint pairs at p >= 0.7 are true); the French data does not, so a French record whose house
number differs from the S1's is most likely a branch / sibling at another number of the same street.

Signature check (US/India holdout labels vs France): on US/India true pairs with disjoint numbers the difference is
mostly one digit added/dropped (indel 45-51%), on false ones 18-24%, on French accepted ones 8.7%; US/India false
merges are distractor records that copy the S1 name and change one number component. French pairs where the record
adds or changes a component (e.g. 62-66 vs 62-87) reach p >= 0.999 at 11-16% vs 57-89% in US/India.

Rule: drop an accepted pair of a country in COUNTRIES when both sides have house numbers (1-4 digits, postal codes
excluded, leading zeros stripped) and the record carries a number the S1 lacks, unless p >= P_KEEP and the difference
looks like a typo (indel1 / prefix). With filler-normalized French names (er.frfill) the p >= 0.999 band of such pairs
is only 17% indel1 (true pairs 45-51%), so model confidence alone no longer protects a pair.
"""
import polars as pl
from rapidfuzz.distance import Levenshtein

from er.common import CACHE, log

COUNTRIES = ("France",)
P_KEEP = 0.999


def house_numbers(split, ids: pl.DataFrame, sources) -> pl.DataFrame:
    lf = pl.concat([pl.scan_parquet(CACHE / "norm_final" / f"{split}_s{s}.parquet").select("entity_id", "a_nums") for s in sources])
    return lf.join(ids.lazy(), on="entity_id", how="semi").collect().with_columns(
        pl.col("a_nums").list.eval(pl.element().filter(pl.element().str.len_chars() <= 4))
        .list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.element().filter(pl.element() != ""))
        .list.unique().alias("hn")).select("entity_id", "hn")


def typo_like(h1, h2) -> bool:
    """Some record number differs from some S1 number by one added/dropped digit, or one is a prefix of the other."""
    for a in h1:
        for b in h2:
            if a != b and (a.startswith(b) or b.startswith(a) or (abs(len(a) - len(b)) == 1 and Levenshtein.distance(a, b) == 1)):
                return True
    return False


def veto(pred: pl.DataFrame, scores: pl.DataFrame, split="test", countries=COUNTRIES, p_keep=P_KEEP):
    """pred: accepted (s1, m); scores: (s1, m, p). Returns (kept, dropped)."""
    cty = pl.read_parquet(CACHE / f"{split}_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    d = pred.join(cty.filter(pl.col("country").is_in(list(countries))), on="s1").join(scores, on=["s1", "m"], how="left") \
        .with_columns(pl.col("p").fill_null(0.0))  # pairs our model never scored (external base) are not protected
    h1 = house_numbers(split, d.select(pl.col("s1").alias("entity_id")).unique(), (1,)).rename({"entity_id": "s1", "hn": "h1"})
    h2 = house_numbers(split, d.select(pl.col("m").alias("entity_id")).unique(), (2, 3)).rename({"entity_id": "m", "hn": "h2"})
    d = d.join(h1, on="s1", how="left").join(h2, on="m", how="left")
    dj = d.filter((pl.col("h1").list.len() > 0) & (pl.col("h2").list.len() > 0) & (pl.col("h2").list.set_difference("h1").list.len() > 0))
    # a confident pair (p >= p_keep) survives only when the number difference looks like a typo (one digit added or
    # dropped, or one number a prefix of the other): the signature of true-pair noise on US/India holdout
    typo = [typo_like(a, b) for a, b in zip(dj["h1"].to_list(), dj["h2"].to_list())]
    dj = dj.with_columns(pl.Series("typo", typo, dtype=pl.Boolean)).filter(~((pl.col("p") >= p_keep) & pl.col("typo")))
    log(f"France house-number veto: dropped {dj.height:,} of {d.height:,} accepted pairs ({dj['s1'].n_unique():,} S1s)")
    return pred.join(dj.select("s1", "m"), on=["s1", "m"], how="anti"), dj
