"""Category-word swaps: accepted pairs whose record name replaces one S1 name word with a *category* word.

Finding (2026-09-26): the generator's name noise "drop a word + add a filler word" uses a small filler list per
country (US/India holdout: center / services / service / partners; substitution pairs are 99.8% true). Filler
words also appear as pure additions (add-only) at ~1/3 of their substitution count. In France, accepted pairs also
contain substitutions to *category* words (club, comite, amicale, ecole, maison, ...) that are NEVER added as
pure additions (add-only ~0). Category words are therefore entity-distinguishing (sibling businesses such as
"Sweet Amicale" / "Sweet Maison"), not noise.

Rule (label-free, per country, from that country's own accepted test pairs):
  category word  = substituted-in >= MIN_SUB times, add-only / substituted-in < MAX_ADD_RATIO,
                   and present in >= MIN_S1_SHARE of the country's S1 names
  drop an accepted pair if its record name is a one-word substitution of the S1 name to a category word.
Report per country, and on the US/India holdout (truth) with holdout-derived category words.
"""
import collections
import json
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from ..io import load_train_pairs
from ..metrics import per_s1_f05, summarize
from ..model import decide

LEG = set("inc llc ltd pvt private limited corp co corporation company pc pllc llp lp sarl sas sa sasu eurl ei snc sci s a r l".split())
MIN_SUB, MAX_ADD_RATIO, MIN_S1_SHARE = 5, 0.05, 0.001
TA, TD = 0.4, 0.75


def toks(s):
    if s is None:
        return []
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in LEG]


def classify(n1, n2):
    """('drop', 'add') for a one-word substitution, ('', 'add') for a one-word pure addition, else None."""
    t1, t2 = toks(n1), toks(n2)
    if not t1 or not t2:
        return None
    dr = [a for a in t1 if max(fuzz.ratio(a, b) for b in t2) < 70]
    ad = [b for b in t2 if max(fuzz.ratio(a, b) for a in t1) < 70]
    if len(dr) == 1 and len(ad) == 1 and len(dr[0]) >= 4 and len(ad[0]) >= 4:
        return dr[0], ad[0]
    if not dr and len(ad) == 1 and len(ad[0]) >= 3:
        return "", ad[0]
    return None


def category_words(acc: pl.DataFrame, s1_names: list) -> set:
    sub, add = collections.Counter(), collections.Counter()
    for n1, n2 in zip(acc["n1"], acc["n2"]):
        c = classify(n1, n2)
        if c:
            (sub if c[0] else add)[c[1]] += 1
    s1c = collections.Counter(w for n in s1_names for w in set(toks(n)))
    return {w for w, k in sub.items()
            if k >= MIN_SUB and add[w] < MAX_ADD_RATIO * k and s1c[w] >= MIN_S1_SHARE * len(s1_names)}


def flag(acc: pl.DataFrame, cats: set) -> pl.Series:
    out = []
    for n1, n2 in zip(acc["n1"], acc["n2"]):
        c = classify(n1, n2)
        out.append(bool(c and c[0] and c[1] in cats))
    return pl.Series("swap", out)


def main():
    rep = {}
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), "country")
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"))
    p = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2.parquet")
    acc = decide(p, TA, TD).join(t1, on="s1").join(rec, on="m")
    drops = []
    for cty in ("US", "India", "France"):
        a = acc.filter(pl.col("country") == cty)
        cats = category_words(a, t1.filter(pl.col("country") == cty)["n1"].to_list())
        a = a.with_columns(flag(a, cats))
        d = a.filter(pl.col("swap"))
        drops.append(d.select("s1", "m"))
        rep[f"test {cty}"] = {"category_words": sorted(cats), "n_category_words": len(cats), "accepted": a.height,
                              "dropped_pairs": d.height, "S1_affected": d["s1"].n_unique(),
                              "S1_affected_share": round(d["s1"].n_unique() / t1.filter(pl.col("country") == cty).height, 4)}
        print(cty, {k: v for k, v in rep[f"test {cty}"].items() if k != "category_words"}, sorted(cats)[:40], flush=True)
    pl.concat(drops).write_parquet(C.WORK / "s2" / "category_swap_drops_test.parquet")
    # holdout (truth): same rule with holdout-derived category words
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
    tr1 = pl.read_parquet(C.CACHE / "train_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), "country")
    trec = pl.concat([pl.read_parquet(C.CACHE / f"train_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"))
    truth = load_train_pairs().select("s1", "m")
    for wd in (20, 0):
        hp = pl.read_parquet(C.WORK / "s2" / f"holdout_pred_w{wd}_e5base2.parquet")
        hacc = decide(hp, TA, TD).join(hold.select("s1"), on="s1").join(tr1, on="s1").join(trec, on="m")
        keep = []
        for cty in ("US", "India"):
            a = hacc.filter(pl.col("country") == cty)
            cats = category_words(a, tr1.join(hold.select("s1"), on="s1").filter(pl.col("country") == cty)["n1"].to_list())
            a = a.with_columns(flag(a, cats))
            dd = a.filter(pl.col("swap")).join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left")
            rep[f"holdout w{wd} {cty}"] = {"category_words": sorted(cats), "dropped_pairs": dd.height,
                                          "dropped_true_rate": (round(dd["y"].fill_null(0).mean(), 3) if dd.height else None)}
            keep.append(a.filter(~pl.col("swap")).select("s1", "m"))
        base = decide(hp, TA, TD).join(hold.select("s1"), on="s1")
        new = pl.concat(keep)
        rep[f"holdout w{wd} F"] = {"base": summarize(per_s1_f05(base, truth, hold))["macro_f05"],
                                  "rule": summarize(per_s1_f05(new, truth, hold))["macro_f05"]}
        print(f"holdout w{wd}", rep[f"holdout w{wd} US"]["dropped_pairs"], rep[f"holdout w{wd} India"]["dropped_pairs"], rep[f"holdout w{wd} F"], flush=True)
    (C.EXP_RUN / "france_category_swap.json").write_text(json.dumps(rep, indent=1))




def write(base: str, countries=("France",)):
    """output/ = archived package `base` minus the category-swap pairs of `countries` (others byte-identical)."""
    import shutil
    arch = C.OUTPUT if base == "output" else C.ROOT / "submissions" / "archive" / base  # "output": apply in place (reproduction)
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    drops = pl.read_parquet(C.WORK / "s2" / "category_swap_drops_test.parquet").join(
        t1.filter(pl.col("country").is_in(list(countries))), on="s1").select("s1", "m")
    pairs = (m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "")
             .rename({c1: "s1", c2: "m"}).join(drops, on=["s1", "m"], how="anti"))
    g = pairs.group_by("s1").agg(pl.col("m").sort().str.join(",").alias(c2))
    out = m.select(c1).join(g.rename({"s1": c1}), on=c1, how="left").with_columns(pl.col(c2).fill_null(""))
    assert out[c1].to_list() == m[c1].to_list()
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    if arch != C.OUTPUT:
        shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    print(f"removed {drops.height:,} pairs ({', '.join(countries)}); rows {out.height:,}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", help="archived base tag: write output/ = base minus France category-swap pairs")
    a = ap.parse_args()
    write(a.write) if a.write else main()
