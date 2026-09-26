"""Category-swap veto BEFORE stage 2 (France): sibling-business pairs contaminate the competition features.

The category-swap rule (ber.experiments.france_category_swap) removes accepted sibling pairs after stage 2
(public +0.003063, sub-2026-09-26-8). Stage 2's competition features are computed from stage-1 scores, so a
sibling pair with p1 ~ 1 also (a) inflates its S1's top/sum/count features and (b) gives the sibling S1's own
true records a strong "best other S1" rival, suppressing them. Here the veto is applied to every France
candidate pair with p1 >= P_MIN (p1 := 0, CE feature removed), competition features are recomputed over the
whole test universe, and the unchanged stage-2 model re-scores. US/India predictions are kept byte-identical;
France decisions come from the re-scored universe with vetoed pairs excluded.

  python -m ber.experiments.france_veto_s2 [--write BASE_TAG]
"""
import argparse
import json
import shutil

import lightgbm as lgb
import polars as pl

from .. import config as C
from ..model import decide
from ..stage2 import MODEL_S2, S2, SFX, _a_null, _with_ce, s2_features
from .france_category_swap import category_words, classify

P_MIN, TA, TD = 0.005, 0.4, 0.75


def main():
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), "country")
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"))
    old = pl.read_parquet(S2 / f"test_pred{SFX}.parquet")
    fr = t1.filter(pl.col("country") == "France")
    acc = decide(old.join(fr.select("s1"), on="s1"), TA, TD).join(fr, on="s1").join(rec, on="m")
    cats = category_words(acc, fr["n1"].to_list())
    st = pl.read_parquet(C.WORK / "pred" / "test.parquet").rename({"p": "p1"})
    cand_fr = st.filter(pl.col("p1") >= P_MIN).join(fr.select("s1", "n1"), on="s1").join(rec, on="m")
    fl = [bool((c := classify(a, b)) and c[0] and c[1] in cats) for a, b in zip(cand_fr["n1"], cand_fr["n2"])]
    veto = cand_fr.filter(pl.Series(fl)).select("s1", "m").with_columns(pl.lit(True).alias("_veto"))
    print(f"France category words {len(cats)}; France candidates p1>={P_MIN}: {cand_fr.height:,}; vetoed {veto.height:,}", flush=True)
    st2 = st.join(veto, on=["s1", "m"], how="left").with_columns(
        pl.when(pl.col("_veto")).then(0.0).otherwise(pl.col("p1")).alias("p1")).drop("_veto")
    cand = pl.read_parquet(C.WORK / "cand" / "test.parquet")
    d = _with_ce(s2_features(st2, cand, _a_null("test")), S2 / f"ce_test{SFX}.parquet")
    d = d.join(veto, on=["s1", "m"], how="left").with_columns(
        [pl.when(pl.col("_veto")).then(None).otherwise(pl.col(c)).alias(c) for c in ("ce", "ce_minus_p1")]).drop("_veto")
    booster = lgb.Booster(model_file=str(MODEL_S2))
    new = d.select("s1", "m").with_columns(pl.Series("p", booster.predict(d.select(booster.feature_name()).to_numpy())))
    new_fr = new.join(fr.select("s1"), on="s1").join(veto, on=["s1", "m"], how="anti")
    out = pl.concat([old.join(fr.select("s1"), on="s1", how="anti"), new_fr])
    out.write_parquet(S2 / f"test_pred{SFX}_frveto.parquet")
    veto.select("s1", "m").write_parquet(S2 / "france_veto_pairs_test.parquet")
    a = decide(old.join(fr.select("s1"), on="s1"), TA, TD)
    swap8 = a.join(veto, on=["s1", "m"], how="anti")  # ~ sub-2026-09-26-8's France decisions (strict swaps among accepted)
    b = decide(new_fr, TA, TD)
    add, rem = b.join(swap8, on=["s1", "m"], how="anti"), swap8.join(b, on=["s1", "m"], how="anti")
    k = fr.select("s1").join(b.group_by("s1").agg(pl.len().alias("k")), on="s1", how="left").with_columns(pl.col("k").fill_null(0))
    rep = {"category_words": len(cats), "vetoed_candidate_pairs": veto.height,
           "vs_post_hoc_drop": {"added": add.height, "removed": rem.height,
                                "S1_changed": pl.concat([add.select("s1"), rem.select("s1")])["s1"].n_unique()},
           "france_k0": round(100 * (k["k"] == 0).mean(), 2), "france_k1": round(100 * (k["k"] == 1).mean(), 2), "france_mean": round(k["k"].mean(), 3)}
    print(json.dumps(rep), flush=True)
    (C.EXP_RUN / "france_veto_s2.json").write_text(json.dumps(rep, indent=1))


def write(base: str):
    """output/ = archived `base` with France rows replaced by the veto re-scored decisions."""
    arch = C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    fr = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias(c1))
    b = decide(pl.read_parquet(S2 / f"test_pred{SFX}_frveto.parquet").join(fr.rename({c1: "s1"}), on="s1"), TA, TD)
    g = b.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("_new")).rename({"s1": c1})
    out = m.join(fr.with_columns(pl.lit(True).alias("_fr")), on=c1, how="left").join(g, on=c1, how="left").with_columns(
        pl.when(pl.col("_fr")).then(pl.col("_new").fill_null("")).otherwise(pl.col(c2).fill_null("")).alias(c2)).select(c1, c2)
    assert out[c1].to_list() == m[c1].to_list()
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    print(f"France rows replaced: {fr.height:,}")


def write2(base: str, t_add_fr: float, relaxed: bool):
    """output/ = `base` with France rows from the veto re-scored universe at t_add_fr, optionally also dropping
    relaxed category swaps (typo'd category words / extra filler words; ber.experiments.france_category_swap)."""
    import collections
    from rapidfuzz import fuzz, process
    from .france_category_swap import toks
    arch = C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), "country")
    fr = t1.filter(pl.col("country") == "France")
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"))
    b = decide(pl.read_parquet(S2 / f"test_pred{SFX}_frveto.parquet").join(fr.select("s1"), on="s1"), TA, t_add_fr)
    if relaxed:
        acc = b.join(fr, on="s1").join(rec, on="m")
        # category words from the PRE-veto accepted pairs (after the veto, swaps are rare and the counts vanish)
        cats = set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"])
        sub, add = collections.Counter(), collections.Counter()
        for n1, n2 in zip(acc["n1"], acc["n2"]):
            c = classify(n1, n2)
            if c:
                (sub if c[0] else add)[c[1]] += 1
        fill = {w for w in set(sub) | set(add) if add[w] >= 5 and add[w] >= 0.15 * max(1, sub[w])}
        catl = list(cats)

        def iscat(w):
            if w in cats:
                return True
            r = process.extractOne(w, catl, scorer=fuzz.ratio) if len(w) >= 4 else None
            return r is not None and r[1] >= 85
        flags = []
        for n1, n2 in zip(acc["n1"], acc["n2"]):
            t1_ = [t for t in toks(n1) if t not in fill]
            t2_ = [t for t in toks(n2) if t not in fill]
            if not t1_ or not t2_:
                flags.append(False)
                continue
            new = [w for w in t2_ if max(fuzz.ratio(w, a) for a in t1_) < 75 and iscat(w)]
            miss = [w for w in t1_ if max(fuzz.ratio(w, x) for x in t2_) < 75 and iscat(w)]
            flags.append(bool(new and miss))
        drop = acc.filter(pl.Series(flags)).select("s1", "m")
        b = b.join(drop, on=["s1", "m"], how="anti")
        print(f"relaxed category swaps dropped: {drop.height:,}")
    g = b.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("_new")).rename({"s1": c1})
    frc = fr.select(pl.col("s1").alias(c1)).with_columns(pl.lit(True).alias("_fr"))
    out = m.join(frc, on=c1, how="left").join(g, on=c1, how="left").with_columns(
        pl.when(pl.col("_fr")).then(pl.col("_new").fill_null("")).otherwise(pl.col(c2).fill_null("")).alias(c2)).select(c1, c2)
    assert out[c1].to_list() == m[c1].to_list()
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    print(f"France pairs: {b.height:,} (t_add_fr {t_add_fr}, relaxed {relaxed})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--write2", help="base tag: France rows from the veto re-score at --t-add-fr, optional --relaxed")
    ap.add_argument("--t-add-fr", type=float, default=TD)
    ap.add_argument("--relaxed", action="store_true")
    a = ap.parse_args()
    if a.write2:
        write2(a.write2, a.t_add_fr, a.relaxed)
    elif a.write:
        write(a.write)
    else:
        main()
