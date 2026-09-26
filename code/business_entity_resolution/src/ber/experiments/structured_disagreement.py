"""Audit disagreements left by the current France vetoes; never change predictions.

Use only supplied raw records and the frozen #27-2 output. Known-country controls
use both fixed e5base2 holdouts. French counts/optimistic bounds are not labels.
Run with BER_RUN=v6 BER_WORK=v6; reports and feature caches have distinct names.
"""
import json
from functools import lru_cache

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from ..io import load_train_pairs
from ..metrics import per_s1_f05
from ..model import decide
from ..progress import Progress, step
from .france_street_diff import _parts, _toks2

DEST = C.CACHE / "diagnostic_structured"
BASE = "sub-2026-09-27-2"


@lru_cache(maxsize=150_000)
def street_parts(value):
    return _parts(value)


def street_features(a, b):
    pp, qq = street_parts(a), street_parts(b)
    if not pp or not qq:
        return (-1., -1., 0, 0, 0, 0, 0, False)
    comparisons = []
    for pn, pw in pp:
        for qn, qw in qq:
            scores = [[fuzz.ratio(x, y) for y in qw] for x in pw]
            row = [max(s) for s in scores]
            col = [max(s[j] for s in scores) for j in range(len(qw))]
            joined = fuzz.token_sort_ratio(" ".join(pw), " ".join(qw))
            up = sum(v < 70 and len(t) >= 4 for t, v in zip(pw, row))
            uq = sum(v < 70 and len(t) >= 4 for t, v in zip(qw, col))
            # Choose the most compatible parsed street pair, conservatively.
            comparisons.append((min(min(row), min(col)), joined, max(row), up, uq,
                                len(pw), len(qw), pn is not None and pn == qn))
    best = max(comparisons, key=lambda x: (min(x[0], x[1]), x[1]))
    return (best[1], best[2], best[3], best[4], best[5], best[6],
            int(best[0] >= 70), best[7])


def name_features(a, b, cats):
    p, q = _toks2(a), _toks2(b)
    if not p or not q:
        return (0, 0, 0, 0, 0.)
    dr = [x for x in p if max(fuzz.ratio(x, y) for y in q) < 70]
    ad = [y for y in q if max(fuzz.ratio(x, y) for x in p) < 70]
    ca, cb = set(p) & cats, set(q) & cats
    # Do not flag morphological extensions such as sport/sportive.
    novel = [y for y in cb if not any(x.startswith(y) or y.startswith(x)
             or fuzz.ratio(x, y) >= 90 for x in p)]
    gone = [x for x in ca if not any(x.startswith(y) or y.startswith(x)
            or fuzz.ratio(x, y) >= 90 for y in q)]
    return (len(dr), len(ad), len(novel), len(gone), fuzz.token_sort_ratio(a or "", b or ""))


def raw(split, source):
    return pl.scan_parquet(C.CACHE / f"{split}_source{source}.parquet")


def attach_text(pairs, split):
    s1 = raw(split, 1).select(pl.col("entity_id").alias("s1"), "country",
        pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    rec = pl.concat([raw(split, s) for s in (2, 3)]).select(pl.col("entity_id").alias("m"),
        pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    return pairs.lazy().join(s1, on="s1").join(rec, on="m").collect()


def features(d, name, cats):
    path = DEST / f"{name}.parquet"
    if path.exists():
        step("structured", f"reuse {path.name}")
        return pl.read_parquet(path)
    progress = Progress(f"structured {name}", d.height, unit="pairs")
    values = []
    for i, (a, b, n, m) in enumerate(d.select("a1", "a2", "n1", "n2").iter_rows()):
        values.append(street_features(a, b) + name_features(n, m, cats))
        if (i + 1) % 10_000 == 0:
            progress.update(10_000)
    progress.update(d.height % 10_000)
    progress.done()
    schema = ["street_ratio", "street_best_word", "street_unmatched_s1", "street_unmatched_m",
              "street_words_s1", "street_words_m", "street_all_words_match", "street_number_same",
              "name_dropped", "name_added", "category_added", "category_dropped", "name_ratio"]
    result = d.hstack(pl.DataFrame(values, schema=schema, orient="row"))
    result.write_parquet(path)
    street_parts.cache_clear()
    return result


def rules():
    both_unmatched = (pl.col("street_unmatched_s1") > 0) & (pl.col("street_unmatched_m") > 0)
    partial = both_unmatched & (pl.col("street_best_word") >= 70)
    out = {f"partial_street_r{t}": partial & (pl.col("street_ratio") < t) for t in (60, 70, 80, 90)}
    out["partial_street_same_number_r80"] = partial & pl.col("street_number_same") & (pl.col("street_ratio") < 80)
    out["all_street_words_weak_r80"] = (pl.col("street_best_word") >= 70) & (pl.col("street_best_word") < 85) & (pl.col("street_ratio") < 80)
    out["multi_name_category"] = (pl.col("name_dropped") >= 1) & (pl.col("name_added") >= 1) & (pl.col("category_added") >= 1)
    out["explicit_category_conflict"] = (pl.col("category_added") >= 1) & (pl.col("category_dropped") >= 1)
    return out


def eval_holdout(d, hold, truth, name):
    base = per_s1_f05(d.select("s1", "m"), truth, hold)
    out = {"baseline": float(base["f05"].mean()), "rules": {}}
    for key, expr in rules().items():
        drop = d.filter(expr)
        new = per_s1_f05(d.filter(~expr).select("s1", "m"), truth, hold)
        delta = new.select("s1", "f05").rename({"f05": "new"}).join(base, on="s1").with_columns((pl.col("new") - pl.col("f05")).alias("delta"))
        by = {c: float(v) for c, v in delta.group_by("country").agg(pl.col("delta").mean()).iter_rows()}
        out["rules"][key] = {"dropped": drop.height,
            "true": drop.join(truth, on=["s1", "m"]).height,
            "delta": float(delta["delta"].mean()), "by_country": by,
            "weighted_delta": .45 * by["US"] + .55 * by["India"],
            "by_half": dict(delta.group_by("half").agg(pl.col("delta").mean()).iter_rows())}
        drop.head(100).write_parquet(DEST / f"{name}_{key}_examples.parquet")
    return out


def eval_france(d):
    counts = d.group_by("s1").len().rename({"len": "P"})
    out = {"accepted": d.height, "rules": {}}
    for key, expr in rules().items():
        drop = d.filter(expr)
        affected = drop.group_by("s1").len().rename({"len": "D"}).join(counts, on="s1")
        # Optimistic bound: all removed pairs false, all retained pairs true.
        gains = affected.with_columns((pl.col("D") / (1.25 * (pl.col("P") - pl.col("D")) + pl.col("D"))).alias("gain"))
        high = drop.filter(pl.col("p") >= .95).join(counts, on="s1")
        out["rules"][key] = {"pairs": drop.height, "s1": affected.height,
            "public_gain_if_every_drop_false_and_every_retained_pair_true": float(gains["gain"].sum() * .15 / 259452),
            "k1_share_p_ge_095": float((high["P"] == 1).mean()) if high.height else None,
            "p_ge_095_n": high.height, "p_median": float(drop["p"].median()) if drop.height else None}
        if drop.height:
            drop.sample(min(100, drop.height), seed=C.SEED).write_parquet(DEST / f"france_{key}_examples.parquet")
    return out


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cats = set(json.loads((C.EXPERIMENTS / "v6/france_category_swap.json").read_text())["test France"]["category_words"])
    sp = pl.read_parquet(C.CACHE / "splits/train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(
        pl.col("entity_id").alias("s1"), "country", pl.col("holdout_half").alias("half"))
    truth = load_train_pairs().select("s1", "m").join(hold.select("s1"), on="s1")
    report = {"base_archive": BASE, "uses_french_labels": False, "france_share_assumption": .15}
    for w in (20, 0):
        step("structured", f"load holdout world {w}")
        p = pl.read_parquet(C.CACHE / "v6/s2" / f"holdout_pred_w{w}_e5base2.parquet")
        pairs = decide(p, .4, .75).join(hold.select("s1"), on="s1").join(p, on=["s1", "m"])
        d = features(attach_text(pairs, "train"), f"w{w}", cats)
        report[f"w{w}"] = eval_holdout(d, hold, truth, f"w{w}")
        (C.EXPERIMENTS / "v6/structured_disagreement.json").write_text(json.dumps(report, indent=2))
        del d, pairs, p
    step("structured", "load exact best France output")
    ids = raw("test", 1).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1")).collect()
    pairs = (pl.read_csv(C.ROOT / "submissions/archive" / BASE / "matching_results.tsv", separator="\t", quote_char=None,
                infer_schema_length=0).rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
             .join(ids, on="s1").with_columns(pl.col("m").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != ""))
    p = pl.scan_parquet(C.CACHE / "v6/s2/test_pred_e5base2.parquet").join(ids.lazy(), on="s1").collect()
    pairs = pairs.join(p, on=["s1", "m"])
    d = features(attach_text(pairs, "test"), "france", cats)
    report["France"] = eval_france(d)
    (C.EXPERIMENTS / "v6/structured_disagreement.json").write_text(json.dumps(report, indent=2))
    step("structured", "DONE: experiments/v6/structured_disagreement.json; no predictions changed")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
