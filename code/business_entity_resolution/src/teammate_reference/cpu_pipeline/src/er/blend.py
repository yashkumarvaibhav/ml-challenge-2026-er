"""Blend of the run-2 and run-3 stage-2 scores (mean logit) -> decide -> France category veto -> output/<tag>.

Holdout (US/India, tune one half / report the other): run 2 0.98444, run 3 0.98453, blend 0.98479; full-holdout
thresholds t_any 0.5, t_add 0.75. Blocking is per country, so the blend and the decision run one country at a time.

  python -m er.blend --tag blend23
"""
import argparse
import json

import polars as pl

from er.common import CACHE, log
from er.model import decide
from er.post import category_veto
from er.write import write

RUNS = ("p22_test.parquet", "p23_test.parquet")
T_ANY, T_ADD = 0.5, 0.75


def _logit(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def main(tag):
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    preds = []
    for c in cty["country"].unique().sort():
        ids = cty.filter(pl.col("country") == c).select("s1").lazy()
        sc = [pl.scan_parquet(CACHE / f).select("s1", "m", pl.col("p").alias(f"p{i}")).join(ids, on="s1", how="semi").collect()
              for i, f in enumerate(RUNS)]
        d = sc[0].join(sc[1], on=["s1", "m"], how="inner")
        assert d.height == sc[0].height == sc[1].height, f"{c}: score files cover different pairs"
        del sc
        d = d.with_columns((1 / (1 + (-(_logit("p0") + _logit("p1")) / 2).exp())).alias("p")).select("s1", "m", "p")
        preds.append(decide(d, T_ANY, T_ADD, "p"))
        log(f"blend {c}: {d.height:,} pairs -> {preds[-1].height:,} accepted")
        del d
    pred = pl.concat(preds)
    kept, dropped, rep = category_veto(pred, "test")
    cand = pl.read_parquet(CACHE / "cand" / "b2_test_w20.parquet", columns=["s1", "m"])
    out = CACHE.parent / "output" / tag
    write(kept, cand, out)
    (out / "meta.json").write_text(json.dumps({"blend": RUNS, "t_any": T_ANY, "t_add": T_ADD,
                                               "veto": {k: {kk: vv for kk, vv in v.items() if kk != "words"} for k, v in rep.items()}}, indent=1))


def scores(path=CACHE / "p2b_test.parquet"):
    """France-only blended scores for er.france_post --scores (veto protection / rescue)."""
    fr = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France") \
        .select(pl.col("entity_id").alias("s1")).lazy()
    sc = [pl.scan_parquet(CACHE / f).select("s1", "m", pl.col("p").alias(f"p{i}")).join(fr, on="s1", how="semi").collect() for i, f in enumerate(RUNS)]
    sc[0].join(sc[1], on=["s1", "m"]).with_columns((1 / (1 + (-(_logit("p0") + _logit("p1")) / 2).exp())).alias("p")) \
        .select("s1", "m", "p").write_parquet(path)
    log(f"blend: France scores -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="blend23")
    ap.add_argument("--france-scores", action="store_true")
    a = ap.parse_args()
    scores() if a.france_scores else main(a.tag)
