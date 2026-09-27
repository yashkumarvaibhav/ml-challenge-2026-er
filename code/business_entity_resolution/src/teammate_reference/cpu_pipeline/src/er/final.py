"""Final assembly: blend stage-2 scores (mean logit) -> expected-F0.5 decision -> France category veto ->
France house-number veto (+ filler rescue when France is not filler-normalized) -> output/<tag>.

  python -m er.final --runs p22 p23b --tag final --frfill       # holdout (US/India) 0.98508 two-way, EF a 1.5 b -0.5 mu 0.35

--frfill takes the France scores from the filler-normalized France split (<run>_testfr.parquet, er.frfill); the rescue
is then skipped, since filler-only name differences are already neutral for the model.
"""
import argparse
import json

import polars as pl

from er import frnum
from er.common import CACHE, log
from er.efdecide import ef_decide
from er.france_post import rescue
from er.post import category_veto
from er.write import write


def _logit(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def blend(files, ids: pl.LazyFrame) -> pl.DataFrame:
    sc = [pl.scan_parquet(CACHE / f).select("s1", "m", pl.col("p").alias(f"p{i}")).join(ids, on="s1", how="semi").collect()
          for i, f in enumerate(files)]
    d = sc[0]
    for x in sc[1:]:
        n = d.height
        d = d.join(x, on=["s1", "m"], how="inner")
        assert d.height == n == x.height, f"score files cover different pairs: {files}"
    z = sum(_logit(f"p{i}") for i in range(len(files))) / len(files)
    return d.select("s1", "m", (1 / (1 + (-z).exp())).alias("p"))


def main(runs, tag, frfill, a, b, mu):
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    preds, fr_scores = [], None
    for c in ("US", "India", "France"):
        ids = cty.filter(pl.col("country") == c).select("s1").lazy()
        files = [f"{r}_testfr.parquet" if (c == "France" and frfill) else f"{r}_test.parquet" for r in runs]
        sc = blend(files, ids)
        preds.append(ef_decide(sc, a, b, mu))
        log(f"final {c}: {files} {sc.height:,} pairs -> {preds[-1].height:,} accepted")
        if c == "France":
            fr_scores = sc
        del sc
    pred = pl.concat(preds)
    kept, _, rep = category_veto(pred, "test")
    kept, dropped = frnum.veto(kept, fr_scores)
    n_res = 0
    if not frfill:
        r = rescue(kept, fr_scores, "feats2")
        n_res = r.height
        kept = pl.concat([kept, r]).unique()
    cand = pl.read_parquet(CACHE / "cand" / "b2_test_w20.parquet", columns=["s1", "m"])
    out = CACHE.parent / "output" / tag
    write(kept, cand, out)
    fr_scores.write_parquet(out / "france_scores.parquet")
    (out / "meta.json").write_text(json.dumps({"runs": runs, "frfill": frfill, "ef": {"a": a, "b": b, "mu": mu},
                                               "category_veto": {k: {kk: vv for kk, vv in v.items() if kk != "words"} for k, v in rep.items()},
                                               "numveto_dropped": dropped.height, "rescued": n_res}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["p22", "p23b"])
    ap.add_argument("--tag", default="final")
    ap.add_argument("--frfill", action="store_true")
    ap.add_argument("--a", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=-0.5)
    ap.add_argument("--mu", type=float, default=0.35)
    x = ap.parse_args()
    main(x.runs, x.tag, x.frfill, x.a, x.b, x.mu)
