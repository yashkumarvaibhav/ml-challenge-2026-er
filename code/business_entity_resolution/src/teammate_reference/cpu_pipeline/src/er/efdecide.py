"""Expected-F0.5 decision: per S1, accept the top-j candidates that maximize the expected F0.5 under the model's
(recalibrated) probabilities, instead of two fixed thresholds.

For an S1 with unique-assigned candidates p_1 >= p_2 >= ... (a record only competes at its best S1):
  E[T]      = sum of p over ALL its candidates (before the unique filter) + mu   (mu: matches outside the candidates)
  E[F | j]  ~ 1.25 * (p_1 + ... + p_j) / (0.25 * E[T] + j)                        (j >= 1)
  E[F | 0]  = P(no true match) ~ prod(1 - p_i) * exp(-mu)
Calibration p' = sigmoid(a * logit(p) + b); a, b, mu are tuned on one holdout half and reported on the other.

  python -m er.efdecide --scores cache/p2b_train_holdout.parquet
"""
import argparse
import itertools
import json

import numpy as np
import polars as pl

from ber.metrics import per_s1_f05, summarize
from er.common import CACHE, hold_s1, log, truth_pairs


def _cal(p, a, b):
    x = pl.col(p).clip(1e-6, 1 - 1e-6)
    return 1 / (1 + (-(a * (x / (1 - x)).log() + b)).exp())


def ef_decide(scored: pl.DataFrame, a=1.0, b=0.0, mu=0.07, p="p") -> pl.DataFrame:
    d = scored.select("s1", "m", _cal(p, a, b).alias("q"))
    s = d.group_by("s1").agg(pl.col("q").sum().alias("et"), (1 - pl.col("q")).log().sum().alias("lp0"))
    u = d.filter(pl.col("q") == pl.col("q").max().over("m")).sort(["s1", "q"], descending=[False, True])
    u = u.with_columns(pl.col("q").cum_sum().over("s1").alias("cs"), pl.int_range(1, pl.len() + 1).over("s1").alias("j"))
    u = u.join(s, on="s1").with_columns((1.25 * pl.col("cs") / (0.25 * (pl.col("et") + mu) + pl.col("j"))).alias("ef"))
    best = u.group_by("s1").agg(pl.col("ef").max().alias("ef_best"), pl.col("j").sort_by("ef").last().alias("j_best"),
                                (pl.col("lp0").first() - mu).exp().alias("ef0"))
    u = u.join(best, on="s1").filter((pl.col("ef_best") > pl.col("ef0")) & (pl.col("j") <= pl.col("j_best")))
    return u.select("s1", "m")


GRID_A = [1.5, 2.0, 2.5, 3.0]
GRID_B = [-1.5, -1.0, -0.5]
GRID_MU = [0.2, 0.35, 0.5]


def tune(scored, truth, hold):
    best = None
    for a, b, mu in itertools.product(GRID_A, GRID_B, GRID_MU):
        f = summarize(per_s1_f05(ef_decide(scored, a, b, mu), truth, hold))["macro_f05"]
        if best is None or f > best[0]:
            best = (f, a, b, mu)
    return best


def main(scores):
    hold = hold_s1()
    truth = truth_pairs().join(hold.select("s1"), on="s1")
    sc = pl.read_parquet(scores).join(hold.select("s1"), on="s1")
    out = {}
    for th, rh in (("A", "B"), ("B", "A")):
        ht, hr = hold.filter(pl.col("holdout_half") == th), hold.filter(pl.col("holdout_half") == rh)
        f, a, b, mu = tune(sc.join(ht.select("s1"), on="s1"), truth, ht)
        rep = summarize(per_s1_f05(ef_decide(sc.join(hr.select("s1"), on="s1"), a, b, mu), truth, hr))
        out[f"tune{th}_rep{rh}"] = (rep["macro_f05"], rep["by_country"], {"a": a, "b": b, "mu": mu, "tuned_f": f})
    f, a, b, mu = tune(sc, truth, hold)
    out["full"] = (f, {"a": a, "b": b, "mu": mu})
    log(f"efdecide {scores}: " + json.dumps(out))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(CACHE / "p2b_train_holdout.parquet"))
    main(ap.parse_args().scores)
