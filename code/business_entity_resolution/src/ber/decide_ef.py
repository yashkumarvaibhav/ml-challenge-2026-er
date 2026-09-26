"""Expected-F0.5-optimal decisions per S1 (replaces the fixed t_any / t_add thresholds).

For one S1 with candidate match probabilities p_1 >= p_2 >= ... (stage-2 scores, calibrated on held-out data),
predicting the top-k set S_k gives the per-S1 score

    k = 0 :  1 if the S1 has no true match, else 0
    k >= 1:  1.25 * TP / (0.25 * T + k),   TP = true matches in S_k,  T = all true matches

Treating candidates as independent Bernoulli(p_i), TP and FN (true matches outside S_k) have Poisson-binomial
distributions, so E[score(k)] is computed exactly by convolution, and the best k is chosen per S1. Only
candidates with p >= P_MIN enter (the rest are treated as p = 0); at most K_MAX are considered.
The probabilities are the model's own; no labels are used at decision time.
"""
import numpy as np
import polars as pl

P_MIN, K_MAX = 0.005, 20


def _pbinom(p: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf of the number of successes."""
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for x in p:
        d[1:] = d[1:] * (1 - x) + d[:-1] * x
        d[0] *= 1 - x
    return d


def best_k(p: np.ndarray, temp: float = 1.0) -> int:
    if temp != 1.0:  # optional calibration temperature on the logit scale
        z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))) / temp
        p = 1 / (1 + np.exp(-z))
    n = len(p)
    best, arg = np.prod(1 - p), 0                       # k = 0: all candidates false
    for k in range(1, n + 1):
        tp = _pbinom(p[:k])                             # P(TP = a), a = 0..k
        fn = _pbinom(p[k:])                             # P(FN = b), b = 0..n-k
        a = np.arange(k + 1)[:, None]
        b = np.arange(n - k + 1)[None, :]
        val = (tp[:, None] * fn[None, :] * (1.25 * a / (0.25 * (a + b) + k))).sum()
        if val > best:
            best, arg = val, k
    return arg


def decide_ef(scored: pl.DataFrame, temp: float = 1.0) -> pl.DataFrame:
    """scored: (s1, m, p) -> accepted (s1, m) pairs."""
    d = scored.filter(pl.col("p") >= P_MIN).sort(["s1", "p"], descending=[False, True]) \
        .with_columns(pl.int_range(pl.len()).over("s1").alias("_r")).filter(pl.col("_r") < K_MAX)
    g = d.group_by("s1", maintain_order=True).agg(pl.col("m"), pl.col("p"))
    out_s1, out_m = [], []
    for s1, ms, ps in g.iter_rows():
        k = best_k(np.asarray(ps, dtype=np.float64), temp)
        if k:
            out_s1.extend([s1] * k)
            out_m.extend(ms[:k])
    return pl.DataFrame({"s1": out_s1, "m": out_m}, schema={"s1": pl.Utf8, "m": pl.Utf8})
