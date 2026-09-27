"""Blocking recall / oracle macro-F0.5 on the world-20 holdout for a cached per-country candidate file.

  python -m er.eval_block India [US]
"""
import sys

import polars as pl

from ber.metrics import per_s1_f05, summarize
from er.common import CACHE, hold_s1, truth_pairs


def main(countries):
    for cty in countries:
        c = pl.read_parquet(CACHE / "cand" / f"train_w20_{cty}.parquet")
        h = hold_s1().filter(pl.col("country") == cty)
        t = truth_pairs().join(h.select("s1"), on="s1")
        n = c["s1"].n_unique()
        for R, K in [(1, 2), (1, 4), (2, 4), (2, 6), (3, 8)]:
            cc = c.filter((pl.col("r_rank") <= R) | (pl.col("s_rank") <= K))
            hh = t.join(cc.select("s1", "m"), on=["s1", "m"], how="semi")
            print(cty, R, K, "pairs/S1", round(cc.height / n, 2), "recall", round(hh.height / t.height, 5),
                  "oracle", summarize(per_s1_f05(hh, t, h))["macro_f05"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["India"])
