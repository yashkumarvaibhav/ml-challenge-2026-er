"""Feature-builder smoke test on ~100k train pairs (world20)."""
import time

import numpy as np
import polars as pl

from ber import config as C
from ber.features import FEATURES, build
from ber.io import load_train_pairs

if __name__ == "__main__":
    cand = pl.read_parquet(C.CACHE / "cand" / "train_w20.parquet")
    s1s = cand.select("s1").unique().sort("s1").head(2200)
    sl = cand.join(s1s, on="s1")
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    world = sp.filter(~pl.col("is_ghost")).select("entity_id")
    t = time.time()
    f = build("train", sl, world, name="features smoke")
    dt = time.time() - t
    print("pairs", f.height, "time", round(dt, 1), "s ->", round(f.height / dt), "pairs/s", flush=True)
    X = f.select(FEATURES).to_numpy()
    print("nan frac (top):", sorted(zip(FEATURES, np.isnan(X).mean(0).round(3)), key=lambda x: -x[1])[:6])
    tr = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
    g = f.join(tr, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    print("pos rate", round(g["y"].mean(), 4))
    print(g.group_by("y").agg([pl.col(c).mean().round(3) for c in
                               ["n_tset", "n_idf_cov1", "n_soft_cov1", "a_tset", "a_num_any", "a_num_conflict", "r_e", "c_sc_rel"]]))
