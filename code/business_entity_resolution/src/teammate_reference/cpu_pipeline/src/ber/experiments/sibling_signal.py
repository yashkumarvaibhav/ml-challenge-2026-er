"""Is sibling evidence informative for the v6 residuals? (reviewer 19:12 W3 condition)
For holdout candidate pairs (s1, m) with 0.01 <= stage-2 p < 0.9 (the uncertain region where the misses sit):
  sib_max  = max bi-encoder cosine between m and the S1's OTHER confidently accepted records (p >= 0.9)
  own_cos  = cosine(m, S1)
Report AUC of p alone, sib_max alone, and a simple combination, overall and for address-null records."""
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from ber import config as C
from ber.io import load_train_pairs

W = C.CACHE / "v6"
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
h = pl.read_parquet(W / "s2" / "holdout_pred_w20_e5base.parquet").join(hold, on="s1")
f = pl.read_parquet(W / "s2" / "s2feats_w20_e5base.parquet", columns=["s1", "m", "m_a_null", "d_cos"]).join(hold, on="s1")
h = h.join(f, on=["s1", "m"], how="left").join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
conf = h.filter(pl.col("p") >= 0.9).select("s1", pl.col("m").alias("m2"))
unc = h.filter((pl.col("p") >= 0.01) & (pl.col("p") < 0.9))
pairs = unc.select("s1", "m").join(conf, on="s1").filter(pl.col("m") != pl.col("m2"))
pids = pl.read_parquet(C.CACHE / "dense" / "ids_train_pool.parquet").select(pl.col("entity_id").alias("m")).with_row_index("j")
EP = np.load(C.CACHE / "emb" / "train_pool.npy", mmap_mode="r")
pairs = pairs.join(pids, on="m").join(pids.rename({"m": "m2", "j": "j2"}), on="m2")
a, b = pairs["j"].to_numpy(), pairs["j2"].to_numpy()
cos = np.empty(len(a), np.float32)
for s in range(0, len(a), 1_000_000):
    cos[s:s + 1_000_000] = (EP[a[s:s + 1_000_000]].astype(np.float32) * EP[b[s:s + 1_000_000]].astype(np.float32)).sum(1)
sib = pairs.select("s1", "m").with_columns(pl.Series("c", cos)).group_by("s1", "m").agg(pl.col("c").max().alias("sib_max"), pl.len().alias("n_sib"))
u = unc.join(sib, on=["s1", "m"], how="left").with_columns(pl.col("sib_max").fill_null(-1.0), pl.col("n_sib").fill_null(0))
print(f"uncertain pairs {u.height:,}, true {int(u['y'].sum()):,}; with >=1 confident sibling {int((u['n_sib'] > 0).sum()):,}")
for name, x in (("all", u), ("address-null", u.filter(pl.col("m_a_null") == 1)), ("address-present", u.filter(pl.col("m_a_null") == 0))):
    y = x["y"].to_numpy()
    if y.min() == y.max():
        continue
    p, s, d = x["p"].to_numpy(), x["sib_max"].to_numpy(), x["d_cos"].fill_null(0).to_numpy()
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    print(f"{name:16s} n={len(y):,} pos={y.mean():.3f}  AUC p={roc_auc_score(y, p):.4f}  sib_max={roc_auc_score(y, s):.4f}  d_cos={roc_auc_score(y, d):.4f}  p+4*sib={roc_auc_score(y, z + 4 * s):.4f}")
