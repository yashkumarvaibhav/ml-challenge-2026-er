"""Targeted test for the retrieved-but-rejected slice (reviewer 09:34 UTC): record-record neighbourhood evidence.
For holdout uncertain pairs (s1, m) (0.01 <= stage-2 p < 0.9, world 20): the record m's 10 nearest POOL records
(bi-encoder, GPU IVF). Each neighbour is 'own' (stage-1 p1 >= 0.5 with s1), 'other' (p1 >= 0.5 with some other S1 of
the world universe) or 'none'. AUC of the neighbourhood features and of p combined with them, overall and for
address-null records. Label-free features; labels only for the evaluation."""
import json

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ber import config as C
from ber.ann import IVF
from ber.io import load_train_pairs

W = C.CACHE / "v6" / "s2"
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "holdout_half")
truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
h = pl.read_parquet(W / "holdout_pred_w20_e5base.parquet").join(hold, on="s1")
f = pl.read_parquet(W / "s2feats_w20_e5base.parquet", columns=["s1", "m", "m_a_null"]).join(hold.select("s1"), on="s1")
unc = h.join(f, on=["s1", "m"], how="left").filter((pl.col("p") >= 0.01) & (pl.col("p") < 0.9)) \
    .join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
st1 = pl.read_parquet(W / "stage1_w20.parquet").filter(pl.col("p1") >= 0.5).select("s1", pl.col("m").alias("n"), "p1")
pids = pl.read_parquet(C.CACHE / "dense" / "ids_train_pool.parquet").with_row_index("j")
EP = np.load(C.CACHE / "emb" / "train_pool.npy", mmap_mode="r")
K = 10
rows = []
for cty in ("US", "India"):
    pc = pids.filter(pl.col("country") == cty).with_row_index("pj")
    qm = unc.select("m").unique().join(pc.select(pl.col("entity_id").alias("m"), "pj", "j"), on="m")
    if qm.height == 0:
        continue
    X = np.ascontiguousarray(EP[pc["j"].to_numpy()])
    idx = IVF(X, seed=C.SEED)
    D, I = idx.search(np.ascontiguousarray(EP[qm["j"].to_numpy()]), K + 1)
    del idx
    ent = pc["entity_id"].to_numpy()
    for r, (m, own_pj) in enumerate(zip(qm["m"].to_list(), qm["pj"].to_list())):
        k = 0
        for c, i in zip(D[r], I[r]):
            if i < 0 or i == own_pj or k >= K:
                continue
            rows.append((m, ent[i], float(c), k))
            k += 1
nb = pl.DataFrame(rows, schema=["m", "n", "cos", "rank"], orient="row")
pairs = unc.select("s1", "m").join(nb, on="m")
pairs = pairs.join(st1.rename({"p1": "p_own"}), on=["s1", "n"], how="left") \
    .join(st1.group_by("n").agg(pl.col("s1").alias("owners")), on="n", how="left") \
    .with_columns(pl.col("p_own").is_not_null().alias("own"),
                  (pl.col("owners").list.len().fill_null(0) - pl.col("p_own").is_not_null().cast(pl.UInt32) > 0).alias("other"))
feat = pairs.group_by("s1", "m").agg(
    (pl.col("own").cast(pl.Float64) * pl.col("cos")).sum().alias("own_w"), (pl.col("other").cast(pl.Float64) * pl.col("cos")).sum().alias("other_w"),
    ((~pl.col("own") & ~pl.col("other")).cast(pl.Float64) * pl.col("cos")).sum().alias("none_w"),
    pl.col("cos").filter(pl.col("own")).max().alias("own_max"), pl.col("cos").filter(pl.col("other")).max().alias("other_max"),
    pl.col("cos").filter(~pl.col("own") & ~pl.col("other")).max().alias("none_max"))
u = unc.join(feat, on=["s1", "m"], how="left").fill_null(0)
rep = {"uncertain_pairs": u.height, "true": int(u["y"].sum())}
cols = ["own_w", "other_w", "none_w", "own_max", "other_max", "none_max"]
for name, x in (("all", u), ("address-null", u.filter(pl.col("m_a_null") == 1))):
    y = x["y"].to_numpy()
    p = x["p"].to_numpy()
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    r = {"n": len(y), "auc_p": round(roc_auc_score(y, p), 4)}
    for c in cols:
        r[f"auc_{c}"] = round(roc_auc_score(y, x[c].to_numpy()), 4)
    # two-way (half A fit -> half B eval and vice versa): does adding the features to p help?
    Xf = np.column_stack([z] + [x[c].to_numpy() for c in cols])
    halves = x["holdout_half"].to_numpy()
    aucs = []
    for a, b in (("A", "B"), ("B", "A")):
        m_ = LogisticRegression(max_iter=2000).fit(Xf[halves == a], y[halves == a])
        aucs.append(round(roc_auc_score(y[halves == b], m_.predict_proba(Xf[halves == b])[:, 1]), 4))
    r["auc_p_plus_neighbour (two-way)"] = aucs
    r["auc_p (per half)"] = [round(roc_auc_score(y[halves == b], p[halves == b]), 4) for b in ("B", "A")]
    rep[name] = r
(C.EXPERIMENTS / "v6" / "neighbour_signal.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
