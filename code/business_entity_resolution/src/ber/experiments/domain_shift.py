"""Covariate-shift diagnostic (26 Sep): how different are France test candidate pairs from US/India pairs in the
stage-1 feature space? Domain classifier (LightGBM, France test pairs vs US/India test pairs) -> AUC, top features,
and the distribution of density-ratio weights p/(1-p) on US/India TRAINING pairs (for importance-weighted training
and a France-like weighted holdout evaluation)."""
import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from ber import config as C
from ber.features import FEATURES
from ber.features import build as build_features

rng = np.random.default_rng(C.SEED)
s1t = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"])
cand = pl.read_parquet(C.WORK / "cand" / "test.parquet")
world = s1t.select("entity_id")
samp = {}
for cty in ("France", "US", "India"):
    ids = s1t.filter(pl.col("country") == cty).sample(30_000, seed=C.SEED).select(pl.col("entity_id").alias("s1"))
    c = cand.join(ids, on="s1")
    samp[cty] = build_features("test", c, world, name=f"domain features {cty}").with_columns(pl.lit(cty).alias("country"))
d = pl.concat(list(samp.values()))
d.write_parquet(C.WORK / "domain_shift_sample.parquet")
y = (d["country"] == "France").to_numpy().astype(int)
X = d.select(FEATURES).to_numpy()
idx = rng.permutation(len(y))
tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
b = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "verbose": -1, "num_threads": C.N_PROCS},
              lgb.Dataset(X[tr], y[tr], feature_name=FEATURES), 300)
p = b.predict(X[te])
imp = sorted(zip(FEATURES, b.feature_importance("gain")), key=lambda x: -x[1])
tot = sum(g for _, g in imp)
rep = {"pairs": {c: int(v.height) for c, v in samp.items()}, "domain_auc": round(float(roc_auc_score(y[te], p)), 4),
       "top_shift_features": [(n, round(g / tot, 3)) for n, g in imp[:12]]}
q = p[y[te] == 0]
w = q / (1 - q)
rep["US/India pair weight quantiles (p/(1-p))"] = {str(k): round(float(np.quantile(w, k)), 4) for k in (0.5, 0.9, 0.99, 0.999)}
rep["effective_sample_share"] = round(float(w.sum() ** 2 / (w ** 2).sum() / len(w)), 4)
(C.EXP_RUN / "domain_shift.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
