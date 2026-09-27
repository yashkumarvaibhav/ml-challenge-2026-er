"""Cross-encoder as an independent judge of France (26 Sep): the CE (multilingual e5-base, raw text) normally scores
only the stage-1 uncertain band, so confident France decisions are never checked by it. Score stratified samples of
v6 ACCEPTED pairs and of REJECTED best candidates -- including the confident region outside the band -- for
France/US/India test and for the labelled US/India holdout (calibration: how often a CE disagreement is a real error).

Cells: accepted & CE<0.1 / CE<0.3  (CE says 'no' to an accepted pair)
       rejected & CE>0.9 / CE>0.7  (CE says 'yes' to a rejected pair)"""
import json

import numpy as np
import polars as pl

from ber import config as C
from ber.cross_encoder import score
from ber.io import load_train_pairs
from ber.model import decide

rng_seed = C.SEED + 31
N_ACC, N_REJ = 120_000, 80_000


def sample(pred, s1c, country, split):
    acc = decide(pred, 0.4, 0.75).join(pred, on=["s1", "m"]).join(s1c, on="s1").filter(pl.col("country") == country)
    # rejected: each S1's best non-accepted candidate with p >= 0.02 (the plausible misses and the empty S1s' best)
    rej = pred.join(s1c, on="s1").filter(pl.col("country") == country).join(acc.select("s1", "m"), on=["s1", "m"], how="anti") \
        .filter(pl.col("p") >= 0.02).sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).first()
    a = acc.sample(min(N_ACC, acc.height), seed=rng_seed).with_columns(pl.lit("acc").alias("kind"))
    r = rej.sample(min(N_REJ, rej.height), seed=rng_seed).with_columns(pl.lit("rej").alias("kind"))
    return pl.concat([a.select("s1", "m", "p", "kind", "country"), r.select("s1", "m", "p", "kind", "country")]), acc.height, rej.height


rep = {}
s1t = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base.parquet")
parts = []
for cty in ("France", "US", "India"):
    x, na, nr = sample(t, s1t, cty, "test")
    x = x.with_columns(pl.Series("ce", score(x.select("s1", "m"), "test", name=f"ce judge test {cty}")))
    parts.append(x.with_columns(pl.lit("test").alias("set")))
    rep[f"test {cty} population"] = {"accepted": na, "rejected_best": nr}
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
h = pl.read_parquet(C.CACHE / "v6" / "s2" / "holdout_pred_w20_e5base.parquet").join(hold.select("s1"), on="s1")
truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
for cty in ("US", "India"):
    x, na, nr = sample(h, hold, cty, "train")
    x = x.with_columns(pl.Series("ce", score(x.select("s1", "m"), "train", name=f"ce judge holdout {cty}")))
    parts.append(x.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0), pl.lit("holdout").alias("set")))
    rep[f"holdout {cty} population"] = {"accepted": na, "rejected_best": nr}
d = pl.concat(parts, how="diagonal_relaxed")
d.write_parquet(C.CACHE / "v6" / "ce_judge_samples.parquet")
for (st, cty), g in d.group_by(["set", "country"]):
    a, r = g.filter(pl.col("kind") == "acc"), g.filter(pl.col("kind") == "rej")
    out = {"acc_ce<0.1": round((a["ce"] < 0.1).mean(), 4), "acc_ce<0.3": round((a["ce"] < 0.3).mean(), 4),
           "acc_ce<0.1 & p>=0.995": round(((a["ce"] < 0.1) & (a["p"] >= 0.995)).mean(), 4),
           "rej_ce>0.9": round((r["ce"] > 0.9).mean(), 4), "rej_ce>0.7": round((r["ce"] > 0.7).mean(), 4)}
    if st == "holdout":
        for name, cell in (("acc_ce<0.1", a.filter(pl.col("ce") < 0.1)), ("acc_ce<0.3", a.filter(pl.col("ce") < 0.3)),
                           ("rej_ce>0.9", r.filter(pl.col("ce") > 0.9)), ("rej_ce>0.7", r.filter(pl.col("ce") > 0.7))):
            out[f"true_rate {name}"] = round(cell["y"].mean(), 4) if cell.height else None
            out[f"n {name}"] = cell.height
        out["true_rate accepted"] = round(a["y"].mean(), 4)
        out["true_rate rejected_best"] = round(r["y"].mean(), 4)
    rep[f"{st} {cty}"] = out
(C.EXPERIMENTS / "v6" / "ce_judge.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(rep, indent=1, default=str))
