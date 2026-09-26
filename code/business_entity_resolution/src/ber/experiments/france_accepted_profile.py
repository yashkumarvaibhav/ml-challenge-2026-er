"""Label-free profile of v6 ACCEPTED test pairs by country (vs the holdout, where precision is known):
stage-2 p, stage-1 p1, CE score (band only), bi-encoder cosine; share of accepted pairs with low CE / low cosine."""
import json

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.model import decide

W = C.CACHE / "v6" / "s2"
s1c = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
t = pl.read_parquet(W / "test_pred_e5base.parquet")
acc = decide(t, 0.4, 0.75).join(t, on=["s1", "m"])
st1 = pl.read_parquet(C.CACHE / "v6" / "pred" / "test.parquet").rename({"p": "p1"})
ce = pl.read_parquet(W / "ce_test_e5base.parquet")
cand = pl.read_parquet(C.CACHE / "v6" / "cand" / "test.parquet", columns=["s1", "m", "d_cos"])
d = acc.join(st1, on=["s1", "m"], how="left").join(ce, on=["s1", "m"], how="left").join(cand, on=["s1", "m"], how="left").join(s1c, on="s1")
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
h = pl.read_parquet(W / "holdout_pred_w20_e5base.parquet").join(hold.select("s1"), on="s1")
hf = pl.read_parquet(W / "s2feats_w20_e5base.parquet", columns=["s1", "m", "p1", "ce", "d_cos"]).join(hold.select("s1"), on="s1")
hacc = decide(h, 0.4, 0.75).join(h, on=["s1", "m"]).join(hf, on=["s1", "m"], how="left").join(hold, on="s1")
truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
hacc = hacc.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))


def prof(x):
    return {"n": x.height, "p<0.9": round((x["p"] < 0.9).mean(), 4), "p1<0.5": round((x["p1"] < 0.5).mean(), 4),
            "ce_present": round(x["ce"].is_not_null().mean(), 4), "ce<0.5 (of all)": round((x["ce"].fill_null(1) < 0.5).mean(), 4),
            "d_cos<0.8": round((x["d_cos"] < 0.8).mean(), 4), "d_cos median": round(float(x["d_cos"].median()), 4)}


rep = {}
for cty in ("US", "India", "France"):
    rep[f"test {cty}"] = prof(d.filter(pl.col("country") == cty))
for cty in ("US", "India"):
    x = hacc.filter(pl.col("country") == cty)
    r = prof(x)
    r["precision"] = round(x["y"].mean(), 4)
    r["precision | ce<0.5"] = round(x.filter(pl.col("ce").fill_null(1) < 0.5)["y"].mean(), 4)
    r["precision | d_cos<0.8"] = round(x.filter(pl.col("d_cos") < 0.8)["y"].mean(), 4)
    rep[f"holdout {cty}"] = r
(C.EXPERIMENTS / "v6" / "france_accepted_profile.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
