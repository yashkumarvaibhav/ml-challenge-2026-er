"""France singleton hypothesis (26 Sep): French names repeat heavily, so a singleton S1 almost always has a same-name
look-alike candidate and may receive a false match (costing 1.0 each). Label-free on test, labelled on the holdout:
per-S1 top stage-2 p distribution by country; S1s whose top p lies in the marginal zone [0.4, t); on the holdout,
how many of those are true singletons and what emptying them would do to macro-F0.5 (US/India)."""
import json

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.metrics import per_s1_f05, summarize
from ber.model import decide

W = C.CACHE / "v6" / "s2"
bins = [0.01, 0.1, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]


def top(d):
    return d.group_by("s1").agg(pl.col("p").max().alias("top"))


def dist(t, n_s1):
    out = {"<0.01 or none": round(1 - t.filter(pl.col("top") >= 0.01).height / n_s1, 4)}
    edges = bins + [1.01]
    for a, b in zip(edges[:-1], edges[1:]):
        out[f"[{a},{b})"] = round(t.filter((pl.col("top") >= a) & (pl.col("top") < b)).height / n_s1, 4)
    return out


rep = {}
s1c = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
t = pl.read_parquet(W / "test_pred_e5base.parquet")
tt = top(t).join(s1c, on="s1")
for cty in ("US", "India", "France"):
    n = s1c.filter(pl.col("country") == cty).height
    rep[f"test {cty} top-p per S1"] = dist(tt.filter(pl.col("country") == cty), n)
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
truth = load_train_pairs().select("s1", "m").join(hold.select("s1"), on="s1")
sing = hold.join(truth.select("s1").unique(), on="s1", how="anti").select("s1")
h = pl.read_parquet(W / "holdout_pred_w20_e5base.parquet").join(hold.select("s1"), on="s1")
ht = top(h).join(hold, on="s1")
for cty in ("US", "India"):
    n = hold.filter(pl.col("country") == cty).height
    x = ht.filter(pl.col("country") == cty)
    r = dist(x, n)
    r["true_singleton_share"] = round(sing.join(hold.filter(pl.col("country") == cty).select("s1"), on="s1").height / n, 4)
    for a, b in ((0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95)):
        z = x.filter((pl.col("top") >= a) & (pl.col("top") < b))
        r[f"singleton share among top in [{a},{b})"] = round(z.join(sing, on="s1").height / max(1, z.height), 3)
    rep[f"holdout {cty} top-p per S1"] = r
# what raising t_any would do on the labelled holdout (US+India)
base = summarize(per_s1_f05(decide(h, 0.4, 0.75), truth, hold))["macro_f05"]
rep["holdout macro-F0.5 by t_any (t_add 0.75)"] = {str(ta): round(summarize(per_s1_f05(decide(h, ta, 0.75), truth, hold))["macro_f05"] - base, 5) for ta in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
(C.EXPERIMENTS / "v6" / "france_singletons.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
