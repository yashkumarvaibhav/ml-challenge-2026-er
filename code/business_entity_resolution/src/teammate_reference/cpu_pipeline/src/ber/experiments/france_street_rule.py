"""Size the France-only 'street disagreement' rule on the v6 test output, and measure what the same rule would
cost on the US/India holdout (where labels exist): accepted pairs whose record and S1 both have rare street tokens
but share none (street 'different'), split by house-number agreement."""
import json
import sys

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.model import decide

RARE = 0.002
sys.path.insert(0, str(C.ROOT / "code/business_entity_resolution/src/ber/experiments"))
from france_siblings import recs, street_tokens  # noqa: E402


def tag(pairs, split):
    r = pl.concat([recs(split, s) for s in C.SOURCES])
    st = street_tokens(r)
    r = r.join(st, on="entity_id", how="left").with_columns(pl.col("st").fill_null([])).select("entity_id", "country", "a_nums", "st")
    a = r.rename({"entity_id": "s1", "a_nums": "n1", "st": "st1"})
    b = r.drop("country").rename({"entity_id": "m", "a_nums": "n2", "st": "st2"})
    d = pairs.join(a, on="s1").join(b, on="m")
    return d.with_columns(
        ((pl.col("st1").list.len() > 0) & (pl.col("st2").list.len() > 0) & (pl.col("st1").list.set_intersection("st2").list.len() == 0)).alias("street_diff"),
        (pl.col("n1").list.set_intersection("n2").list.len() > 0).alias("num_shared"))


rep = {}
# test: v6 accepted pairs
t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base.parquet")
acc = decide(t, 0.4, 0.75).join(t, on=["s1", "m"])
d = tag(acc, "test")
for cty in ("US", "India", "France"):
    x = d.filter(pl.col("country") == cty)
    sd = x.filter(pl.col("street_diff"))
    rep[f"test {cty}"] = {"accepted": x.height, "street_diff": sd.height, "share": round(sd.height / x.height, 4),
                          "street_diff_num_shared": sd.filter(pl.col("num_shared")).height,
                          "street_diff_p_ge_0.99": sd.filter(pl.col("p") >= 0.99).height}
# holdout: what would the rule remove on US/India, and how many of those are true?
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
h = pl.read_parquet(C.CACHE / "v6" / "s2" / "holdout_pred_w20_e5base.parquet").join(hold, on="s1")
hacc = decide(h, 0.4, 0.75).join(h, on=["s1", "m"])
truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
hd = tag(hacc, "train").join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
for cty in ("US", "India"):
    x = hd.filter(pl.col("country") == cty)
    sd = x.filter(pl.col("street_diff"))
    rep[f"holdout {cty}"] = {"accepted": x.height, "street_diff": sd.height, "share": round(sd.height / x.height, 4),
                             "street_diff_true_rate": round(sd["y"].mean(), 4) if sd.height else None,
                             "street_diff_num_shared_true_rate": round(sd.filter(pl.col("num_shared"))["y"].mean(), 4) if sd.filter(pl.col("num_shared")).height else None}
(C.EXPERIMENTS / "v6" / "france_street_rule.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
