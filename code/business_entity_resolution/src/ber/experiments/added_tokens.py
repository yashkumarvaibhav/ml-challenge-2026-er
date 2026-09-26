"""Idea 3 (26 Sep): label-free 'generator-added' name tokens per country = tokens much more frequent in S2/S3
names than in S1 names. Report the top added tokens per country and, for accepted / rejected-best test pairs,
the share whose record name contains an added token that the S1 name lacks."""
import json

import polars as pl

from ber import config as C
from ber.model import decide


def toks(split, s):
    return pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "country", "n_tok"])


s1 = toks("test", 1)
pool = pl.concat([toks("test", 2), toks("test", 3)])
n1 = s1.group_by("country").len().rename({"len": "N1"})
npl = pool.group_by("country").len().rename({"len": "NP"})
f1 = s1.select("country", pl.col("n_tok").list.unique()).explode("n_tok").group_by("country", "n_tok").len().rename({"len": "c1"})
fp = pool.select("country", pl.col("n_tok").list.unique()).explode("n_tok").group_by("country", "n_tok").len().rename({"len": "cp"})
f = fp.join(f1, on=["country", "n_tok"], how="left").with_columns(pl.col("c1").fill_null(0)).join(n1, on="country").join(npl, on="country") \
    .with_columns((pl.col("cp") / pl.col("NP")).alias("rp"), (pl.col("c1") / pl.col("N1")).alias("r1")) \
    .with_columns(((pl.col("rp") + 1e-5) / (pl.col("r1") + 1e-5)).alias("ratio"))
added = f.filter((pl.col("rp") >= 0.002) & (pl.col("ratio") >= 5))
rep = {}
for cty in ("US", "India", "France"):
    x = added.filter(pl.col("country") == cty).sort("rp", descending=True)
    rep[f"{cty} added tokens (pool share, ratio)"] = [(t, round(rp, 4), round(r, 1)) for t, rp, r in x.select("n_tok", "rp", "ratio").head(20).iter_rows()]
    rep[f"{cty} n_added"] = x.height
addset = added.select("country", "n_tok").with_columns(pl.lit(True).alias("is_added"))
t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base.parquet")
acc = decide(t, 0.4, 0.75)
s1n = s1.rename({"entity_id": "s1", "n_tok": "t1"})
pn = pool.drop("country").rename({"entity_id": "m", "n_tok": "t2"})
d = acc.join(s1n, on="s1").join(pn, on="m").with_columns(pl.col("t2").list.set_difference("t1").alias("extra")) \
    .select("s1", "m", "country", "extra").explode("extra").join(addset, left_on=["country", "extra"], right_on=["country", "n_tok"], how="left") \
    .group_by("s1", "m", "country").agg(pl.col("is_added").fill_null(False).any().alias("has_added"))
for cty in ("US", "India", "France"):
    x = d.filter(pl.col("country") == cty)
    rep[f"{cty} accepted pairs whose record adds a generator token"] = round(x["has_added"].mean(), 4)
(C.EXPERIMENTS / "v6" / "added_tokens.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
print(json.dumps(rep, indent=1, ensure_ascii=False))
