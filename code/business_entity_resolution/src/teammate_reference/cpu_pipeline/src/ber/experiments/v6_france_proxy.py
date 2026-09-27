"""Label-free France retrieval check for v6 (plan-discussion 20:23 UTC): proxy positives = S1/record pairs with the
same exact normalized full name AND a shared address number, where that name+number key is unique to one S1 and
<= 3 records. Share retrieved by (a) the old v5 candidate set, (b) the v6 set, (c) v6 dense channels alone."""
import json
import sys

import polars as pl

from ber import config as C

nk = pl.col("n_tok").list.unique().list.sort().list.join(" ")


def keys(split, s):
    d = pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "country", "n_tok", "a_nums"])
    return d.select("entity_id", "country", nk.alias("n"), pl.col("a_nums").list.unique().alias("t")).filter(pl.col("n") != "").explode("t") \
        .drop_nulls("t").select("entity_id", "country", (pl.col("n") + "#" + pl.col("t")).alias("k"))


s1 = keys("test", 1)
pool = pl.concat([keys("test", 2), keys("test", 3)])
u1 = s1.group_by("k").agg(pl.len().alias("n1")).filter(pl.col("n1") == 1)
up = pool.group_by("k").agg(pl.len().alias("np")).filter(pl.col("np") <= 3)
proxy = s1.join(u1, on="k").rename({"entity_id": "s1"}).join(pool.join(up, on="k").select(pl.col("entity_id").alias("m"), "k"), on="k").select("s1", "m", "country").unique()
old = pl.read_parquet(C.CACHE / "cand" / "test.parquet", columns=["s1", "m"])
new = pl.read_parquet(C.CACHE / "v6" / "cand" / "test.parquet", columns=["s1", "m", "r_df", "r_dr"])
dense = new.filter(pl.col("r_df").is_not_null() | pl.col("r_dr").is_not_null()).select("s1", "m")
rep = {}
for cty in sorted(proxy["country"].unique().to_list()):
    p = proxy.filter(pl.col("country") == cty)
    rep[cty] = {"proxy_pairs": p.height,
                "old_v5_set": round(p.join(old, on=["s1", "m"]).height / p.height, 4),
                "v6_set": round(p.join(new.select("s1", "m"), on=["s1", "m"]).height / p.height, 4),
                "v6_dense_only": round(p.join(dense, on=["s1", "m"]).height / p.height, 4)}
cnt = new.join(pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1").group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("s1s"))
rep["v6_cand_per_s1"] = {r["country"]: round(r["pairs"] / r["s1s"], 2) for r in cnt.iter_rows(named=True)}
(C.EXPERIMENTS / "v6" / "france_proxy_retrieval.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
