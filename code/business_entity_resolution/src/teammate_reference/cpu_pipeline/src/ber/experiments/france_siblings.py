"""Label-free France precision hypothesis (26 Sep): does the data contain 'sibling' businesses (same exact
name + same house number, different street) more often in France?  (a) among S1 entities of each country
(test and train); (b) among v6 accepted pairs: share whose record has name+number equal but no rare street
token in common with the S1 (street 'different'), and the stage-2 p of those pairs."""
import json
import sys

import polars as pl

from ber import config as C

RARE = 0.002


def recs(split, s):
    return pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "country", "n_tok", "a_tok", "a_nums"])


def street_tokens(d):
    n = d.group_by("country").len().rename({"len": "N"})
    t = d.select("entity_id", "country", pl.col("a_tok").list.unique()).explode("a_tok").drop_nulls().filter(~pl.col("a_tok").str.contains(r"\d"))
    rare = t.group_by("country", "a_tok").len().join(n, on="country").filter(pl.col("len") < RARE * pl.col("N")).select("country", "a_tok")
    return t.join(rare, on=["country", "a_tok"]).group_by("entity_id").agg(pl.col("a_tok").alias("st"))


def s1_siblings(split):
    d = recs(split, 1)
    d = d.join(street_tokens(d), on="entity_id", how="left").with_columns(pl.col("st").fill_null([]))
    k = d.select("entity_id", "country", pl.col("n_tok").list.unique().list.sort().list.join(" ").alias("n"), "a_nums", "st").explode("a_nums").drop_nulls("a_nums")
    p = k.join(k, on=["country", "n", "a_nums"], suffix="_2").filter(pl.col("entity_id") < pl.col("entity_id_2"))
    p = p.with_columns(pl.col("st").list.set_intersection("st_2").list.len().alias("shared"))
    sib = p.filter((pl.col("shared") == 0) & (pl.col("st").list.len() > 0) & (pl.col("st_2").list.len() > 0)).select("entity_id", "country")
    tot = d.group_by("country").len().rename({"len": "n_s1"})
    out = pl.concat([sib, sib.rename({"entity_id": "x"}).rename({"x": "entity_id"})]).unique().group_by("country").len().join(tot, on="country")
    return {r["country"]: round(r["len"] / r["n_s1"], 4) for r in out.iter_rows(named=True)}


rep = {"S1_with_same_name_same_number_diff_street_sibling": {"train": s1_siblings("train"), "test": s1_siblings("test")}}
(C.EXPERIMENTS / "v6" / "france_siblings.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
