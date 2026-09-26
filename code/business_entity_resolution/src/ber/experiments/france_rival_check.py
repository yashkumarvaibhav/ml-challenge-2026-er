"""Reviewer 09:25 UTC scope check for the France probe.
(a) corrected S1 sibling share: BOTH members of each same-name + shared-number + no-shared-rare-address-token pair;
(b) of the 63,872 France pairs removed by france_street_variant.py: how many have a RIVAL S1 (not the matched S1)
    with the same normalized full name and a house number shared with the record, AND >= 1 rare address token
    shared between the record and the rival (address evidence favoring the rival); affected S1 counts; examples."""
import json

import polars as pl

from ber import config as C
from ber.io import load_source
from ber.model import decide

RARE = 0.002


def recs(split, s):
    return pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "country", "n_tok", "a_tok", "a_nums"])


def with_rare(d):
    n = d.group_by("country").len().rename({"len": "N"})
    t = d.select("entity_id", "country", pl.col("a_tok").list.unique()).explode("a_tok").drop_nulls().filter(~pl.col("a_tok").str.contains(r"\d"))
    rare = t.group_by("country", "a_tok").len().join(n, on="country").filter(pl.col("len") < RARE * pl.col("N")).select("country", "a_tok")
    st = t.join(rare, on=["country", "a_tok"]).group_by("entity_id").agg(pl.col("a_tok").alias("st"))
    return d.join(st, on="entity_id", how="left").with_columns(pl.col("st").fill_null([]),
                                                               pl.col("n_tok").list.unique().list.sort().list.join(" ").alias("nm"))


rep = {}
for split in ("train", "test"):
    d = with_rare(recs(split, 1))
    k = d.select("entity_id", "country", "nm", "a_nums", "st").explode("a_nums").drop_nulls("a_nums")
    p = k.join(k, on=["country", "nm", "a_nums"], suffix="_2").filter(pl.col("entity_id") != pl.col("entity_id_2"))
    p = p.filter((pl.col("st").list.len() > 0) & (pl.col("st_2").list.len() > 0) & (pl.col("st").list.set_intersection("st_2").list.len() == 0))
    tot = d.group_by("country").len().rename({"len": "n"})
    rep[f"S1 with a sibling ({split}, both members)"] = {r["country"]: round(r["len"] / r["n"], 4) for r in
                                                         p.select("entity_id", "country").unique().group_by("country").len().join(tot, on="country").iter_rows(named=True)}

allr = pl.concat([recs("test", s) for s in C.SOURCES])
allr = with_rare(allr)
s1 = allr.filter(pl.col("entity_id").str.starts_with("S1-")).select(pl.col("entity_id").alias("s1"), "country", pl.col("nm").alias("nm1"), pl.col("a_nums").alias("n1"), pl.col("st").alias("st1"))
rec = allr.filter(~pl.col("entity_id").str.starts_with("S1-")).select(pl.col("entity_id").alias("m"), pl.col("nm").alias("nm2"), pl.col("a_nums").alias("n2"), pl.col("st").alias("st2"))
t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base.parquet")
acc = decide(t, 0.4, 0.75)
d = acc.join(s1, on="s1").join(rec, on="m").filter(pl.col("country") == "France")
removed = d.filter((pl.col("st1").list.len() > 0) & (pl.col("st2").list.len() > 0) & (pl.col("st1").list.set_intersection("st2").list.len() == 0)
                   & (pl.col("n1").list.set_intersection("n2").list.len() > 0))
rep["removed_pairs"] = removed.height
rep["removed_affected_S1"] = removed["s1"].n_unique()
rep["removed_name_equal_to_matched_S1"] = int((removed["nm1"] == removed["nm2"]).sum())
# rivals: other France S1 with the record's normalized name and a number shared with the record
rv = removed.select("s1", "m", "nm2", "n2", "st2").explode("n2").drop_nulls("n2") \
    .join(s1.filter(pl.col("country") == "France").select(pl.col("s1").alias("rival"), pl.col("nm1").alias("nm2"), pl.col("n1"), pl.col("st1").alias("st_r")).explode("n1").rename({"n1": "n2"}), on=["nm2", "n2"]) \
    .filter(pl.col("rival") != pl.col("s1"))
rv = rv.with_columns((pl.col("st2").list.set_intersection("st_r").list.len() > 0).alias("addr_favours_rival"))
g = rv.group_by("s1", "m").agg(pl.len().alias("n_rivals"), pl.col("addr_favours_rival").any().alias("fav"))
rep["removed_with_any_same_name_number_rival_S1"] = g.height
rep["removed_with_rival_and_address_evidence_for_rival"] = int(g["fav"].sum())
rep["removed_S1_affected_with_rival_evidence"] = g.filter("fav")["s1"].n_unique()
(C.EXPERIMENTS / "v6" / "france_rival_check.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
raw1 = load_source("test", 1).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
rawp = pl.concat([load_source("test", s) for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(12); pl.Config.set_tbl_width_chars(250)
print("with rival evidence:"); print(g.filter("fav").sample(min(5, int(g["fav"].sum())), seed=1).join(raw1, on="s1").join(rawp, on="m").select("n1", "a1", "n2", "a2"))
print("no rival at all:"); print(removed.join(g.select("s1", "m"), on=["s1", "m"], how="anti").sample(5, seed=1).join(raw1, on="s1").join(rawp, on="m").select("n1", "a1", "n2", "a2"))
