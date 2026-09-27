"""Refined France check (after the withdrawn probe): accepted pairs sharing a house number whose rare address tokens
have NO exact overlap, split by the best fuzzy similarity between the two sides' rare tokens (RapidFuzz ratio):
typo-like (>= 75) vs genuinely different (< 60). Test by country vs holdout (with precision)."""
import json

import polars as pl
from rapidfuzz import fuzz

from ber import config as C
from ber.io import load_train_pairs
from ber.model import decide

RARE = 0.002


def recs(split, s):
    return pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "country", "a_tok", "a_nums"])


def with_rare(d):
    n = d.group_by("country").len().rename({"len": "N"})
    t = d.select("entity_id", "country", pl.col("a_tok").list.unique()).explode("a_tok").drop_nulls().filter(~pl.col("a_tok").str.contains(r"\d"))
    rare = t.group_by("country", "a_tok").len().join(n, on="country").filter(pl.col("len") < RARE * pl.col("N")).select("country", "a_tok")
    st = t.join(rare, on=["country", "a_tok"]).group_by("entity_id").agg(pl.col("a_tok").alias("st"))
    return d.join(st, on="entity_id", how="left").with_columns(pl.col("st").fill_null([]))


def tag(pairs, split):
    r = with_rare(pl.concat([recs(split, s) for s in C.SOURCES]))
    a = r.select(pl.col("entity_id").alias("s1"), "country", pl.col("a_nums").alias("n1"), pl.col("st").alias("st1"))
    b = r.select(pl.col("entity_id").alias("m"), pl.col("a_nums").alias("n2"), pl.col("st").alias("st2"))
    d = pairs.join(a, on="s1").join(b, on="m").filter(
        (pl.col("st1").list.len() > 0) & (pl.col("st2").list.len() > 0) & (pl.col("st1").list.set_intersection("st2").list.len() == 0)
        & (pl.col("n1").list.set_intersection("n2").list.len() > 0))
    best = [max((fuzz.ratio(x, y) for x in s1 for y in s2), default=0) for s1, s2 in zip(d["st1"].to_list(), d["st2"].to_list())]
    return d.with_columns(pl.Series("fz", best)), pairs.join(a.select("s1", "country"), on="s1")


rep = {}
t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base.parquet")
acc = decide(t, 0.4, 0.75)
d, allacc = tag(acc, "test")
for cty in ("US", "India", "France"):
    x, n = d.filter(pl.col("country") == cty), allacc.filter(pl.col("country") == cty).height
    rep[f"test {cty}"] = {"accepted": n, "typo_like(>=75)": round(x.filter(pl.col("fz") >= 75).height / n, 4),
                          "different(<60)": round(x.filter(pl.col("fz") < 60).height / n, 4), "different_pairs": x.filter(pl.col("fz") < 60).height}
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
h = pl.read_parquet(C.CACHE / "v6" / "s2" / "holdout_pred_w20_e5base.parquet").join(hold, on="s1")
hacc = decide(h, 0.4, 0.75)
truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
hd, hall = tag(hacc, "train")
hd = hd.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
for cty in ("US", "India"):
    x, n = hd.filter(pl.col("country") == cty), hall.filter(pl.col("country") == cty).height
    diff = x.filter(pl.col("fz") < 60)
    rep[f"holdout {cty}"] = {"accepted": n, "typo_like(>=75)": round(x.filter(pl.col("fz") >= 75).height / n, 4),
                             "different(<60)": round(diff.height / n, 4), "different_pairs": diff.height,
                             "precision_different": round(diff["y"].mean(), 4) if diff.height else None,
                             "precision_typo_like": round(x.filter(pl.col("fz") >= 75)["y"].mean(), 4)}
(C.EXPERIMENTS / "v6" / "france_fuzzy_street.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
