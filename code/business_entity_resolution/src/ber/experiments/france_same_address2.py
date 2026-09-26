"""Refine the same-address / unrelated-name cell (26 Sep): split by (a) another world S1 has exactly the record's
normalized address (co-located business) and (b) the record name is the S1's initials. True rates on the labelled
US/India holdout; sizes in France/US/India test (samples from ce_judge_samples.parquet, rejected best candidates)."""
import json

import polars as pl

from ber import config as C

LEGAL = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "llc", "inc", "ltd", "corp", "co", "pvt", "private", "limited", "llp", "pllc", "lp", "pc"}
d = pl.read_parquet(C.CACHE / "v6" / "ce_judge_samples.parquet").filter(pl.col("kind") == "rej")
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
worlds = {"test": None, "train": sp.filter(~pl.col("is_ghost")).select("entity_id")}


def norm(split):
    return pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "n_tok", "a_tok", "a_nums", "a_null"]) for s in C.SOURCES])


def initials(toks):
    t = [x for x in toks if x]
    full = "".join(x[0] for x in t)
    core = "".join(x[0] for x in t if x not in LEGAL)
    return full, core


rep = {}
for split, st in (("test", "test"), ("train", "holdout")):
    r = norm(split).with_columns(pl.col("a_tok").list.unique().list.sort().list.join(" ").alias("akey"))
    s1w = r.filter(pl.col("entity_id").str.starts_with("S1-"))
    if worlds[split] is not None:
        s1w = s1w.join(worlds[split], on="entity_id")
    keycount = s1w.group_by("akey").len().rename({"len": "n_s1_at_key"})
    a = r.select(pl.col("entity_id").alias("s1"), pl.col("n_tok").alias("n1"), pl.col("a_tok").alias("t1"), pl.col("a_nums").alias("u1"), pl.col("akey").alias("k1"))
    b = r.select(pl.col("entity_id").alias("m"), pl.col("n_tok").alias("n2"), pl.col("a_tok").alias("t2"), pl.col("a_nums").alias("u2"), pl.col("a_null").alias("z2"), pl.col("akey").alias("k2"))
    x = d.filter(pl.col("set") == st).join(a, on="s1").join(b, on="m").with_columns(
        (pl.col("t1").list.set_intersection("t2").list.len() / pl.col("t1").list.set_union("t2").list.len().clip(lower_bound=1)).alias("ajac"),
        (pl.col("u1").list.sort() == pl.col("u2").list.sort()).alias("num_eq"),
        (pl.col("n1").list.set_intersection("n2").list.len() == 0).alias("name_disjoint"))
    cell = x.filter(pl.col("num_eq") & (pl.col("u1").list.len() > 0) & (pl.col("ajac") >= 0.8) & pl.col("name_disjoint") & ~pl.col("z2"))
    cell = cell.join(keycount.rename({"akey": "k2"}), on="k2", how="left").with_columns(pl.col("n_s1_at_key").fill_null(0))
    cell = cell.with_columns(((pl.col("n_s1_at_key") - (pl.col("k1") == pl.col("k2")).cast(pl.UInt32)) > 0).alias("other_s1_at_address"))
    ini = [any(len(t) >= 2 and (t == f or t == c) for t in n2) for n2, (f, c) in zip(cell["n2"].to_list(), [initials(n) for n in cell["n1"].to_list()])]
    cell = cell.with_columns(pl.Series("is_initials", ini))
    for cty in sorted(cell["country"].unique().to_list()):
        cc = cell.filter(pl.col("country") == cty)
        n_rej = x.filter(pl.col("country") == cty).height
        out = {}
        for name, sub in (("other_S1_at_address", cc.filter(pl.col("other_s1_at_address"))),
                          ("no_other_S1_at_address", cc.filter(~pl.col("other_s1_at_address"))),
                          ("no_other & initials", cc.filter(~pl.col("other_s1_at_address") & pl.col("is_initials"))),
                          ("no_other & not initials", cc.filter(~pl.col("other_s1_at_address") & ~pl.col("is_initials")))):
            out[f"{name} share_of_rejected"] = round(sub.height / max(1, n_rej), 4)
            if st == "holdout":
                out[f"{name} true_rate"] = round(sub["y"].mean(), 4) if sub.height else None
                out[f"{name} n"] = sub.height
        rep[f"{st} {cty}"] = out
(C.EXPERIMENTS / "v6" / "france_same_address2.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
