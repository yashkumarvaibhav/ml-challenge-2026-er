"""France recall hypothesis (26 Sep, from the CE judge): rejected best candidates whose record has the SAME normalized
address as the S1 but an unrelated name (initials / pseudo-word / domain) and CE > 0.9 look like true matches.
Cell = rejected-best pair with: numbers equal (non-empty), address-token Jaccard >= 0.8, name-token overlap = 0.
Report its size per country (test) and its true rate on the labelled US/India holdout, by CE level."""
import json

import polars as pl

from ber import config as C

d = pl.read_parquet(C.CACHE / "v6" / "ce_judge_samples.parquet").filter(pl.col("kind") == "rej")


def norm(split):
    return pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "n_tok", "a_tok", "a_nums", "a_null"]) for s in C.SOURCES])


rep = {}
for split, st in (("test", "test"), ("train", "holdout")):
    r = norm(split)
    a = r.rename({"entity_id": "s1", "n_tok": "n1", "a_tok": "t1", "a_nums": "u1", "a_null": "z1"})
    b = r.rename({"entity_id": "m", "n_tok": "n2", "a_tok": "t2", "a_nums": "u2", "a_null": "z2"})
    x = d.filter(pl.col("set") == st).join(a, on="s1").join(b, on="m").with_columns(
        (pl.col("t1").list.set_intersection("t2").list.len() / pl.col("t1").list.set_union("t2").list.len().clip(lower_bound=1)).alias("ajac"),
        (pl.col("u1").list.sort() == pl.col("u2").list.sort()).alias("num_eq"),
        (pl.col("n1").list.set_intersection("n2").list.len() == 0).alias("name_disjoint"))
    cell = x.filter(pl.col("num_eq") & (pl.col("u1").list.len() > 0) & (pl.col("ajac") >= 0.8) & pl.col("name_disjoint") & ~pl.col("z2"))
    for cty in sorted(x["country"].unique().to_list()):
        xc, cc = x.filter(pl.col("country") == cty), cell.filter(pl.col("country") == cty)
        out = {"rejected_best_sampled": xc.height, "cell_share": round(cc.height / max(1, xc.height), 4)}
        for lo in (0.5, 0.9):
            c2 = cc.filter(pl.col("ce") > lo)
            out[f"cell&ce>{lo} share"] = round(c2.height / max(1, xc.height), 4)
            if st == "holdout":
                out[f"cell&ce>{lo} true_rate"] = round(c2["y"].mean(), 4) if c2.height else None
                out[f"cell&ce>{lo} n"] = c2.height
        if st == "holdout":
            out["cell (any ce) true_rate"] = round(cc["y"].mean(), 4) if cc.height else None
        rep[f"{st} {cty}"] = out
(C.EXPERIMENTS / "v6" / "france_same_address.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
