"""Size two evidence-backed France rules on the FULL France test output (v6 decisions), and the same rules on the
labelled US/India holdout (what they would do there, as a calibration).
R1 singleton: an S1 with exactly one accepted match whose stage-2 p < 0.95 -> predict empty.
R2 disguised same-address: a rejected candidate (p >= 0.02) with the same house numbers, address-token Jaccard >= 0.8,
   disjoint name tokens, record address present, and NO other world S1 at the record's exact normalized address
   (CE is added in a second pass for the France candidates)."""
import json

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.metrics import per_s1_f05, summarize
from ber.model import decide


def norm(split):
    return pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "n_tok", "a_tok", "a_nums", "a_null"]) for s in C.SOURCES]) \
        .with_columns(pl.col("a_tok").list.unique().list.sort().list.join(" ").alias("akey"))


def rules(pred, split, world_ids):
    acc = decide(pred, 0.4, 0.75).join(pred, on=["s1", "m"])
    k = acc.group_by("s1").agg(pl.len().alias("k"), pl.col("p").max().alias("pmax"))
    r1 = k.filter((pl.col("k") == 1) & (pl.col("pmax") < 0.95)).select("s1")
    r = norm(split)
    s1w = r.filter(pl.col("entity_id").str.starts_with("S1-")).join(world_ids, on="entity_id")
    kc = s1w.group_by("akey").len().rename({"len": "n_s1_at_key"})
    a = r.select(pl.col("entity_id").alias("s1"), pl.col("n_tok").alias("n1"), pl.col("a_tok").alias("t1"), pl.col("a_nums").alias("u1"), pl.col("akey").alias("k1"))
    b = r.select(pl.col("entity_id").alias("m"), pl.col("n_tok").alias("n2"), pl.col("a_tok").alias("t2"), pl.col("a_nums").alias("u2"), pl.col("a_null").alias("z2"), pl.col("akey").alias("k2"))
    rej = pred.filter(pl.col("p") >= 0.02).join(acc.select("s1", "m"), on=["s1", "m"], how="anti")
    x = rej.join(a, on="s1").join(b, on="m").filter(~pl.col("z2") & (pl.col("u1").list.len() > 0)
        & (pl.col("u1").list.sort() == pl.col("u2").list.sort())
        & (pl.col("n1").list.set_intersection("n2").list.len() == 0)
        & ((pl.col("t1").list.set_intersection("t2").list.len() / pl.col("t1").list.set_union("t2").list.len().clip(lower_bound=1)) >= 0.8))
    x = x.join(kc.rename({"akey": "k2"}), on="k2", how="left").with_columns(pl.col("n_s1_at_key").fill_null(0)) \
        .filter((pl.col("n_s1_at_key") - (pl.col("k1") == pl.col("k2")).cast(pl.UInt32)) == 0)
    return acc, r1, x.select("s1", "m", "p")


rep = {}
s1t = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base.parquet")
acc, r1, r2 = rules(t, "test", s1t.select(pl.col("s1").alias("entity_id")))
for cty in ("US", "India", "France"):
    ids = s1t.filter(pl.col("country") == cty).select("s1")
    rep[f"test {cty}"] = {"S1s": ids.height, "R1_S1s_emptied": r1.join(ids, on="s1").height, "R2_pairs_added": r2.join(ids, on="s1").height,
                          "R2_S1s": r2.join(ids, on="s1")["s1"].n_unique()}
r2.join(s1t.filter(pl.col("country") == "France"), on="s1").write_parquet(C.CACHE / "v6" / "france_r2_candidates.parquet")
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
h = pl.read_parquet(C.CACHE / "v6" / "s2" / "holdout_pred_w20_e5base.parquet").join(hold.select("s1"), on="s1")
hacc, hr1, hr2 = rules(h, "train", sp.filter(~pl.col("is_ghost")).select("entity_id"))
truth = load_train_pairs().select("s1", "m").join(hold.select("s1"), on="s1")
sing = hold.join(truth.select("s1").unique(), on="s1", how="anti").select("s1")
base = hacc.select("s1", "m")
f0 = summarize(per_s1_f05(base, truth, hold))["macro_f05"]
f1 = summarize(per_s1_f05(base.join(hr1, on="s1", how="anti"), truth, hold))["macro_f05"]
f2 = summarize(per_s1_f05(pl.concat([base, hr2.select("s1", "m")]).unique(), truth, hold))["macro_f05"]
rep["holdout US+India"] = {"R1_S1s": hr1.height, "R1_singleton_share": round(hr1.join(sing, on="s1").height / max(1, hr1.height), 4),
                           "R1_delta_macro": round(f1 - f0, 5), "R2_pairs": hr2.height,
                           "R2_true_rate": round(hr2.join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left")["y"].fill_null(0).mean(), 4) if hr2.height else None,
                           "R2_delta_macro": round(f2 - f0, 5)}
(C.EXPERIMENTS / "v6" / "france_rules_size.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
