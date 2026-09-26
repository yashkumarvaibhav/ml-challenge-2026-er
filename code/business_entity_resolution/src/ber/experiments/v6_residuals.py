"""v6 residual errors (holdout, world 20, stage 2 + e5-base at 0.4/0.75): where do the remaining losses sit?
Misses: not retrieved vs retrieved-but-rejected (by stage-2 p band, CE present?, address-null, name equal,
whether the S1 got ANY match, whether a sibling record of the same S1 was accepted). False pairs: by type."""
import json
import sys

import polars as pl

from ber import config as C
from ber.io import load_source, load_train_pairs
from ber.metrics import per_s1_f05
from ber.model import decide

W = C.CACHE / "v6"
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
truth = load_train_pairs().select("s1", "m").join(hold.select("s1"), on="s1").with_columns(pl.lit(1).alias("y"))
h = pl.read_parquet(W / "s2" / "holdout_pred_w20_e5base.parquet")
feat = pl.read_parquet(W / "s2" / "s2feats_w20_e5base.parquet", columns=["s1", "m", "p1", "ce", "m_best_other", "s_top", "s_n_hi", "m_a_null", "d_cos"]).join(hold.select("s1"), on="s1")
acc = decide(h, 0.4, 0.75).with_columns(pl.lit(1).alias("a"))
d = h.join(feat, on=["s1", "m"], how="left").join(truth, on=["s1", "m"], how="left").join(acc, on=["s1", "m"], how="left") \
    .with_columns(pl.col("y").fill_null(0), pl.col("a").fill_null(0))
n_acc = acc.group_by("s1").len().rename({"len": "n_acc"})
n_true_acc = acc.join(truth, on=["s1", "m"]).group_by("s1").len().rename({"len": "n_true_acc"})
d = d.join(n_acc, on="s1", how="left").join(n_true_acc, on="s1", how="left").with_columns(pl.col("n_acc").fill_null(0), pl.col("n_true_acc").fill_null(0))
rep = {}
miss_nc = truth.join(h.select("s1", "m"), on=["s1", "m"], how="anti").height
fn = d.filter((pl.col("y") == 1) & (pl.col("a") == 0))
fp = d.filter((pl.col("y") == 0) & (pl.col("a") == 1))
rep["missed_not_retrieved"] = miss_nc
rep["missed_rejected"] = fn.height
rep["false_pairs"] = fp.height
band = lambda c: pl.when(pl.col(c) < 0.05).then(pl.lit("<0.05")).when(pl.col(c) < 0.2).then(pl.lit("0.05-0.2")).when(pl.col(c) < 0.4).then(pl.lit("0.2-0.4")).when(pl.col(c) < 0.75).then(pl.lit("0.4-0.75")).otherwise(pl.lit(">=0.75"))
rep["rejected_by_p2"] = dict(fn.group_by(band("p").alias("b")).len().sort("b").iter_rows())
rep["rejected_ce_present"] = round(fn["ce"].is_not_null().mean(), 3)
rep["rejected_record_address_null"] = round(fn["m_a_null"].fill_null(0).cast(pl.Float64).mean(), 3)
rep["rejected_S1_got_no_match_at_all"] = round((fn["n_acc"] == 0).mean(), 3)
rep["rejected_S1_has_other_true_accepted"] = round((fn["n_true_acc"] > 0).mean(), 3)
rep["rejected_record_claimed_by_other_S1_ge_0.5"] = round((fn["m_best_other"].fill_null(0) >= 0.5).mean(), 3)
rep["false_by_p2"] = dict(fp.group_by(band("p").alias("b")).len().sort("b").iter_rows())
rep["false_on_S1_with_no_true_match"] = fp.join(truth.select("s1").unique(), on="s1", how="anti").height
rep["false_record_address_null"] = round(fp["m_a_null"].fill_null(0).cast(pl.Float64).mean(), 3)
# macro-F0.5 loss attribution per S1
f = per_s1_f05(acc.select("s1", "m"), truth.select("s1", "m"), hold)
loss = f.with_columns((1 - pl.col("f05")).alias("loss")) if "f05" in f.columns else None
if loss is not None:
    tot = loss["loss"].sum()
    kinds = loss.join(fn.group_by("s1").len().rename({"len": "n_fn"}), on="s1", how="left").join(fp.group_by("s1").len().rename({"len": "n_fp"}), on="s1", how="left") \
        .with_columns(pl.col("n_fn").fill_null(0), pl.col("n_fp").fill_null(0))
    rep["loss_share_S1_with_only_FN"] = round(kinds.filter((pl.col("n_fn") > 0) & (pl.col("n_fp") == 0))["loss"].sum() / tot, 3)
    rep["loss_share_S1_with_FP"] = round(kinds.filter(pl.col("n_fp") > 0)["loss"].sum() / tot, 3)
    rep["loss_share_other(not retrieved only)"] = round(kinds.filter((pl.col("n_fn") == 0) & (pl.col("n_fp") == 0))["loss"].sum() / tot, 3)
(C.EXPERIMENTS / "v6" / "residuals_w20.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
s1t = load_source("train", 1).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
pool = pl.concat([load_source("train", s) for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(260)
print(fn.sample(22, seed=5).join(s1t, on="s1").join(pool, on="m").select(pl.col("p").round(3), pl.col("p1").round(3), pl.col("ce").round(3), "n_acc", "n1", "n2", "a1", "a2"))
