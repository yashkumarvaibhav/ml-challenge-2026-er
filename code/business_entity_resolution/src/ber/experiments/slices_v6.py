"""Reviewer 00:25 UTC gate: singleton / false-merge slices, Package A (v4 universe, stage 2 + e5-small) vs v6
(stage 2 + e5-base), both ghost worlds, full holdout at each run's own tuned thresholds."""
import json

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.metrics import per_s1_f05, summarize
from ber.model import decide

sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
truth = load_train_pairs().select("s1", "m")
tcount = truth.join(hold, on="s1").group_by("s1").len().rename({"len": "T"})
runs = {"A": (C.CACHE / "s2" / "holdout_pred_w{w}.parquet", 0.4, 0.75),
        "v6": (C.CACHE / "v6" / "s2" / "holdout_pred_w{w}_e5base.parquet", 0.4, 0.75)}
rep = {}
for w in (20, 0):
    for name, (path, ta, tb) in runs.items():
        p = C.ROOT / str(path).format(w=w)
        if not p.exists():
            rep[f"w{w} {name}"] = "missing (A world-0 holdout preds were not saved)"
            continue
        pred = decide(pl.read_parquet(p), ta, tb).join(hold.select("s1"), on="s1")
        tp = pred.join(truth, on=["s1", "m"])
        fp = pred.join(truth, on=["s1", "m"], how="anti")
        ent = hold.join(tcount, on="s1", how="left").fill_null(0).join(pred.group_by("s1").len().rename({"len": "P"}), on="s1", how="left").fill_null(0)
        sing = ent.filter(pl.col("T") == 0)
        rep[f"w{w} {name}"] = {
            "macro_f05": summarize(per_s1_f05(pred, truth.join(hold.select("s1"), on="s1"), hold))["macro_f05"],
            "singletons": sing.height, "singleton_correct_empty": round((sing["P"] == 0).mean(), 4),
            "false_pairs": fp.height, "false_pairs_on_singletons": fp.join(sing.select("s1"), on="s1").height,
            "true_pairs": tp.height, "missed_true_pairs": truth.join(hold.select("s1"), on="s1").height - tp.height,
            "records_claimed_by_gt1_s1": pred.group_by("m").len().filter(pl.col("len") > 1).height,
            "nonsingleton_left_empty": round(ent.filter((pl.col("T") > 0) & (pl.col("P") == 0)).height / max(1, ent.filter(pl.col("T") > 0).height), 4)}
(C.EXPERIMENTS / "v6" / "slices_v6_vs_A.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
