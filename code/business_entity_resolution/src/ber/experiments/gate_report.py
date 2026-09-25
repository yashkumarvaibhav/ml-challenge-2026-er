"""Gate report for a run's saved holdout predictions (two-way A/B, both ghost worlds).

Adds what the reviewer asked to see next to macro-F0.5: per-country scores and singleton
behaviour (true singletons predicted empty vs given a false match), plus non-singletons
predicted empty.

  python code/business_entity_resolution/src/ber/experiments/gate_report.py <pred_dir> <decision_params.json>
  e.g. cache/runs/v2 experiments/v2/decision_params.json   or   cache/pred experiments/decision_params.json
"""
import json
import sys
from pathlib import Path

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.metrics import per_s1_f05, summarize
from ber.model import decide, tune

if __name__ == "__main__":
    pred_dir, params_path = Path(sys.argv[1]), Path(sys.argv[2])
    prm = json.loads(params_path.read_text())
    truth = load_train_pairs().select("s1", "m")
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    out = {}
    for w in (20, 0):
        sc = pl.read_parquet(pred_dir / f"train_holdout_w{w}.parquet")
        for th, rh in (("A", "B"), ("B", "A")):
            best = tune(sc, truth, hold.filter(pl.col("holdout_half") == th))
            hr = hold.filter(pl.col("holdout_half") == rh)
            per = per_s1_f05(decide(sc.join(hr.select("s1"), on="s1"), best["t_any"], best["t_add"]), truth, hr)
            rep = summarize(per)
            sing = per.filter(pl.col("T") == 0)
            nons = per.filter(pl.col("T") > 0)
            rep.update({
                "params": {k: best[k] for k in ("t_any", "t_add")},
                "singletons": sing.height,
                "singleton_correct_empty": round((sing["P"] == 0).mean(), 4),
                "singleton_false_match": int((sing["P"] > 0).sum()),
                "nonsingleton_predicted_empty": round((nons["P"] == 0).mean(), 4),
            })
            out[f"w{w}_tune{th}_rep{rh}"] = rep
        full = per_s1_f05(decide(sc, prm["t_any"], prm["t_add"]), truth, hold)
        out[f"w{w}_full_at_final_params"] = summarize(full)
    print(json.dumps(out, indent=1))
    (pred_dir / "gate_report.json").write_text(json.dumps(out, indent=1))
