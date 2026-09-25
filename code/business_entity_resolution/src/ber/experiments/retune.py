"""Re-tune decision thresholds on saved holdout predictions with a wider grid (two-way A/B, both worlds)."""
import json

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.metrics import per_s1_f05, summarize
from ber.model import decide, tune

if __name__ == "__main__":
    truth = load_train_pairs().select("s1", "m")
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    ga = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gb = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    out = {}
    sc = {w: pl.read_parquet(C.CACHE / "pred" / f"train_holdout_w{w}.parquet") for w in (20, 0)}
    for w in (20, 0):
        for th, rh in (("A", "B"), ("B", "A")):
            best = tune(sc[w], truth, hold.filter(pl.col("holdout_half") == th), ga, gb)
            hr = hold.filter(pl.col("holdout_half") == rh)
            rep = summarize(per_s1_f05(decide(sc[w].join(hr.select("s1"), on="s1"), best["t_any"], best["t_add"]), truth, hr))
            out[f"w{w}_tune{th}_rep{rh}"] = {"params": best, "report": rep["macro_f05"], "by_country": rep["by_country"]}
            print(f"w{w} tune{th} rep{rh}", best, rep["macro_f05"], rep["by_country"], flush=True)
    # full-holdout table for the average of both worlds
    grid = []
    for ta in ga:
        for tb in gb:
            fs = [summarize(per_s1_f05(decide(sc[w], ta, tb), truth, hold))["macro_f05"] for w in (20, 0)]
            grid.append((round(sum(fs) / 2, 5), ta, tb))
    grid.sort(reverse=True)
    print("top full-holdout (avg of worlds):", grid[:8])
    out["full_grid_top"] = grid[:15]
    (C.EXPERIMENTS / "b5_retune_wide.json").write_text(json.dumps(out, indent=2))
