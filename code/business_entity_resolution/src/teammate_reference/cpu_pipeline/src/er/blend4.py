"""Four-source blend on the team's stage-2 universe: team e5base2 (pt) + ours (po) + the team's Qwen2.5-7B LoRA
pair scores (q7, stage-2 bands 0.05-0.95 and 0.95-0.995) + Qwen2.5-1.5B (q15, same bands plus the low band).
z = (wt*lt + wo*lo + w7*l7 + w15*l15) / (sum of weights); a missing source takes the team logit.
The LLMs were trained on leakage-safe train pairs (never holdout), so holdout scores are clean.

  python -m er.blend4 eval
  python -m er.blend4 build --w 0.4,0.2,0.3,0.1 --t 0.5,0.75
"""
import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from ber.metrics import per_s1_f05, summarize
from er.common import CACHE, ROOT, hold_s1, log, truth_pairs
from er.final import blend as our_blend
from er.france_post import read_pairs
from er.model import decide
from er.teamblend import TEAM
from er.write import write_pairs

L = CACHE / "llm"
SUB11 = ROOT.parent / "best_sub11_0.989213"


def _lg(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def llm(split, tag, bands):
    fs = [L / f"llm_{b}_{split}_{tag}.parquet" for b in bands]
    return pl.concat([pl.read_parquet(f).select("s1", "m", pl.col("llm").cast(pl.Float64)) for f in fs if f.exists()]).unique(["s1", "m"])


def sources(split, ids: pl.DataFrame):
    if split == "hold":
        team = pl.read_parquet(TEAM / "holdout_pred_w20_e5base2.parquet").select("s1", "m", "p")
        ours = our_blend(["p22_train_holdout.parquet", "p23b_train_holdout.parquet"], ids.lazy())
        q7 = pl.concat([pl.read_parquet(L / "llm_s3band_w20_qwen7b_r3.parquet"), pl.read_parquet(L / "llm_hiband_holdout_qwen7b_r3.parquet")])
        q15 = pl.concat([pl.read_parquet(L / f) for f in ("llm_s3band_w20_qwen15_r3.parquet", "llm_hiband_holdout_qwen15_r3.parquet",
                                                           "llm_loband_holdout_qwen15_r3.parquet")])
    else:
        team = pl.scan_parquet(TEAM / "test_pred_e5base2.parquet").select("s1", "m", "p").join(ids.lazy(), on="s1", how="semi").collect()
        ours = our_blend(["p22_test.parquet", "p23b_test.parquet"], ids.lazy())
        q7 = pl.concat([pl.read_parquet(L / "llm_s3band_test_qwen7b_r3.parquet"), pl.read_parquet(L / "llm_hiband_test_qwen7b_r3.parquet")])
        q15 = pl.concat([pl.read_parquet(L / f) for f in ("llm_s3band_test_qwen15_r3.parquet", "llm_hiband_test_qwen15_r3.parquet",
                                                           "llm_loband_test_qwen15_r3.parquet")])
    team = team.join(ids, on="s1", how="semi")
    q7 = q7.select("s1", "m", pl.col("llm").cast(pl.Float64).alias("q7")).unique(["s1", "m"])
    q15 = q15.select("s1", "m", pl.col("llm").cast(pl.Float64).alias("q15")).unique(["s1", "m"])
    d = team.rename({"p": "pt"}).join(ours.rename({"p": "po"}), on=["s1", "m"], how="left").join(q7, on=["s1", "m"], how="left") \
        .join(q15, on=["s1", "m"], how="left")
    return d.with_columns([pl.coalesce(c, "pt").alias(c) for c in ("po", "q7", "q15")])


def score(d, w):
    wt, wo, w7, w15 = w
    z = (wt * _lg("pt") + wo * _lg("po") + w7 * _lg("q7") + w15 * _lg("q15")) / (wt + wo + w7 + w15)
    return d.select("s1", "m", (1 / (1 + (-z).exp())).alias("p"))


GRID_T = list(itertools.product([0.4, 0.5, 0.6, 0.7], [0.7, 0.75, 0.8]))
GRID_W = [(1, 0, 0, 0), (0.7, 0.3, 0, 0), (0.7, 0, 0.3, 0), (0.5, 0.25, 0.25, 0), (0.45, 0.25, 0.2, 0.1), (0.4, 0.3, 0.3, 0),
          (0.4, 0.2, 0.4, 0), (0.35, 0.25, 0.3, 0.1), (0.5, 0.3, 0.2, 0), (0.3, 0.3, 0.4, 0), (0.4, 0.25, 0.25, 0.1)]


def evaluate():
    hold = hold_s1()
    truth = truth_pairs().join(hold.select("s1"), on="s1")
    d = sources("hold", hold.select("s1"))
    log(f"blend4 eval: {d.height:,} pairs")
    res = []
    for w in GRID_W:
        sc = score(d, w)
        two = []
        for th, rh in (("A", "B"), ("B", "A")):
            ht, hr = hold.filter(pl.col("holdout_half") == th), hold.filter(pl.col("holdout_half") == rh)
            st, sr = sc.join(ht.select("s1"), on="s1"), sc.join(hr.select("s1"), on="s1")
            f, g = max((summarize(per_s1_f05(decide(st, *g, "p"), truth, ht))["macro_f05"], g) for g in GRID_T)
            rep = summarize(per_s1_f05(decide(sr, *g, "p"), truth, hr))
            two.append((rep["macro_f05"], g))
        full = max((summarize(per_s1_f05(decide(sc, *g, "p"), truth, hold))["macro_f05"], g) for g in GRID_T)
        r = {"w": w, "two_way": two, "mean": round((two[0][0] + two[1][0]) / 2, 5), "full": full}
        res.append(r)
        log("blend4 eval " + json.dumps(r))
        print(json.dumps(r), flush=True)


def build(w, t, out):
    """US/India: sub-11 rows + (blend4 decisions vs team cut 0.4/0.7) changes; France: sub11_plus France rows."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    s11 = read_pairs(SUB11 / "matching_results.tsv")
    parts = []
    for c in ("US", "India"):
        ids = cty.filter(pl.col("country") == c).select("s1")
        d = sources("test", ids)
        b = decide(score(d, w), *t, "p")
        tc = decide(d.select("s1", "m", pl.col("pt").alias("p")), 0.4, 0.7, "p")
        add, rem = b.join(tc, on=["s1", "m"], how="anti"), tc.join(b, on=["s1", "m"], how="anti")
        base = s11.join(ids, on="s1", how="semi")
        add = add.join(base, on=["s1", "m"], how="anti")
        n_rem = base.join(rem, on=["s1", "m"], how="semi").height
        new = base.join(rem, on=["s1", "m"], how="anti").join(add.select("m"), on="m", how="anti")
        parts.append(pl.concat([new, add]).unique())
        log(f"blend4 build {c}: sub11 {base.height:,} -> {parts[-1].height:,} (removed {n_rem:,}, added {add.height:,})")
    fr = cty.filter(pl.col("country") == "France").select("s1")
    parts.append(read_pairs(CACHE.parent / "output" / "sub11_plus" / "matching_results.tsv").join(fr, on="s1", how="semi"))
    final = pl.concat(parts)
    write_pairs(cty.select("s1"), final, "matched_entity_ids", out / "matching_results.tsv")
    from er.candunion import main as cu
    cu(str(out / "matching_results.tsv"), str(out / "candidate_pairs.tsv"), str(SUB11 / "candidate_pairs.tsv"))
    shutil.copyfile(CACHE.parent / "output" / "sub7_blend" / "code.zip", out / "code.zip")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")],
                       capture_output=True, text=True)
    log(f"blend4 build -> {out}: {final.height:,} pairs; validator: {r.stdout.strip()[-100:]}")
    print(r.stdout.strip()[-200:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eval", "build"])
    ap.add_argument("--w", default="0.4,0.25,0.25,0.1")
    ap.add_argument("--t", default="0.5,0.75")
    ap.add_argument("--out", default=str(CACHE.parent / "output" / "blend4"))
    a = ap.parse_args()
    if a.step == "eval":
        evaluate()
    else:
        build(tuple(float(x) for x in a.w.split(",")), tuple(float(x) for x in a.t.split(",")), a.out)
