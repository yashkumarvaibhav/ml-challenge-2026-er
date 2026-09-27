"""Three-model blend: team e5base2 + ours + our cross-encoder (scored on uncertain pairs only).
z = wt*logit(team) + wo*logit(ours) + wc*logit(ce), each missing score replaced by the team score (weights renormalised).

  python -m er.blend3 eval --ce-hold ce/ce_hold.parquet
  python -m er.blend3 build --ce-test ce/ce_test.parquet --wt 0.5 --wo 0.2 --wc 0.3 --t-any 0.6 --t-add 0.75
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

SUB11 = ROOT.parent / "best_sub11_0.989213"


def _lg(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def combine3(team, ours, ce, wt, wo, wc):
    d = team.rename({"p": "pt"}).join(ours.rename({"p": "po"}), on=["s1", "m"], how="left").join(ce, on=["s1", "m"], how="left")
    d = d.with_columns(pl.coalesce("po", "pt").alias("po"), pl.coalesce("ce", "pt").alias("ce"))
    z = (wt * _lg("pt") + wo * _lg("po") + wc * _lg("ce")) / (wt + wo + wc)
    return d.select("s1", "m", (1 / (1 + (-z).exp())).alias("p"))


GRID_T = list(itertools.product([0.5, 0.6, 0.7], [0.7, 0.75, 0.8]))
GRID_W = [(0.7, 0.3, 0.0), (0.5, 0.2, 0.3), (0.4, 0.2, 0.4), (0.5, 0.3, 0.2), (0.3, 0.2, 0.5), (0.6, 0.0, 0.4), (0.4, 0.3, 0.3)]


def evaluate(ce_path):
    hold = hold_s1()
    truth = truth_pairs().join(hold.select("s1"), on="s1")
    team = pl.read_parquet(TEAM / "holdout_pred_w20_e5base2.parquet").select("s1", "m", "p").join(hold.select("s1"), on="s1")
    ours = our_blend(["p22_train_holdout.parquet", "p23b_train_holdout.parquet"], hold.select("s1").lazy())
    ce = pl.read_parquet(ce_path).select("s1", "m", "ce")
    out = []
    for w in GRID_W:
        sc = combine3(team, ours, ce, *w)
        two = []
        for th, rh in (("A", "B"), ("B", "A")):
            ht, hr = hold.filter(pl.col("holdout_half") == th), hold.filter(pl.col("holdout_half") == rh)
            st, sr = sc.join(ht.select("s1"), on="s1"), sc.join(hr.select("s1"), on="s1")
            f, g = max((summarize(per_s1_f05(decide(st, *g, "p"), truth, ht))["macro_f05"], g) for g in GRID_T)
            rep = summarize(per_s1_f05(decide(sr, *g, "p"), truth, hr))
            two.append((rep["macro_f05"], g, rep["by_country"]))
        full = max((summarize(per_s1_f05(decide(sc, *g, "p"), truth, hold))["macro_f05"], g) for g in GRID_T)
        r = {"w": w, "two_way": two, "mean": round((two[0][0] + two[1][0]) / 2, 5), "full": full}
        log("blend3 eval " + json.dumps(r))
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


def build(ce_path, w, t_any, t_add, out):
    """US/India: sub-11 + (blend3 decisions vs team cut 0.4/0.7) changes; France: sub11_plus France rows."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    s11 = read_pairs(SUB11 / "matching_results.tsv")
    ce_all = pl.read_parquet(ce_path).select("s1", "m", "ce")
    parts = []
    for c in ("US", "India"):
        ids = cty.filter(pl.col("country") == c).select("s1")
        team = pl.scan_parquet(TEAM / "test_pred_e5base2.parquet").select("s1", "m", "p").join(ids.lazy(), on="s1", how="semi").collect()
        b = decide(combine3(team, our_blend(["p22_test.parquet", "p23b_test.parquet"], ids.lazy()), ce_all.join(ids, on="s1", how="semi"), *w),
                   t_any, t_add, "p")
        t = decide(team, 0.4, 0.7, "p")
        add, rem = b.join(t, on=["s1", "m"], how="anti"), t.join(b, on=["s1", "m"], how="anti")
        base = s11.join(ids, on="s1", how="semi")
        add = add.join(base, on=["s1", "m"], how="anti")
        n_rem = base.join(rem, on=["s1", "m"], how="semi").height
        new = base.join(rem, on=["s1", "m"], how="anti").join(add.select("m"), on="m", how="anti")
        parts.append(pl.concat([new, add]).unique())
        log(f"blend3 build {c}: sub11 {base.height:,} -> {parts[-1].height:,} (removed {n_rem:,}, added {add.height:,})")
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
    log(f"blend3 build -> {out}: {final.height:,} pairs; validator: {r.stdout.strip()[-100:]}")
    print(r.stdout.strip()[-200:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eval", "build"])
    ap.add_argument("--ce-hold")
    ap.add_argument("--ce-test")
    ap.add_argument("--wt", type=float, default=0.5)
    ap.add_argument("--wo", type=float, default=0.2)
    ap.add_argument("--wc", type=float, default=0.3)
    ap.add_argument("--t-any", type=float, default=0.6)
    ap.add_argument("--t-add", type=float, default=0.75)
    ap.add_argument("--out", default=str(CACHE.parent / "output" / "blend3"))
    a = ap.parse_args()
    if a.step == "eval":
        evaluate(a.ce_hold)
    else:
        build(a.ce_test, (a.wt, a.wo, a.wc), a.t_any, a.t_add, a.out)
