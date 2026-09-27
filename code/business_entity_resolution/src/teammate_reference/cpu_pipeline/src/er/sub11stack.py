"""Final: sub-11 + stacker v2 changes (US/India) + sub11_plus France.

Stacker v2 (er.stack2; team e5base2 + ours + Qwen-7B / Qwen-1.5B LLM scores + competition features, 4-fold cross-fitted
on the holdout): holdout two-way 0.99063 / 0.99045 vs team alone 0.98979 / 0.98975 and team+ours blend 0.99026 / 0.99012.
US/India rows = sub-11 - (team-cut 0.4/0.7 pairs the stacker rejects) + (stacker pairs the team cut rejects).

  python -m er.sub11stack --t 0.7,0.75 --out output/sub11_stack
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from er.common import CACHE, ROOT, log
from er.france_post import read_pairs
from er.model import decide
from er.teamblend import TEAM
from er.write import write_pairs

SUB11 = ROOT.parent / "best_sub11_0.989213"


def main(t, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    s11 = read_pairs(SUB11 / "matching_results.tsv")
    st = pl.read_parquet(CACHE / "stack2_test.parquet")
    parts = []
    for c in ("US", "India"):
        ids = cty.filter(pl.col("country") == c).select("s1")
        team = pl.scan_parquet(TEAM / "test_pred_e5base2.parquet").select("s1", "m", "p").join(ids.lazy(), on="s1", how="semi").collect()
        b = decide(st.join(ids, on="s1", how="semi"), *t, "p")
        tc = decide(team, 0.4, 0.7, "p")
        add, rem = b.join(tc, on=["s1", "m"], how="anti"), tc.join(b, on=["s1", "m"], how="anti")
        base = s11.join(ids, on="s1", how="semi")
        add = add.join(base, on=["s1", "m"], how="anti")
        n_rem = base.join(rem, on=["s1", "m"], how="semi").height
        new = base.join(rem, on=["s1", "m"], how="anti").join(add.select("m"), on="m", how="anti")
        parts.append(pl.concat([new, add]).unique())
        log(f"sub11stack {c}: stacker {b.height:,} vs team cut {tc.height:,}; sub11 {base.height:,} -> {parts[-1].height:,} "
            f"(removed {n_rem:,}, added {add.height:,})")
    fr = cty.filter(pl.col("country") == "France").select("s1")
    parts.append(read_pairs(CACHE.parent / "output" / "sub11_plus" / "matching_results.tsv").join(fr, on="s1", how="semi"))
    final = pl.concat(parts)
    assert final.height == final.unique().height
    write_pairs(cty.select("s1"), final, "matched_entity_ids", out / "matching_results.tsv")
    from er.candunion import main as cu
    cu(str(out / "matching_results.tsv"), str(out / "candidate_pairs.tsv"), str(SUB11 / "candidate_pairs.tsv"))
    shutil.copyfile(CACHE.parent / "output" / "sub7_blend" / "code.zip", out / "code.zip")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")],
                       capture_output=True, text=True)
    log(f"sub11stack -> {out}: {final.height:,} pairs; validator: {r.stdout.strip()[-100:]}")
    print(r.stdout.strip()[-200:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", default="0.7,0.75")
    ap.add_argument("--out", default=str(CACHE.parent / "output" / "sub11_stack"))
    a = ap.parse_args()
    main(tuple(float(x) for x in a.t.split(",")), a.out)
