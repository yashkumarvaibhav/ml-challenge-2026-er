"""Final candidate on top of the team's sub-11 (public 0.989213):
  US/India: + our holdout-validated blend changes (team+ours w 0.7, 0.6/0.75 vs the team cut 0.4/0.7) not already in sub-11;
  France:   - the extra house-number conflicts our veto drops (typo-protected, team p) that sub-11's veto kept.

  python -m er.sub11plus --out output/sub11_plus
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from er.common import CACHE, ROOT, log
from er.final import blend as our_blend
from er.france_post import read_pairs
from er.model import decide
from er.sub7blend import REF, T_ADD, T_ANY, W
from er.teamblend import TEAM, combine
from er.write import write_pairs

SUB11 = Path(os.environ.get("ER_SUB11", ROOT.parent / "best_sub11_0.989213"))   # team sub-2026-09-27-11 TSVs


def main(out, veto=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    s11 = read_pairs(SUB11 / "matching_results.tsv")
    parts = []
    for c in ("US", "India"):
        ids = cty.filter(pl.col("country") == c).select("s1")
        team = pl.scan_parquet(TEAM / "test_pred_e5base2.parquet").select("s1", "m", "p").join(ids.lazy(), on="s1", how="semi").collect()
        b = decide(combine(team, our_blend(["p22_test.parquet", "p23b_test.parquet"], ids.lazy()), W), T_ANY, T_ADD, "p")
        t = decide(team, *REF, "p")
        add, rem = b.join(t, on=["s1", "m"], how="anti"), t.join(b, on=["s1", "m"], how="anti")
        base = s11.join(ids, on="s1", how="semi")
        add = add.join(base, on=["s1", "m"], how="anti")
        n_rem = base.join(rem, on=["s1", "m"], how="semi").height
        new = base.join(rem, on=["s1", "m"], how="anti").join(add.select("m"), on="m", how="anti")
        new = pl.concat([new, add]).unique()
        log(f"sub11plus {c}: sub11 {base.height:,} -> {new.height:,} (removed {n_rem:,}, added {add.height:,})")
        parts.append(new)
    fr = cty.filter(pl.col("country") == "France").select("s1")
    base = s11.join(fr, on="s1", how="semi")
    veto = veto or CACHE.parent / "output" / "sub7_fr" / "france_numveto_dropped.parquet"   # veto list from run_sub11_plus.sh step 2
    drop = pl.read_parquet(veto).select("s1", "m")
    new = base.join(drop, on=["s1", "m"], how="anti")
    log(f"sub11plus France: sub11 {base.height:,} -> {new.height:,} (extra house-number conflicts removed {base.height - new.height:,})")
    parts.append(new)
    final = pl.concat(parts)
    assert final.height == final.unique().height and final["m"].n_unique() >= final.height - s11.height + s11["m"].n_unique() - 5000
    write_pairs(cty.select("s1"), final, "matched_entity_ids", out / "matching_results.tsv")
    cand = out / "candidate_pairs.tsv"
    from er.candunion import main as cu
    cu(str(out / "matching_results.tsv"), str(cand), str(SUB11 / "candidate_pairs.tsv"))
    if (CACHE.parent / "output" / "sub7_blend" / "code.zip").exists():
        shutil.copyfile(CACHE.parent / "output" / "sub7_blend" / "code.zip", out / "code.zip")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(cand), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")], capture_output=True, text=True)
    log(f"sub11plus -> {out}: {final.height:,} pairs; validator: {r.stdout.strip()[-120:]}")
    print(r.stdout.strip()[-200:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CACHE.parent / "output" / "sub11_plus"))
    ap.add_argument("--veto", help="france_numveto_dropped.parquet (default: output/sub7_fr/)")
    a = ap.parse_args()
    main(a.out, a.veto)
