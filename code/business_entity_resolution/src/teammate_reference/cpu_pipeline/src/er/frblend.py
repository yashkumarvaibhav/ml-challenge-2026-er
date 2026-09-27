"""Last-slot variant: sub7_blend + the same team/ours blend applied to France as changes on top of sub7_fr's France rows
(team cut 0.4/0.7 vs blend w 0.7, 0.6/0.75), then the France category veto and house-number veto on the result.

  python -m er.frblend --out output/sub7_blend_fr
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from er import frnum
from er.common import CACHE, ROOT, log
from er.final import blend as our_blend
from er.france_post import read_pairs
from er.model import decide
from er.post import category_veto
from er.sub7blend import REF, T_ADD, T_ANY, W
from er.teamblend import TEAM, combine
from er.write import write_pairs


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    fr = cty.filter(pl.col("country") == "France").select("s1")
    base = read_pairs(CACHE.parent / "output" / "sub7_blend" / "matching_results.tsv")
    usin = base.join(fr, on="s1", how="anti")
    frows = base.join(fr, on="s1", how="semi")
    team = pl.read_parquet(CACHE / "team_fr_test.parquet").select("s1", "m", "p")
    ours = our_blend(["p22_test.parquet", "p23b_test.parquet"], fr.lazy())
    b = decide(combine(team, ours, W), T_ANY, T_ADD, "p")
    t = decide(team, *REF, "p")
    add, rem = b.join(t, on=["s1", "m"], how="anti"), t.join(b, on=["s1", "m"], how="anti")
    new = frows.join(rem, on=["s1", "m"], how="anti").join(add.select("m"), on="m", how="anti")
    new = pl.concat([new, add]).unique()
    kept, _, _ = category_veto(new, "test")
    kept, dropped = frnum.veto(kept, team)
    log(f"frblend France: blend {b.height:,} vs team cut {t.height:,}: +{add.height:,} / -{rem.height:,}; "
        f"sub7_fr France {frows.height:,} -> {new.height:,} -> after vetoes {kept.height:,}")
    final = pl.concat([usin, kept])
    write_pairs(cty.select("s1"), final, "matched_entity_ids", out / "matching_results.tsv")
    cand = out / "candidate_pairs.tsv"
    from er.candunion import main as cu
    cu(str(out / "matching_results.tsv"), str(cand), str(ROOT.parent / "best_sub7_0.988599_tsvs" / "candidate_pairs.tsv"))
    shutil.copyfile(CACHE.parent / "output" / "sub7_blend" / "code.zip", out / "code.zip")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(cand), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")], capture_output=True, text=True)
    log(f"frblend -> {out}: {final.height:,} pairs; validator: {r.stdout.strip()[-120:]}")
    print(r.stdout.strip()[-200:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CACHE.parent / "output" / "sub7_blend_fr"))
    main(ap.parse_args().out)
