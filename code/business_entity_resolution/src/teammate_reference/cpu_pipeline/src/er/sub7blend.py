"""US/India: apply the holdout-validated team+ours blend as changes on top of sub-7; France: rows of output/sub7_fr.

Holdout (US/India, 160k S1, tune one half / report the other): team e5base2 alone 0.98977; blend 0.7*logit(team) +
0.3*logit(ours) where we scored the pair (team score alone elsewhere), t_any 0.6 / t_add 0.75: 0.99019.
Sub-7's US/India rows are close to the team scores cut at 0.4 / 0.7, so the final US/India rows are
sub-7 - (team-cut pairs the blend rejects) + (blend pairs the team cut rejects); each record keeps one S1.

  python -m er.sub7blend --out output/sub7_blend
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from er.common import CACHE, ROOT, log
from er.final import blend as our_blend
from er.france_post import read_pairs
from er.model import decide
from er.teamblend import TEAM, combine
from er.write import write_pairs

SUB7 = Path(os.environ.get("ER_SUB7", ROOT.parent / "best_sub7_0.988599_tsvs"))   # team sub-2026-09-27-7 TSVs
W, T_ANY, T_ADD, REF = 0.7, 0.6, 0.75, (0.4, 0.7)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    sub7 = read_pairs(SUB7 / "matching_results.tsv")
    fr_rows = read_pairs(CACHE.parent / "output" / "sub7_fr" / "matching_results.tsv").join(cty.filter(pl.col("country") == "France"), on="s1", how="semi")
    parts = [fr_rows]
    for c in ("US", "India"):
        ids = cty.filter(pl.col("country") == c).select("s1")
        team = pl.scan_parquet(TEAM / "test_pred_e5base2.parquet").select("s1", "m", "p").join(ids.lazy(), on="s1", how="semi").collect()
        ours = our_blend(["p22_test.parquet", "p23b_test.parquet"], ids.lazy())
        b = decide(combine(team, ours, W), T_ANY, T_ADD, "p")
        t = decide(team, *REF, "p")
        add, rem = b.join(t, on=["s1", "m"], how="anti"), t.join(b, on=["s1", "m"], how="anti")
        base = sub7.join(ids, on="s1", how="semi")
        new = base.join(rem, on=["s1", "m"], how="anti").join(add.select("m"), on="m", how="anti")   # a re-assigned record leaves its old S1
        new = pl.concat([new, add]).unique()
        log(f"sub7blend {c}: blend {b.height:,} vs team cut {t.height:,}: +{add.height:,} / -{rem.height:,}; sub7 {base.height:,} -> {new.height:,}")
        parts.append(new)
    final = pl.concat(parts)
    assert final.height == final.unique().height
    write_pairs(cty.select("s1"), final, "matched_entity_ids", out / "matching_results.tsv")
    # candidates: the team's file plus any final pair it lacks
    cand = out / "candidate_pairs.tsv"
    subprocess.run([sys.executable, "-c", f"from er.candunion import main; main({str(out / 'matching_results.tsv')!r}, {str(cand)!r}, "
                                          f"{str(SUB7 / 'candidate_pairs.tsv')!r})"], check=True, env=os.environ)
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(cand), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")], capture_output=True, text=True)
    log(f"sub7blend -> {out}: {final.height:,} pairs; validator: {r.stdout.strip()[-160:]}")
    print(r.stdout.strip()[-300:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CACHE.parent / "output" / "sub7_blend"))
    main(ap.parse_args().out)
