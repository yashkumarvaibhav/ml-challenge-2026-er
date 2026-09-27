"""Splice France rows of one submission into another: team US/India rows + our France rows (matches and candidates).

  python -m er.splice --base ../team_files --france output/final --out output/team_us_in_our_fr
"""
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

from er.common import CACHE, ROOT, log


def main(base, france, out):
    base, france, out = Path(base), Path(france), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fr = set(pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"])
             .filter(pl.col("country") == "France")["entity_id"].to_list())
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        ours = {}
        with open(france / name) as f:
            next(f)
            for line in f:
                k = line.split("\t", 1)[0]
                if k in fr:
                    ours[k] = line
        n = 0
        with open(base / name) as f, open(out / name, "w") as g:
            g.write(next(f))
            for line in f:
                k = line.split("\t", 1)[0]
                if k in fr:
                    line, n = ours.get(k, k + "\t\n"), n + 1
                g.write(line if line.endswith("\n") else line + "\n")
        log(f"splice {name}: {n:,} France rows replaced")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")], capture_output=True, text=True)
    log(f"splice: {base} + France from {france} -> {out}; validator: {r.stdout.strip()[-200:]}")
    print(r.stdout.strip()[-300:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--france", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.base, a.france, a.out)
