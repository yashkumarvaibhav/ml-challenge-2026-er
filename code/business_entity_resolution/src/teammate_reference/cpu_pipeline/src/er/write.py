"""Write output/matching_results.tsv and output/candidate_pairs.tsv (one row per test S1, sorted IDs)."""
import filecmp
import json
import os
import subprocess
import sys

import polars as pl

from er.common import CACHE, ROOT, log


def write_pairs(s1: pl.DataFrame, pairs: pl.DataFrame, col: str, path):
    g = pairs.group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias(col))
    out = s1.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null("")).rename({"s1": "source1_entity_id"})
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def write(pred: pl.DataFrame, cand: pl.DataFrame, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    assert pred.join(cand, on=["s1", "m"], how="anti").height == 0, "prediction outside candidate set"
    write_pairs(s1, pred, "matched_entity_ids", outdir / "matching_results.tsv")
    write_pairs(s1, cand.select("s1", "m"), "candidate_entity_ids", outdir / "candidate_pairs.tsv")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(outdir / "matching_results.tsv"), "--candidate", str(outdir / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")], capture_output=True, text=True)
    log(f"validator: {r.stdout.strip()[-400:]} {r.stderr.strip()[-400:]}")
    ref, dst = CACHE.parent / "output" / "run2" / "candidate_pairs.tsv", outdir / "candidate_pairs.tsv"
    if ref.exists() and not ref.samefile(dst) and filecmp.cmp(ref, dst, shallow=False):
        dst.unlink()
        os.link(ref, dst)  # identical candidate file: share one copy on disk
    n = pred.group_by("s1").len()
    log(f"wrote {outdir}: {pred.height:,} matches, {n.height:,} non-empty S1 of {s1.height:,}; "
        f"{cand.height / s1.height:.2f} cand/S1")
    return r.returncode == 0
