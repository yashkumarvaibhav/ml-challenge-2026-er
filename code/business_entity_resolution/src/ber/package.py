"""Validate outputs and build the portal upload package (PLAN.md B6, §8).

  python -m ber.package --tag sub-2026-09-25-1

1. Runs the organizer validator with --check-ids; any "matched IDs not present in
   candidate_pairs.tsv" warning is treated as a failure.
2. Builds the code ZIP (code/business_entity_resolution: src, README, requirements) from
   the committed HEAD (refuses a dirty tree for tracked code).
3. Archives matching_results.tsv, candidate_pairs.tsv and the code ZIP with SHA-256
   hashes in submissions/archive/<tag>/ and prints the log row.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone

from . import config as C


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate() -> str:
    cmd = [sys.executable, "student_resource/utils/validate_submission.py",
           "--matching", str(C.OUTPUT / "matching_results.tsv"),
           "--candidate", str(C.OUTPUT / "candidate_pairs.tsv"),
           "--test-dir", "student_resource/dataset/test", "--check-ids"]
    r = subprocess.run(cmd, cwd=C.ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0 or "not present in candidate_pairs.tsv" in out or "PASS" not in out:
        print(out)
        raise SystemExit("validator FAILED (or matched-not-in-candidates warning)")
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build_zip(dest) -> str:
    dirty = git("status", "--porcelain", "--", "code/")
    if dirty:
        raise SystemExit(f"refusing to package: uncommitted changes under code/:\n{dirty}")
    head = git("rev-parse", "HEAD")
    files = git("ls-files", "code/business_entity_resolution").splitlines()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(C.ROOT / f, arcname=f.replace("code/", "", 1))
        z.writestr("business_entity_resolution/GIT_COMMIT.txt", head + "\n")
    return head


def main(tag: str):
    out = validate()
    arch = C.ROOT / "submissions" / "archive" / tag
    arch.mkdir(parents=True, exist_ok=False)
    head = build_zip(arch / "code.zip")
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(C.OUTPUT / name, arch / name)
    now = datetime.now(timezone.utc)
    meta = {
        "tag": tag, "commit": head, "utc": now.strftime("%Y-%m-%d %H:%M"),
        "ist": (now + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M"),
        "sha256": {n: sha256(arch / n) for n in ("matching_results.tsv", "candidate_pairs.tsv", "code.zip")},
        "validator": [l for l in out.splitlines() if l.strip()][-3:],
    }
    decision = C.EXPERIMENTS / "decision_params.json"
    if decision.exists():
        meta["holdout"] = json.loads(decision.read_text())
    (arch / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    with zipfile.ZipFile(arch / "code.zip") as z:
        print(f"code.zip: {len(z.namelist())} files, e.g. {z.namelist()[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    main(ap.parse_args().tag)
