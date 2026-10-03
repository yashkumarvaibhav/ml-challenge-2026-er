#!/usr/bin/env python3
"""Check tracked/staged files and optionally all reachable history for public release.

Uses only the Python standard library. Reports rule/path/line, never matched values.
The file policy deliberately allows only reviewed source and documentation paths.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 256 * 1024
EXACT_PATHS = {
    ".gitignore", ".gitattributes", "README.md", "LICENSE", "NOTICE.md",
    ".github/workflows/public-release.yml", "scripts/check_public_release.py",
    "docs/methodology.md", "docs/reproduction.md", "docs/public-content.md",
    "code/business_entity_resolution/README.md",
    "code/business_entity_resolution/requirements.txt",
    "code/business_entity_resolution/src/teammate_reference/cpu_pipeline/README.md",
    "code/business_entity_resolution/src/teammate_reference/cpu_pipeline/requirements.txt",
}
SOURCE_PATH = re.compile(r"code/business_entity_resolution/src/(?:ber|teammate_reference)/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py\Z")
SHELL_PATH = re.compile(r"code/business_entity_resolution/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.sh\Z")
RULES = {
    "credential-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{24,})\b"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential-url": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    "private-ip": re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
    "personal-path": re.compile(r"/(?:home|Users)/[A-Za-z][A-Za-z0-9_-]+/"),
    "record-id": re.compile(r"\bS[123]-(?:[0-9a-f]{12,}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\b", re.I),
    "personal-email": re.compile(r"\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|outlook|hotmail|iiitd\.ac)\.(?:com|in)\b", re.I),
    "assigned-secret": re.compile(r"\b(?:password|api_key|access_token|client_secret)\s*=\s*['\"][A-Za-z0-9_+/=-]{12,}['\"]", re.I),
}
NO_REPLY = re.compile(r"(?:\d+\+)?[A-Za-z0-9_-]+@users\.noreply\.github\.com\Z")


def git(*args: str, input: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, input=input,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def check(path: str, data: bytes) -> list[str]:
    issues = []
    if path not in EXACT_PATHS and not SOURCE_PATH.fullmatch(path) and not SHELL_PATH.fullmatch(path):
        issues.append(f"unapproved-path: {path}")
    parts = Path(path).parts
    if any(p in {"artifacts", "models", "models2", "models3", "cache", "output", "handoff", "student_resource"} for p in parts):
        issues.append(f"private-runtime-directory: {path}")
    if len(data) > MAX_BYTES:
        issues.append(f"file-size-limit: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return issues + [f"binary-file: {path}"]
    if b"\0" in data:
        issues.append(f"binary-file: {path}")
    for rule, pattern in RULES.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            issues.append(f"{rule}: {path}:{line}")
    return issues


def index_checks() -> tuple[int, list[str]]:
    issues = []
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    count = 0
    for entry in filter(None, entries):
        meta, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = meta.decode().split()
        path = raw_path.decode("utf-8")
        count += 1
        if mode not in {"100644", "100755"} or stage != "0":
            issues.append(f"unsupported-git-entry: {path}")
            continue
        issues.extend(check(path, git("cat-file", "blob", oid)))
        local = ROOT / path
        if local.is_symlink():
            issues.append(f"symlink: {path}")
        elif local.is_file():
            # Also inspect unstaged edits so local verification is not stale.
            issues.extend(check(path, local.read_bytes()))
        else:
            issues.append(f"missing-worktree-file: {path}")
    return count, issues


def history_checks() -> tuple[int, list[str]]:
    commits = git("rev-list", "--all").decode().splitlines()
    issues = []
    blobs: dict[tuple[str, str], bool] = {}
    for commit in commits:
        for line in git("show", "-s", "--format=%ae%n%ce", commit).decode().splitlines():
            if not NO_REPLY.fullmatch(line):
                issues.append(f"non-noreply-commit-email: {commit[:12]}")
        for entry in filter(None, git("ls-tree", "-r", "-z", commit).split(b"\0")):
            meta, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            path = raw_path.decode()
            if kind != "blob" or mode not in {"100644", "100755"}:
                issues.append(f"unsupported-history-entry: {path}")
                continue
            blobs[(oid, path)] = True
    for oid, path in blobs:
        issues.extend(check(path, git("cat-file", "blob", oid)))
    # Annotated tag text can also contain information excluded from file scans.
    for line in git("for-each-ref", "--format=%(objecttype) %(objectname)", "refs/tags").decode().splitlines():
        kind, oid = line.split()
        if kind == "tag":
            tag = git("cat-file", "tag", oid).decode()
            for rule, pattern in RULES.items():
                if pattern.search(tag):
                    issues.append(f"{rule}: annotated-tag:{oid[:12]}")
    return len(commits), issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="include all reachable commits and annotated tags")
    args = parser.parse_args()
    count, issues = index_checks()
    if count == 0:
        issues.append("empty-index: stage the intended source release first")
    commits = 0
    if args.history:
        commits, more = history_checks()
        issues.extend(more)
    issues = sorted(set(issues))
    if issues:
        print("Public-content check FAILED (matched values are redacted):")
        for issue in issues:
            print(f"  {issue}")
        return 1
    print(f"Public-content check PASS: {count} tracked files; {commits} reachable commits checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
