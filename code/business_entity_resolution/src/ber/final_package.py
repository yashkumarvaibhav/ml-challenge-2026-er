"""Final submission package (organizer layout) from a validated portal archive, with an audit.

  <team>_submission.zip
  ├── output/matching_results.tsv, output/candidate_pairs.tsv     (byte-identical to submissions/archive/<tag>/)
  ├── code/business_entity_resolution/                             (the archive's code.zip = committed code at the tag)
  │   └── artifacts/                                               (small learned files + MODELS.md manifest)
  └── Documentation_template.md                                    (docs/Documentation.md at HEAD)

Large fine-tuned encoder weights are NOT inside by default (MODELS.md lists their SHA-256 and how to regenerate
them from the supplied data); --with-weights adds them.

  python -m ber.final_package --tag sub-2026-09-26-3 [--team GitBoosters] [--with-weights]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import config as C

SMALL = {  # learned artifacts needed to run inference without retraining the LightGBM models
    "cache/maps/maps.json": "learned normalization maps (training-portion pairs)",
    "cache/maps/self_maps_test.json": "self-trained France normalization maps (test pseudo-pairs, FAQ-allowed)",
    "models/lgb_v4.txt": "v4 LightGBM (prerequisite: France pseudo-pairs, encoder hard negatives)",
    "models/lgb_v6.txt": "v6 stage-1 LightGBM",
    "models/lgb_s2_v6_e5base.txt": "v6 stage-2 LightGBM (with e5-base CE feature)",
    "experiments/v6/decision_params_s2_e5base.json": "decision thresholds used for matching_results.tsv",
}
LARGE = {
    "models/dense_e5small/model.pt": ("fine-tuned bi-encoder (dense retrieval)", "intfloat/multilingual-e5-small", "MIT", "117.7M",
                                      "614241f622f53c4eeff9890bdc4f31cfecc418b3", "python -m ber.dense --steps data,tokenize,train"),
    "models/ce_e5base/model.pt": ("fine-tuned cross-encoder (stage-2 band feature)", "intfloat/multilingual-e5-base", "MIT", "278.0M",
                                  "d128750597153bb5987e10b1c3493a34e5a4502a", "BER_CE=e5base python -m ber.cross_encoder --steps data,train"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def models_md() -> str:
    lines = ["# Models and learned artifacts", "",
             "All models are fine-tuned or trained only on the supplied training data and run offline. No external data or APIs.", "",
             "## Pretrained encoders (downloaded once, pinned)", "",
             "| Fine-tuned file | Role | Base model | Licence | Parameters | Base revision | SHA-256 of fine-tuned weights | Regenerate |",
             "|---|---|---|---|---|---|---|---|"]
    for f, (role, base, lic, n, rev, cmd) in LARGE.items():
        p = C.ROOT / f
        lines.append(f"| `{f}` | {role} | `{base}` | {lic} | {n} | `{rev}` | `{sha256(p) if p.exists() else 'missing'}` | `{cmd}` |")
    lines += ["", "## Included small artifacts (`artifacts/`)", "", "| File | What | SHA-256 |", "|---|---|---|"]
    for f, what in SMALL.items():
        p = C.ROOT / f
        lines.append(f"| `{Path(f).name}` | {what} | `{sha256(p)}` |")
    lines += ["", "LightGBM (MIT) models are trained locally by `ber.pipeline` / `ber.stage2`."]
    return "\n".join(lines) + "\n"


def main(tag: str, team: str, with_weights: bool):
    arch = C.ROOT / "submissions" / "archive" / tag
    meta = json.loads((arch / "meta.json").read_text())
    for n in ("matching_results.tsv", "candidate_pairs.tsv", "code.zip"):
        assert sha256(arch / n) == meta["sha256"][n], f"archive file changed since packaging: {n}"
    out_dir = C.ROOT / "submissions" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"{team}_submission{'_with_weights' if with_weights else ''}.zip"
    doc = subprocess.run(["git", "show", "HEAD:docs/Documentation.md"], cwd=C.ROOT, capture_output=True, text=True, check=True).stdout
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(arch / n, f"output/{n}")
        with zipfile.ZipFile(arch / "code.zip") as cz:
            for info in cz.infolist():
                if not info.is_dir():
                    z.writestr(f"code/{info.filename}", cz.read(info.filename))
        for f in SMALL:
            z.write(C.ROOT / f, f"code/business_entity_resolution/artifacts/{Path(f).name}")
        z.writestr("code/business_entity_resolution/artifacts/MODELS.md", models_md())
        if with_weights:
            for f in LARGE:
                z.write(C.ROOT / f, f"code/business_entity_resolution/artifacts/{f.replace('/', '__')}")
        z.writestr("Documentation_template.md", doc)
    audit(zpath, arch, meta, tag)


def audit(zpath: Path, arch: Path, meta: dict, tag: str):
    """Structure, byte-identity of the outputs, official validator on the zipped outputs."""
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        assert z.testzip() is None, "corrupt member"
        need = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt"]
        missing = [n for n in need if n not in names]
        assert not missing, f"missing {missing}"
        assert any(n.startswith("code/business_entity_resolution/src/ber/") for n in names), "no source"
        with tempfile.TemporaryDirectory(dir=C.ROOT / "cache") as td:
            z.extract("output/matching_results.tsv", td)
            z.extract("output/candidate_pairs.tsv", td)
            for n in ("matching_results.tsv", "candidate_pairs.tsv"):
                assert sha256(Path(td) / "output" / n) == meta["sha256"][n], f"{n} differs from the archive"
            v = subprocess.run(["python3", "utils/validate_submission.py", "--matching", str(Path(td) / "output/matching_results.tsv"),
                                "--candidate", str(Path(td) / "output/candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                               cwd=C.ROOT / "student_resource", capture_output=True, text=True)
    ok = "PASS" in v.stdout
    rep = {"zip": str(zpath.relative_to(C.ROOT)), "sha256": sha256(zpath), "bytes": zpath.stat().st_size, "members": len(names),
           "from_archive": tag, "archive_commit": meta["commit"], "outputs_identical_to_archive": True,
           "validator": [l for l in v.stdout.splitlines() if l.strip()][-2:], "validator_pass": ok,
           "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    (zpath.with_suffix(".audit.json")).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    assert ok, "official validator did not pass on the zipped outputs"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--team", default="GitBoosters")
    ap.add_argument("--with-weights", action="store_true")
    a = ap.parse_args()
    main(a.tag, a.team, a.with_weights)
