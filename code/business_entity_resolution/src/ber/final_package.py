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
    "models/lgb_s2_v6_e5base2.txt": "v6 stage-2 LightGBM (with the e5base2 CE feature)",
    "experiments/v6/decision_params_s2_e5base2.json": "decision thresholds used for matching_results.tsv (t_any 0.4, t_add 0.75)",
    "experiments/v6/france_category_swap.json": "France category words mined label-free from accepted test pairs (category-swap rule)",
}
STAGE3 = {  # sub-2026-09-27-5 (--stage3): US/India stage 3 on the stage-2 uncertain band
    "models/lgb_s3_llm_qwen7b_r3.txt": "stage-3 LightGBM (logit p2, logit LLM, null flag, S1 rank/top, record's best other; fit on holdout half A, world 0)",
    "experiments/v6/decision_params_s3_qwen7b_r3.json": "stage-3 decision thresholds for US/India (t_any 0.5, t_add 0.75); France rows = sub-2026-09-27-2",
    "models/llm_ce_qwen7b_r3/adapter_config.json": "LoRA adapter config of the Qwen2.5-7B pair classifier (r=16, sequence-classification head)",
    "models/llm_ce_qwen7b_r3/adapter_model.safetensors": "LoRA adapter weights + score head of the Qwen2.5-7B pair classifier (3 rounds: 80k + 160k + 160k CE pairs)",
}
STAGE3V4 = {  # sub-2026-09-27-9/-10 (--stage3v4)
    "models/lgb_s3v4_llm_qwen7b_r3.txt": "stage-3 v4 LightGBM (band 0.01-0.995; stage-2 context, S2X score, Qwen-7B and Qwen-1.5B scores; fit on holdout half A, world 0)",
    "experiments/v6/decision_params_s3_qwen7b_r3.json": "stage-3 decision thresholds (t_any 0.5, t_add 0.75)",
    "models/llm_ce_qwen7b_r3/adapter_config.json": "LoRA adapter config of the Qwen2.5-7B pair classifier",
    "models/llm_ce_qwen7b_r3/adapter_model.safetensors": "LoRA adapter + score head of the Qwen2.5-7B pair classifier (80k + 160k + 160k CE pairs)",
    "models/llm_ce_qwen15_r3/adapter_config.json": "LoRA adapter config of the Qwen2.5-1.5B pair classifier",
    "models/llm_ce_qwen15_r3/adapter_model.safetensors": "LoRA adapter + score head of the Qwen2.5-1.5B pair classifier (150k + 300k + 300k CE pairs)",
    "cache/v6/s2/france_llm_veto_scores.parquet": "Qwen2.5-7B scores of the France accepted pairs with stage-2 p in (0.95, 0.999] (France LLM veto)",
}
FRANCE2711 = {  # scored sub-2026-09-27-11 (--france-2711)
    "cache/v6/s2/france_numveto_2711_drops.parquet": "team-generated France house-number veto, limited to stage-2 p <= 0.999 (7,367 pairs)",
    "cache/v6/s2/france_filler_2711_adds.parquet": "team-generated France filler rescue (339 pairs)",
}
LARGE = {
    "models/dense_e5small/model.pt": ("fine-tuned bi-encoder (dense retrieval)", "intfloat/multilingual-e5-small", "MIT", "117.7M",
                                      "614241f622f53c4eeff9890bdc4f31cfecc418b3", "python -m ber.dense --steps data,tokenize,train"),
    "models/ce_e5base2/model.pt": ("fine-tuned cross-encoder (stage-2 band feature; continued from ce_e5base on v6 hard negatives)",
                                   "intfloat/multilingual-e5-base", "MIT", "278.0M", "d128750597153bb5987e10b1c3493a34e5a4502a",
                                   "BER_CE=e5base python -m ber.cross_encoder --steps data,train && BER_CE=e5base2 python -m ber.cross_encoder --steps data,train"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def models_md(stage3: bool = False, v4: bool = False, france_2711: bool = False) -> str:
    lines = ["# Models and learned artifacts", "",
             "All models use only supplied challenge data and run offline. Stage-3 LightGBM is fitted on holdout half A; half B is the evaluation half. No external identity data or hosted APIs.", "",
             "## Pretrained encoders (downloaded once, pinned)", "",
             "| Fine-tuned file | Role | Base model | Licence | Parameters | Base revision | SHA-256 of fine-tuned weights | Regenerate |",
             "|---|---|---|---|---|---|---|---|"]
    for f, (role, base, lic, n, rev, cmd) in LARGE.items():
        p = C.ROOT / f
        lines.append(f"| `{f}` | {role} | `{base}` | {lic} | {n} | `{rev}` | `{sha256(p) if p.exists() else 'missing'}` | `{cmd}` |")
    if stage3:
        lines.append("| `models/llm_ce_qwen7b_r3/` (LoRA adapter, included) | stage-3 pair classifier on the US/India stage-2 uncertain band | `Qwen/Qwen2.5-7B` | Apache-2.0 | 7.6B | `d149729398750b98c0af14eb82c78cfe92750796` | see below | README \"Stage 3\" |")
    if v4:
        lines.append("| `models/llm_ce_qwen7b_r3/` (LoRA adapter, included) | stage-3 pair classifier | `Qwen/Qwen2.5-7B` | Apache-2.0 | 7.6B | `d149729398750b98c0af14eb82c78cfe92750796` | see below | README \"Stage 3\" |")
        lines.append("| `models/llm_ce_qwen15_r3/` (LoRA adapter, included) | stage-3 second pair classifier | `Qwen/Qwen2.5-1.5B` | Apache-2.0 | 1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | see below | README \"Stage 3 v4\" |")
    lines += ["", "## Included small artifacts (`artifacts/`)", "", "| File | What | SHA-256 |", "|---|---|---|"]
    v4_files = {f: what for f, what in STAGE3V4.items() if not (france_2711 and f.endswith("france_llm_veto_scores.parquet"))}
    for f, what in {**SMALL, **(STAGE3 if stage3 else {}), **(v4_files if v4 else {}),
                    **(FRANCE2711 if france_2711 else {})}.items():
        p = C.ROOT / f
        lines.append(f"| `{Path(f).name}` | {what} | `{sha256(p)}` |")
    lines += ["", "LightGBM (MIT) models are trained locally by `ber.pipeline` / `ber.stage2`."]
    return "\n".join(lines) + "\n"


def main(tag: str, team: str, with_weights: bool, stage3: bool = False, v4: bool = False,
         france_2711: bool = False):
    assert not france_2711 or v4, "#27-11 uses stage 3 v4 for US/India"
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
        if stage3:
            for f in STAGE3:
                sub = "llm_ce_qwen7b_r3/" if "llm_ce_qwen7b_r3/" in f else ""
                z.write(C.ROOT / f, f"code/business_entity_resolution/artifacts/{sub}{Path(f).name}")
        if v4:
            for f in STAGE3V4:
                if france_2711 and f.endswith("france_llm_veto_scores.parquet"):
                    continue
                sub = next((d + "/" for d in ("llm_ce_qwen7b_r3", "llm_ce_qwen15_r3") if f"/{d}/" in f), "")
                z.write(C.ROOT / f, f"code/business_entity_resolution/artifacts/{sub}{Path(f).name}")
        if france_2711:
            for f in FRANCE2711:
                z.write(C.ROOT / f, f"code/business_entity_resolution/artifacts/{Path(f).name}")
        z.writestr("code/business_entity_resolution/artifacts/MODELS.md", models_md(stage3 and not v4, v4, france_2711))
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
    ap.add_argument("--stage3", action="store_true", help="include the stage-3 (Qwen2.5-7B LoRA) artifacts of sub-2026-09-27-5/-7")
    ap.add_argument("--stage3v4", action="store_true", help="include the stage-3 v4 artifacts of sub-2026-09-27-9/-10")
    ap.add_argument("--france-2711", action="store_true", help="include exact #27-11 France veto and rescue lists")
    a = ap.parse_args()
    main(a.tag, a.team, a.with_weights, a.stage3 and not a.stage3v4, a.stage3v4, a.france_2711)
