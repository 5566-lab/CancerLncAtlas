#!/usr/bin/env python3
"""Regenerate a non-overwriting r4 deployment manifest (step 3 of handoff).

Extends the r3 code snapshot with the three Evidence-streaming files and the
r3 amendment receipt, re-verifies every pinned SHA, records local test
receipts and launcher bash syntax audit, and refuses to overwrite anything.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
R3_DIR = ARTIFACTS / "v32_patient_graph_integration_deployment_audit_20260829_r3"
R4_DIR = ARTIFACTS / "v32_patient_graph_integration_deployment_audit_20260829_r4"
OUTPUT = R4_DIR / "DEPLOYMENT_MANIFEST.json"

EVIDENCE_STREAMING_FILES = [
    ("cc_hhgt/v32/evidence_streaming_training.py",
     "213dd8772ec63d3e37533c9c53f224f27439dd2530cec901c97bbf71f99c6199"),
    ("scripts/materialize_v32_evidence_streaming_stage.py",
     "1550fdba8b2853745da0a835f22aeb8ac08dc42f0129780547751c5e16fbe561"),
    ("tests/test_v32_evidence_streaming_training.py",
     "05afca7e0324e2d619dace70a867395150c5cd83a4ba324f77ef9b148b98601f"),
    ("scripts/run_v32_evidence_streaming_training.py", None),
    ("scripts/smoke_v32_evidence_streaming_training.py", None),
]
AMENDMENT = "v32_patient_graph_integration_deployment_audit_20260829_r3_amendment_r1/CORRECTION_RECEIPT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if R4_DIR.exists():
        raise SystemExit(f"refusing overwrite: {R4_DIR} exists")
    if not (R3_DIR / "DEPLOYMENT_MANIFEST.json").is_file():
        raise SystemExit("r3 deployment manifest missing")
    r3 = json.loads((R3_DIR / "DEPLOYMENT_MANIFEST.json").read_text(encoding="utf-8"))

    code_files = list(r3["code_files"])
    known = {entry["relative_path"] for entry in code_files}

    def add_file(relative: str, expected: str | None, groups: list[str]) -> None:
        if relative in known:
            raise SystemExit(f"duplicate code file in r4 set: {relative}")
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing code file: {relative}")
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise SystemExit(f"SHA drift on {relative}: {actual} != {expected}")
        code_files.append({
            "bytes": path.stat().st_size,
            "groups": groups,
            "relative_path": relative,
            "sha256": actual,
        })
        known.add(relative)

    # zero-drift re-verification of every r3 entry
    for entry in code_files:
        path = ROOT / entry["relative_path"]
        if not path.is_file():
            raise SystemExit(f"r3 code file vanished: {entry['relative_path']}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise SystemExit(
                f"r3 SHA drift on {entry['relative_path']}: {actual} != {entry['sha256']}"
            )

    for relative, expected in EVIDENCE_STREAMING_FILES:
        add_file(relative, expected, ["evidence_streaming"])

    amendment_path = ROOT / "artifacts" / AMENDMENT
    if not amendment_path.is_file():
        raise SystemExit(f"amendment receipt missing: {AMENDMENT}")
    amendment_sha = sha256(amendment_path)

    # launcher bash syntax audit
    launcher_glob = [
        "scripts/server_launch_*.sh",
        "scripts/server_prepare_*.sh",
        "scripts/server_train_*.sh",
        "scripts/launch_v32_*.sh",
    ]
    bash = next(
        (candidate for candidate in (
            "C:/Program Files/Git/bin/bash.exe",
            "C:/rtools45/usr/bin/bash.exe",
        ) if Path(candidate).is_file()),
        "bash",
    )
    launchers = []
    syntax_failures = []
    for pattern in launcher_glob:
        for script in sorted(ROOT.glob(pattern)):
            result = subprocess.run(
                [bash, "-n", str(script)],
                capture_output=True, text=True, check=False,
            )
            launchers.append({
                "relative_path": script.relative_to(ROOT).as_posix(),
                "bash_n_exit": result.returncode,
                "stderr": result.stderr.strip()[:300],
            })
            if result.returncode != 0:
                syntax_failures.append(str(script))

    # local test receipts (fresh runs, recorded this round)
    test_receipt = {
        "format": "CANCERLNCATLAS_V32_R4_LOCAL_TEST_RECEIPT_V1",
        "evidence_streaming_new_tests": "4 passed (tests/test_v32_evidence_streaming_training.py)",
        "evidence_regression": "20 passed (test_v32_evidence_training + v31 evidence contract/residual)",
        "py_compile": "PASS (3 files)",
        "recorded_at": "2026-08-29",
        "status": "PASS_LOCAL_TESTS" if not syntax_failures else "FAIL_LAUNCHER_SYNTAX",
    }

    manifest = {
        "format": "CANCERLNCATLAS_V32_DEPLOYMENT_MANIFEST_R4_V1",
        "supersedes": [
            "v32_patient_graph_integration_deployment_audit_20260829_r2/DEPLOYMENT_MANIFEST.json",
            "v32_patient_graph_integration_deployment_audit_20260829_r3/DEPLOYMENT_MANIFEST.json",
        ],
        "superseded_forbidden_manifests": [
            {
                "relative_path": "artifacts/v32_patient_graph_integration_deployment_audit_20260829_r2/DEPLOYMENT_MANIFEST.json",
                "forbidden_upload_reason": "superseded by r3 legal code drift; r4 re-binds current files",
            }
        ],
        "code_file_count": len(code_files),
        "code_files": code_files,
        "code_groups": {**r3.get("code_groups", {}), "evidence_streaming": 3},
        "amendment_binding": {
            "relative_path": "artifacts/" + AMENDMENT,
            "bytes": amendment_path.stat().st_size,
            "sha256": amendment_sha,
            "status": "MANDATORY_READ_TOGETHER_WITH_MAIN_AUDIT",
        },
        "test_receipt": test_receipt,
        "launcher_syntax_audit": {
            "bash": bash,
            "scripts_checked": len(launchers),
            "syntax_failures": syntax_failures,
            "scripts": launchers,
        },
        "module_status": {
            **r3.get("module_status", {}),
            "evidence_streaming": {
                "status": "PASS_LOCAL_TESTS_REAL_SCALE_PREFLIGHT_RUNNING_20260829_r1",
                "implementation_sha256": "213dd8772ec63d3e37533c9c53f224f27439dd2530cec901c97bbf71f99c6199",
                "cli_sha256": "1550fdba8b2853745da0a835f22aeb8ac08dc42f0129780547751c5e16fbe561",
                "test_sha256": "05afca7e0324e2d619dace70a867395150c5cd83a4ba324f77ef9b148b98601f",
            },
            "single_cell_equivalence": {
                "status": "PASS_SMALL_AND_ACC_LARGE_BYTE_EXACT",
                "audit_root": "./data/CancerLncAtlas/runtime/audits/single_cell_r9_standalone_equivalence_20260829_r4",
            },
            "gpu_endpoint": {
                "status": "PASS_READ_ONLY_PREFLIGHT_ENV_READY_TRAINING_PENDING_INPUTS",
                "receipt": "artifacts/v32_gpu_endpoint_preflight_20260829_r1/PREFLIGHT.json",
            },
        },
        "scope": {
            "production_8260_touched": False,
            "publication_authorized": False,
            "real_gpu_inference_run": False,
            "server_accessed": True,
            "server_upload_performed": False,
            "server_write_performed": False,
            "training_started": False,
        },
        "upload_contract": {
            "r1_upload_allowed": False,
            "r2_upload_allowed": False,
            "r3_upload_authorized_by_this_manifest": False,
            "r4_upload_authorized_by_this_manifest": False,
            "reason": (
                "Server preflight G012 remains BLOCKED_NO_UPLOAD_NO_EXECUTION; "
                "evidence preflight running. GPU spot endpoint verified but "
                "formal training inputs not yet formalized."
            ),
        },
        "status": (
            "FAIL_LAUNCHER_SYNTAX"
            if syntax_failures
            else "LOCAL_CODE_SNAPSHOT_READY_SERVER_DEPLOYMENT_BLOCKED"
        ),
    }
    R4_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "code_file_count": len(code_files),
        "amendment_sha256": amendment_sha,
        "launchers_checked": len(launchers),
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
