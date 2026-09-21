from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "artifacts/web_authoritative_candidate_20260829_r1/CANDIDATE_MANIFEST.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_candidate_manifest_is_strict_current_and_does_not_claim_remote_pass() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candidate_root = MANIFEST.parent.resolve()
    paths = [row["path"] for row in payload["files"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    for row in payload["files"]:
        path = (candidate_root / row["path"]).resolve()
        assert candidate_root in path.parents
        assert path.is_file()
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == _sha256(path)

    authoritative_app = candidate_root / "authoritative/website/backend/app.py"
    assert payload["authoritative_base"]["candidate_patched_app_sha256"] == _sha256(
        authoritative_app
    )
    regression = payload["test_evidence"]["formal_first_component_regression"]
    assert regression["status"] == "PASS"
    assert (regression["passed"], regression["failed"], regression["errors"]) == (
        9,
        0,
        0,
    )
    deployment = payload["deployment"]
    assert deployment["production_deployable"] is False
    assert deployment["production_modified"] is False
    assert deployment["local_candidate"]["status"] == "validated"
    isolated = deployment["isolated_candidate"]
    assert isolated["status"] == "remote_reprobe_required"
    assert isolated["formal_first_patch_uploaded"] is False
    assert isolated["formal_first_patch_http_validated"] is False
    assert isolated["production_promotion_authorized_by_manifest"] is False
