#!/usr/bin/env python3
"""Add a pinned Gene Set closure to an otherwise complete runtime manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_BINDINGS_V1"
STATUS = "SERVER_UPLOAD_PENDING_RUNTIME_VALIDATION"
CLOSURE_FORMAT = "CANCERLNCATLAS_V32_GENE_SET_RUNTIME_METADATA_CLOSURE_V1"
EXPECTED_EXISTING_COMPONENTS = {
    "clinical",
    "evidence",
    "evidence_direction",
    "state_gene_set",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RuntimeClosureError(RuntimeError):
    """A pinned input or no-overwrite runtime-manifest gate failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeClosureError(message)


def _load(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(SHA256.fullmatch(expected) is not None, f"{role} SHA256 invalid")
    _require(path.is_file() and not path.is_symlink(), f"{role} missing or symlinked")
    _require(sha256_file(path) == expected, f"{role} SHA256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeClosureError(f"{role} invalid JSON") from exc
    _require(isinstance(value, dict), f"{role} must be a JSON object")
    return value


def build(
    base_path: Path,
    base_sha256: str,
    closure_path: Path,
    closure_sha256: str,
    *,
    server_root: str,
) -> dict[str, Any]:
    _require(
        server_root.startswith(
            "./data/CancerLncAtlas/runtime/authorized_acceptance/"
        ),
        "Closed runtime server_root escaped authorized_acceptance",
    )
    base = _load(base_path.resolve(strict=True), base_sha256, "base runtime manifest")
    closure = _load(
        closure_path.resolve(strict=True), closure_sha256, "Gene Set closure receipt"
    )
    _require(base.get("format") == FORMAT and base.get("status") == STATUS, "Base runtime contract drifted")
    _require(
        base.get("main_score_changed") is False
        and base.get("production_deployed") is False
        and base.get("release_ready") is False,
        "Base runtime publication flags drifted",
    )
    _require(
        isinstance(base.get("bindings"), dict)
        and set(base["bindings"]) == EXPECTED_EXISTING_COMPONENTS,
        "Base runtime component set drifted",
    )
    _require(
        closure.get("format") == CLOSURE_FORMAT
        and closure.get("status") == "PASS"
        and closure.get("metadata_closure", {}).get("transitive_metadata_closed")
        is True
        and closure.get("metadata_closure", {}).get(
            "portable_rebinder_pending_was_runtime_false_negative"
        )
        is True
        and closure.get("exact_probes", {}).get("cancers_typed") == 33
        and closure.get("exact_probes", {}).get("available_cancers") == 31
        and set(closure.get("exact_probes", {}).get("typed_unavailable", {}))
        == {"CHOL", "UCS"}
        and closure.get("family_to_exact_broadcast") is False
        and closure.get("changes_primary_ranking") is False
        and closure.get("production_deployed") is False
        and closure.get("release_ready") is False,
        "Gene Set closure did not pass the strict runtime gate",
    )
    declaration = closure.get("runtime_binding_declaration")
    _require(isinstance(declaration, dict), "Gene Set runtime declaration is missing")
    for key in ("binding_path", "audit_binding_path"):
        _require(
            str(declaration.get(key, "")).startswith(
                "./data/CancerLncAtlas/runtime/authorized_acceptance/"
            ),
            f"Gene Set runtime {key} escaped the acceptance root",
        )
    for key in ("binding_sha256", "audit_binding_sha256"):
        _require(SHA256.fullmatch(str(declaration.get(key, ""))) is not None, f"Gene Set {key} invalid")
    result = deepcopy(base)
    result["server_root"] = server_root
    result["bindings"]["gene_set_ranked_subtype"] = deepcopy(declaration)
    result["gene_set_ranked_subtype"] = {
        "status": "RUNTIME_TRANSITIVE_METADATA_CLOSED",
        "closure_receipt": {
            "path": str(closure_path.resolve()),
            "sha256": str(closure_sha256).lower(),
            "format": CLOSURE_FORMAT,
            "status": "PASS",
        },
        "legacy_gene_set_transitive_unresolved": 24,
        "legacy_ranked_subtype_transitive_unresolved": 72,
        "legacy_pending_classified_as_runtime_false_negative": True,
        "production_deployed": False,
        "release_ready": False,
    }
    return result


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"Refusing to overwrite runtime manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime-bindings", required=True)
    parser.add_argument("--base-runtime-bindings-sha256", required=True)
    parser.add_argument("--gene-set-closure", required=True)
    parser.add_argument("--gene-set-closure-sha256", required=True)
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    payload = build(
        Path(args.base_runtime_bindings),
        args.base_runtime_bindings_sha256,
        Path(args.gene_set_closure),
        args.gene_set_closure_sha256,
        server_root=args.server_root,
    )
    _write(output, payload)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
