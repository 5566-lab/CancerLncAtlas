"""Hash-bound C graph overlay for the frozen A candidate/label payload.

The large prepared fold stays byte-for-byte unchanged.  Only its graph bundle
and graph authority are replaced in memory, after both inputs have been hashed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


OVERLAY_FORMAT = "CANCERLNCATLAS_V32_C_GRAPH_OVERLAY_V1"
TYPED_GENERATION = "TYPED_ASSAY_CLASS_V1"


def _sha256_file(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _identity(stat: os.stat_result) -> tuple[int, ...]:
    return (
        int(stat.st_dev), int(stat.st_ino), int(stat.st_mode),
        int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns),
    )


def _sha(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError(f"C graph overlay lacks a valid {label} SHA256")
    return digest


def load_c_graph_overlay(
    torch: Any,
    *,
    input_manifest_path: str | Path,
    overlay_path: str | Path,
    fold: int,
    parent_sha256: str,
    old_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Load exactly one manifest-authorized sidecar through a stable handle."""
    manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
    rows = manifest.get("fold_inputs")
    if not isinstance(rows, list):
        raise RuntimeError("C graph overlay requires fold_inputs in the input manifest")
    matches = [row for row in rows if isinstance(row, Mapping)
               and int(row.get("fold", -1)) == int(fold)]
    if len(matches) != 1 or not isinstance(matches[0].get("graph_overlay"), Mapping):
        raise RuntimeError(f"C graph overlay requires one fold {fold} sidecar record")
    record = matches[0]["graph_overlay"]
    path = Path(overlay_path).resolve()
    if Path(str(record.get("path", ""))).resolve() != path:
        raise RuntimeError("C graph overlay path differs from the authorized manifest")
    size_only = record.get("verification_mode") == "SIZE_AND_GRAPH_SEMANTICS_V1"
    expected = None if size_only else _sha(record.get("sha256"), "sidecar")
    parent = _sha(parent_sha256, "parent prepared fold")
    if _sha(record.get("source_prepared_sha256"), "declared parent") != parent:
        raise RuntimeError("C graph overlay parent SHA256 differs from the loaded fold")
    receipt = _sha(record.get("graph_authority_receipt_sha256"), "graph receipt")
    if str(record.get("format")) != OVERLAY_FORMAT:
        raise RuntimeError("C graph overlay manifest format mismatch")
    if not path.is_file():
        raise RuntimeError(f"C graph overlay sidecar is absent: {path}")
    with path.open("rb") as handle:
        before = _identity(os.fstat(handle.fileno()))
        expected_bytes = record.get("bytes")
        reuse = os.environ.get("CANCERLNCATLAS_REUSE_VERIFIED_INPUTS_NO_REHASH") == "1"
        if size_only:
            if type(expected_bytes) is not int or before[3] != expected_bytes:
                raise RuntimeError("Global C overlay byte size differs from authorization")
            observed = None
        elif reuse:
            if type(expected_bytes) is not int or before[3] != expected_bytes:
                raise RuntimeError("C graph overlay sidecar byte size differs from authorization")
            observed = expected
        else:
            observed = _sha256_file(handle)
            if _identity(os.fstat(handle.fileno())) != before or observed != expected:
                raise RuntimeError("C graph overlay sidecar hash or inode changed")
        handle.seek(0)
        overlay = torch.load(handle, map_location="cpu", weights_only=False)
        if _identity(os.fstat(handle.fileno())) != before:
            raise RuntimeError("C graph overlay sidecar changed during loading")
    if not isinstance(overlay, Mapping):
        raise RuntimeError("C graph overlay sidecar must be a mapping")
    result = apply_c_graph_overlay(
        old_payload, overlay,
        fold=fold,
        source_sha256=parent,
        overlay_sha256=observed,
        overlay_bytes=before[3],
        receipt_sha256=receipt,
        global_binding_source_path=(
            Path(record["global_binding_source_path"])
            if size_only and "global_binding_source_path" in record else None
        ),
    )
    if size_only:
        bundle = result["bundle"]
        new_edges = record.get("expected_graph_edges")
        binding_edges = record.get("expected_binding_edges")
        if type(new_edges) is not int or len(bundle.edges) != new_edges:
            raise RuntimeError("Global C overlay graph edge count differs from authorization")
        typed = bundle.edges.edge_role.astype(str).str.startswith(
            "static_global_lnc_protein_binding"
        )
        if type(binding_edges) is not int or int(typed.sum()) != binding_edges:
            raise RuntimeError("Global C overlay binding edge count differs from authorization")
    return result, observed


def apply_c_graph_overlay(
    old_payload: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    fold: int,
    source_sha256: str,
    overlay_sha256: str | None,
    overlay_bytes: int | None = None,
    receipt_sha256: str,
    global_binding_source_path: Path | None = None,
) -> dict[str, Any]:
    """Replace only the graph after checking node indices and C evidence."""
    from .formal_graph_authority import validate_formal_graph_payload_binding

    if overlay.get("format") != OVERLAY_FORMAT:
        raise RuntimeError("C graph overlay payload format mismatch")
    if int(overlay.get("patient_fold", -1)) != int(fold):
        raise RuntimeError("C graph overlay fold mismatch")
    if overlay.get("formal_graph_variant") != "G2":
        raise RuntimeError("C graph overlay must target G2")
    if _sha(overlay.get("source_prepared_sha256"), "parent") != _sha(source_sha256, "parent"):
        raise RuntimeError("C graph overlay is bound to a different A payload")
    if _sha(overlay.get("graph_authority_receipt_sha256"), "receipt") != _sha(receipt_sha256, "receipt"):
        raise RuntimeError("C graph overlay receipt mismatch")
    old_binding = old_payload.get("formal_graph_authority")
    new_binding = overlay.get("formal_graph_authority")
    old_bundle = old_payload.get("bundle")
    bundle = overlay.get("bundle")
    if not isinstance(old_binding, Mapping) or not isinstance(new_binding, Mapping):
        raise RuntimeError("C graph overlay requires both graph bindings")
    if old_payload.get("formal_graph_variant") != "G2" or int(old_payload.get("patient_fold", -1)) != int(fold):
        raise RuntimeError("C graph overlay A payload variant or fold mismatch")
    if new_binding.get("binding_generation") != TYPED_GENERATION:
        raise RuntimeError("C graph overlay is not the typed binding generation")
    if new_binding.get("receipt", {}).get("sha256") != receipt_sha256:
        raise RuntimeError("C graph overlay binding receipt SHA256 mismatch")
    old_node_sha = old_binding.get("graph", {}).get("node_sha256")
    new_node_sha = new_binding.get("graph", {}).get("node_sha256")
    if not old_node_sha or old_node_sha != new_node_sha:
        raise RuntimeError("C graph overlay node universe differs from A")
    if getattr(old_bundle, "node_maps", None) != getattr(bundle, "node_maps", None):
        raise RuntimeError("C graph overlay node index maps differ from A")
    validate_formal_graph_payload_binding(
        new_binding, outer_fold=fold, variant="G2", bundle=bundle,
    )
    edges = getattr(bundle, "edges", None)
    if edges is None or "relation_type" not in edges:
        raise RuntimeError("C graph overlay lacks canonical edge relations")
    counts = edges.relation_type.astype(str).value_counts()
    if int(counts.get("binds_protein_eclip", 0)) <= 0:
        raise RuntimeError("C graph overlay lacks active eCLIP edges")
    if int(counts.get("binds_protein_predicted", 0)) != 0:
        raise RuntimeError("C graph overlay unexpectedly admits predicted binding")
    if int(counts.get("binds_protein", 0)) != 0:
        raise RuntimeError("C graph overlay still has the legacy flat binding relation")
    global_policy = overlay.get("binding_context_policy") == "GLOBAL_PHYSICAL_BINDING"
    if overlay_sha256 is None and not global_policy:
        raise RuntimeError("Size-only C overlay must declare global binding policy")
    if global_policy:
        source = overlay.get("global_binding_source")
        correction = new_binding.get("global_binding_correction")
        if not isinstance(source, Mapping) or not isinstance(correction, Mapping):
            raise RuntimeError("Global C overlay lacks binding-source provenance")
        source_path = Path(str(source.get("path", "")))
        source_bytes = source.get("bytes")
        staged_source = global_binding_source_path or source_path
        if (type(source_bytes) is not int or source_bytes <= 0
                or not staged_source.is_absolute() or not staged_source.is_file()
                or staged_source.stat().st_size != source_bytes
                or correction.get("source_path") != str(source_path)
                or correction.get("source_bytes") != source_bytes):
            raise RuntimeError("Global C binding source path or size drift")
        typed = edges.edge_role.astype(str).str.startswith(
            ("static_global_lnc_protein_binding", "static_context_lnc_rbp_eclip")
        )
        if edges.loc[typed, "cancer_id"].notna().any() or edges.loc[
            typed, "is_context_specific"
        ].any():
            raise RuntimeError("Global C overlay retains cancer-specific binding")
        if edges.edge_role.astype(str).eq("static_context_lnc_rbp_eclip").any():
            raise RuntimeError("Global C overlay retains a contextual binding role")
    original_hashes = old_payload.get("artifact_hashes")
    if not isinstance(original_hashes, Mapping):
        raise RuntimeError("A prepared fold lacks its original authorization hashes")
    result = dict(old_payload)
    result["bundle"] = bundle
    result["formal_graph_authority"] = new_binding
    result["graph"] = None
    # The source mapping remains immutable and inspectable; the new lineage is
    # reported separately in every training checkpoint and SUCCESS receipt.
    lineage = dict(result.get("input_authority_hashes", original_hashes))
    lineage["parent_formal_graph_authority_receipt_sha256"] = old_binding.get("receipt", {}).get("sha256")
    lineage["formal_graph_authority_receipt_sha256"] = receipt_sha256
    lineage["source_prepared_sha256"] = source_sha256
    if overlay_sha256 is None:
        if type(overlay_bytes) is not int or overlay_bytes <= 0:
            raise RuntimeError("Global C overlay lacks an observed byte size")
        lineage["c_graph_overlay_bytes"] = overlay_bytes
    else:
        lineage["c_graph_overlay_sha256"] = overlay_sha256
    result["input_authority_hashes"] = lineage
    result["graph_overlay_authority"] = {
        "format": OVERLAY_FORMAT,
        "source_prepared_sha256": source_sha256,
        "overlay_sha256": overlay_sha256,
        "verification_mode": (
            "SIZE_AND_GRAPH_SEMANTICS_V1" if overlay_sha256 is None else "SHA256"
        ),
        "graph_authority_receipt_sha256": receipt_sha256,
        "verified": True,
    }
    return result
