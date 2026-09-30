from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cc_hhgt.v32.c_graph_overlay import (  # noqa: E402
    OVERLAY_FORMAT, apply_c_graph_overlay, load_c_graph_overlay,
)


PARENT_SHA = "a" * 64
RECEIPT_SHA = "b" * 64


def _inputs():
    rows = [{"relation_type": "binds_protein_eclip"}]
    old_bundle = SimpleNamespace(node_maps={"lncRNA": {"L1": 0}})
    new_bundle = SimpleNamespace(
        node_maps={"lncRNA": {"L1": 0}}, edges=pd.DataFrame(rows),
    )
    old = {
        "patient_fold": 0, "formal_graph_variant": "G2",
        "bundle": old_bundle,
        "formal_graph_authority": {
            "graph": {"node_sha256": "c" * 64},
            "receipt": {"sha256": "d" * 64},
        },
        "artifact_hashes": {"code_sha256": "e" * 64},
        "train_batches": [object()],
    }
    overlay = {
        "format": OVERLAY_FORMAT,
        "patient_fold": 0, "formal_graph_variant": "G2",
        "source_prepared_sha256": PARENT_SHA,
        "graph_authority_receipt_sha256": RECEIPT_SHA,
        "bundle": new_bundle,
        "formal_graph_authority": {
            "binding_generation": "TYPED_ASSAY_CLASS_V1",
            "graph": {"node_sha256": "c" * 64},
            "receipt": {"sha256": RECEIPT_SHA},
        },
    }
    return old, overlay


def test_c_graph_overlay_reuses_batches_and_binds_lineage(monkeypatch):
    from cc_hhgt.v32 import formal_graph_authority

    calls = []
    monkeypatch.setattr(
        formal_graph_authority, "validate_formal_graph_payload_binding",
        lambda binding, **kw: calls.append((binding, kw)),
    )
    old, overlay = _inputs()
    result = apply_c_graph_overlay(
        old, overlay, fold=0, source_sha256=PARENT_SHA,
        overlay_sha256="f" * 64, receipt_sha256=RECEIPT_SHA,
    )
    assert result["train_batches"] is old["train_batches"]
    assert old["bundle"] is not result["bundle"]
    assert result["input_authority_hashes"]["c_graph_overlay_sha256"] == "f" * 64
    assert result["input_authority_hashes"]["parent_formal_graph_authority_receipt_sha256"] == "d" * 64
    assert calls and calls[0][1]["variant"] == "G2"


@pytest.mark.parametrize("drift", ["node_map", "receipt", "predicted", "missing_eclip"])
def test_c_graph_overlay_rejects_wrong_graph(monkeypatch, drift):
    from cc_hhgt.v32 import formal_graph_authority

    monkeypatch.setattr(
        formal_graph_authority, "validate_formal_graph_payload_binding",
        lambda *_args, **_kwargs: None,
    )
    old, overlay = _inputs()
    if drift == "node_map":
        overlay["bundle"].node_maps = {"lncRNA": {"L1": 1}}
    elif drift == "receipt":
        overlay["formal_graph_authority"]["receipt"]["sha256"] = "0" * 64
    elif drift == "predicted":
        overlay["bundle"].edges.loc[0, "relation_type"] = "binds_protein_predicted"
    else:
        overlay["bundle"].edges.loc[0, "relation_type"] = "encoded_by"
    with pytest.raises(RuntimeError):
        apply_c_graph_overlay(
            old, overlay, fold=0, source_sha256=PARENT_SHA,
            overlay_sha256="f" * 64, receipt_sha256=RECEIPT_SHA,
        )


def test_manifest_binds_sidecar_and_parent_before_substitution(tmp_path, monkeypatch):
    from cc_hhgt.v32 import formal_graph_authority

    monkeypatch.setattr(
        formal_graph_authority, "validate_formal_graph_payload_binding",
        lambda *_args, **_kwargs: None,
    )
    old, overlay = _inputs()
    path = tmp_path / "C_G2_PATIENT_FOLD_0.pt"
    torch.save(overlay, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(json.dumps({"fold_inputs": [{
        "fold": 0,
        "graph_overlay": {
            "format": OVERLAY_FORMAT,
            "path": str(path), "sha256": digest,
            "source_prepared_sha256": PARENT_SHA,
            "graph_authority_receipt_sha256": RECEIPT_SHA,
        },
    }]}))
    result, observed = load_c_graph_overlay(
        torch, input_manifest_path=manifest, overlay_path=path,
        fold=0, parent_sha256=PARENT_SHA, old_payload=old,
    )
    assert observed == digest
    assert result["bundle"].edges.relation_type.iloc[0] == "binds_protein_eclip"
    with pytest.raises(RuntimeError, match="parent SHA256"):
        load_c_graph_overlay(
            torch, input_manifest_path=manifest, overlay_path=path,
            fold=0, parent_sha256="1" * 64, old_payload=old,
        )


def test_global_binding_overlay_uses_size_and_graph_semantics_without_file_hash(
    tmp_path, monkeypatch,
):
    from cc_hhgt.v32 import formal_graph_authority

    monkeypatch.setattr(
        formal_graph_authority, "validate_formal_graph_payload_binding",
        lambda *_args, **_kwargs: None,
    )
    old, overlay = _inputs()
    binding_source = tmp_path / "global_binding.parquet"
    binding_source.write_bytes(b"already materialized global binding")
    overlay["binding_context_policy"] = "GLOBAL_PHYSICAL_BINDING"
    overlay["global_binding_source"] = {
        "path": str(binding_source), "bytes": binding_source.stat().st_size,
    }
    overlay["formal_graph_authority"]["global_binding_correction"] = {
        "source_path": str(binding_source),
        "source_bytes": binding_source.stat().st_size,
    }
    overlay["bundle"].edges = pd.DataFrame([{
        "relation_type": "binds_protein_eclip",
        "edge_role": "static_global_lnc_protein_binding_eclip",
        "cancer_id": pd.NA, "is_context_specific": False,
    }])
    path = tmp_path / "G2_GLOBAL_FOLD_0.pt"
    torch.save(overlay, path)
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(json.dumps({"fold_inputs": [{
        "fold": 0,
        "graph_overlay": {
            "format": OVERLAY_FORMAT, "path": str(path),
            "bytes": path.stat().st_size,
            "verification_mode": "SIZE_AND_GRAPH_SEMANTICS_V1",
            "expected_graph_edges": 1,
            "expected_binding_edges": 1,
            "source_prepared_sha256": PARENT_SHA,
            "graph_authority_receipt_sha256": RECEIPT_SHA,
        },
    }]}))
    result, digest = load_c_graph_overlay(
        torch, input_manifest_path=manifest, overlay_path=path,
        fold=0, parent_sha256=PARENT_SHA, old_payload=old,
    )
    assert digest is None
    assert result["input_authority_hashes"]["c_graph_overlay_bytes"] == path.stat().st_size
    assert result["graph_overlay_authority"]["verification_mode"] == "SIZE_AND_GRAPH_SEMANTICS_V1"
    overlay["bundle"].edges.loc[0, "cancer_id"] = "BRCA"
    with pytest.raises(RuntimeError, match="cancer-specific binding"):
        apply_c_graph_overlay(
            old, overlay, fold=0, source_sha256=PARENT_SHA,
            overlay_sha256=None, overlay_bytes=path.stat().st_size,
            receipt_sha256=RECEIPT_SHA,
        )
    overlay["bundle"].edges.loc[0, "cancer_id"] = pd.NA
    torch.save(overlay, path)
    staged_source = tmp_path / "cloud_inputs" / "global_binding.parquet"
    staged_source.parent.mkdir()
    staged_source.write_bytes(binding_source.read_bytes())
    binding_source.unlink()
    authorized = json.loads(manifest.read_text())
    authorized["fold_inputs"][0]["graph_overlay"]["bytes"] = path.stat().st_size
    authorized["fold_inputs"][0]["graph_overlay"][
        "global_binding_source_path"
    ] = str(staged_source)
    manifest.write_text(json.dumps(authorized))
    result, digest = load_c_graph_overlay(
        torch, input_manifest_path=manifest, overlay_path=path,
        fold=0, parent_sha256=PARENT_SHA, old_payload=old,
    )
    assert digest is None
    assert result["graph_overlay_authority"]["verified"] is True


def test_user_directed_no_rehash_reuses_manifest_digest_and_checks_bytes(
    tmp_path, monkeypatch,
):
    from cc_hhgt.v32 import formal_graph_authority

    monkeypatch.setattr(
        formal_graph_authority, "validate_formal_graph_payload_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("CANCERLNCATLAS_REUSE_VERIFIED_INPUTS_NO_REHASH", "1")
    old, overlay = _inputs()
    path = tmp_path / "C_G2_PATIENT_FOLD_0.pt"
    torch.save(overlay, path)
    carried_digest = "f" * 64
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(json.dumps({"fold_inputs": [{
        "fold": 0,
        "graph_overlay": {
            "format": OVERLAY_FORMAT,
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": carried_digest,
            "source_prepared_sha256": PARENT_SHA,
            "graph_authority_receipt_sha256": RECEIPT_SHA,
        },
    }]}))
    result, observed = load_c_graph_overlay(
        torch, input_manifest_path=manifest, overlay_path=path,
        fold=0, parent_sha256=PARENT_SHA, old_payload=old,
    )
    assert observed == carried_digest
    assert result["graph_overlay_authority"]["verified"] is True
