"""Current-C three-arm graph sidecars, consumed by training and frozen export."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .current_binding_graph import ARMS, canonical_json_sha, binding_arm_runtime_bundle
from .contracts import dataframe_sha256
from .safe_graph import EDGE_KEYS, build_safe_graph
from .current_core_export import checked_sha

FORMAT = "CURRENT_C_BINDING_EXPANSION_OVERLAY_V1"
BINDING_FORMAT = "CURRENT_C_BINDING_EXPANSION_AUTHORITY_V1"
GRAPH_FIELDS = {"bundle", "graph", "formal_graph_authority", "input_authority_hashes",
                "graph_overlay_authority", "binding_expansion_authority"}


def _hash_value(digest, value):
    """Length-delimited values; no pickle/repr identities or silent omissions."""
    import torch
    def token(raw):
        digest.update(str(len(raw)).encode()+b":"+raw)
    if torch.is_tensor(value):
        if value.layout != torch.strided: raise ValueError("Sparse payload hashing is not supported")
        array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        token(b"torch"); token(str(value.dtype).encode()); _hash_value(digest, tuple(value.shape))
        token(str(array.nbytes).encode()); digest.update(memoryview(array))
    elif isinstance(value, np.ndarray):
        token(b"numpy"); token(str(value.dtype).encode()); _hash_value(digest, value.shape)
        if value.dtype.hasobject:
            for item in value.flat: _hash_value(digest, item)
        else:
            array = np.ascontiguousarray(value)
            token(str(array.nbytes).encode()); digest.update(memoryview(array).cast("B"))
    elif isinstance(value, pd.DataFrame):
        token(b"dataframe"); _hash_value(digest, value.columns.tolist()); _hash_value(digest, value.index.tolist())
        for col in value: _hash_value(digest, value[col])
    elif isinstance(value, pd.Series):
        token(b"series"); token(str(value.dtype).encode()); _hash_value(digest, value.name)
        _hash_value(digest, value.index.tolist()); _hash_value(digest, value.to_numpy())
    elif isinstance(value, Mapping):
        token(b"mapping"); token(str(len(value)).encode())
        if not all(isinstance(k, str) for k in value): raise ValueError("Payload mapping keys must be strings")
        for key in sorted(value): _hash_value(digest, key); _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        token(type(value).__name__.encode()); token(str(len(value)).encode())
        for item in value: _hash_value(digest, item)
    elif value is pd.NA: token(b"pandas_NA")
    elif value is None: token(b"none")
    elif isinstance(value, (bool, np.bool_)): token(b"bool"); token(str(bool(value)).encode())
    elif isinstance(value, (int, np.integer)): token(b"int"); token(str(int(value)).encode())
    elif isinstance(value, (float, np.floating)): token(b"float"); token(float(value).hex().encode())
    elif isinstance(value, (str, Path)): token(b"text"); token(str(value).encode())
    else: raise ValueError(f"Unsupported invariant payload type: {type(value).__name__}")


def value_sha(value):
    digest = hashlib.sha256(); _hash_value(digest, value); return digest.hexdigest()


def non_graph_payload_sha(payload):
    return value_sha({k: v for k, v in payload.items() if k not in GRAPH_FIELDS})


def comparison_config_sha(config):
    """Bind every setting except arm identity, output location and execution switch.

The separate training guard still validates the execution switch. Runtime,
early stopping, optimizer, candidate batching and all scientific choices remain
in this hash, including unknown future settings.
"""
    common = deepcopy(config)
    common.pop("execution_control", None)
    for key in ("binding_expansion_arm", "binding_comparison_config_sha256"):
        common.get("task_contract", {}).pop(key, None)
    for key in ("binding_expansion_overlay_pattern", "output_root"):
        common.get("training_io", {}).pop(key, None)
    return canonical_json_sha(common)


def _verify_base(payload, fold):
    from .formal_graph_authority import validate_formal_graph_payload_binding
    if payload.get("graph_overlay_authority", {}).get("verified") is not True:
        from .native_graph_input import validate_native_base
        validate_native_base(payload, fold, bundle=False)
    if payload.get("binding_expansion_authority"):
        raise ValueError("An already expanded graph cannot become the current C baseline")
    validate_formal_graph_payload_binding(payload["formal_graph_authority"], outer_fold=fold,
        variant="G2", bundle=payload["bundle"])


def _input_authority(payload):
    from .training import _prepared_input_authority_hashes
    return _prepared_input_authority_hashes(payload)


def prepare_binding_overlay(payload, arms, arm, comparison, *, non_graph_sha=None):
    fold = int(arms.audit["fold"]); _verify_base(payload, fold)
    if comparison.get("format") != "CURRENT_C_THREE_ARM_COMPARISON_V1":
        raise ValueError("Missing current-C comparison specification")
    checked_sha(comparison.get("config_sha256"), "comparison config")
    bundle = binding_arm_runtime_bundle(arms, arm)
    identity = non_graph_sha or non_graph_payload_sha(payload)
    binding = {
        "format": BINDING_FORMAT, "variant": "G2", "outer_fold": fold, "binding_expansion_arm": arm,
        "base_authority": deepcopy(payload["formal_graph_authority"]),
        "comparison": deepcopy(comparison), "comparison_sha256": canonical_json_sha(comparison),
        "non_graph_payload_sha256": identity,
        "graph": {"node_sha256": dataframe_sha256(bundle.nodes, ["node_type", "canonical_id"]),
            "edge_sha256": dataframe_sha256(bundle.edges, list(EDGE_KEYS)), "edge_count": len(bundle.edges),
            "master_edge_sha256": arms.audit["master_edge_sha256"], "same_relation_schema_all_variants": True},
        "receipt": {"sha256": canonical_json_sha({"audit": arms.audit, "comparison": comparison})},
        "node_feature_sha256": value_sha(bundle.node_degrees),
        "relation_schema_sha256": dataframe_sha256(bundle.relation_schema_edges, list(bundle.relation_schema_edges.columns)),
        "runtime_schedule_sha256": dataframe_sha256(bundle.runtime_schedule, list(EDGE_KEYS)),
        "base_graph_sha256": arms.audit["base_edge_sha256"],
    }
    return {"format": FORMAT, "patient_fold": fold, "binding_expansion_arm": arm,
        "bundle": bundle, "formal_graph_authority": binding,
        "base_authority_sha256": canonical_json_sha(payload["formal_graph_authority"]),
        "base_input_authority_sha256": canonical_json_sha(_input_authority(payload)),
        "non_graph_payload_sha256": identity, "audit": arms.audit}


def validate_expansion_binding(binding, *, outer_fold, variant, bundle=None):
    from .formal_graph_authority import validate_formal_graph_payload_binding
    if (binding.get("format") != BINDING_FORMAT or binding.get("binding_expansion_arm") not in ARMS
            or binding.get("variant") != variant or variant != "G2"
            or binding.get("outer_fold") != outer_fold):
        raise ValueError("Binding expansion authority arm/fold mismatch")
    parent = binding["base_authority"]
    if parent.get("format") == BINDING_FORMAT: raise ValueError("Recursive expansion authority is forbidden")
    validate_formal_graph_payload_binding(parent, outer_fold=outer_fold, variant="G2")
    comparison = binding["comparison"]
    if canonical_json_sha(comparison) != checked_sha(binding.get("comparison_sha256"), "comparison"):
        raise ValueError("Binding comparison identity mismatch")
    for key in ("non_graph_payload_sha256", "node_feature_sha256", "relation_schema_sha256", "runtime_schedule_sha256", "base_graph_sha256"):
        checked_sha(binding.get(key), key)
    if binding["base_graph_sha256"] != parent["graph"]["edge_sha256"]:
        raise ValueError("Expansion parent is not its declared current C graph")
    graph = binding["graph"]
    for key in ("node_sha256", "edge_sha256", "master_edge_sha256"): checked_sha(graph.get(key), key)
    if bundle is not None:
        observed = {"node_sha256": dataframe_sha256(bundle.nodes, ["node_type", "canonical_id"]),
            "edge_sha256": dataframe_sha256(bundle.edges, list(EDGE_KEYS)), "edge_count": len(bundle.edges)}
        if any(graph.get(key) != value for key, value in observed.items()):
            raise ValueError("Expanded graph differs from its selected arm")
        if (value_sha(bundle.node_degrees) != binding["node_feature_sha256"]
                or dataframe_sha256(bundle.relation_schema_edges, list(bundle.relation_schema_edges.columns)) != binding["relation_schema_sha256"]
                or dataframe_sha256(bundle.runtime_schedule, list(EDGE_KEYS)) != binding["runtime_schedule_sha256"]):
            raise ValueError("Expanded features/schema/schedule changed")
        arm = binding["binding_expansion_arm"]
        expected_dropout = {"binds_protein_predicted": comparison["predicted_dropout"]} if arm == "experimental_predicted" else {}
        if bundle.binding_evidence_arm != arm or bundle.relation_edge_dropout != expected_dropout:
            raise ValueError("Expanded runtime arm/dropout policy mismatch")
        build_safe_graph(bundle.edges, outer_fold=outer_fold)
    return {"status": "PASS_CURRENT_C_BINDING_EXPANSION", "outer_fold": outer_fold,
        "variant": variant, "binding_expansion_arm": binding["binding_expansion_arm"],
        "receipt_sha256": binding["receipt"]["sha256"], "edge_sha256": graph["edge_sha256"]}


def apply_binding_overlay(payload, overlay, *, fold, arm, overlay_sha, comparison_sha, config):
    _verify_base(payload, fold)
    checked_sha(overlay_sha, "expansion sidecar")
    if overlay.get("format") != FORMAT or overlay.get("patient_fold") != fold or overlay.get("binding_expansion_arm") != arm:
        raise ValueError("Expansion sidecar format/fold/arm mismatch")
    if (overlay["base_authority_sha256"] != canonical_json_sha(payload["formal_graph_authority"])
            or overlay["base_input_authority_sha256"] != canonical_json_sha(_input_authority(payload))):
        raise ValueError("Expansion sidecar current C parent changed")
    binding = overlay["formal_graph_authority"]; bundle = overlay["bundle"]; base = payload["bundle"]
    if binding["binding_expansion_arm"] != arm or binding["comparison_sha256"] != comparison_sha:
        raise ValueError("Expansion sidecar comparison or arm substitution")
    if comparison_config_sha(config) != binding["comparison"]["config_sha256"]:
        raise ValueError("Comparison training settings/budget drift")
    if non_graph_payload_sha(payload) != overlay["non_graph_payload_sha256"] or overlay["non_graph_payload_sha256"] != binding["non_graph_payload_sha256"]:
        raise ValueError("Comparison candidates, labels or non-graph inputs changed")
    validate_expansion_binding(binding, outer_fold=fold, variant="G2", bundle=bundle)
    for kind, mapping in base.node_maps.items():
        if any(bundle.node_maps.get(kind, {}).get(key) != value for key, value in mapping.items()):
            raise ValueError("Expansion changed an existing candidate/node index")
        if not np.array_equal(base.node_degrees[kind], bundle.node_degrees[kind][:len(mapping)]):
            raise ValueError("Expansion changed current C node features")
    retained = bundle.edges.loc[bundle.edges.edge_id.isin(base.edges.edge_id), base.edges.columns]
    if dataframe_sha256(retained, list(EDGE_KEYS)) != dataframe_sha256(base.edges, list(EDGE_KEYS)):
        raise ValueError("Expansion modified or removed a current C edge")
    added = bundle.edges.loc[~bundle.edges.edge_id.isin(base.edges.edge_id)]
    if (not added.edge_role.str.startswith("static_global_lnc_protein_binding_").all()
            or (arm == "current_c" and len(added))
            or (arm != "experimental_predicted" and added.relation_type.eq("binds_protein_predicted").any())):
        raise ValueError("Comparison arm contains forbidden added relations")
    if bundle.runtime_num_chunks != base.runtime_num_chunks: raise ValueError("Comparison runtime chunk count changed")
    result = dict(payload)  # All original tensors/labels remain the same objects.
    result.update(bundle=bundle, graph=None, formal_graph_authority=binding)
    result["binding_expansion_authority"] = {"verified": True, "arm": arm,
        "sidecar_sha256": overlay_sha, "comparison_sha256": comparison_sha,
        "non_graph_payload_sha256": binding["non_graph_payload_sha256"]}
    result["input_authority_hashes"] = {**_input_authority(payload),
        "binding_expansion_sidecar_sha256": overlay_sha, "binding_comparison_sha256": comparison_sha,
        "binding_expansion_graph_receipt_sha256": binding["receipt"]["sha256"]}
    return result


def load_binding_overlay(torch, *, input_manifest_path, overlay_path, fold, arm, payload, config):
    """Hash and deserialize through the same handle; no size-only mode."""
    from .c_graph_overlay import _identity, _sha256_file
    manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
    rows = [row for row in manifest.get("fold_inputs", []) if row.get("fold") == fold]
    if len(rows) != 1 or not isinstance(rows[0].get("binding_expansion"), Mapping):
        raise ValueError("Exactly one authorized expansion per patient fold is required")
    record = rows[0]["binding_expansion"]; path = Path(overlay_path).resolve()
    if (Path(record["path"]).resolve() != path or record.get("format") != FORMAT
            or record.get("arm") != arm): raise ValueError("Unauthorized expansion sidecar/arm")
    expected = checked_sha(record.get("sha256"), "expansion sidecar")
    with path.open("rb") as handle:
        before = _identity(os.fstat(handle.fileno())); digest = _sha256_file(handle)
        if digest != expected or before[3] != record.get("bytes") or _identity(os.fstat(handle.fileno())) != before:
            raise ValueError("Expansion sidecar hash/size changed")
        handle.seek(0); overlay = torch.load(handle, map_location="cpu", weights_only=False)
        if _identity(os.fstat(handle.fileno())) != before: raise ValueError("Expansion changed while loading")
    return apply_binding_overlay(payload, overlay, fold=fold, arm=arm, overlay_sha=digest,
        comparison_sha=record["comparison_sha256"], config=config)


def load_declared_binding_expansion(torch, *, payload, config, input_manifest_path, fold, root, task_seed=None):
    manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
    records = [r for r in manifest.get("fold_inputs", []) if r.get("fold") == fold]
    declared = any(r.get("binding_expansion") for r in records)
    arm = config.get("task_contract", {}).get("binding_expansion_arm")
    if bool(arm) != declared: raise ValueError("Training config and manifest disagree on binding expansion")
    if not arm: return payload
    if arm not in ARMS: raise ValueError("Unknown comparison arm")
    if task_seed is not None and task_seed != int(config.get("crossfit", {}).get("seed", 20260726)):
        raise ValueError("Comparison task seed differs from the common config")
    pattern = config.get("training_io", {}).get("binding_expansion_overlay_pattern")
    if not isinstance(pattern, str) or pattern.count("{fold}") != 1:
        raise ValueError("Binding expansion requires exactly one explicit fold placeholder")
    path = Path(pattern.format(fold=fold))
    if not path.is_absolute(): path = Path(root)/path
    return load_binding_overlay(torch, input_manifest_path=input_manifest_path, overlay_path=path,
        fold=fold, arm=arm, payload=payload, config=config)
