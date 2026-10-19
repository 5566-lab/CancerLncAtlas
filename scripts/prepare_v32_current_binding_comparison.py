#!/usr/bin/env python3
"""Prepare current-C, experimental and experimental+predicted sidecars on 149.

No optimizer or training dispatch. Outputs retain the existing guarded trainer
entry and blocked tasks. Input CURRENT_C_BINDING_COMPARISON_INPUT_V1 declares a
hash-bound config, five fold_inputs (prepared/current graph_overlay/input_manifest,
optional global_binding_source), and named native_october or rbns_sequence sources.
Native direct G2 uses graph_input_mode=NATIVE_GLOBAL_PHYSICAL_BINDING in each
fold record and its input manifest; it declares neither historical sidecar role.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import socket

from cc_hhgt.v32.current_core_export import sha256, verified_path
from cc_hhgt.v32.native_graph_input import NATIVE_MODE, C_OVERLAY_MODES, validate_native_base
from cc_hhgt.v32.current_binding_graph import (
    ARMS, adapt_october_edges, normalize_binding_source, build_current_binding_arms, canonical_json_sha,
)
from cc_hhgt.v32.current_binding_overlay import (
    FORMAT, prepare_binding_overlay, comparison_config_sha, non_graph_payload_sha,
)


def load_sources(specs, root):
    import pandas as pd
    if not specs or len({s["source_id"] for s in specs}) != len(specs):
        raise ValueError("Unique, nonempty expansion sources are required")
    tables = []; records = []
    for source in specs:
        kind = source["kind"]; source_id = source["source_id"]
        if kind == "native_october":
            paths = {role: verified_path(source[role], root) for role in ("edges", "lineage")}
            table = adapt_october_edges(pd.read_parquet(paths["edges"]), pd.read_parquet(paths["lineage"]))
        elif kind == "rbns_sequence":
            audit_path = verified_path(source["audit"], root)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (audit.get("format") != "RBNS_SEQUENCE_ANALYSIS_AUDIT_V1" or audit.get("status") != "PASS"
                    or audit.get("lncrna_binding_measured") is not False or not audit.get("annotation_id")):
                raise ValueError("RBNS admission requires the completed sequence-analysis audit")
            path = verified_path(source["binding"], root)
            if source["binding"]["sha256"] != audit["outputs"]["RBNS_TYPED_BINDING.parquet"]["sha256"]:
                raise ValueError("RBNS binding table does not belong to the audited sequence analysis")
            table = pd.read_parquet(path)
            if not table.graph_assay_class.eq("predicted").all(): raise ValueError("RBNS sequence scores are predicted binding only")
        else: raise ValueError(f"Unsupported expansion source kind: {kind}")
        tables.append(normalize_binding_source(table, source_id=source_id))
        records.append(deepcopy(source))
    return tables, records


def load_current_fold(record, root):
    import torch
    from cc_hhgt.v32.c_graph_overlay import apply_c_graph_overlay
    from cc_hhgt.v32.training import _validate_prepared
    fold = record["fold"]
    direct = record.get("graph_input_mode") == NATIVE_MODE
    if direct and ({"graph_overlay", "global_binding_source"} & set(record)):
        raise ValueError("Native direct G2 cannot declare historical overlay sources")
    roles = ("prepared", "input_manifest") if direct else ("prepared", "graph_overlay", "input_manifest")
    paths = {key: verified_path(record[key], root) for key in roles}
    if "global_binding_source" in record:
        paths["global_binding_source"] = verified_path(record["global_binding_source"], root)
    source = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
    matches = [r for r in source["fold_inputs"] if r["fold"] == fold]
    if len(matches) != 1 or matches[0]["sha256"] != record["prepared"]["sha256"]:
        raise ValueError("Current source fold differs from its manifest")
    authorized = deepcopy(matches[0])
    if direct:
        if (authorized.get("graph_input_mode") != NATIVE_MODE or authorized.get("graph_overlay")
                or authorized.get("binding_expansion") or authorized.get("bytes") != paths["prepared"].stat().st_size):
            raise ValueError("Native direct graph differs from its manifest")
        payload = torch.load(paths["prepared"], map_location="cpu", weights_only=False)
        validate_native_base(payload, fold)
        from cc_hhgt.v32.fold_local_graph import replay_binding
        replay_binding(payload["formal_graph_authority"], root, fold)
        _validate_prepared(payload, fold=fold, artifact_hashes=payload.get("artifact_hashes", {}))
        authorized["path"] = str(paths["prepared"])
        return payload, authorized
    if authorized.get("graph_input_mode") == NATIVE_MODE:
        raise ValueError("Native direct graph cannot be loaded as a historical C overlay")
    original_overlay = authorized["graph_overlay"]
    size_only = original_overlay.get("verification_mode") == "SIZE_AND_GRAPH_SEMANTICS_V1"
    if ((size_only and paths["graph_overlay"].stat().st_size != original_overlay.get("bytes"))
            or (not size_only and record["graph_overlay"]["sha256"] != original_overlay.get("sha256"))):
        raise ValueError("Current C sidecar differs from its manifest")
    payload = torch.load(paths["prepared"], map_location="cpu", weights_only=False)
    overlay = torch.load(paths["graph_overlay"], map_location="cpu", weights_only=False)
    payload = apply_c_graph_overlay(payload, overlay, fold=fold,
        source_sha256=record["prepared"]["sha256"],
        overlay_sha256=None if size_only else record["graph_overlay"]["sha256"],
        overlay_bytes=paths["graph_overlay"].stat().st_size,
        receipt_sha256=original_overlay["graph_authority_receipt_sha256"],
        global_binding_source_path=paths.get("global_binding_source"))
    _validate_prepared(payload, fold=fold, artifact_hashes=payload.get("artifact_hashes", {}), authorized_graph_overlay=True)
    authorized["path"] = str(paths["prepared"]); authorized["bytes"] = paths["prepared"].stat().st_size
    authorized["graph_overlay"]["path"] = str(paths["graph_overlay"])
    if "global_binding_source" in paths:
        authorized["graph_overlay"]["global_binding_source_path"] = str(paths["global_binding_source"])
    return payload, authorized


def _pattern(records, field):
    paths = [Path(r[field]).resolve() for r in records]
    pattern = str(paths[0]).replace("FOLD_0", "FOLD_{fold}")
    if pattern.count("{fold}") != 1 or any(pattern.format(fold=i) != str(p) for i, p in enumerate(paths)):
        raise ValueError("Prepared/current overlay files require one consistent FOLD_{fold} pattern")
    return pattern


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--edge-chunk-size", type=int, default=250000)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149": raise RuntimeError("Binding comparison preparation must execute on host 149")
    import torch
    import yaml
    from cc_hhgt.v32.orchestration import build_task_manifest, write_task_manifest
    if not re.fullmatch("[a-z0-9-]+", args.release): raise ValueError("Invalid release token")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")); root = args.manifest.resolve().parent
    if manifest.get("format") != "CURRENT_C_BINDING_COMPARISON_INPUT_V1": raise ValueError("Invalid comparison input format")
    folds = manifest["fold_inputs"]
    if len(folds) != 5 or any(type(r.get("fold")) is not int for r in folds) or {r["fold"] for r in folds} != set(range(5)):
        raise ValueError("Exactly five unique integer patient folds are required")
    folds = sorted(folds, key=lambda r: r["fold"])
    config_path = verified_path(manifest["config"], root)
    common = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mode = common.get("task_contract", {}).get("rbp_evidence_mode")
    if mode not in C_OVERLAY_MODES | {NATIVE_MODE}:
        raise ValueError("Comparison must start with the current typed C config")
    direct = mode == NATIVE_MODE
    if any((r.get("graph_input_mode") == NATIVE_MODE) != direct for r in folds):
        raise ValueError("All comparison folds must use the configured graph input mode")
    if direct and (any({"graph_overlay", "global_binding_source"} & set(r) for r in folds)
                   or common.get("training_io", {}).get("c_graph_overlay_pattern")):
        raise ValueError("Native direct comparison cannot mix historical C overlays")
    if common["task_contract"].get("graph_variant") != "G2":
        raise ValueError("Current C comparison requires the G2 graph variant")
    common["execution_control"] = {"execution_mode": "CODE_ONLY", "training_authorized": False,
        "paid_enabled": False, "max_paid_hours": 0., "max_cost_cny": 0.}
    common.setdefault("runtime_profile", {}).update(device="cpu", target_host="149", mixed_precision=False)
    # Only resolve the common patterns here. load_current_fold hashes the large
    # inputs once when they are consumed, rather than repeating a full read.
    relocated = [{"prepared": str((root/Path(r["prepared"]["path"])).resolve()),
                  **({"overlay": str((root/Path(r["graph_overlay"]["path"])).resolve())} if not direct else {})} for r in folds]
    if any(Path(r["prepared"]).parent.name != "G2" for r in relocated):
        raise ValueError("Prepared folds must retain their registered G2 parent directory")
    common.setdefault("training_io", {}).update(prepared_fold_pattern=_pattern(relocated, "prepared"))
    if not direct: common["training_io"]["c_graph_overlay_pattern"] = _pattern(relocated, "overlay")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    preflight = {"target_host": "149", "training_started": False,
                 "stage": "PREPARE_CURRENT_C_THREE_ARM_COMPARISON", "base_graph_input_mode": mode}
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(preflight, indent=2))
    tables, source_records = load_sources(manifest["sources"], root)
    comparison = {"format": "CURRENT_C_THREE_ARM_COMPARISON_V1", "arms": list(ARMS),
        "source_manifest_sha256": sha256(args.manifest), "config_sha256": comparison_config_sha(common),
        "sources": source_records, "predicted_dropout": .5, "predicted_weight_cap": .25,
        "edge_chunk_size": args.edge_chunk_size, "target_host": "149", "training_started": False,
        "base_graph_input_mode": mode}
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({**preflight,
        "comparison_sha256": canonical_json_sha(comparison)}, indent=2))
    arm_rows = {arm: [] for arm in ARMS}; audits = []
    for record in folds:
        fold = record["fold"]; payload, authorized = load_current_fold(record, root)
        arms = build_current_binding_arms(payload["bundle"], tables, fold=fold, edge_chunk_size=args.edge_chunk_size)
        identity = non_graph_payload_sha(payload); audits.append({**arms.audit, "non_graph_payload_sha256": identity})
        evidence = output/f"SOURCE_ADMISSION_FOLD_{fold}.parquet"
        arms.source_rows.to_parquet(evidence, index=False, compression="zstd")
        for arm in ARMS:
            folder = output/arm; folder.mkdir(exist_ok=True)
            sidecar = prepare_binding_overlay(payload, arms, arm, comparison, non_graph_sha=identity)
            path = folder/f"BINDING_FOLD_{fold}.pt"; torch.save(sidecar, path)
            item = deepcopy(authorized)
            item["binding_expansion"] = {"format": FORMAT, "arm": arm, "path": str(path),
                "sha256": sha256(path), "bytes": path.stat().st_size, "comparison_sha256": canonical_json_sha(comparison),
                "source_admission": {"path": str(evidence), "sha256": sha256(evidence)}}
            arm_rows[arm].append(item)
            del sidecar
        del payload, arms
        print(json.dumps({"fold": fold, "status": "THREE_ARMS_PREPARED", "training_started": False}), flush=True)
    commands = []
    for arm in ARMS:
        folder = output/arm; config = deepcopy(common)
        config["task_contract"].update(binding_expansion_arm=arm,
            binding_comparison_config_sha256=comparison["config_sha256"])
        config["training_io"].update(binding_expansion_overlay_pattern=str(folder/"BINDING_FOLD_{fold}.pt"),
            output_root=str(folder/"training_outputs"))
        if comparison_config_sha(config) != comparison["config_sha256"]: raise ValueError("Arm config changed the common budget")
        (folder/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (folder/"INPUT_MANIFEST.json").write_text(json.dumps({"format": "CURRENT_C_COMPARISON_TRAINING_INPUT_V1",
            "graph_variant": "G2", "fold_inputs": arm_rows[arm], "comparison": comparison}, indent=2))
        run_id = f"v32-g012-g2-binding-{arm.replace('_', '-')}-{args.release}"
        seed = int(config.get("crossfit", {}).get("seed", 20260726))
        tasks = build_task_manifest(run_id=run_id, seed=seed, owner="server149", hardware_class="SERVER149_CPU")
        write_task_manifest(folder/"TASK_MANIFEST.tsv", tasks)
        for task in tasks:
            commands.append({"arm": arm, "fold": task["patient_fold"], "status": "BLOCKED_CODE_ONLY", "argv": [
                "python", "scripts/v32_pipeline.py", "run-shard", "--repo-root", str(Path(__file__).resolve().parents[1]),
                "--config", str(folder/"config.yaml"), "--input-manifest", str(folder/"INPUT_MANIFEST.json"),
                "--task-manifest", str(folder/"TASK_MANIFEST.tsv"), "--approval", str(folder/"TRAINING_APPROVAL.json"),
                "--run-id", run_id, "--task-id", task["task_id"], "--endpoint-id", "server149", "--hardware-class", "SERVER149_CPU"]})
    (output/"COMPARISON.json").write_text(json.dumps({**comparison, "fold_audits": audits,
        "status": "PREPARED_NOT_TRAINED", "commands": commands}, indent=2))
    (output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "training_started": False,
        "comparison_sha256": sha256(output/"COMPARISON.json"), "sidecars": 15, "blocked_tasks": 15}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
