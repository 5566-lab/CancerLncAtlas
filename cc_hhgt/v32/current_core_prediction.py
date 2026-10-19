"""Full-candidate inference from the actual selected V4 core; no label evaluation."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import (sha256, verified_path, validate_weights, validate_checkpoint,
    validate_training_receipt, CHECKPOINT_FORMAT)
from .evidence_rule_summary import KEYS, candidate_aliases, stable_id

FORMAT = "CURRENT_CORE_PREDICTION_RELEASE_V1"
INPUT_FORMAT = "CURRENT_CORE_PREDICTION_INPUT_V1"
CHUNK_ESTIMAND = "SIGMOID_OF_REGISTERED_CHUNK_WEIGHTED_MEAN_LOGIT"
FOLD_ESTIMAND = "ARITHMETIC_MEAN_OF_FIVE_FOLD_PROBABILITIES"
VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
NUMERIC = ("association_membership_probability", "association_direction_probability",
           "l1_probability", "graph_residual", "graph_gate")


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def implementation_hashes():
    root = Path(__file__).parent
    names = ["current_core_prediction.py", "current_core_export.py", "model.py", "training.py", "integrated_model.py",
             "prediction_scope.py", "fold_local_scope.py", "fold_local_graph.py", "native_graph_input.py"]
    return {**{name: sha256(root/name) for name in names},
        "export_v32_current_core_embeddings.py": sha256(root.parents[1]/"scripts/export_v32_current_core_embeddings.py"),
        "gnn.py": sha256(root.parent/"gnn.py")}


def candidate_table(manifest, root):
    from .evidence_rule_sources import read_table
    from .prediction_scope import manifest_roles, annotated_candidates, validate_annotations
    manifest_roles(manifest)
    frame = read_table(verified_path(manifest["candidate_authority"], root))
    required = [*KEYS, "pathway_family_id"]
    candidate_aliases(frame)
    if "fold_local_scope" in manifest:
        frame = annotated_candidates(manifest, root, frame)
    else:
        if not set(required).issubset(frame): raise ValueError("Current candidates need exact pathway/family annotations")
        frame = frame[required].copy()
        scope = read_table(verified_path(manifest["lncrna_scope"], root))[["cancer_id", "lncrna_id", "shared_or_local_scope"]]
        if scope.duplicated(["cancer_id", "lncrna_id"]).any(): raise ValueError("Ambiguous current lncRNA scope")
        frame = frame.merge(scope, on=["cancer_id", "lncrna_id"], how="left", validate="many_to_one")
    validate_annotations(frame)
    if (frame.pathway_family_id.isna().any() or frame.pathway_family_id.eq("").any()
            or frame.pathway_family_id.eq(frame.pathway_id).any()
            or frame.groupby("pathway_id").pathway_family_id.nunique().gt(1).any()):
        raise ValueError("Incomplete or ambiguous current candidate annotations")
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)


def prediction_batches(payload, candidates):
    """Strip to whitelisted train-derived features; check parity without reading labels."""
    import torch
    if any(str(k).lower().startswith("test") or k in {"sealed_test_labels", "heldout_test_labels"} for k in payload):
        raise ValueError("Current prediction does not accept sealed test payloads")
    bundle = payload["bundle"]; maps = bundle.node_maps
    inverse = {}
    for kind in ("lncRNA", "pathway", "cancer"):
        mapping = maps[kind]
        if any(type(v) is not int for v in mapping.values()) or sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Candidate node maps must be one-to-one and zero-based")
        inverse[kind] = {v: str(k) for k,v in mapping.items()}
    trains = payload["train_batches"]; validations = payload["validation_batches"]
    if not trains or len(trains) != len(validations): raise ValueError("Current feature views have different batch coverage")
    index = pd.MultiIndex.from_frame(candidates[KEYS]); seen = np.zeros(len(candidates), dtype=bool); batches = []
    for train, validation in zip(trains, validations):
        batch = {"candidate_batch": {}}
        for field in ("base_logit", "conservation_context", "graph_available"):
            left = train.get(field); right = validation.get(field)
            if field == "graph_available" and left is None and right is None:
                left = right = torch.ones(len(train["base_logit"]), dtype=torch.bool)
            if (not torch.is_tensor(left) or not torch.is_tensor(right) or left.dtype != right.dtype
                    or left.shape != right.shape or not torch.equal(left, right)):
                raise ValueError("Train/validation prediction feature drift: "+field)
            batch[field] = left.detach().cpu()
        if set(train["candidate_batch"]) != {"l", "p", "c"} or set(validation["candidate_batch"]) != {"l", "p", "c"}:
            raise ValueError("Prediction candidate inputs may contain only node indices")
        keys = {}
        for field, kind, column in (("l", "lncRNA", "lncrna_id"), ("p", "pathway", "pathway_id"), ("c", "cancer", "cancer_id")):
            left = train["candidate_batch"][field]; right = validation["candidate_batch"][field]
            if (not torch.is_tensor(left) or left.dtype != torch.int64 or left.ndim != 1
                    or not torch.is_tensor(right) or right.dtype != left.dtype or not torch.equal(left, right)):
                raise ValueError("Train/validation candidate node alignment differs")
            try: keys[column] = [inverse[kind][int(v)] for v in left]
            except KeyError as error: raise ValueError("Prediction references an unknown node") from error
            batch["candidate_batch"][field] = left.detach().cpu()
        frame = pd.DataFrame(keys)
        positions = index.get_indexer(pd.MultiIndex.from_frame(frame[KEYS]))
        if (len(frame) == 0 or frame.duplicated(KEYS).any() or (positions < 0).any() or seen[positions].any()):
            raise ValueError("Prediction features duplicate or leave the candidate authority")
        n = len(frame)
        if (batch["base_logit"].shape != (n,) or batch["conservation_context"].shape != (n, int(payload.get("conservation_context_features", 4)))
                or batch["graph_available"].shape != (n,) or batch["graph_available"].dtype != torch.bool
                or not torch.isfinite(batch["base_logit"]).all() or not torch.isfinite(batch["conservation_context"]).all()):
            raise ValueError("Prediction features are not finite aligned vectors")
        seen[positions] = True; batch["candidate_positions"] = positions; batches.append(batch)
    if not seen.all(): raise ValueError("Prediction features do not cover every authorized candidate")
    return batches


def stream_predictions(model, bundle, batches, chunks, weights, graph_for_chunk, *, candidate_rows, device="cpu"):
    """One encoder per chunk and compensated logit sums, without a chunk stack."""
    import torch
    from .integrated_model import freeze_v32_core, assert_v32_core_unchanged
    validate_weights(chunks, weights)
    before = freeze_v32_core(model)
    columns = ("final_logit", "direction_logit", "graph_residual", "graph_gate")
    total = {k: np.zeros(candidate_rows, dtype=np.float64) for k in columns}
    compensation = {k: np.zeros(candidate_rows, dtype=np.float64) for k in columns}
    base = np.empty(candidate_rows, dtype=np.float64); available = np.empty(candidate_rows, dtype=bool)
    seen = np.zeros(candidate_rows, dtype=bool)
    for batch in batches:
        positions = batch["candidate_positions"]
        if (len(np.unique(positions)) != len(positions) or (positions < 0).any()
                or (positions >= candidate_rows).any() or seen[positions].any()):
            raise ValueError("Streaming prediction candidate indices overlap or exceed authority")
        seen[positions] = True; base[positions] = batch["base_logit"].numpy(); available[positions] = batch["graph_available"].numpy()
    if not seen.all(): raise ValueError("Streaming prediction omitted candidates")
    with torch.inference_mode():
        for chunk in chunks:
            graph = graph_for_chunk(bundle, chunk, device=device); encoded = model.encoder.encode(graph)
            for batch in batches:
                positions = batch["candidate_positions"]
                output = model(graph, {k:v.to(device) for k,v in batch["candidate_batch"].items()},
                    batch["base_logit"].to(device), batch["conservation_context"].to(device), batch["graph_available"].to(device),
                    admitted=True, encoded=encoded)
                for key in columns:
                    value = output[key].detach().cpu().to(torch.float64).numpy()
                    if value.shape != (len(positions),) or not np.isfinite(value).all():
                        raise ValueError("Invalid current prediction output: "+key)
                    correction = value*weights[chunk] - compensation[key][positions]
                    updated = total[key][positions] + correction
                    compensation[key][positions] = (updated-total[key][positions])-correction
                    total[key][positions] = updated
            del graph, encoded
    assert_v32_core_unchanged(model, before)
    tolerance = 2*np.finfo(np.float32).eps*np.maximum(np.abs(base), 1.)
    if (np.abs(total["final_logit"]-base-total["graph_residual"]) > tolerance).any():
        raise ValueError("Current prediction residual does not compose with its L1 base")
    if not np.array_equal(total["final_logit"][~available], base[~available]):
        # Weighted float arithmetic can differ from base by roundoff. Restore the
        # mathematically exact fallback only after checking it is unmodified.
        if not np.allclose(total["final_logit"][~available], base[~available], rtol=0, atol=1e-12):
            raise ValueError("Unavailable graph changed the primary fallback")
    total["final_logit"][~available] = base[~available]
    from .model import probability_from_logit
    frame = pd.DataFrame({"association_membership_probability": probability_from_logit(total["final_logit"]),
        "association_direction_probability": probability_from_logit(total["direction_logit"]),
        "l1_probability": probability_from_logit(base), "graph_residual": total["graph_residual"],
        "graph_gate": total["graph_gate"], "graph_available": available})
    if (not frame.graph_gate.between(0,1).all() or frame.graph_residual.abs().gt(1+2e-6).any()):
        raise ValueError("Current graph residual/gate violates its bounded model")
    return frame, {"core_parameter_sha256": before, "core_unchanged_after_prediction": True,
        "encoder_forward_calls": len(chunks), "decoder_forward_calls": len(chunks)*len(batches),
        "chunk_estimand": CHUNK_ESTIMAND, "labels_used_for_prediction": False, "optimizer_started": False}


def _fold_source(manifest, record, root, candidates, device):
    import torch
    import yaml
    from scripts.export_v32_current_core_embeddings import _validated_payload, _build_model
    from .training import _runtime_chunk_contract, _runtime_graph_for_chunk
    roles = ["config", "input_manifest", "prepared", "checkpoint", "training_receipt"]
    roles += [k for k in ("graph_overlay", "global_binding_source", "binding_expansion") if k in record]
    paths = {k: verified_path(record[k], root) for k in roles}; fold = record["fold"]
    receipt = _json(paths["training_receipt"])
    validate_training_receipt(receipt, run_id=manifest["core_run_id"], fold=fold,
        checkpoint_sha=record["checkpoint"]["sha256"], variant=manifest["graph_variant"])
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, receipt, fold=fold, prepared_sha=record["prepared"]["sha256"],
        config_sha=record["config"]["sha256"], input_manifest_sha=record["input_manifest"]["sha256"])
    candidate_sha = receipt["artifact_hashes"].get("formal_candidate_authority_sha256")
    if candidate_sha is not None and candidate_sha != manifest["candidate_authority"]["sha256"]:
        raise ValueError("Current prediction candidates differ from the training-authorized file")
    payload = _validated_payload(paths, record, fold)
    batches = prediction_batches(payload, candidates)
    from .prediction_scope import validate_prepared_scope, fold_annotations
    scope_source = validate_prepared_scope(manifest, root, payload, candidates, batches, fold)
    model = _build_model(payload, yaml.safe_load(paths["config"].read_text()), checkpoint, device)
    chunks, weights = _runtime_chunk_contract(payload["bundle"])
    if ({str(c):weights[c] for c in chunks} != receipt["runtime_chunk_weights"] or len(chunks) != receipt["runtime_num_chunks"]):
        raise ValueError("Current inference runtime estimator differs from the trained checkpoint")
    frame, audit = stream_predictions(model, payload["bundle"], batches, chunks, weights, _runtime_graph_for_chunk,
        candidate_rows=len(candidates), device=device)
    frame = pd.concat([fold_annotations(candidates, fold), frame], axis=1)
    frame["patient_fold_id"] = fold; frame["training_run_id"] = manifest["core_run_id"]
    frame["analysis_version"] = VERSION; frame["pathway_target_level"] = "exact_pathway"
    return frame, {**audit, "patient_fold": fold, "checkpoint_format": CHECKPOINT_FORMAT,
        "runtime_chunk_weights": {str(c):weights[c] for c in chunks},
        **({"training_detection_scope": scope_source} if scope_source else {}),
        "sources": {k:{"path":str(paths[k]), "sha256":record[k]["sha256"]} for k in roles}}


def ensemble_predictions(paths, candidates, run_id):
    from .prediction_scope import fold_annotations
    result = candidates.copy(); sums = {k:np.zeros(len(candidates)) for k in NUMERIC}
    for fold, path in enumerate(paths):
        frame = pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)
        expected = fold_annotations(candidates, fold)
        if (not frame[expected.columns].equals(expected) or not frame.patient_fold_id.eq(fold).all()
                or not frame.training_run_id.eq(run_id).all()):
            raise ValueError("Current fold prediction keys, annotations or identity drifted")
        for key in NUMERIC:
            values = frame[key].to_numpy(float)
            if not np.isfinite(values).all(): raise ValueError("Nonfinite current fold predictions")
            if key != "graph_residual" and ((values < 0)|(values > 1)).any(): raise ValueError("Invalid current fold probability/gate")
            if key == "graph_residual" and (np.abs(values)>1+2e-6).any(): raise ValueError("Invalid current fold bounded residual")
            sums[key] += values
    if len(paths) != 5: raise ValueError("Current ensemble requires every patient fold")
    for key in NUMERIC: result[key] = sums[key]/5
    for key in ("association_membership_probability", "association_direction_probability", "l1_probability", "graph_gate"):
        if not result[key].between(0,1).all(): raise ValueError("Current probability/gate is outside [0,1]")
    result["association_direction"] = np.select([result.association_direction_probability.gt(.5), result.association_direction_probability.lt(.5)],
        ["positive", "negative"], default="unknown")
    result["pathway_target_level"] = "exact_pathway"; result["n_folds_available"] = 5
    result["training_run_id"] = run_id; result["analysis_version"] = VERSION
    result["prediction_scope"] = "FULL_CANDIDATE_DISCOVERY_NOT_HELDOUT_EVALUATION"
    return result


def materialize_current_predictions(manifest_path, output, *, expected_run_id, device="cpu"):
    import socket
    if socket.gethostname() != "149": raise RuntimeError("Current candidate inference requires host 149")
    if device not in {"cpu", "cuda"}: raise ValueError("Unknown inference device")
    manifest_path = Path(manifest_path).resolve(); manifest = _json(manifest_path); root = manifest_path.parent
    source_sha = sha256(manifest_path)
    if (manifest.get("format") != INPUT_FORMAT or manifest.get("core_run_id") != expected_run_id
            or manifest.get("graph_variant") not in {"G0", "G1", "G2"}):
        raise ValueError("Invalid current prediction inputs or requested run")
    records = manifest["fold_inputs"]
    if len(records) != 5 or any(type(r.get("fold")) is not int for r in records) or {r["fold"] for r in records} != set(range(5)):
        raise ValueError("Current prediction requires five unique integer folds")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149", "training_started":False,
        "stage":"FROZEN_FULL_CANDIDATE_INFERENCE", "core_run_id":expected_run_id, "device":device, "input_manifest_sha256":source_sha}))
    candidates = candidate_table(manifest, root); folds = []; paths = []
    for record in sorted(records, key=lambda r:r["fold"]):
        frame, audit = _fold_source(manifest, record, root, candidates, device)
        path = output/f"FOLD_{record['fold']}.parquet"; frame.to_parquet(path,index=False,compression="zstd")
        paths.append(path); sources = audit["sources"]
        folds.append({**audit,"prediction_path":str(path),"prediction_sha256":sha256(path),
            "checkpoint_path":sources["checkpoint"]["path"],"checkpoint_sha256":sources["checkpoint"]["sha256"],
            "success_path":sources["training_receipt"]["path"],"success_sha256":sources["training_receipt"]["sha256"]})
        print(json.dumps({"fold":record["fold"],"rows":len(frame),"status":"PREDICTED_WITHOUT_FITTING"}),flush=True)
    ensemble = ensemble_predictions(paths, candidates, expected_run_id)
    prediction = output/"CURRENT_CORE_PREDICTIONS.parquet"; ensemble.to_parquet(prediction,index=False,compression="zstd")
    if sha256(manifest_path) != source_sha: raise ValueError("Current prediction manifest changed during inference")
    implementation = implementation_hashes()
    from .prediction_scope import manifest_roles
    binding = {"format":FORMAT,"status":"PASS","target_host":"149","training_run_id":expected_run_id,
        "run_id":stable_id("CURRENT-PREDICTION:",[source_sha,implementation]),"graph_variant":manifest["graph_variant"],
        "input_manifest":{"path":str(manifest_path),"sha256":source_sha}, "implementation_sha256":implementation,
        "input_artifacts":[{"path":str(verified_path(manifest[k],root)),"sha256":manifest[k]["sha256"]} for k in manifest_roles(manifest)],
        "ensemble_source_records":folds,"prediction":{"path":str(prediction),"sha256":sha256(prediction),"rows":len(ensemble)},
        "chunk_estimand":CHUNK_ESTIMAND,"fold_estimand":FOLD_ESTIMAND,"labels_used_for_prediction":False,
        "test_payloads_loaded":False,"metrics_computed":False,"training_started":False,"production_deployed":False,
        "available_score_columns":["association_membership_probability","l1_probability"],
        "ridge_available":False,"ridge_unavailable_reason":"NO_CURRENT_BOUND_RIDGE_PRODUCER"}
    path = output/"CURRENT_PREDICTION_BINDING.json"; path.write_text(json.dumps(binding,indent=2)+"\n")
    load_current_predictions(path,expected_sha=sha256(path),require_success=False)
    (output/"SUCCESS.json").write_text(json.dumps({"status":"PASS","binding_sha256":sha256(path),"training_started":False}))
    return path


def load_current_predictions(binding_path, *, expected_sha, require_success=True):
    path = verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()); binding = _json(path)
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS" or binding.get("target_host") != "149"
            or binding.get("chunk_estimand") != CHUNK_ESTIMAND or binding.get("fold_estimand") != FOLD_ESTIMAND
            or any(binding.get(k) is not False for k in ("labels_used_for_prediction","test_payloads_loaded","metrics_computed","training_started"))):
        raise ValueError("Invalid current candidate prediction release")
    if require_success:
        success = _json(path.parent/"SUCCESS.json")
        if success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha: raise ValueError("Current candidate release is incomplete")
    manifest_path = verified_path(binding["input_manifest"],path.parent); manifest = _json(manifest_path)
    if (manifest.get("format") != INPUT_FORMAT or manifest.get("core_run_id") != binding["training_run_id"]
            or manifest.get("graph_variant") != binding["graph_variant"]
            or binding["run_id"] != stable_id("CURRENT-PREDICTION:",[binding["input_manifest"]["sha256"],binding["implementation_sha256"]])):
        raise ValueError("Current candidate release changed its training identity")
    authorized = {r["fold"]:r for r in manifest["fold_inputs"]}
    if len(manifest["fold_inputs"]) != 5 or set(authorized) != set(range(5)):
        raise ValueError("Current candidate source lacks all five selected folds")
    candidates = candidate_table(manifest,manifest_path.parent)
    from .prediction_scope import manifest_roles, fold_source_record
    expected_inputs = [{"path": str(verified_path(manifest[k], manifest_path.parent)), "sha256": manifest[k]["sha256"]}
                       for k in manifest_roles(manifest)]
    if binding["input_artifacts"] != expected_inputs: raise ValueError("Current prediction input annotations drifted")
    records = binding["ensemble_source_records"]
    if len(records) != 5 or {r["patient_fold"] for r in records} != set(range(5)): raise ValueError("Incomplete current fold predictions")
    paths = []
    for item in sorted(records,key=lambda r:r["patient_fold"]):
        if item.get("core_unchanged_after_prediction") is not True or item.get("optimizer_started") is not False:
            raise ValueError("Current prediction lacks frozen-core evidence")
        authority = authorized[item["patient_fold"]]
        expected_scope = fold_source_record(manifest, manifest_path.parent, item["patient_fold"])
        if item.get("training_detection_scope") != expected_scope:
            raise ValueError("Current prediction training detection scope drifted")
        roles = {"config", "input_manifest", "prepared", "checkpoint", "training_receipt"}
        roles |= {k for k in ("graph_overlay", "global_binding_source", "binding_expansion") if k in authority}
        expected_sources = {k:{"path":str(verified_path(authority[k],manifest_path.parent)),"sha256":authority[k]["sha256"]} for k in roles}
        if item["sources"] != expected_sources: raise ValueError("Current prediction fold sources differ from their manifest")
        if (item["checkpoint_sha256"] != expected_sources["checkpoint"]["sha256"]
                or item["checkpoint_path"] != expected_sources["checkpoint"]["path"]
                or item["success_sha256"] != expected_sources["training_receipt"]["sha256"]
                or item["success_path"] != expected_sources["training_receipt"]["path"]):
            raise ValueError("Current prediction checkpoint/receipt aliases drifted")
        receipt = _json(verified_path({"path":item["success_path"],"sha256":item["success_sha256"]},path.parent))
        validate_training_receipt(receipt,run_id=binding["training_run_id"],fold=item["patient_fold"],
            checkpoint_sha=item["checkpoint_sha256"],variant=binding["graph_variant"])
        weights = item["runtime_chunk_weights"]; validate_weights(tuple(weights),weights)
        if (weights != receipt["runtime_chunk_weights"] or len(weights) != receipt["runtime_num_chunks"]
                or item["encoder_forward_calls"] != len(weights) or item["chunk_estimand"] != CHUNK_ESTIMAND):
            raise ValueError("Current prediction changed chunk weights or coverage")
        paths.append(verified_path({"path":item["prediction_path"],"sha256":item["prediction_sha256"]},path.parent))
    expected = ensemble_predictions(paths,candidates,binding["training_run_id"])
    observed = pd.read_parquet(verified_path(binding["prediction"],path.parent))
    if len(observed) != binding["prediction"]["rows"]: raise ValueError("Current prediction row count drift")
    try: pd.testing.assert_frame_equal(observed,expected,check_dtype=False,check_exact=True)
    except AssertionError as error: raise ValueError("Current prediction differs from actual fold ensemble") from error
    return observed,binding


def audit_current_predictions(binding_path, *, expected_sha, replay_models=False, device="cpu"):
    """Optionally replay actual frozen checkpoints, not just the fold ensemble."""
    import socket
    if socket.gethostname() != "149": raise RuntimeError("Current prediction audit requires host 149")
    _, binding = load_current_predictions(binding_path, expected_sha=expected_sha)
    if replay_models:
        if binding["implementation_sha256"] != implementation_hashes():
            raise ValueError("Frozen checkpoint replay requires the recorded implementation")
        manifest_path = Path(binding["input_manifest"]["path"]); manifest = _json(manifest_path)
        candidates = candidate_table(manifest, manifest_path.parent)
        outputs = {r["patient_fold"]:r for r in binding["ensemble_source_records"]}
        for record in sorted(manifest["fold_inputs"],key=lambda r:r["fold"]):
            expected, audit = _fold_source(manifest, record, manifest_path.parent, candidates, device)
            item = outputs[record["fold"]]
            if audit["core_parameter_sha256"] != item["core_parameter_sha256"]:
                raise ValueError("Replayed core parameters differ from prediction source")
            observed = pd.read_parquet(item["prediction_path"])
            try: pd.testing.assert_frame_equal(observed, expected, check_dtype=False, rtol=0, atol=2e-6)
            except AssertionError as error: raise ValueError("Current fold scores disagree with frozen checkpoint replay") from error
    return {"status":"PASS", "target_host":"149", "binding_sha256":expected_sha,
        "fold_ensemble_recomputed":True,"frozen_checkpoints_replayed":bool(replay_models),
        "replay_precision_absolute_tolerance":2e-6 if replay_models else None,
        "training_started":False,"metrics_computed":False,"independent_biological_validation":False}
