"""One source-bound score snapshot for all downstream consumers; never fits or chooses by test results."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import verified_path, sha256
from .evidence_rule_summary import KEYS, candidate_aliases, stable_id
from .external_validation_current import producer_scores
from .external_validation_metrics import validate_scores

FORMAT = "CURRENT_SELECTED_SCORE_RELEASE_V1"


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def implementation_hashes():
    root = Path(__file__).parent
    names = ["selected_score.py", "external_validation_current.py", "external_validation_metrics.py",
        "exact_release.py", "multimodal_fusion.py", "evidence_rule_release.py", "evidence_rule_summary.py"]
    return {name: sha256(root/name) for name in names}


def selection_provenance(selection, model, reference, root):
    kind = selection.get("kind")
    if kind == "EXPLICIT_SOURCE_REGISTRATION":
        if not isinstance(selection.get("rationale"), str) or not selection["rationale"].strip():
            raise ValueError("Explicit score registration needs its rationale; it is not a best-model claim")
        return {"kind": kind, "rationale": selection["rationale"], "validated_winner_claimed": False,
            "external_metrics_used_for_selection": False, "new_model_selection_performed": False}
    if kind != "G012_VALIDATION_WINNER": raise ValueError("Unknown score selection authority")
    if model["producer_kind"] != "exact_release" or model["score_column"] != "association_membership_probability":
        raise ValueError("A G012 lock cannot select a fusion score or sparse baseline")
    path = verified_path(selection["binding"], root); lock = _json(path)
    from .sealed_test_inference import _validate_winner_declaration
    prepared, checkpoints = _validate_winner_declaration(lock, expected_graph_variant=lock.get("graph_variant"))
    rows = lock.get("selection_table", [])
    if len(rows) != 3 or {r["graph_variant"] for r in rows} != {"G0", "G1", "G2"}:
        raise ValueError("G012 selection table does not cover the declared comparison")
    for row in rows:
        losses = row["fold_validation_losses"]
        if len(losses) != 5 or not np.isfinite(losses).all() or np.min(losses)<0 or not np.isclose(np.mean(losses), row["mean_validation_loss"], rtol=0, atol=1e-12):
            raise ValueError("G012 validation summary does not match its five fold values")
    winner = min(rows, key=lambda r: (r["mean_validation_loss"], ["G0", "G1", "G2"].index(r["graph_variant"])))
    if winner["graph_variant"] != lock["winner_id"]: raise ValueError("G012 selection winner differs from the declared metric")
    reference_path = verified_path(reference["producer_binding"], root); lineage = _json(reference_path)
    if lock["candidate_authority_sha256"] != lineage["input_artifacts"][0]["sha256"]:
        raise ValueError("G012 selection candidate authority differs from the exact producer")
    verified_path(lineage["input_artifacts"][0], reference_path.parent)
    actual = {r["patient_fold"]: r["checkpoint_sha256"] for r in lineage["ensemble_source_records"]}
    if actual != {fold: r["sha256"] for fold, r in checkpoints.items()}:
        raise ValueError("Selected exact predictions are not the validation-locked checkpoints")
    for record in [*prepared.values(), *checkpoints.values()]: verified_path(record, path.parent)
    for item in lineage["ensemble_source_records"]:
        success_path = verified_path({"path": item["success_path"], "sha256": item["success_sha256"]}, reference_path.parent)
        expected = _json(success_path)["artifact_hashes"]
        observed = checkpoints[item["patient_fold"]]["training_artifact_hashes"]
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError("G012 checkpoint training authority differs from exact producer receipts")
    return {"kind": kind, "binding": {"path": str(path), "sha256": sha256(path)}, "winner_id": lock["winner_id"],
        "validated_winner_claimed": True, "external_metrics_used_for_selection": False, "new_model_selection_performed": False}


def prepare_selected_scores(manifest_path):
    manifest_path = Path(manifest_path).resolve(); root = manifest_path.parent; manifest = _json(manifest_path)
    if manifest.get("format") != "CURRENT_SELECTED_ANALYSIS_INPUT_V1": raise ValueError("Unknown selected-analysis manifest")
    chosen = manifest["selected_model"]; reference = manifest["reference_exact_model"]
    if reference["producer_kind"] != "exact_release" or reference["score_column"] != "association_membership_probability":
        raise ValueError("Candidate annotations/direction require a bound exact primary producer")
    if chosen["producer_kind"] not in {"exact_release", "fusion_release"}: raise ValueError("Selected score must come from an actual supported producer")
    cache = {}
    primary, primary_source = producer_scores(reference, root, cache=cache)
    selected, selected_source = producer_scores(chosen, root, cache=cache)
    validate_scores({"primary": primary, "selected": selected})
    primary_binding_path = Path(primary_source["producer_binding"]["path"]); primary_binding = _json(primary_binding_path)
    scientific_status = "EXACT_PRODUCER_REGISTERED_NO_NEW_VALIDATION"
    performance_outcome = "NOT_REESTIMATED"
    if chosen["producer_kind"] == "exact_release":
        if selected_source["producer_binding"]["sha256"] != primary_source["producer_binding"]["sha256"]:
            raise ValueError("Exact selected scores and annotations must come from the same producer")
    else:
        fusion_path = Path(selected_source["producer_binding"]["path"]); fusion = _json(fusion_path)
        audit_path = verified_path(fusion["artifacts"]["fusion_input_audit"], fusion_path.parent)
        audit = _json(audit_path)
        if (audit.get("status") != "PASS" or audit.get("core_run_id") != reference["run_id"]
                or audit.get("primary_source_sha256") != primary_source["prediction"]["sha256"]):
            raise ValueError("Selected fusion and exact annotations come from different primary runs")
        if not any(r["sha256"] == primary_source["prediction"]["sha256"] for r in fusion["source_inputs"]):
            raise ValueError("The selected fusion producer did not consume this exact primary")
        if "fusion_input_private" not in fusion["artifacts"]:
            raise ValueError("Selected fusion requires its actual private input snapshot; rematerialize the release")
        lineage = _json(verified_path(fusion["artifacts"]["module_lineage"], fusion_path.parent))
        endpoint = "discovery" if chosen["score_column"] == "discovery_adjusted_probability" else "confidence"
        scientific_status = lineage[endpoint+"_scientific_status"]
        performance_outcome = lineage[endpoint+"_performance_outcome"]
    selection = selection_provenance(manifest["selection"], chosen, reference, root)
    public = pd.read_parquet(primary_source["prediction"]["path"])
    required = [*KEYS, "pathway_family_id", "shared_or_local_scope", "association_membership_probability", "association_direction_probability"]
    if not set(required).issubset(public): raise ValueError("Primary annotations or direction probability are missing")
    frame = public[required].copy(); candidate_aliases(frame)
    if (frame.pathway_family_id.isna().any() or frame.pathway_family_id.astype(str).eq("").any()
            or frame.pathway_id.eq(frame.pathway_family_id).any()
            or frame.groupby("pathway_id").pathway_family_id.nunique().gt(1).any()):
        raise ValueError("Exact pathway/family annotations are ambiguous")
    if not frame.shared_or_local_scope.isin(["shared", "cancer_local"]).all():
        raise ValueError("Selected analysis requires explicit shared/cancer_local scope")
    direction = pd.to_numeric(frame.association_direction_probability, errors="coerce")
    if not np.isfinite(direction).all() or not direction.between(0, 1).all(): raise ValueError("Invalid primary direction probability")
    frame["association_direction"] = np.select([direction.gt(.5), direction.lt(.5)], ["positive", "negative"], default="unknown")
    frame["direction_available"] = direction.ne(.5)
    frame["direction_source_run_id"] = reference["run_id"]
    frame["direction_semantics"] = "PRIMARY_MODEL_ASSOCIATION_DIRECTION_NOT_CAUSAL_REGULATION"
    frame = frame.merge(selected.rename(columns={"score": "selected_score"}), on=KEYS, how="left", validate="one_to_one")
    if chosen["producer_kind"] == "fusion_release":
        from dataclasses import fields
        from .multimodal_fusion import ResidualFusionModel, apply_residual_fusion_model
        inputs = pd.read_parquet(verified_path(fusion["artifacts"]["fusion_input_private"], fusion_path.parent))
        primary_keys = primary.sort_values(KEYS).reset_index(drop=True)
        inputs = inputs.sort_values(KEYS).reset_index(drop=True)
        if not inputs[KEYS].equals(primary_keys[KEYS]) or not np.array_equal(inputs.primary_probability.to_numpy(), primary_keys.score.to_numpy()):
            raise ValueError("Actual fusion inputs differ from the declared exact primary scores")
        endpoint = "discovery" if chosen["score_column"] == "discovery_adjusted_probability" else "confidence"
        parameters = _json(verified_path(fusion["artifacts"]["checkpoint"], fusion_path.parent))[endpoint]
        model = ResidualFusionModel(**{f.name: parameters[f.name] for f in fields(ResidualFusionModel) if f.name in parameters})
        applied = apply_residual_fusion_model(model, inputs)
        auxiliary = applied[KEYS].copy()
        auxiliary["selected_primary_fallback"] = applied.primary_fallback
        for expert in model.expert_ids:
            auxiliary[f"selected_aux_{expert}_available"] = applied[expert+"_available"]
            auxiliary[f"selected_aux_{expert}_contributed"] = applied[expert+"_available"] & applied[expert+"_logit_contribution"].ne(0)
        frame = frame.merge(auxiliary, on=KEYS, how="left", validate="one_to_one")
    frame["score_source_model_id"] = chosen["model_id"]; frame["score_source_run_id"] = chosen["run_id"]
    frame["score_source_column"] = chosen["score_column"]
    frame["score_semantics"] = "FUSED_CONFIDENCE" if chosen["score_column"] == "fused_confidence_probability" else "ASSOCIATION_SCORE"
    frame["score_scientific_status"] = scientific_status
    frame["score_performance_outcome"] = performance_outcome
    # Actual per-fold estimates exist only for exact outputs. Never replicate a
    # final fusion score five times to invent fold support or uncertainty.
    fold_sources = []
    for item in sorted(primary_binding["ensemble_source_records"], key=lambda r: r["patient_fold"]):
        fold_sources.append({"fold": item["patient_fold"], "path": item["prediction_path"], "sha256": item["prediction_sha256"]})
    if manifest.get("rule_evidence_binding"):
        from .evidence_rule_release import attach_rule_annotations
        path = verified_path(manifest["rule_evidence_binding"], root)
        frame = attach_rule_annotations(frame, binding_path=path, expected_sha=sha256(path))
    return manifest, frame.sort_values(KEYS).reset_index(drop=True), {
        "selected_producer": selected_source, "reference_exact_producer": primary_source, "selection": selection,
        "source_scientific_status": scientific_status, "source_performance_outcome": performance_outcome,
        "selected_fold_estimates_available": chosen["producer_kind"] == "exact_release",
        "reference_fold_sources": fold_sources, "selected_fold_score_column": chosen["score_column"] if chosen["producer_kind"] == "exact_release" else None}


def materialize_selected_scores(manifest_path, output):
    import socket
    if socket.gethostname() != "149": raise RuntimeError("Selected score materialization requires host 149")
    manifest_path = Path(manifest_path).resolve(); source_sha = sha256(manifest_path)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False, "input_manifest_sha256": source_sha}))
    manifest, frame, source = prepare_selected_scores(manifest_path)
    implementation = implementation_hashes()
    run_id = stable_id("SELECTED:", [source_sha, implementation])
    frame["selected_score_run_id"] = run_id
    frame["training_run_id"] = manifest["selected_model"]["run_id"]
    frame["analysis_version"] = "CancerLncAtlas_V3.2_SELECTED_SCORE"
    path = output/"SELECTED_SCORES.parquet"; frame.to_parquet(path, index=False, compression="zstd")
    if sha256(manifest_path) != source_sha: raise ValueError("Selected policy changed during materialization")
    binding = {"format": FORMAT, "status": "PASS", "target_host": "149", "run_id": run_id,
        "training_run_id": manifest["selected_model"]["run_id"], "analysis_version": "CancerLncAtlas_V3.2_SELECTED_SCORE",
        "input_manifest": {"path": str(manifest_path), "sha256": source_sha}, "sources": source,
        "scores": {"path": str(path), "sha256": sha256(path), "rows": len(frame)},
        "selected_score_column": "selected_score", "training_started": False, "source_scores_unchanged": True,
        "primary_direction_relabelled_as_selected_direction": False, "model_use_inventory_completeness_verified": False,
        "production_deployed": False, "implementation_sha256": implementation}
    binding_path = output/"SELECTED_SCORE_BINDING.json"; binding_path.write_text(json.dumps(binding, indent=2)+"\n")
    (output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "binding_sha256": sha256(binding_path), "training_started": False}))
    return binding_path


def load_selected_scores(binding_path, *, expected_sha, recompute=True):
    binding_path = verified_path({"path": str(binding_path), "sha256": expected_sha}, Path.cwd()); binding = _json(binding_path)
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS" or binding.get("target_host") != "149"
            or binding.get("training_started") is not False or binding.get("source_scores_unchanged") is not True
            or binding.get("primary_direction_relabelled_as_selected_direction") is not False
            or binding.get("model_use_inventory_completeness_verified") is not False):
        raise ValueError("Invalid selected score binding")
    if binding["run_id"] != stable_id("SELECTED:", [binding["input_manifest"]["sha256"], binding["implementation_sha256"]]):
        raise ValueError("Selected score version does not match its bound inputs and implementation")
    success = _json(binding_path.parent/"SUCCESS.json")
    if success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha: raise ValueError("Selected score materialization is incomplete")
    path = verified_path(binding["scores"], binding_path.parent); frame = pd.read_parquet(path)
    candidate_aliases(frame)
    if (len(frame) != binding["scores"]["rows"] or not frame.selected_score_run_id.eq(binding["run_id"]).all()
            or not frame.training_run_id.eq(binding["training_run_id"]).all()
            or not frame.analysis_version.eq(binding["analysis_version"]).all()):
        raise ValueError("Selected score row identities drifted")
    if recompute:
        if binding["implementation_sha256"] != implementation_hashes():
            raise ValueError("Selected score replay requires its recorded implementation")
        manifest = verified_path(binding["input_manifest"], binding_path.parent)
        _, expected, sources = prepare_selected_scores(manifest)
        if sources != binding["sources"]: raise ValueError("Selected producer provenance drift")
        if set(frame) != set(expected)|{"selected_score_run_id", "training_run_id", "analysis_version"}:
            raise ValueError("Selected score schema contains undeclared or missing fields")
        try: pd.testing.assert_frame_equal(frame[expected.columns], expected, check_dtype=False, check_exact=True)
        except AssertionError as error: raise ValueError("Selected scores differ from actual producers") from error
    return frame, binding
