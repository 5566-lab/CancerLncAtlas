"""Hash-bound external evaluation of existing exact/fusion producer outputs.

Isolation is computed against the registered model-use inventory. Completeness
of that inventory is not inferred from a passing hash check or source overlap.
"""
from __future__ import annotations
from dataclasses import fields
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .current_core_export import verified_path, sha256
from .evidence_rule_sources import read_table
from .evidence_rule_summary import KEYS, stable_id, candidate_aliases
from .external_validation_sources import identity_maps, read_sources, isolate_sources
from .external_validation_metrics import evaluate_external, validate_scores


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare_scores(observed, expected, column):
    left = observed.sort_values(KEYS).reset_index(drop=True); right = expected.sort_values(KEYS).reset_index(drop=True)
    if not left[KEYS].equals(right[KEYS]) or not np.allclose(left[column], right[column], rtol=0, atol=1e-12):
        raise ValueError("External scores differ from their producer recomputation")


def producer_scores(spec, root, *, cache=None):
    binding_path = verified_path(spec["producer_binding"], root); binding = _json(binding_path)
    kind = spec["producer_kind"]; column = spec["score_column"]
    if not spec.get("run_id") or binding.get("training_run_id") != spec["run_id"]:
        raise ValueError("External model run identity differs from its producer")
    allowed = {"exact_release": {"association_membership_probability", "l1_probability", "ridge_probability"},
        "current_core_release": {"association_membership_probability", "l1_probability"},
        "fusion_release": {"discovery_adjusted_probability", "fused_confidence_probability"},
        "selected_release": {"selected_score"}}
    if kind not in allowed or column not in allowed[kind]: raise ValueError("Unsupported external prediction producer/score")
    cache_key = (str(binding_path), spec["producer_binding"]["sha256"], spec["run_id"])
    if cache is not None and kind in {"exact_release", "current_core_release"} and cache_key in cache:
        observed, source = cache[cache_key]
        verified_path(source["prediction"], root)
        return observed[[*KEYS, column]].rename(columns={column: "score"}), {**source, "model_id": spec["model_id"], "score_column": column}
    if kind == "exact_release":
        if (binding.get("module_id") != "exact_pathway" or binding.get("five_fold_ensemble") is not True
                or binding.get("training_status") != "SUCCESS" or binding.get("folds") != 5):
            raise ValueError("An exact external score requires its completed five-fold producer")
        if column not in {"association_membership_probability", "l1_probability", "ridge_probability"}:
            raise ValueError("Unsupported exact association score")
        path = verified_path({"path": binding["prediction_path"], "sha256": binding["prediction_sha256"]}, binding_path.parent)
        records = binding["ensemble_source_records"]
        if len(records) != 5 or {r["patient_fold"] for r in records} != set(range(5)):
            raise ValueError("External score source needs every patient fold")
        folds = []
        for item in sorted(records, key=lambda r: r["patient_fold"]):
            for role in ("prediction", "checkpoint", "success", "metrics"):
                source = verified_path({"path": item[role+"_path"], "sha256": item[role+"_sha256"]}, binding_path.parent)
                if role == "prediction": folds.append(source)
        from .exact_release import ensemble_fold_predictions
        expected = ensemble_fold_predictions(folds, run_id=spec["run_id"])
        observed = pd.read_parquet(path)
        inputs = [verified_path(record, binding_path.parent) for record in binding["input_artifacts"]]
        candidates = read_table(inputs[0])[KEYS]
        candidate_aliases(candidates)
        if not candidates.sort_values(KEYS).reset_index(drop=True).equals(expected[KEYS].sort_values(KEYS).reset_index(drop=True)):
            raise ValueError("Exact producer predictions do not cover the bound candidate authority")
        for score_column in allowed[kind]: _compare_scores(observed, expected, score_column)
        # Directions and candidate annotations drive downstream Gene Sets too.
        try:
            pd.testing.assert_frame_equal(observed[expected.columns].sort_values(KEYS).reset_index(drop=True),
                expected.sort_values(KEYS).reset_index(drop=True), check_dtype=False, check_exact=True)
        except (AssertionError, KeyError) as error:
            raise ValueError("Exact producer annotations differ from five-fold recomputation") from error
    elif kind == "current_core_release":
        from .current_core_prediction import load_current_predictions
        observed, binding = load_current_predictions(binding_path, expected_sha=spec["producer_binding"]["sha256"])
        path = verified_path(binding["prediction"], binding_path.parent)
    elif kind == "fusion_release":
        if (binding.get("format") != "CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1"
                or binding.get("status") != "SUCCESS_SECONDARY_FUSION"):
            raise ValueError("External fusion score lacks its completed producer")
        if column not in {"discovery_adjusted_probability", "fused_confidence_probability"}:
            raise ValueError("Unsupported fusion score")
        paths = {key: verified_path(record, binding_path.parent) for key, record in binding["artifacts"].items()}
        for source in binding["source_inputs"]: verified_path(source, binding_path.parent)
        path = paths["secondary_scores"]; observed = pd.read_parquet(path)
        checkpoint = _json(paths["checkpoint"])
        endpoint = "discovery" if column == "discovery_adjusted_probability" else "confidence"
        from .multimodal_fusion import ResidualFusionModel, apply_residual_fusion_model
        record = checkpoint[endpoint]
        model = ResidualFusionModel(**{field.name: record[field.name] for field in fields(ResidualFusionModel) if field.name in record})
        if "fusion_input_private" in paths:
            inputs = pd.read_parquet(paths["fusion_input_private"])
        else:
            # Legacy releases retain native public inputs but lack the complete
            # private input snapshot. Selected current releases require the latter.
            inputs = observed[[*KEYS, "primary_probability"]].copy()
            for index, name in enumerate(model.expert_ids):
                transform = model.feature_transforms[index] if model.feature_transforms else "probability_logit"
                suffix = "probability" if transform == "probability_logit" else "value"
                inputs[name+"_probability"] = observed[name+"_native_"+suffix]
                inputs[name+"_available"] = observed[name+"_native_available"]
        _compare_scores(observed, apply_residual_fusion_model(model, inputs), column)
    elif kind == "selected_release":
        from .selected_score import load_selected_scores
        observed, binding = load_selected_scores(binding_path, expected_sha=spec["producer_binding"]["sha256"])
        path = verified_path(binding["scores"], binding_path.parent)
    else: raise ValueError("Unsupported external prediction producer; a bare score table is insufficient")
    if "training_run_id" not in observed or not observed.training_run_id.eq(spec["run_id"]).all():
        raise ValueError("External prediction rows have another run identity")
    source = {"model_id": spec["model_id"],
        "run_id": spec["run_id"], "producer_binding": {"path": str(binding_path), "sha256": sha256(binding_path)},
        "prediction": {"path": str(path), "sha256": sha256(path)}, "score_column": column, "producer_recomputed": True}
    if cache is not None and kind in {"exact_release", "current_core_release"}: cache[cache_key] = observed, source
    return observed[[*KEYS, column]].rename(columns={column: "score"}), source


def evaluate_manifest(path):
    path = Path(path).resolve(); root = path.parent; manifest = _json(path)
    native_use=manifest.get("format")=="CURRENT_EXTERNAL_VALIDATION_INPUT_V2"
    if manifest.get("format") not in {"CURRENT_EXTERNAL_VALIDATION_INPUT_V1","CURRENT_EXTERNAL_VALIDATION_INPUT_V2"}:
        raise ValueError("Unknown current external evaluation manifest")
    if not native_use and "model_use_inventory" in manifest:raise ValueError("Native model-use binding requires external input V2")
    if not native_use and "patient_identity_crosswalk" in manifest:raise ValueError("Patient isolation requires native external input V2")
    maps = identity_maps(read_table(verified_path(manifest["identity_crosswalk"], root)))
    models = {}; model_sources = []; producer_cache = {}
    for spec in manifest["models"]:
        if not isinstance(spec.get("model_id"), str) or not spec["model_id"] or spec["model_id"] in models:
            raise ValueError("External model IDs must be nonempty unique strings")
        model, source = producer_scores(spec, root, cache=producer_cache); models[spec["model_id"]] = model; model_sources.append(source)
    producer_cache.clear()
    validate_scores(models)
    uses = manifest.get("model_use_sources",[])
    for spec in uses:
        if not spec.get("model_ids") or not set(spec["model_ids"]).issubset(models): raise ValueError("Every used source must be assigned to evaluated models")
    if not native_use:
        for name in models:
            if not any(name in s["model_ids"] for s in uses): raise ValueError("Missing model-use inventory for a compared model")
    # The union across compared models is excluded from every model's evaluation.
    # Otherwise a comparator could have seen the primary model's test evidence.
    frames=[read_sources(uses,maps,root,usage=True)] if uses else []
    derived={};dependency_complete=None;dependency_hashes=()
    if native_use:
        from .current_model_use import load_current_model_use,SCOPE
        source=verified_path(manifest["model_use_inventory"],root)
        use_binding,derived=load_current_model_use(source,expected_sha=manifest["model_use_inventory"]["sha256"],models=manifest["models"],root=root)
        frames.append(derived["usage"])
        dependency_complete=use_binding["model_use_inventory_completeness_verified"]
        dependency_hashes=derived["dependencies"].sha256
    usage=pd.concat(frames,ignore_index=True)
    patients=None
    if native_use:
        from .patient_source_identity import ExternalPatientIsolation
        patients=ExternalPatientIsolation(manifest.get("patient_identity_crosswalk"),root,use_binding,derived)
    observations = read_sources(manifest["external_sources"], maps, root,patients=patients)
    if observations.empty: raise ValueError("No external source records were registered")
    isolated, used = isolate_sources(observations, usage,dependency_complete=dependency_complete,dependency_hashes=dependency_hashes)
    settings = manifest["evaluation"]
    required = {"primary_model", "cutoffs", "lncrna_aggregation", "bootstrap_replicates", "seed"}
    if set(settings) != required: raise ValueError("External evaluation policy must explicitly declare all settings")
    metrics, details, coverage = evaluate_external(isolated, used, models, **settings)
    tables={"isolation":isolated,"model_use":used,"metrics":metrics,"details":details,"coverage":coverage}
    if native_use:
        coverage["independence_scope"]=SCOPE
        tables.update(model_dependencies=derived["dependencies"].drop(columns="path"),model_dependency_gaps=derived["gaps"])
        tables["model_use_completeness"]=derived["completeness"]
        tables["model_transformation_replays"]=derived["transformation_replays"]
        tables.update(patient_mapping=derived["patient_mapping"],patient_context_overlap=derived["patient_context_overlap"])
    return manifest,tables,model_sources


def materialize_current_external(manifest_path, output):
    import socket
    if socket.gethostname() != "149": raise RuntimeError("External evaluation must execute on host 149")
    manifest_path = Path(manifest_path).resolve(); digest = sha256(manifest_path)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False,
        "input_manifest_sha256": digest}, indent=2))
    manifest, tables, model_sources = evaluate_manifest(manifest_path)
    outputs = {}
    for name, frame in tables.items():
        path = output/(name.upper()+".parquet"); frame.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {"path": str(path), "sha256": sha256(path), "rows": len(frame)}
    if sha256(manifest_path) != digest: raise ValueError("External policy changed during evaluation")
    from . import external_validation_sources, external_validation_metrics
    native_use=manifest["format"]=="CURRENT_EXTERNAL_VALIDATION_INPUT_V2"
    native_binding=None;scope="REGISTERED_MODEL_USE_ONLY";inventory_complete=False
    implementation_files=[__file__,external_validation_sources.__file__,external_validation_metrics.__file__]
    if native_use:
        from . import current_model_use,patient_source_identity,model_use_completeness
        source=verified_path(manifest["model_use_inventory"],manifest_path.parent)
        native_binding={"path":str(source),"sha256":sha256(source)};scope=current_model_use.SCOPE
        implementation_files.append(current_model_use.__file__)
        implementation_files.append(patient_source_identity.__file__)
        implementation_files.append(model_use_completeness.__file__)
        inventory_complete=model_use_completeness.is_complete(tables["model_use_completeness"])
    implementation = {str(Path(p).resolve()): sha256(Path(p)) for p in
        implementation_files}
    binding = {"format": "CURRENT_EXTERNAL_VALIDATION_RELEASE_V2" if native_use else "CURRENT_EXTERNAL_VALIDATION_RELEASE_V1", "status": "PASS", "target_host": "149",
        "run_id": stable_id("EXTERNAL:", [digest, implementation]), "input_manifest": {"path": str(manifest_path), "sha256": digest},
        "outputs": outputs, "models": model_sources, "implementation_sha256": implementation, "evaluation": manifest["evaluation"],
        "training_started": False, "used_for_model_selection": False, "negative_labels_synthesized": False,
        "isolation_scope":scope,"native_model_use_inventory":native_binding,"model_use_inventory_completeness_verified":inventory_complete,
        "independent_external_validation_attested": False, "production_deployed": False}
    path = output/"EXTERNAL_BINDING.json"; path.write_text(json.dumps(binding, indent=2)+"\n")
    # Recompute from the original raw tables, not from already filtered outputs.
    audit = audit_current_external(path, expected_sha=sha256(path))
    audit_path = output/"RECOMPUTATION_AUDIT.json"; audit_path.write_text(json.dumps(audit, indent=2)+"\n")
    (output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "binding_sha256": sha256(path),
        "audit_sha256": sha256(audit_path), "training_started": False}, indent=2))
    return {"binding_path": str(path), "binding_sha256": sha256(path), "run_id": binding["run_id"], "training_started": False}


def _load_binding(path, expected_sha):
    path = verified_path({"path": str(path), "sha256": expected_sha}, Path.cwd()); binding = _json(path)
    from .current_model_use import SCOPE
    scopes={"CURRENT_EXTERNAL_VALIDATION_RELEASE_V1":"REGISTERED_MODEL_USE_ONLY","CURRENT_EXTERNAL_VALIDATION_RELEASE_V2":SCOPE}
    if (binding.get("format") not in scopes or binding.get("status") != "PASS"
            or binding.get("target_host") != "149" or binding.get("training_started") is not False
            or binding.get("used_for_model_selection") is not False or binding.get("negative_labels_synthesized") is not False
            or binding.get("independent_external_validation_attested") is not False
            or binding.get("isolation_scope") != scopes.get(binding.get("format"))):
        raise ValueError("Invalid external evaluation binding")
    inventory_complete=False
    if binding["format"]=="CURRENT_EXTERNAL_VALIDATION_RELEASE_V2":
        from .current_model_use import load_current_model_use
        source=verified_path(binding["native_model_use_inventory"],path.parent)
        native,_=load_current_model_use(source,expected_sha=sha256(source),replay=False)
        inventory_complete=native["model_use_inventory_completeness_verified"]
    elif binding.get("native_model_use_inventory") is not None:raise ValueError("Legacy external release has a hidden native inventory")
    if binding.get("model_use_inventory_completeness_verified") is not inventory_complete:
        raise ValueError("External completeness claim differs from native input evidence")
    allowed={"isolation","model_use","metrics","details","coverage"}
    if binding["format"]=="CURRENT_EXTERNAL_VALIDATION_RELEASE_V2":
        allowed|={"model_dependencies","model_dependency_gaps","model_use_completeness","model_transformation_replays","patient_mapping","patient_context_overlap"}
    if not set(binding["outputs"])<=allowed or any(r.get("private") is True for r in binding["outputs"].values()):
        raise ValueError("External release may publish only approved aggregate tables")
    outputs = {name: verified_path(record, path.parent) for name, record in binding["outputs"].items()}
    import pyarrow.parquet as pq
    private_columns={"patient_token","canonical_patient_id","patient_id","sample_id","donor_id","query_donor_id","canonical_donor_id"}
    if any(private_columns & set(pq.ParquetFile(file).schema_arrow.names) for file in outputs.values()):
        raise ValueError("External public tables contain private patient identity columns")
    return path, binding, outputs


def audit_current_external(path, *, expected_sha):
    path, binding, outputs = _load_binding(path, expected_sha)
    source = verified_path(binding["input_manifest"], path.parent)
    manifest, expected, model_sources = evaluate_manifest(source)
    if manifest["format"]=="CURRENT_EXTERNAL_VALIDATION_INPUT_V2":
        native=verified_path(manifest["model_use_inventory"],source.parent)
        if binding.get("native_model_use_inventory")!={"path":str(native),"sha256":sha256(native)}:
            raise ValueError("External native model-use provenance drift")
        from .model_use_completeness import is_complete
        if binding["model_use_inventory_completeness_verified"] is not is_complete(expected["model_use_completeness"]):
            raise ValueError("External completeness differs from recomputed input evidence")
    elif binding["format"]!="CURRENT_EXTERNAL_VALIDATION_RELEASE_V1":raise ValueError("External input/release inventory mode differs")
    if model_sources != binding["models"]: raise ValueError("External model provenance drift")
    if set(expected) != set(outputs): raise ValueError("External output tables differ from the registered analysis")
    for name, frame in expected.items():
        observed = pd.read_parquet(outputs[name])
        try: pd.testing.assert_frame_equal(observed, frame, check_dtype=False, rtol=0, atol=1e-12)
        except AssertionError as error: raise ValueError("External source recomputation disagrees: "+name) from error
    return {"status": "PASS", "binding_sha256": expected_sha, "raw_sources_recomputed": True,
        "same_implementation_replay": True, "independent_statistical_implementation": False,
        "biological_independence_attested": False, "training_started": False}


class CurrentExternalValidationQuery:
    def __init__(self, binding_path, *, expected_sha):
        path, self.binding, self.paths = _load_binding(binding_path, expected_sha)
        success = _json(path.parent/"SUCCESS.json")
        if (success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha
                or success.get("audit_sha256") != sha256(path.parent/"RECOMPUTATION_AUDIT.json")):
            raise ValueError("External evaluation lacks completed source recomputation")
        audit = _json(path.parent/"RECOMPUTATION_AUDIT.json")
        if audit.get("status") != "PASS" or audit.get("binding_sha256") != expected_sha:
            raise ValueError("External recomputation audit belongs to a different binding")
        self.binding_sha256 = expected_sha

    def query(self, *, table="metrics", filters=None, limit=100, offset=0):
        if table not in self.paths: raise ValueError("Unknown external evaluation table")
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 1000 or not 0 <= offset <= 1000000:
            raise ValueError("Invalid external pagination")
        filters = filters or {}
        allowed = {"cancer_id", "lncrna_id", "pathway_id", "source_id", "source_group", "endpoint", "isolation_scope", "model_id", "baseline_model_id", "metric",
                   "phase", "use_view", "role", "reason", "status", "dependency_context","source_role","patient_identity_status",
                   "left_model_id","right_model_id","left_dependency_context","right_dependency_context"}
        if not set(filters).issubset(allowed) or not all(isinstance(v, str) for v in filters.values()):
            raise ValueError("External filters require exact string identities")
        connection = duckdb.connect()
        try:
            path = str(self.paths[table]); columns = [r[0] for r in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path]).fetchall()]
            if not set(filters).issubset(columns): raise ValueError("Filter does not belong to this external table")
            where = " AND ".join('"'+key+'"=?' for key in filters) or "true"
            order = ",".join('"'+key+'"' for key in columns if key in allowed) or "1"
            frame = connection.execute(f"SELECT * FROM read_parquet(?) WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [path, *filters.values(), limit, offset]).df()
        finally: connection.close()
        return {"run_id": self.binding["run_id"], "binding_sha256": self.binding_sha256,
            "isolation_scope":self.binding["isolation_scope"],"independent_external_validation_attested":False,
            "model_use_inventory_completeness_verified":self.binding["model_use_inventory_completeness_verified"],
            "rows": json.loads(frame.to_json(orient="records"))}
