"""Recompute native local-CNV statistics from the files actually consumed.

This verifies numerical transformations and can capture statistical patient
scopes, not original segment acquisition or learned model parameters.
"""
from pathlib import Path

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "NATIVE_LOCAL_CNV_NUMERIC_REPLAY_V1"


def code_paths(producer):
    runners = {"association": "scripts/run_v32_local_cnv_confounding_audit_r2.py",
               "validation": "scripts/materialize_v32_local_cnv_validation_associations.py"}
    names = [runners[producer], "cc_hhgt/v32/local_cnv_sources.py", "cc_hhgt/v32/local_cnv_source_replay.py",
        "cc_hhgt/v32/local_cnv_confounding.py", "cc_hhgt/v32/local_cnv_cohort.py",
        "cc_hhgt/v32/bulk_coexpression_release.py", "cc_hhgt/v32/directional_cnv.py",
        "cc_hhgt/v32/cnv_input_sources.py", "cc_hhgt/stats.py", "cc_hhgt/v32/associations.py",
        "cc_hhgt/v32/fold_local_expression.py", "cc_hhgt/v32/fold_local_activity.py",
        "cc_hhgt/v32/local_cnv_core_preparation.py", "cc_hhgt/v32/local_cnv_patient_use.py"]
    return [ROOT/name for name in names]


def compare(actual, expected):
    """Compare every column, preserving nulls and exact discrete decisions."""
    if list(actual) != list(expected) or len(actual) != len(expected):
        raise ValueError("Local-CNV numerical replay schema/row coverage differs")
    for column in expected:
        a, b = actual[column], expected[column]
        try:
            pd.testing.assert_series_equal(a, b, check_dtype=True,
                check_exact=not pd.api.types.is_float_dtype(b.dtype), rtol=1e-10, atol=1e-12)
        except AssertionError as error:
            raise ValueError("Local-CNV numerical replay differs: "+column) from error


def replay(binding, root, *, cache=None, patient_capture=None, patient_cache=None):
    require_host()
    from .local_cnv_sources import validate
    receipt_path = verified_path(binding["receipt"], root); receipt = read(receipt_path)
    path, value = validate(binding["sources"], root)
    output = verified_path(binding["output"], root)
    if (receipt.get("status") != "PASS" or receipt.get("cancer_id") != value["cancer_id"]
            or receipt.get("consumed_sources") != binding["sources"] or not receipt.get("run_id")):
        raise ValueError("Local-CNV numerical replay producer identity differs")
    matches = [r for r in value["outputs"] if verified_path(r, path.parent) == output]
    if len(matches) != 1: raise ValueError("Local-CNV numerical replay output is not unique")
    target = matches[0]; kind = target["output_kind"]; fold = target["fold"]
    producer_outputs = receipt.get("outputs", []) if kind == "validation" else receipt.get("pair_outputs" if kind == "pair" else "g0_outputs", [])
    if [r for r in producer_outputs if r.get("fold") == fold] != [target]:
        raise ValueError("Local-CNV numerical replay producer/output metadata differs")
    if value.get("numeric_replay_format") is None: return None
    expected_code = {p.name: record(p)["sha256"] for p in code_paths(value["producer"])}
    if {Path(r["path"]).name:r["sha256"] for r in value["implementations"]} != expected_code:
        raise ValueError("Local-CNV numerical replay needs the pinned native implementation")
    result = {"format": FORMAT, "patient_fold": fold, "output_kind": kind,
        "source_release_sha256": binding["sources"]["sha256"], "output_sha256": binding["output"]["sha256"],
        "folds_recomputed": 1, "outputs_compared": 1, "training_started": False,
        "model_forwards_executed": False, "raw_study_ancestry_complete": False}
    # The validation parent is outside the source table receipt. Verify it on
    # every call, including cache hits, before reusing numerical proof.
    parent = None
    if kind == "validation":
        from .local_cnv_core_preparation import validate_local_cnv_audit
        parent = receipt["local_cnv_authority"]
        if validate_local_cnv_audit(parent["root"], expected_success_sha256=parent["success_sha256"]) != parent:
            raise ValueError("Local-CNV numerical replay validation parent differs")
    key = (FORMAT, binding["sources"]["sha256"], binding["receipt"]["sha256"], binding["output"]["sha256"])
    if cache is not None and key in cache:
        if patient_capture is None: return result
        if patient_cache is not None and key in patient_cache:
            patient_capture(patient_cache[key]); return result
    sources = value["records"]; cancer = value["cancer_id"]
    def source(role):
        found = [r for r in sources if r["role"] == role and r.get("fold", -1) in {-1, fold}]
        if len(found) != 1: raise ValueError("Local-CNV replay source role/fold is not unique: "+role)
        return verified_path(found[0], path.parent)
    config_path = source("source_config"); config = read(config_path)
    patient_path = source("patient_map")
    trace = None
    if patient_capture is not None:
        from .local_cnv_patient_use import Trace
        trace = Trace(patient_path, record(patient_path), cancer, fold)
    patients = pd.read_csv(patient_path, sep="\t")
    from .local_cnv_confounding import canonical_patient
    patients["cancer_id"] = patients.cancer_id.astype(str)
    patients["patient_id"] = patients.patient_id.map(canonical_patient)
    patients["patient_fold_id"] = pd.to_numeric(patients.patient_fold_id, errors="raise").astype(int)
    covariates = pd.read_parquet(source("covariates"))
    covariates["patient_id"] = covariates.patient_id.map(canonical_patient)
    covariates["cancer_id"] = covariates.cancer_id.astype(str).str.upper()
    compact = {verified_path(r, path.parent).parent for r in sources if r["role"] == "compact"}
    if len(compact) != 1: raise ValueError("Local-CNV numerical replay compact scope differs")
    from .directional_cnv import DirectionalCNVStore
    cnv = DirectionalCNVStore(compact.pop())
    from .fold_local_expression import expression_path
    from .fold_local_activity import activity_path
    from scripts import run_v32_local_cnv_confounding_audit_r2 as association
    from scripts import materialize_v32_local_cnv_validation_associations as validation
    cov_path, cov_binding = (validation._bind_covariate_authority(config_path, config["sources"])
        if kind == "validation" else association.bind_covariate_authority(config_path, config["sources"], source("covariates")))
    if cov_path != source("covariates") or cov_binding != receipt["covariate_contract"]:
        raise ValueError("Local-CNV numerical replay covariate authority differs")
    def matrix(role, config_key, entity, column):
        actual = source(role)
        declared = (activity_path(config["sources"], cancer, fold, patient_path) if entity == "pathway_id"
                    else expression_path(config["sources"], config_key, cancer, fold, patient_path))
        if declared.resolve() != actual: raise ValueError("Local-CNV replay matrix differs from declared source")
        if kind == "validation": return validation._matrix(actual, cancer, entity, column)
        return association.patient_matrix(actual, cancer, entity, column)[0]
    lnc = matrix("lncrna_expression", "lnc_expression_root", "lncrna_id", "logcpm")
    if kind != "graph":
        pairs = pd.read_parquet(source("candidates"), columns=["cancer_id", "lncrna_id", "pathway_id"],
                                filters=[("cancer_id", "==", cancer)]).astype(str)
        from .local_cnv_cohort import from_config, require_receipt
        cohort = from_config(config, config_path.parent, source("candidates"), patient_path)
        require_receipt(receipt, cohort)
        pathway = matrix("pathway_activity", "pathway_activity_root", "pathway_id", "activity_score")
    if kind == "pair":
        expected = association.association_fold(cancer, fold, pairs, lnc, pathway, patients, cnv, covariates,
                                                run_id=receipt["run_id"], patient_trace=trace)
        fields = ("covariates_used", "missing_covariate_patients", "base_design_rank",
            "a0_train_design_rank", "a0_validation_design_rank", "validation_covariates_used",
            "validation_missing_covariate_patients", "validation_base_design_rank")
        metadata = {"rows": len(expected), **{k:expected.attrs[k] for k in fields}}
    elif kind == "graph":
        genes = matrix("gene_expression", "gene_expression_root", "gene_id", "logcpm")
        allowed = set()
        for name in ("signed_membership", "protein_gene_encoding"):
            allowed.update(pd.read_parquet(source(name), columns=["gene_id"]).gene_id.astype(str))
        expected = association.g0_fold(cancer, fold, lnc, genes, patients, covariates, cnv, allowed,
                                      run_id=receipt["run_id"], patient_trace=trace)
        train_n = len(association.split_patients(patients, cancer, fold))
        metadata = {"rows": len(expected), "declared_train_patients": train_n,
            "status": "TYPED_UNAVAILABLE_INSUFFICIENT_TRAIN_PATIENTS" if train_n < 40 else
                "PASS_EMPTY_NO_SIGNIFICANT_EDGES" if expected.empty else "PASS_REAL_G0_EDGE_COMPARISON"}
    else:
        expected, metadata = validation.validation_fold(cancer, fold, pairs, lnc, pathway, patients, cnv, covariates,
            run_id=receipt["run_id"], audit_success_sha256=parent["success_sha256"], patient_trace=trace)
    compare(pd.read_parquet(output), expected)
    if any(target.get(k) != v for k,v in metadata.items()):
        raise ValueError("Local-CNV numerical replay statistical metadata differs")
    # Recheck every byte after calculation, including on future cached calls.
    validate(binding["sources"], root)
    verified_path(binding["receipt"], root)
    if parent is not None and validate_local_cnv_audit(parent["root"], expected_success_sha256=parent["success_sha256"]) != parent:
        raise ValueError("Local-CNV validation parent changed during numerical replay")
    for p in code_paths(value["producer"]):
        if record(p)["sha256"] != expected_code[p.name]: raise ValueError("Local-CNV replay implementation changed")
    if cache is not None: cache.add(key)
    if trace is not None:
        captured = trace.payload()
        if patient_cache is not None: patient_cache[key] = captured
        patient_capture(captured)
    return result


def inspect(inventory, binding, root, model, role, fold):
    from .local_cnv_patient_use import attach
    result = replay(binding, root, cache=inventory.visited,
        patient_capture=lambda payload:attach(inventory,payload,binding,root,model,role,fold),
        patient_cache=inventory.statistical_patient_cache)
    if result is None:
        inventory.gap(model, role, "LOCAL_CNV_NUMERIC_REPLAY_NOT_BOUND", fold, binding["sources"]["sha256"])
        return False
    if result["patient_fold"] != fold: raise ValueError("Local-CNV numerical replay consumed fold differs")
    if result["output_kind"] == "validation":
        parent = read(verified_path(binding["receipt"], root))["local_cnv_authority"]
        for name, key in (("SUCCESS.json", "success_sha256"),
                          ("LOCAL_CNV_INDEPENDENT_AUDIT.json", "independent_audit_sha256"),
                          ("LOCAL_CNV_DECISION.json", "decision_sha256")):
            inventory.bind({"path":str(Path(parent["root"])/name), "sha256":parent[key]}, root, model,
                           role+".validation_parent."+key, fold)
    inventory.transform_replays.append({"model_id": model, "dependency_context": inventory.context_id,
        "patient_fold_id": fold, "role": role+"."+result["output_kind"], "format": FORMAT,
        "source_release_sha256": result["source_release_sha256"], "normalization_sha256": "", "activity_sha256": "",
        "folds_recomputed": 1, "outputs_compared": 1, "model_training_started": False})
    return True
