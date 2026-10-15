"""Bind native Drug analyses to the core and score snapshot actually selected.

The replay below runs descriptive statistics and frozen forwards, never fits.
It verifies producer outputs, including subtype and point-dose context. Public
model probabilities outside attribution queries are source-bound, not replayed.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from . import drug_training as base
from .current_core_export import sha256, validate_current_export
from .drug_sparse_query import (DrugSparseQueryBundle, _bounded_duckdb,
    _parquet_relation, _parquet_file_relation, PROBABILITY_COLUMN)
from .drug_training_streaming import StreamingDrugTrainingConfig
from .input_lineage import artifact_sha256
from .integrated_model import build_private_auxiliary_head, module_state_sha256
from .selected_genesets import SelectedAnalysisQuery, audit_selected_analysis

INPUT_FORMAT = "CURRENT_DRUG_CONTEXT_INPUT_V1"
KEYS = ["cancer_id", "lncrna_id", "drug_id"]
SCAN_ROLES = {"exact":"exact_candidates", "response":"raw_drug_response",
    "expression":"raw_lncrna_expression", "mapping":"cell_line_map", "targets":"drug_gene_target"}


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


class DrugSources:
    def __init__(self): self.records = {}

    def bind(self, record, root):
        path = Path(record["path"])
        path = (root/path).resolve() if not path.is_absolute() else path.resolve()
        if artifact_sha256(path) != record["sha256"]:
            raise ValueError("Drug source hash mismatch: "+str(path))
        self.records[str(path)] = {"path":str(path), "sha256":record["sha256"]}
        return path

    def document(self, record, root):
        path = self.bind(record, root)
        return path, read_json(path)

    def verify(self):
        for record in list(self.records.values()): self.bind(record, Path.cwd())


def same_relation(left, right):
    """Bounded, order-independent multiset comparison, including nulls."""
    with _bounded_duckdb() as con:
        if con.execute(f"SELECT * FROM (({left}) EXCEPT ALL ({right})) LIMIT 1").fetchone() is not None:
            raise ValueError("Drug native replay differs from published rows")
        if con.execute(f"SELECT * FROM (({right}) EXCEPT ALL ({left})) LIMIT 1").fetchone() is not None:
            raise ValueError("Drug native replay differs from published rows")


def report_tables(audit_path, sources, *, kind):
    audit = read_json(audit_path)
    success_path = audit_path.parent/"SUCCESS.json"; success = read_json(success_path)
    sources.bind({"path":str(success_path), "sha256":sha256(success_path)}, audit_path.parent)
    if (audit.get("status") != "PASS" or audit.get("target_host") != "149"
            or audit.get("training_started") is not False or success.get("status") != "PASS"
            or success.get("audit_sha256") != sha256(audit_path)):
        raise ValueError("Drug analysis lacks a completed, source-bound native report")
    groups = {}
    for name, record in audit["outputs"].items():
        path = sources.bind(record, audit_path.parent)
        if kind == "attribution":
            match = re.fullmatch(r"(baseline|attribution)_fold[0-4]_part[0-9]+\.parquet", name)
            if not match: raise ValueError("Unknown native Drug attribution output")
            group = match.group(1)
        else:
            if name.startswith("PRIVATE_"):
                if record.get("public_download_eligible") is not False:
                    raise ValueError("Private drug measurement dispositions must not be public")
                continue
            if name in {"CANCER_SUBTYPE_COVERAGE.parquet", "DOSE_CONTEXT_COVERAGE.parquet"}:
                group = name.removesuffix(".parquet").lower()
            elif re.fullmatch(r"(associations|dose_associations|dose_contrasts)/[^/]+\.parquet", name):
                group = name.split("/")[0]
            else: raise ValueError("Unknown public native Drug response output")
            if record.get("public_download_eligible") is False:
                raise ValueError("Public Drug aggregate marked private")
        groups.setdefault(group, []).append(path)
    return audit, groups


def current_drug_sources(manifest_path):
    import torch
    from .current_core_prediction import FORMAT as CURRENT_CORE_PREDICTION
    from .drug_pharmacology import load_pharmacology, FEATURES, TARGET_POOLS

    path = Path(manifest_path).resolve(); manifest = read_json(path); root = path.parent
    if (manifest.get("format") != INPUT_FORMAT or not manifest.get("core_run_id")
            or not manifest.get("drug_run_id")):
        raise ValueError("Invalid current Drug context manifest")
    sources = DrugSources()
    selected_path = sources.bind(manifest["selected_analysis"], root)
    selected = SelectedAnalysisQuery(selected_path, expected_sha=manifest["selected_analysis"]["sha256"])
    audit_selected_analysis(selected_path, expected_sha=selected.binding_sha256)
    score = read_json(selected.score_binding_path)
    primary_path, primary = sources.document(score["sources"]["reference_exact_producer"]["producer_binding"], selected.score_binding_path.parent)
    if primary.get("format") != CURRENT_CORE_PREDICTION or primary.get("training_run_id") != manifest["core_run_id"]:
        raise ValueError("Drug and selected scores use different current core runs")
    core_path = sources.bind(manifest["core_manifest"], root)
    core, _, _ = validate_current_export(core_path, expected_run_id=manifest["core_run_id"])
    folds = primary["ensemble_source_records"]
    if len(folds) != 5 or {r["patient_fold"] for r in folds} != set(range(5)):
        raise ValueError("Selected Drug context requires five current core folds")
    for row in folds:
        parent = core["folds"][str(row["patient_fold"])]
        if any(row[k] != parent[k] for k in ("checkpoint_sha256", "core_parameter_sha256")):
            raise ValueError("Drug and selected scores use different core checkpoints")
        sources.bind(parent["training_receipt"], core_path.parent)
        for record in parent["exports"].values(): sources.bind(record, core_path.parent)
    candidate_path = sources.bind(primary["input_artifacts"][0], primary_path.parent)
    am_path, am = sources.document(manifest["attribution_manifest"], root)
    if (am.get("format") != "DRUG_MODEL_ATTRIBUTION_INPUT_V1"
            or am.get("core_run_id") != manifest["core_run_id"] or am.get("drug_run_id") != manifest["drug_run_id"]):
        raise ValueError("Drug attribution belongs to another run")
    paths = {role:sources.bind(record, am_path.parent) for role,record in am["artifacts"].items()}
    if artifact_sha256(paths["core_embedding_manifest"]) != manifest["core_manifest"]["sha256"]:
        raise ValueError("Drug attribution uses another core export")
    lineage = read_json(paths["drug_lineage"])
    if (lineage.get("module_id") != "drug" or lineage.get("training_status") != "SUCCESS"
            or lineage.get("training_run_id") != manifest["drug_run_id"] or lineage.get("folds") != 5
            or lineage.get("input_hashes_start") != lineage.get("input_hashes_end")
            or lineage.get("checkpoint_manifest_sha256") != artifact_sha256(paths["checkpoint_manifest"])
            or lineage.get("config_sha256") != artifact_sha256(paths["config"])):
        raise ValueError("Native Drug lineage is incomplete or belongs to another run")
    for key in ("private_head_trained_from_scratch", "core_parameters_frozen", "all_five_folds_have_optimizer_updates"):
        if lineage.get(key) is not True: raise ValueError("Native Drug isolation missing: "+key)
    for key in ("old_checkpoint_loaded", "old_predictions_used_as_features", "old_rankings_used_as_outputs",
                "old_gdsc_prism_association_tables_used", "changes_primary_ranking"):
        if lineage.get(key) is not False: raise ValueError("Native Drug isolation invalid: "+key)
    for role in [*SCAN_ROLES.values(), "exact_pathway_membership", "core_embedding_manifest"]:
        if lineage["input_hashes_start"].get(role) != am["artifacts"][role]["sha256"]:
            raise ValueError("Drug analysis differs from actual trained inputs: "+role)
    columns = "cancer_id,lncrna_id,pathway_id"
    same_relation(f"SELECT DISTINCT {columns} FROM {_parquet_relation(candidate_path)}",
                  f"SELECT DISTINCT {columns} FROM {_parquet_relation(paths['exact_candidates'])}")
    config = StreamingDrugTrainingConfig(**read_json(paths["config"])["training"]); config.validate()
    pharmacology = None; features = list(base._DOMAIN_FEATURES)
    if bool(lineage.get("pharmacology_features")) != ("pharmacology_manifest" in paths):
        raise ValueError("Drug context dropped its trained pharmacology inputs")
    if "pharmacology_manifest" in paths:
        pharmacology = load_pharmacology(paths["pharmacology_manifest"],
            targets=base.normalise_drug_targets(base._read_table(paths["drug_gene_target"])),
            response=base.normalise_raw_response(base._read_table(paths["raw_drug_response"])),
            mapping=base.normalise_cell_line_map(base._read_table(paths["cell_line_map"])))
        if pharmacology.receipt() != lineage["pharmacology_features"]:
            raise ValueError("Drug pharmacology source or encoder differs from the trained head")
        features += list(FEATURES)
    records = read_json(paths["checkpoint_manifest"])["records"]
    if len(records) != 5 or {r["cell_line_fold"] for r in records} != set(range(5)):
        raise ValueError("Current Drug context requires all five native heads")
    for record in records:
        fold = record["cell_line_fold"]; parent = core["folds"][str(fold)]
        cp_record = {"path":am.get("checkpoint_paths", {}).get(str(fold), record["path"]), "sha256":record["sha256"]}
        cp_path = sources.bind(cp_record, am_path.parent)
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
        if (type(fold) is not int or record.get("optimizer_steps",0) <= 0
                or checkpoint.get("optimizer_steps") != record["optimizer_steps"]
                or checkpoint.get("checkpoint_format") != base.PRIVATE_CHECKPOINT_FORMAT
                or checkpoint.get("cell_line_fold") != fold or checkpoint.get("module_id") != "drug"
                or checkpoint.get("old_checkpoint_loaded") is not False or checkpoint.get("old_predictions_used") is not False
                or checkpoint.get("domain_features") != features
                or checkpoint.get("pharmacology_features") != lineage.get("pharmacology_features")):
            raise ValueError("Native Drug head identity, features or training receipt differs")
        for key in ("checkpoint_sha256", "parameter_sha256"):
            source_key = "core_"+key
            expected = parent["checkpoint_sha256" if key == "checkpoint_sha256" else "core_parameter_sha256"]
            if checkpoint.get(source_key) != expected or record.get(source_key) != expected:
                raise ValueError("Drug head is bound to another core fold")
        embeddings = base.load_fold_core_embeddings(core_path, core, fold)
        head, init = build_private_auxiliary_head("drug", core_features=embeddings.lncrna.values.shape[1]*(3+len(TARGET_POOLS) if pharmacology else 3),
            domain_features=len(features), hidden_features=config.hidden_features, dropout=config.dropout, seed=config.seed+fold*101)
        if checkpoint.get("initialization") != init.as_dict() or record.get("initial_parameter_sha256") != init.initial_parameter_sha256:
            raise ValueError("Native Drug initialization differs from actual architecture")
        state = checkpoint["model_state"]
        if not state or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Native Drug tensors are invalid")
        head.load_state_dict(state, strict=True); digest = module_state_sha256(head)
        if (digest == init.initial_parameter_sha256 or digest != checkpoint.get("final_parameter_sha256")
                or digest != record.get("final_parameter_sha256")):
            raise ValueError("Native Drug head did not change or its parameter hash differs")
        for key in ("domain_mean", "domain_scale"):
            values = np.asarray(checkpoint[key])
            if values.shape != (len(features),) or not np.isfinite(values).all() or (key == "domain_scale" and (values <= 0).any()):
                raise ValueError("Drug feature normalization is invalid")
    sparse_path = sources.bind(manifest["sparse_query_manifest"], root)
    sparse = DrugSparseQueryBundle(sparse_path, expected_manifest_sha256=manifest["sparse_query_manifest"]["sha256"])
    if (sparse.manifest["training_run_id"] != manifest["drug_run_id"] or sparse.manifest["model_run_status"] != "SUCCESS"
            or lineage.get("drug_sparse_query_manifest_sha256") != sparse.manifest_sha256
            or lineage.get("prediction_sha256") != artifact_sha256(sparse.paths["available_predictions"])):
        raise ValueError("Drug sparse predictions differ from the native trained release")
    for source in sparse.paths.values(): sources.bind({"path":str(source),"sha256":artifact_sha256(source)}, sparse_path.parent)
    same_relation(f"SELECT DISTINCT {columns} FROM {_parquet_relation(candidate_path)}",
                  f"SELECT {columns} FROM {_parquet_relation(sparse.paths['exact_candidates'])}")
    ap = sources.bind(manifest["attribution_audit"], root); attribution, attr_tables = report_tables(ap,sources,kind="attribution")
    if (attribution.get("format") != "DRUG_MODEL_TARGET_ABLATION_V1"
            or attribution.get("core_run_id") != manifest["core_run_id"] or attribution.get("drug_run_id") != manifest["drug_run_id"]
            or attribution.get("input_manifest_sha256") != sha256(am_path)
            or attribution.get("attributions_additive") is not False or attribution.get("biological_causality_claimed") is not False):
        raise ValueError("Drug attribution report identity or semantics differ")
    rp = sources.bind(manifest["response_scan_audit"], root); scan, scan_tables = report_tables(rp,sources,kind="scan")
    if (scan.get("format") != "NATIVE_DRUG_RESPONSE_CONTEXT_SCAN_V1" or scan.get("requires_target_annotation") is not False
            or scan.get("core_model_prediction") is not False or scan.get("causal_claimed") is not False
            or scan.get("tcga_patient_treatment_response_claimed") is not False):
        raise ValueError("Drug measured response report semantics differ")
    for role, record in scan["inputs"].items():
        sources.bind(record, rp.parent)
        if role in SCAN_ROLES and record["sha256"] != am["artifacts"][SCAN_ROLES[role]]["sha256"]:
            raise ValueError("Measured response scan uses different native inputs: "+role)
    return dict(manifest=manifest, path=path, sources=sources, selected=selected, sparse=sparse,
        attribution_manifest=am_path, attribution_audit=attribution, attribution_tables=attr_tables,
        scan=scan, scan_tables=scan_tables, pharmacology=pharmacology)


def replay_native_reports(context):
    """Recompute native aggregates/ablation and compare actual table contents."""
    from scripts.analyze_v32_drug_model_attribution import main as attribution_main
    from scripts.analyze_v32_native_drug_response import main as response_main
    scan = context["scan"]; config = scan["config"]
    # The native CLI currently has one fixed split seed.
    from .drug_response_analysis import DrugResponseScanConfig
    if config["seed"] != DrugResponseScanConfig().seed: raise ValueError("Unsupported native response scan seed")
    with tempfile.TemporaryDirectory(prefix="current-drug-replay-") as temp:
        root = Path(temp)
        attribution_main(["--manifest",str(context["attribution_manifest"]),"--output",str(root/"ablation")])
        args = [arg for key,record in scan["inputs"].items() for arg in ["--"+key,record["path"]]]
        dose = scan.get("dose_response_analysis")
        if dose:
            dose_record = dose["sources"]["point_manifest"]
            dose_path = context["sources"].bind(dose_record, context["path"].parent)
            args += ["--dose-manifest",str(dose_path)]
        response_main([*args,"--output",str(root/"response"),"--min-models",str(config["min_total_cell_lines"]),
            "--min-models-per-dataset",str(config["min_cell_lines_per_dataset"]),"--chunk-rows",str(config["candidate_chunk_rows"])])
        for label,kind,expected in [("ablation","attribution",context["attribution_tables"]),("response","scan",context["scan_tables"])]:
            _, replayed = report_tables(root/label/"AUDIT.json", DrugSources(), kind=kind)
            if set(replayed) != set(expected): raise ValueError("Drug native replay output coverage differs")
            for group in expected:
                same_relation("SELECT * FROM "+_parquet_file_relation(expected[group]),
                              "SELECT * FROM "+_parquet_file_relation(replayed[group]))
    # Verify that ablation's actual head baseline has the public ensemble value
    # and fold mask. The rest of the sparse release is not claimed as replayed.
    if "baseline" in context["attribution_tables"]:
        relation = _parquet_file_relation(context["attribution_tables"]["baseline"])
        predictions = _parquet_relation(context["sparse"].paths["available_predictions"])
        with _bounded_duckdb() as con:
            bad = con.execute(f"""WITH b AS (SELECT {','.join(KEYS)},
                avg(baseline_probability) FILTER (WHERE available) AS p,
                sum(CASE WHEN available THEN (1 << cell_line_fold) ELSE 0 END) AS mask,
                count(*) AS folds, count(DISTINCT cell_line_fold) AS unique_folds
                FROM {relation} GROUP BY ALL)
                SELECT 1 FROM b LEFT JOIN {predictions} p USING ({','.join(KEYS)})
                WHERE b.folds<>5 OR b.unique_folds<>5 OR
                    (b.mask>0 AND (p.{PROBABILITY_COLUMN} IS NULL OR abs(b.p-p.{PROBABILITY_COLUMN})>1e-6 OR b.mask<>p.prediction_fold_mask))
                    OR (b.mask=0 AND p.{PROBABILITY_COLUMN} IS NOT NULL) LIMIT 1""").fetchone()
        if bad is not None: raise ValueError("Drug ablation baseline disagrees with the public ensemble/fold mask")
    context["sources"].verify()
    if context["pharmacology"] is not None: context["pharmacology"].verify_sources()
