"""Publish tumour-state context from the joint Pathway/State mainline.

The adapter accepts five verified R6.2 formal completions and their signed
outer-test evaluations.  It has no path for the retired frozen-core private
State head or its release format.
"""
from __future__ import annotations

import json
import re
import socket
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .current_core_export import sha256, verified_path
from .table_replay import assert_same_table


INPUT_FORMAT = "CURRENT_JOINT_STATE_CONTEXT_INPUT_V1"
FORMAT = "CURRENT_JOINT_STATE_CONTEXT_RELEASE_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_JOINT"
KEYS = ["cancer_id", "lncrna_id", "state_id"]
CATALOG_COLUMNS = ["gene_set_id", "gene_set_name", "gene_set_type", "scope",
    "cancer_id", "state_id", "direction", "member_count", "selection_rule"]
MEMBER_COLUMNS = ["gene_set_id", "member_rank", "member_id", "member_gene_id",
    "membership_score", "state_effect", "cancer_count", "direction_consistency",
    "source_cancer_id"]
OUTPUT_NAMES = {"state_oof_predictions", "state_release", "selected_member_states",
    "selected_geneset_state_profiles", "selected_state_coverage",
    "state_relation_consensus", "state_cross_cancer_candidates_not_final",
    "state_target_context_subtypes", "state_geneset_catalog", "state_geneset_members",
    "gmt"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def _sigmoid(values):
    x = np.asarray(values, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    result[~positive] = exp / (1.0 + exp)
    return result


def _adapt_engine_state_oof(frame, patient_fold_id):
    """Adapt the exact table written by ``cla_engine.evaluate.evaluate_fold``."""
    required = {"patient_fold_id", "cancer_id", "task", "lncrna_id", "target_id",
        "model_logit", "baseline_logit", "predicted_direction", "heldout_effect",
        "heldout_p", "heldout_q", "heldout_n", "availability"}
    if required - set(frame):
        raise ValueError("Joint State OOF table lacks the R62 evaluate schema")
    frame = frame.loc[frame.task.astype(str).eq("state")].copy()
    if frame.empty or not frame.patient_fold_id.eq(patient_fold_id).all():
        raise ValueError("Joint State OOF table lacks its typed fold rows")
    if frame[["cancer_id", "lncrna_id", "target_id"]].duplicated().any():
        raise ValueError("Joint State OOF table duplicates a cancer/lncRNA/state key")
    available = frame.availability.astype(bool)
    probabilities = _sigmoid(pd.to_numeric(frame.model_logit, errors="raise"))
    baseline = _sigmoid(pd.to_numeric(frame.baseline_logit, errors="raise"))
    frame["state_membership_probability"] = pd.array(
        np.where(available, probabilities, np.nan), dtype="Float64")
    frame["baseline_probability"] = pd.array(
        np.where(available, baseline, np.nan), dtype="Float64")
    effects = pd.to_numeric(frame.heldout_effect, errors="coerce").where(available)
    frame["state_effect"] = pd.array(effects, dtype="Float64")
    frame["state_id"] = frame.pop("target_id").astype(str)
    frame["availability"] = available
    frame["analysis_version"] = ANALYSIS_VERSION
    frame["model_family"] = "cc_hhgt_shared_state_head"
    frame["state_gradients_into_core"] = True
    frame["changes_primary_ranking"] = False
    columns = ["patient_fold_id", *KEYS, "state_membership_probability",
        "baseline_probability", "predicted_direction", "state_effect", "heldout_p",
        "heldout_q", "heldout_n", "availability", "analysis_version", "model_family",
        "state_gradients_into_core", "changes_primary_ranking"]
    return frame[columns].sort_values(
        ["patient_fold_id", *KEYS], kind="stable").reset_index(drop=True)


def _manifest(path):
    source = Path(path).resolve()
    value = _read(source)
    if (set(value) != {"format", "release_id", "selected_analysis", "folds"}
            or value.get("format") != INPUT_FORMAT
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", str(value.get("release_id", "")))
            or not isinstance(value.get("folds"), list) or len(value["folds"]) != 5):
        raise ValueError("Joint State context requires one selected analysis and five explicit folds")
    selected = verified_path(value["selected_analysis"], source.parent)
    from .selected_genesets import SelectedAnalysisQuery
    SelectedAnalysisQuery(selected, expected_sha=value["selected_analysis"]["sha256"])
    folds = []
    for item in value["folds"]:
        if (not isinstance(item, dict)
                or set(item) != {"patient_fold_id", "prepared_manifest", "training_receipt", "evaluation_receipt"}):
            raise ValueError("Joint State fold binding fields differ")
        fold = int(item["patient_fold_id"])
        if type(item["patient_fold_id"]) is not int or fold not in range(5):
            raise ValueError("Joint State patient_fold_id must be 0..4")
        paths = {key: verified_path(item[key], source.parent) for key in
                 ("prepared_manifest", "training_receipt", "evaluation_receipt")}
        if paths["prepared_manifest"].name != "MANIFEST.json":
            raise ValueError("Joint State prepared binding must name MANIFEST.json")
        if paths["training_receipt"].name != "TRAINING_RECEIPT.json":
            raise ValueError("Joint State training binding must name TRAINING_RECEIPT.json")
        if paths["evaluation_receipt"].name != "EVALUATION_RECEIPT.json":
            raise ValueError("Joint State evaluation binding must name EVALUATION_RECEIPT.json")
        folds.append({"patient_fold_id": fold, **paths, "records": item})
    if {item["patient_fold_id"] for item in folds} != set(range(5)):
        raise ValueError("Joint State fold bindings must cover folds 0..4 exactly once")
    return source, value, selected, sorted(folds, key=lambda item: item["patient_fold_id"])


def _validate_fold(item):
    from cla_engine.common import read_json as engine_read, sha as engine_sha, verify_manifest
    from cla_engine.lineage import verify_completion, verify_signed_file
    prepared = item["prepared_manifest"].parent
    training = item["training_receipt"].parent
    evaluation = item["evaluation_receipt"].parent
    receipt, lineage = verify_completion(prepared, training)
    spec = engine_read(prepared / "SPEC.json")
    config = engine_read(prepared / "CONFIG.json")
    if (spec.get("fold") != item["patient_fold_id"]
            or spec.get("architecture") != "CC_HHGT_PYG_FULL_R62_JOINT"
            or spec.get("graph_variant") != "G2"
            or receipt.get("state") != "TRAINED_FORMAL_CANDIDATE"
            or receipt.get("source_profile") != "formal"
            or receipt.get("synthetic_fixture") is not False
            or receipt.get("task_weights") != {"pathway": 1.0, "state": 0.5}
            or receipt.get("gradient_proof", {}).get("state_max_shared_grad", 0) <= 0
            or receipt.get("gradient_proof", {}).get("pathway_max_shared_grad", 0) <= 0):
        raise ValueError("State publication requires a formal G2 joint completion with both shared gradients")
    verify_manifest(evaluation)
    verify_signed_file(evaluation / "MANIFEST.json", config, "OUTER_EVALUATION")
    outer = engine_read(item["evaluation_receipt"])
    if (outer.get("fold") != item["patient_fold_id"]
            or outer.get("state") != "OUTER_TEST_EVALUATED_NOT_DEPLOYED"
            or outer.get("profile") != "formal"
            or outer.get("synthetic_fixture") is not False
            or outer.get("training_state") != "TRAINED_FORMAL_CANDIDATE"
            or outer.get("training_receipt_sha256") != engine_sha(item["training_receipt"])
            or outer.get("prepared_sha256") != engine_sha(item["prepared_manifest"])):
        raise ValueError("Joint State outer evaluation is not bound to its completed formal fold")
    table = evaluation / "OOF_ASSOCIATIONS.tsv.gz"
    if not table.is_file():
        raise ValueError("Joint State outer evaluation lacks OOF_ASSOCIATIONS.tsv.gz")
    frame = _adapt_engine_state_oof(
        pd.read_csv(table, sep="\t"), item["patient_fold_id"])
    return frame, {
        "prepared_manifest": item["records"]["prepared_manifest"],
        "training_receipt": item["records"]["training_receipt"],
        "evaluation_receipt": item["records"]["evaluation_receipt"],
        "training_signature": receipt["signature"],
        "lineage": lineage,
    }, config


def _aggregate_state_oof(oof):
    """Keep the complete candidate universe, including keys unavailable in all folds."""
    working = oof.copy()
    working["available_fold"] = working.availability.astype(int)
    working["available_heldout_n"] = pd.to_numeric(
        working.heldout_n, errors="coerce").where(working.availability, 0).fillna(0)
    aggregate = (working.groupby(KEYS, observed=True, as_index=False, dropna=False)
        .agg(state_membership_probability=("state_membership_probability", "mean"),
             baseline_probability=("baseline_probability", "mean"),
             state_effect=("state_effect", "mean"),
             folds_available=("available_fold", "sum"),
             heldout_n_total=("available_heldout_n", "sum")))
    aggregate["availability"] = aggregate.folds_available.gt(0)
    effect = aggregate.state_effect
    aggregate["association_direction"] = np.select(
        [aggregate.availability & effect.gt(0), aggregate.availability & effect.lt(0),
         aggregate.availability & effect.eq(0)],
        ["positive", "negative", "neutral"], default=None)
    aggregate["folds_expected"] = 5
    aggregate["analysis_version"] = ANALYSIS_VERSION
    aggregate["model_family"] = "cc_hhgt_shared_state_head"
    aggregate["state_gradients_into_core"] = True
    aggregate["changes_primary_ranking"] = False
    return aggregate.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _selected_state_tables(selected_path, aggregate):
    from .selected_genesets import SelectedAnalysisQuery
    selected = SelectedAnalysisQuery(selected_path, expected_sha=sha256(selected_path))
    members = pd.read_parquet(selected.paths["geneset_members"])
    coverage = pd.read_parquet(selected.paths["geneset_coverage"])
    selected_columns = ["geneset_id", "geneset_rank", "cancer_id", "lncrna_id",
        "pathway_id", "selected_score", "association_direction"]
    if not set(selected_columns).issubset(members):
        raise ValueError("Selected member schema is incomplete")
    scores = pd.to_numeric(members.selected_score, errors="coerce")
    if (not np.isfinite(scores).all() or not scores.between(0, 1).all()
            or members.duplicated(["geneset_id", "lncrna_id"]).any()):
        raise ValueError("Selected State members are invalid or duplicated")
    state_ids = sorted(aggregate.state_id.astype(str).unique())
    states = pd.DataFrame({"state_id": state_ids})
    data = members[selected_columns].rename(columns={
        "association_direction": "pathway_association_direction"}).merge(states, how="cross")
    native = aggregate.rename(columns={"association_direction": "state_association_direction"})
    data = data.merge(native, how="left", on=KEYS, validate="many_to_one")
    data["availability"] = data.availability.eq(True)
    data["folds_available"] = data.folds_available.fillna(0).astype(int)
    data["folds_expected"] = 5
    data["state_effect_semantics"] = "OUTER_FOLD_ASSOCIATION_NOT_CAUSAL"
    data["available_score_weight"] = data.selected_score.where(data.availability, 0.0)
    data["weighted_membership"] = data.state_membership_probability.fillna(0) * data.available_score_weight
    data["weighted_effect"] = data.state_effect.fillna(0) * data.available_score_weight
    profile_keys = ["geneset_id", "cancer_id", "pathway_id", "state_id",
        "pathway_association_direction"]
    profiles = data.groupby(profile_keys, sort=True, dropna=False).agg(
        member_count=("lncrna_id", "size"), available_members=("availability", "sum"),
        total_score_weight=("selected_score", "sum"),
        available_score_weight=("available_score_weight", "sum"),
        weighted_membership=("weighted_membership", "sum"),
        weighted_effect=("weighted_effect", "sum")).reset_index()
    denominator = profiles.available_score_weight.where(profiles.available_score_weight.gt(0))
    profiles["mean_state_membership_probability"] = profiles.weighted_membership / denominator
    profiles["mean_held_out_state_effect"] = profiles.weighted_effect / denominator
    profiles["member_coverage"] = profiles.available_members / profiles.member_count
    profiles["selected_weight_coverage"] = profiles.available_score_weight / profiles.total_score_weight.where(
        profiles.total_score_weight.gt(0))
    profiles["availability"] = profiles.available_score_weight.gt(0)
    profiles["availability_reason"] = np.where(
        profiles.availability, None, "NO_MEMBERS_WITH_AVAILABLE_JOINT_STATE")
    effect = profiles.mean_held_out_state_effect
    profiles["descriptive_effect_direction"] = np.select(
        [effect.gt(0), effect.lt(0), effect.eq(0)], ["positive", "negative", "neutral"],
        default=None)
    profiles["summary_semantics"] = "SELECTED_SCORE_WEIGHTED_MEMBER_DESCRIPTIVE_MEAN_NOT_JOINT_GENESET_EFFECT"
    profiles["p_value_available"] = False
    profiles = profiles.drop(columns=["weighted_membership", "weighted_effect"])
    data = data.drop(columns=["available_score_weight", "weighted_membership", "weighted_effect"])
    coverage_columns = ["cancer_id", "pathway_id", "association_direction",
        "geneset_available", "unavailable_reason", "exported_members"]
    if not set(coverage_columns).issubset(coverage):
        raise ValueError("Selected GeneSet coverage schema is incomplete")
    selected_coverage = coverage[coverage_columns].copy().rename(columns={
        "association_direction": "pathway_association_direction",
        "unavailable_reason": "geneset_unavailable_reason"}).merge(states, how="cross")
    selected_coverage = selected_coverage.merge(
        profiles[["geneset_id", *profile_keys[1:], "available_members", "member_coverage",
                  "selected_weight_coverage", "availability", "availability_reason"]],
        on=profile_keys[1:], how="left", validate="one_to_one")
    selected_coverage.loc[~selected_coverage.geneset_available, "availability_reason"] = \
        "SELECTED_GENESET_UNAVAILABLE"
    selected_coverage["availability"] = selected_coverage.availability.eq(True)
    selected_coverage["available_members"] = selected_coverage.available_members.fillna(0).astype(int)
    return {"selected_member_states": data,
        "selected_geneset_state_profiles": profiles,
        "selected_state_coverage": selected_coverage}


def _joint_state_sets(evaluation_roots, destination):
    """Run the imported R62 OOF set producer and adapt its State rows for the web."""
    from cla_engine.postprocess import make_sets
    destination = Path(destination)
    make_sets(evaluation_roots, destination)
    relation = pd.read_csv(destination / "RELATION_CONSENSUS.tsv.gz", sep="\t")
    cancer = pd.read_csv(destination / "CANCER_GENE_SETS_EXPLORATORY.tsv", sep="\t")
    cross = pd.read_csv(destination / "CROSS_CANCER_CANDIDATES_NOT_FINAL.tsv", sep="\t")
    subtypes = pd.read_csv(destination / "TARGET_CONTEXT_SUBTYPES_EXPLORATORY.tsv", sep="\t")
    relation = relation.loc[relation.task.astype(str).eq("state")].copy()
    cancer = cancer.loc[cancer.task.astype(str).eq("state")].copy()
    cross = cross.loc[cross.task.astype(str).eq("state")].copy()
    subtypes = subtypes.loc[subtypes.task.astype(str).eq("state")].copy()
    catalogs, members = [], []
    for (cancer_id, state_id), group in cancer.groupby(
            ["cancer_id", "target_id"], observed=True, sort=True):
        group = group.sort_values(["set_rank", "lncrna_id"], kind="stable")
        gene_set_id = f"JOINT_STATE::CANCER::{cancer_id}::{state_id}"
        signs = np.sign(pd.to_numeric(group.effect, errors="coerce").dropna())
        direction = "positive" if len(signs) and (signs >= 0).all() else (
            "negative" if len(signs) and (signs <= 0).all() else "mixed")
        catalogs.append({"gene_set_id": gene_set_id, "gene_set_name": gene_set_id,
            "gene_set_type": "joint_state_oof_descriptive", "scope": "CANCER_SPECIFIC",
            "cancer_id": str(cancer_id), "state_id": str(state_id), "direction": direction,
            "member_count": len(group),
            "selection_rule": "R62_DESCRIPTIVE_OOF_CONSENSUS_TOP50"})
        for row in group.itertuples(index=False):
            members.append({"gene_set_id": gene_set_id, "member_rank": int(row.set_rank),
                "member_id": str(row.lncrna_id),
                "member_gene_id": re.sub(r"^LNC:", "", str(row.lncrna_id), flags=re.I),
                "membership_score": float(_sigmoid([row.mean_model_logit])[0]),
                "state_effect": float(row.effect), "cancer_count": 1,
                "direction_consistency": float(row.direction_consistency),
                "source_cancer_id": str(cancer_id)})
    catalog = pd.DataFrame(catalogs, columns=CATALOG_COLUMNS)
    member = pd.DataFrame(members, columns=MEMBER_COLUMNS)
    lines = []
    for row in catalog.itertuples(index=False):
        ids = member.loc[member.gene_set_id.eq(row.gene_set_id)].sort_values(
            "member_rank").member_gene_id.tolist()
        values = [row.gene_set_id, "R62_DESCRIPTIVE_OOF_SETS_NOT_VALIDATED_RELEASE", *ids]
        if any(any(char in str(value) for char in "\t\r\n") for value in values):
            raise ValueError("Unsafe joint State GMT identity")
        lines.append("\t".join(map(str, values)) + "\n")
    tables = {"state_relation_consensus": relation,
        "state_cross_cancer_candidates_not_final": cross,
        "state_target_context_subtypes": subtypes,
        "state_geneset_catalog": catalog, "state_geneset_members": member}
    return tables, "".join(lines)


def _tables(folds, selected_path, engine_sets_output):
    parts, lineage, configs = [], [], []
    for item in folds:
        frame, record, config = _validate_fold(item)
        parts.append(frame); lineage.append(record); configs.append(config)
    oof = pd.concat(parts, ignore_index=True)
    aggregate = _aggregate_state_oof(oof)
    tables = {"state_oof_predictions": oof, "state_release": aggregate,
        **_selected_state_tables(selected_path, aggregate)}
    sets, gmt = _joint_state_sets(
        [item["evaluation_receipt"].parent for item in folds], engine_sets_output)
    tables.update(sets)
    for frame in tables.values():
        frame["analysis_version"] = ANALYSIS_VERSION
        frame["model_family"] = "cc_hhgt_shared_state_head"
        frame["state_gradients_into_core"] = True
        frame["changes_primary_ranking"] = False
    return tables, gmt, lineage, configs


def materialize_current_state_context(manifest_path, output):
    if socket.gethostname() != "149":
        raise RuntimeError("Joint State context must execute on target host 149")
    source, manifest, selected_path, folds = _manifest(manifest_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False,
        "stage": "CURRENT_JOINT_STATE_CONTEXT", "input_manifest_sha256": sha256(source)})
    try:
        # The gate and R62 set producer independently reverify the same signed
        # evaluations; neither path fits or changes a model.
        validated = [_validate_fold(item) for item in folds]
        configs = [item[2] for item in validated]
        from cla_engine.evaluate import gate
        gate_root = output / "formal_gate_replay"
        gate([item["evaluation_receipt"].parent for item in folds], configs[0], gate_root)
        gate_record = _read(gate_root / "GLOBAL_GATE.json")
        if gate_record.get("state") != "PREDICTION_GATE_PASS_NOT_DEPLOYED" or gate_record.get("tasks", {}).get("state", {}).get("pass") is not True:
            raise ValueError("Joint State formal prediction gate has not passed")
        tables, gmt, lineage, _ = _tables(
            folds, selected_path,
            output / "joint_oof_sets_replay")
        outputs = {}
        for name, frame in tables.items():
            path = output / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            outputs[name] = {**_record(path), "rows": len(frame)}
        gmt_path = output / "state_genesets.gmt"
        gmt_path.write_text(gmt, encoding="utf-8")
        outputs["gmt"] = {**_record(gmt_path), "rows": len(gmt.splitlines())}
        binding = {"format": FORMAT, "status": "PASS", "target_host": "149",
            "training_started": False, "input_manifest": _record(source),
            "selected_analysis": manifest["selected_analysis"], "release_id": manifest["release_id"],
            "analysis_version": ANALYSIS_VERSION, "model_family": "cc_hhgt_shared_state_head",
            "initialization_policy": "JOINT_SHARED_ENCODER_FROM_SCRATCH",
            "task_weights": {"pathway": 1.0, "state": 0.5},
            "state_gradients_into_core": True, "outputs": outputs,
            "fold_lineage": lineage, "formal_gate_replay": _record(gate_root / "GLOBAL_GATE.json"),
            "joint_set_status": "DESCRIPTIVE_OOF_SETS_NOT_VALIDATED_RELEASE",
            "gene_set_effect_inference_performed": False,
            "context_changes_ranking": False, "production_deployed": False}
        binding_path = output / "CURRENT_STATE_CONTEXT_BINDING.json"
        _write(binding_path, binding)
        audit = audit_current_state_context(binding_path, expected_sha=sha256(binding_path))
        audit_path = output / "RECOMPUTATION_AUDIT.json"; _write(audit_path, audit)
        result = {"status": "PASS", "target_host": "149", "training_started": False,
            "binding_path": str(binding_path), "binding_sha256": sha256(binding_path),
            "audit_sha256": sha256(audit_path)}
        _write(output / "SUCCESS.json", result)
        return result
    except Exception as error:
        _write(output / "RUN_FAILED.json", {"target_host": "149", "training_started": False,
            "error": str(error)})
        raise


def _load(binding_path, expected_sha):
    path = verified_path({"path": str(binding_path), "sha256": expected_sha}, Path.cwd())
    binding = _read(path)
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS"
            or binding.get("target_host") != "149" or binding.get("training_started") is not False
            or binding.get("analysis_version") != ANALYSIS_VERSION
            or binding.get("model_family") != "cc_hhgt_shared_state_head"
            or binding.get("initialization_policy") != "JOINT_SHARED_ENCODER_FROM_SCRATCH"
            or binding.get("task_weights") != {"pathway": 1.0, "state": 0.5}
            or binding.get("state_gradients_into_core") is not True
            or binding.get("joint_set_status") != "DESCRIPTIVE_OOF_SETS_NOT_VALIDATED_RELEASE"
            or binding.get("gene_set_effect_inference_performed") is not False
            or binding.get("context_changes_ranking") is not False
            or binding.get("production_deployed") is not False):
        raise ValueError("Invalid current joint State context binding")
    outputs = {name: verified_path(item, path.parent) for name, item in binding.get("outputs", {}).items()}
    if set(outputs) != OUTPUT_NAMES:
        raise ValueError("Joint State output inventory differs")
    gate = _read(verified_path(binding["formal_gate_replay"], path.parent))
    if gate.get("state") != "PREDICTION_GATE_PASS_NOT_DEPLOYED" or gate.get("tasks", {}).get("state", {}).get("pass") is not True:
        raise ValueError("Joint State context lacks a passing formal gate")
    return path, binding, outputs


def audit_current_state_context(binding_path, *, expected_sha):
    path, binding, outputs = _load(binding_path, expected_sha)
    manifest_path = verified_path(binding["input_manifest"], path.parent)
    _, manifest, selected_path, folds = _manifest(manifest_path)
    with tempfile.TemporaryDirectory(prefix="joint-state-sets-audit-") as temporary:
        tables, gmt, lineage, _ = _tables(folds, selected_path, Path(temporary) / "sets")
    if binding["selected_analysis"] != manifest["selected_analysis"] or binding["fold_lineage"] != lineage:
        raise ValueError("Joint State input identity changed")
    for name, expected in tables.items():
        observed = pd.read_parquet(outputs[name])
        if len(observed) != binding["outputs"][name]["rows"]:
            raise ValueError("Joint State output row count changed")
        try:
            assert_same_table(observed, expected)
        except AssertionError as error:
            raise ValueError("Joint State source replay differs: " + name) from error
    if outputs["gmt"].read_text(encoding="utf-8") != gmt:
        raise ValueError("Joint State GMT source replay differs")
    return {"status": "PASS", "binding_sha256": expected_sha,
        "joint_completion_reverified": True, "outer_oof_rows_recomputed": True,
        "joint_oof_sets_recomputed": True, "selected_member_profiles_recomputed": True,
        "state_gradients_into_core": True, "training_started": False}


class CurrentStateContextQuery:
    def __init__(self, binding_path, *, expected_sha, selected_analysis_sha):
        path, self.binding, self.paths = _load(binding_path, expected_sha)
        if self.binding["selected_analysis"]["sha256"] != selected_analysis_sha:
            raise ValueError("State query requested a different selected analysis")
        success = _read(path.parent / "SUCCESS.json")
        audit_path = path.parent / "RECOMPUTATION_AUDIT.json"
        audit = _read(audit_path)
        if (success.get("binding_sha256") != expected_sha
                or success.get("audit_sha256") != sha256(audit_path)
                or audit.get("status") != "PASS" or audit.get("binding_sha256") != expected_sha):
            raise ValueError("Joint State context lacks completed source replay")
        self.sha = expected_sha

    def download(self, name):
        if name not in self.paths:
            raise ValueError("Unknown current State download")
        return {**self.binding["outputs"][name], "binding_sha256": self.sha,
            "selected_analysis_sha256": self.binding["selected_analysis"]["sha256"]}

    def query(self, *, table="state_release", filters=None, limit=100, offset=0):
        if table == "gmt" or table not in self.paths or type(limit) is not int or type(offset) is not int \
                or not 1 <= limit <= 1000 or not 0 <= offset <= 1_000_000:
            raise ValueError("Invalid State table or pagination")
        filters = filters or {}
        allowed = {"patient_fold_id", "cancer_id", "lncrna_id", "state_id",
                   "pathway_id", "geneset_id", "gene_set_id", "direction", "scope",
                   "pathway_association_direction", "state_association_direction",
                   "association_direction", "predicted_direction", "availability"}
        if not set(filters).issubset(allowed) or not all(isinstance(v, (str, bool, int)) for v in filters.values()):
            raise ValueError("State filters require exact typed identities")
        connection = duckdb.connect()
        try:
            columns = [row[0] for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(self.paths[table])]).fetchall()]
            if not set(filters).issubset(columns):
                raise ValueError("State filter is not a column of this table")
            where = " AND ".join('"' + key + '"=?' for key in filters) or "true"
            order = ",".join('"' + key + '"' for key in ["patient_fold_id", "cancer_id",
                "pathway_id", "geneset_id", "gene_set_id", "state_id", "member_rank",
                "lncrna_id"] if key in columns) or "1"
            data = connection.execute(
                f"SELECT * FROM read_parquet(?) WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [str(self.paths[table]), *filters.values(), limit, offset]).df()
        finally:
            connection.close()
        return {"binding_sha256": self.sha,
            "selected_analysis_sha256": self.binding["selected_analysis"]["sha256"],
            "rows": data.astype(object).where(data.notna(), None).to_dict("records")}


__all__ = ["ANALYSIS_VERSION", "CurrentStateContextQuery", "FORMAT", "INPUT_FORMAT",
           "_adapt_engine_state_oof", "_aggregate_state_oof",
           "audit_current_state_context", "materialize_current_state_context"]
