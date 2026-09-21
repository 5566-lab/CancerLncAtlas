"""V3.2 experiment-to-Evidence anti-double-counting bridge.

The experiment materialisation is not a sixth model head.  Its exact events
are already members of the fresh Evidence Transformer event bags.  This
module proves that identity, preserves a downloadable experiment-native view,
and measures source attribution by removing those events at inference from
the *same held-out-fold checkpoint*.  It never retrains a head, changes a
fold, or writes a Primary/Discovery score.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from .evidence_training import (
    ANALYSIS_VERSION,
    BagExample,
    _batch_tensors,
    build_bag_examples,
    build_fresh_private_eventset_head,
    load_core_feature_bundle,
    model_parameter_sha256,
)


MODULE_ID = "experiment_perturbation_evidence_bridge"
BRIDGE_FORMAT = "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_BINDING_V1"
ABSORPTION_STATUS = "ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER"
CONFIDENCE_ROUTE = "EXISTING_EVIDENCE_TRANSFORMER_ONLY"
ABLATION_METHOD = "SAME_FOLD_CHECKPOINT_INFERENCE_EVENT_SOURCE_REMOVAL_NO_RETRAIN"
EXPERIMENT_BINDING_SHA256 = (
    "ec85aff0debb2d374a00efcb009b0b65c7606598fae4e47fb57338f317bcbb2f"
)
EVIDENCE_BINDING_SHA256 = (
    "444b5d779615ff94c93890d56ab8e807fd1b9ab1592a2bba9e610a0e44b06748"
)
FUSION_BINDING_SHA256 = (
    "0d2b5a34d0464016438725ef1e2fcea1ff04db611c9a03e571b01f87b780b059"
)
FUSION_POST_AUDIT_BINDING_SHA256 = (
    "9bcedbdc186f044bb70eef96feb8d1ab04fd1b5382b0cfeeaa50758d8664c3cd"
)
EXPERIMENT_EXACT_SHA256 = (
    "3b1d0e4b953a1d16db378405e17ca4f6a9a0ccb0c77238972b524de203e5afde"
)
EVENT_LINEAGE_SHA256 = (
    "8fc3c592678b5ef279a6b06a27ccbe43ffc996893879477e690955fae9d51528"
)
PHYSICAL_FACT_SHA256 = (
    "d3111e8cd40ee11a4b1f5a0a89b6705c17dfccf171bc5f7257bc033028082700"
)
EXACT_KEYS = ["cancer_id", "lncrna_id", "pathway_id"]
ARTIFACT_FILENAMES = {
    "lineage_bridge": "experiment_evidence_lineage_bridge.parquet",
    "exact_query": "experiment_exact_query.parquet",
    "source_removal_ablation": "source_removal_ablation.parquet",
    "source_removal_metrics": "source_removal_metrics.parquet",
}


class ExperimentEvidenceBridgeError(RuntimeError):
    """Raised when a bridge, lineage, checkpoint, or no-double-count invariant fails."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path, expected_sha256: str, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ExperimentEvidenceBridgeError(f"{label} is missing or unsafe: {source}")
    observed = file_sha256(source)
    if observed != expected_sha256:
        raise ExperimentEvidenceBridgeError(
            f"{label} SHA-256 mismatch: {observed} != {expected_sha256}"
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentEvidenceBridgeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ExperimentEvidenceBridgeError(f"{label} must be a JSON object")
    return source, value


def _declared_file(
    declaration: Mapping[str, Any], label: str, *, expected_sha256: str | None = None
) -> Path:
    path = Path(str(declaration.get("path", ""))).resolve()
    if not path.is_file() or path.is_symlink():
        raise ExperimentEvidenceBridgeError(f"{label} is missing or unsafe: {path}")
    declared = str(declaration.get("sha256", ""))
    if expected_sha256 is not None and declared != expected_sha256:
        raise ExperimentEvidenceBridgeError(
            f"{label} declared SHA mismatch: {declared} != {expected_sha256}"
        )
    observed = file_sha256(path)
    if observed != declared:
        raise ExperimentEvidenceBridgeError(
            f"{label} content drift: {observed} != {declared}"
        )
    return path


def _sql_path(path: str | Path) -> str:
    return "'" + Path(path).resolve().as_posix().replace("'", "''") + "'"


def _artifact_metadata(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }


def _canonical_lnc(value: Any) -> str:
    text = re.sub(r"^(?:LNC|LNCRNA|GENE):", "", str(value).strip(), flags=re.I)
    return "LNC:" + re.sub(r"\.\d+$", "", text.upper())


def build_absorption_bridge(
    experiment_exact: pd.DataFrame, evidence_lineage: pd.DataFrame
) -> pd.DataFrame:
    """Join experiment-native events to their identical Evidence lineage rows."""

    required_experiment = {
        "perturbation_event_id",
        "source_event_id",
        "source_record_id",
        "source_row_sha256",
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "evidence_pathway_id",
        "partner_id",
        "mapping_route",
        "assay_family",
        "assay_detail",
        "assay_detail_available",
        "assay_detail_unavailable_reason",
        "perturbation_methods",
        "perturbation_method_available",
        "readout_assays",
        "readout_assay_available",
        "assay_detail_match_route",
        "assay_detail_evidence_strength",
        "assay_detail_manual_review_required",
        "direction_class",
        "pmid",
        "cell_line",
        "tissue",
        "family_broadcast_used",
    }
    required_evidence = {
        "event_id",
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "partner_id",
        "route_type",
        "source_record_id",
        "source_database",
        "source_dataset",
        "experiment_type",
        "relation_type",
        "is_physical",
        "physical_fact_id",
        "leakage_fold",
        "family_broadcast_used",
    }
    if missing := sorted(required_experiment - set(experiment_exact.columns)):
        raise ExperimentEvidenceBridgeError(f"Experiment exact table lacks columns: {missing}")
    if missing := sorted(required_evidence - set(evidence_lineage.columns)):
        raise ExperimentEvidenceBridgeError(f"Evidence lineage lacks columns: {missing}")
    if experiment_exact.perturbation_event_id.duplicated().any():
        raise ExperimentEvidenceBridgeError("Experiment materialised event IDs are not unique")
    evidence = evidence_lineage.rename(
        columns={
            "event_id": "evidence_event_id",
            "cancer_id": "evidence_cancer_id",
            "lncrna_id": "evidence_lncrna_id_join",
            "pathway_id": "evidence_pathway_id_join",
            "partner_id": "evidence_partner_id",
            "source_record_id": "evidence_source_record_id",
            "family_broadcast_used": "evidence_family_broadcast_used",
        }
    )
    merged = experiment_exact.merge(
        evidence,
        left_on="perturbation_event_id",
        right_on="evidence_event_id",
        how="left",
        validate="one_to_one",
        suffixes=("_experiment", "_evidence"),
    )
    if merged.evidence_event_id.isna().any():
        raise ExperimentEvidenceBridgeError(
            f"{int(merged.evidence_event_id.isna().sum())} experiment events are absent from Evidence lineage"
        )
    signature_mismatch = (
        merged.cancer_id.astype(str).ne(merged.evidence_cancer_id.astype(str))
        | merged.lncrna_id.map(_canonical_lnc).ne(
            merged.evidence_lncrna_id_join.map(_canonical_lnc)
        )
        | merged.evidence_pathway_id.astype(str).ne(
            merged.evidence_pathway_id_join.astype(str)
        )
        | merged.partner_id.astype(str).ne(merged.evidence_partner_id.astype(str))
        | merged.source_record_id.astype(str).ne(
            merged.evidence_source_record_id.astype(str)
        )
        | merged.mapping_route.astype(str).ne(merged.route_type.astype(str))
    )
    if signature_mismatch.any():
        raise ExperimentEvidenceBridgeError(
            f"{int(signature_mismatch.sum())} event identities/signatures disagree"
        )
    if (
        merged.family_broadcast_used.fillna(True).astype(bool).any()
        or merged.evidence_family_broadcast_used.fillna(True).astype(bool).any()
    ):
        raise ExperimentEvidenceBridgeError("Family broadcast entered the bridge")
    output = pd.DataFrame(
        {
            "experiment_native_event_id": merged.perturbation_event_id.astype(str),
            "experiment_source_event_id": merged.source_event_id_experiment.astype(str)
            if "source_event_id_experiment" in merged
            else merged.source_event_id.astype(str),
            "experiment_source_record_id": merged.source_record_id.astype(str),
            "experiment_source_row_sha256": merged.source_row_sha256.astype(str),
            "evidence_event_id": merged.evidence_event_id.astype(str),
            "evidence_source_record_id": merged.evidence_source_record_id.astype(str),
            "cancer_id": merged.cancer_id.astype(str),
            "lncrna_id": merged.lncrna_id.map(_canonical_lnc),
            "pathway_id": merged.pathway_id.astype(str),
            "evidence_pathway_id": merged.evidence_pathway_id.astype(str),
            "partner_id": merged.partner_id.astype(str),
            "mapping_route": merged.mapping_route.astype(str),
            "source_database": merged.source_database_evidence.astype(str)
            if "source_database_evidence" in merged
            else merged.source_database.astype(str),
            "source_dataset": merged.source_dataset_evidence.astype(str)
            if "source_dataset_evidence" in merged
            else merged.source_dataset.astype(str),
            "assay_family": merged.assay_family.astype(str),
            "assay_detail": merged.assay_detail,
            "assay_detail_available": merged.assay_detail_available.astype(bool),
            "assay_detail_unavailable_reason": merged.assay_detail_unavailable_reason,
            "perturbation_methods": merged.perturbation_methods,
            "perturbation_method_available": merged.perturbation_method_available.astype(bool),
            "readout_assays": merged.readout_assays,
            "readout_assay_available": merged.readout_assay_available.astype(bool),
            "assay_detail_match_route": merged.assay_detail_match_route,
            "assay_detail_evidence_strength": merged.assay_detail_evidence_strength,
            "assay_detail_manual_review_required": (
                merged.assay_detail_manual_review_required.astype(bool)
            ),
            "experiment_type": merged.experiment_type.astype(str),
            "relation_type": merged.relation_type_evidence.astype(str)
            if "relation_type_evidence" in merged
            else merged.relation_type.astype(str),
            "direction_class": merged.direction_class,
            "pmid": merged.pmid_experiment
            if "pmid_experiment" in merged
            else merged.pmid,
            "cell_line": merged.cell_line_experiment
            if "cell_line_experiment" in merged
            else merged.cell_line,
            "tissue": merged.tissue_experiment
            if "tissue_experiment" in merged
            else merged.tissue,
            "is_physical": merged.is_physical.fillna(False).astype(bool),
            "physical_fact_id": merged.physical_fact_id,
            "evidence_leakage_fold": pd.to_numeric(
                merged.leakage_fold, errors="raise"
            ).astype("int8"),
            "absorption_status": ABSORPTION_STATUS,
            "confidence_route": CONFIDENCE_ROUTE,
            "separate_fusion_forbidden": True,
            "used_for_separate_fusion": False,
            "family_broadcast_used": False,
            "changes_primary_ranking": False,
            "changes_discovery_ranking": False,
        }
    )
    return output.sort_values(
        EXACT_KEYS + ["experiment_native_event_id"], kind="stable"
    ).reset_index(drop=True)


def deterministic_checkpoint_predictions(
    model: Any,
    examples: Sequence[BagExample],
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> pd.DataFrame:
    """Deterministic inference used for paired source-removal attribution."""

    import torch

    columns = EXACT_KEYS + [
        "deterministic_probability",
        "deterministic_direction",
        "selected_event_count",
        "evidence_leakage_fold",
    ]
    if not examples:
        return pd.DataFrame(columns=columns)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    rows: list[dict[str, Any]] = []
    names = np.asarray(["negative", "neutral", "positive"], dtype=object)
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            subset = list(examples[start : start + batch_size])
            batch = _batch_tensors(subset, list(range(len(subset))), torch_device)
            result = model(batch["events"], batch["mask"], batch["core"])
            probability = torch.sigmoid(result["confidence_logit"]).cpu().numpy()
            direction = torch.argmax(result["direction_logits"], dim=-1).cpu().numpy()
            for example, value, direction_index in zip(
                subset, probability, direction, strict=True
            ):
                rows.append(
                    {
                        **dict(zip(EXACT_KEYS, example.key, strict=True)),
                        "deterministic_probability": float(value),
                        "deterministic_direction": str(names[int(direction_index)]),
                        "selected_event_count": int(example.event_count),
                        "evidence_leakage_fold": int(example.leakage_fold),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def compute_ablation_metrics(ablation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate private target metrics without returning pair-level targets."""

    required = {
        "full_probability",
        "source_removed_probability",
        "fusion_target_private",
        "evidence_leakage_fold",
    }
    if missing := sorted(required - set(ablation.columns)):
        raise ExperimentEvidenceBridgeError(f"Ablation frame lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    eps = 1e-7
    groups: Iterable[tuple[Any, pd.DataFrame]] = [
        ("ALL_PAIR_BLOCKED_OOF", ablation)
    ] + list(ablation.groupby("evidence_leakage_fold", sort=True))
    for group_id, frame in groups:
        target = pd.to_numeric(frame.fusion_target_private, errors="raise").to_numpy(float)
        full = np.clip(
            pd.to_numeric(frame.full_probability, errors="raise").to_numpy(float), eps, 1 - eps
        )
        blocked = np.clip(
            pd.to_numeric(frame.source_removed_probability, errors="raise").to_numpy(float),
            eps,
            1 - eps,
        )
        full_logloss = float(np.mean(-(target * np.log(full) + (1 - target) * np.log(1 - full))))
        blocked_logloss = float(
            np.mean(-(target * np.log(blocked) + (1 - target) * np.log(1 - blocked)))
        )
        full_brier = float(np.mean((full - target) ** 2))
        blocked_brier = float(np.mean((blocked - target) ** 2))
        rows.append(
            {
                "evaluation_group": str(group_id),
                "evidence_leakage_fold": (
                    pd.NA if group_id == "ALL_PAIR_BLOCKED_OOF" else int(group_id)
                ),
                "evaluated_exact_keys": int(len(frame)),
                "full_logloss": full_logloss,
                "source_removed_logloss": blocked_logloss,
                "source_removal_minus_full_logloss": blocked_logloss - full_logloss,
                "full_brier": full_brier,
                "source_removed_brier": blocked_brier,
                "source_removal_minus_full_brier": blocked_brier - full_brier,
                "mean_probability_delta_full_minus_removed": float(
                    np.mean(full - blocked)
                ),
                "mean_absolute_probability_delta": float(np.mean(np.abs(full - blocked))),
                "same_fold_checkpoint": True,
                "retrained": False,
                "fold_switched": False,
                "ablation_method": ABLATION_METHOD,
            }
        )
    return pd.DataFrame(rows)


def _load_same_fold_model(
    checkpoint_path: Path,
    declaration: Mapping[str, Any],
    patient_fold: int,
    core_dim: int,
):
    import torch

    observed_checkpoint_sha = file_sha256(checkpoint_path)
    if observed_checkpoint_sha != declaration.get("sha256"):
        raise ExperimentEvidenceBridgeError(
            f"Evidence checkpoint {patient_fold} hash drift: {observed_checkpoint_sha}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(payload.get("patient_fold", -1)) != patient_fold:
        raise ExperimentEvidenceBridgeError(
            f"Checkpoint-fold mismatch: requested {patient_fold}, payload {payload.get('patient_fold')}"
        )
    if payload.get("initialized_from_checkpoint") is not False:
        raise ExperimentEvidenceBridgeError("Historical initialization entered Evidence checkpoint")
    state = payload.get("private_model_state")
    if not isinstance(state, dict):
        raise ExperimentEvidenceBridgeError("Evidence checkpoint lacks private model state")
    event_dim = int(state["event_projection.0.weight"].shape[1])
    hidden_dim = int(state["event_projection.0.weight"].shape[0])
    state_core_dim = int(state["core_query.0.weight"].shape[1])
    if state_core_dim != core_dim:
        raise ExperimentEvidenceBridgeError(
            f"Core feature dimension mismatch: {state_core_dim} != {core_dim}"
        )
    model = build_fresh_private_eventset_head(
        event_feature_dim=event_dim,
        core_feature_dim=core_dim,
        hidden_dim=hidden_dim,
        dropout=0.20,
        seed=int(payload["seed"]),
    )
    model.load_state_dict(state, strict=True)
    observed_parameter_sha = model_parameter_sha256(model)
    expected_parameter_sha = str(payload.get("final_parameter_sha256", ""))
    if observed_parameter_sha != expected_parameter_sha:
        raise ExperimentEvidenceBridgeError(
            f"Checkpoint {patient_fold} parameter hash mismatch"
        )
    if expected_parameter_sha != declaration.get("final_parameter_sha256"):
        raise ExperimentEvidenceBridgeError(
            f"Checkpoint {patient_fold} binding parameter SHA mismatch"
        )
    return model, event_dim, observed_parameter_sha, payload


def _top_events_for_fold(
    connection: duckdb.DuckDBPyConnection,
    event_lineage_path: Path,
    fold: int,
    *,
    remove_experiment_source: bool,
) -> pd.DataFrame:
    anti = (
        "AND NOT EXISTS (SELECT 1 FROM experiment_event_ids x WHERE x.event_id=e.event_id)"
        if remove_experiment_source
        else ""
    )
    return connection.execute(
        f"""
        WITH ranked AS (
          SELECT e.*,
                 row_number() OVER (
                   PARTITION BY e.cancer_id,e.lncrna_id,e.pathway_id
                   ORDER BY e.event_id
                 ) AS event_order
          FROM read_parquet({_sql_path(event_lineage_path)}) e
          INNER JOIN affected_keys k
             ON e.cancer_id=k.cancer_id
            AND e.lncrna_id=k.evidence_lncrna_id
            AND e.pathway_id=k.evidence_pathway_id
          WHERE e.leakage_fold=? {anti}
        )
        SELECT * EXCLUDE(event_order) FROM ranked WHERE event_order<=64
        ORDER BY cancer_id,lncrna_id,pathway_id,event_id
        """,
        [fold],
    ).fetchdf()


def _counts_for_affected_keys(
    connection: duckdb.DuckDBPyConnection, event_lineage_path: Path
) -> pd.DataFrame:
    return connection.execute(
        f"""
        WITH ranked AS (
          SELECT e.cancer_id,e.lncrna_id,e.pathway_id,e.event_id,e.leakage_fold,
                 (x.event_id IS NOT NULL) AS is_experiment_event,
                 row_number() OVER (
                   PARTITION BY e.cancer_id,e.lncrna_id,e.pathway_id ORDER BY e.event_id
                 ) AS full_order,
                 CASE WHEN x.event_id IS NULL THEN row_number() OVER (
                   PARTITION BY e.cancer_id,e.lncrna_id,e.pathway_id,(x.event_id IS NULL)
                   ORDER BY e.event_id
                 ) END AS nonexperiment_order
          FROM read_parquet({_sql_path(event_lineage_path)}) e
          INNER JOIN affected_keys k
             ON e.cancer_id=k.cancer_id
            AND e.lncrna_id=k.evidence_lncrna_id
            AND e.pathway_id=k.evidence_pathway_id
          LEFT JOIN experiment_event_ids x ON e.event_id=x.event_id
        )
        SELECT k.cancer_id,k.lncrna_id,k.pathway_id,
               min(leakage_fold)::TINYINT AS evidence_leakage_fold,
               count(*)::BIGINT AS all_evidence_event_count,
               count_if(is_experiment_event)::BIGINT AS experiment_event_count,
               count_if(NOT is_experiment_event)::BIGINT AS source_removed_event_count,
               count_if(full_order<=64)::BIGINT AS full_selected_event_count,
               count_if(full_order<=64 AND is_experiment_event)::BIGINT
                 AS experiment_selected_in_full_top64_count,
               least(count_if(NOT is_experiment_event),64)::BIGINT
                 AS source_removed_selected_event_count,
               count(DISTINCT leakage_fold)::INTEGER AS fold_count
        FROM ranked r
        INNER JOIN affected_keys k
          ON r.cancer_id=k.cancer_id
         AND r.lncrna_id=k.evidence_lncrna_id
         AND r.pathway_id=k.evidence_pathway_id
        GROUP BY k.cancer_id,k.lncrna_id,k.pathway_id
        ORDER BY cancer_id,lncrna_id,pathway_id
        """
    ).fetchdf()


def materialize_experiment_evidence_bridge(
    *,
    experiment_binding_path: str | Path,
    evidence_binding_path: str | Path,
    fusion_binding_path: str | Path,
    fusion_post_audit_binding_path: str | Path,
    core_embedding_root: str | Path,
    output_dir: str | Path,
    runner_path: str | Path | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Materialise the formal bridge and same-fold source-removal attribution."""

    experiment_binding_path, experiment_binding = _load_json(
        experiment_binding_path, EXPERIMENT_BINDING_SHA256, "Experiment binding"
    )
    evidence_binding_path, evidence_binding = _load_json(
        evidence_binding_path, EVIDENCE_BINDING_SHA256, "Evidence output binding"
    )
    fusion_binding_path, fusion_binding = _load_json(
        fusion_binding_path, FUSION_BINDING_SHA256, "Transparent fusion binding"
    )
    fusion_post_audit_binding_path, fusion_post_audit = _load_json(
        fusion_post_audit_binding_path,
        FUSION_POST_AUDIT_BINDING_SHA256,
        "Transparent fusion independent audit binding",
    )
    if experiment_binding.get("module_id") != "experiment_perturbation":
        raise ExperimentEvidenceBridgeError("Unexpected experiment binding module")
    if evidence_binding.get("five_fresh_private_heads_verified") is not True:
        raise ExperimentEvidenceBridgeError("Fresh Evidence heads are not verified")
    if evidence_binding.get("historical_checkpoints_used") is not False:
        raise ExperimentEvidenceBridgeError("Historical Evidence checkpoint use detected")
    if fusion_binding.get("primary_ranking_unchanged") is not True:
        raise ExperimentEvidenceBridgeError("Transparent fusion did not preserve Primary")
    if fusion_post_audit.get("accepted_for_api_integration") is not True:
        raise ExperimentEvidenceBridgeError("Transparent fusion independent audit is not accepted")

    output = Path(output_dir).resolve()
    if output.exists():
        raise ExperimentEvidenceBridgeError(
            f"Output directory exists; refusing lineage mixing: {output}"
        )
    core_root = Path(core_embedding_root).resolve()
    if not core_root.is_dir():
        raise ExperimentEvidenceBridgeError(f"Core embedding root is missing: {core_root}")

    exact_path = _declared_file(
        experiment_binding["artifacts"]["v32_perturbation_exact_pathway"],
        "Experiment exact-pathway artifact",
        expected_sha256=EXPERIMENT_EXACT_SHA256,
    )
    event_lineage_path = _declared_file(
        evidence_binding["artifacts"]["event_lineage"],
        "Evidence event lineage",
        expected_sha256=EVENT_LINEAGE_SHA256,
    )
    physical_path = _declared_file(
        evidence_binding["artifacts"]["physical_facts"],
        "Evidence physical facts",
        expected_sha256=PHYSICAL_FACT_SHA256,
    )
    evidence_predictions_path = _declared_file(
        evidence_binding["artifacts"]["evidence_predictions"],
        "Evidence predictions",
    )
    fusion_scores_path = _declared_file(
        fusion_binding["artifacts"]["secondary_scores"], "Transparent fusion scores"
    )
    confidence_private_path = _declared_file(
        fusion_binding["artifacts"]["confidence_oof_private"],
        "Private confidence OOF",
    )

    connection = duckdb.connect(database=":memory:")
    try:
        experiment_exact = connection.execute(
            f"SELECT * FROM read_parquet({_sql_path(exact_path)})"
        ).fetchdf()
        experiment_ids = experiment_exact[["perturbation_event_id"]].rename(
            columns={"perturbation_event_id": "event_id"}
        )
        affected_keys = experiment_exact[
            [
                "cancer_id",
                "evidence_lncrna_id",
                "evidence_pathway_id",
                "lncrna_id",
                "pathway_id",
            ]
        ].drop_duplicates()
        if affected_keys.duplicated(
            ["cancer_id", "evidence_lncrna_id", "evidence_pathway_id"]
        ).any():
            raise ExperimentEvidenceBridgeError(
                "Evidence canonical key maps to more than one formal display key"
            )
        connection.register("experiment_event_ids", experiment_ids)
        connection.register("affected_keys", affected_keys)
        evidence_overlap = connection.execute(
            f"""
            SELECT e.* FROM read_parquet({_sql_path(event_lineage_path)}) e
            INNER JOIN experiment_event_ids x ON e.event_id=x.event_id
            ORDER BY e.cancer_id,e.lncrna_id,e.pathway_id,e.event_id
            """
        ).fetchdf()
        bridge = build_absorption_bridge(experiment_exact, evidence_overlap)
        counts = _counts_for_affected_keys(connection, event_lineage_path)
        if len(counts) != 14_452 or counts.fold_count.ne(1).any():
            raise ExperimentEvidenceBridgeError(
                f"Affected-key/fold invariant failed: rows={len(counts)}, bad_folds={int(counts.fold_count.ne(1).sum())}"
            )

        fold_frames: list[pd.DataFrame] = []
        checkpoint_audit: dict[str, Any] = {}
        for fold in range(5):
            core = load_core_feature_bundle(core_root, fold)
            declaration = evidence_binding["checkpoints"][str(fold)]
            checkpoint_path = Path(str(declaration["path"])).resolve()
            model, event_dim, parameter_sha, payload = _load_same_fold_model(
                checkpoint_path, declaration, fold, core.combined_dim
            )
            full_events = _top_events_for_fold(
                connection, event_lineage_path, fold, remove_experiment_source=False
            )
            removed_events = _top_events_for_fold(
                connection, event_lineage_path, fold, remove_experiment_source=True
            )
            full_examples, full_missing = build_bag_examples(
                full_events, core, event_feature_dim=event_dim, max_events=64
            )
            removed_examples, removed_missing = build_bag_examples(
                removed_events, core, event_feature_dim=event_dim, max_events=64
            )
            if not full_missing.empty or not removed_missing.empty:
                raise ExperimentEvidenceBridgeError(
                    f"Fold {fold} has missing core identifiers in source-removal audit"
                )
            full_prediction = deterministic_checkpoint_predictions(
                model, full_examples, batch_size=batch_size
            ).rename(
                columns={
                    "deterministic_probability": "full_probability",
                    "deterministic_direction": "full_direction",
                    "selected_event_count": "full_selected_event_count_model",
                }
            )
            removed_prediction = deterministic_checkpoint_predictions(
                model, removed_examples, batch_size=batch_size
            ).rename(
                columns={
                    "deterministic_probability": "source_removed_probability",
                    "deterministic_direction": "source_removed_direction",
                    "selected_event_count": "source_removed_selected_event_count_model",
                }
            )
            merge_keys = EXACT_KEYS + ["evidence_leakage_fold"]
            paired = full_prediction.merge(
                removed_prediction,
                on=merge_keys,
                how="outer",
                validate="one_to_one",
            )
            paired = paired.rename(
                columns={
                    "lncrna_id": "evidence_lncrna_id",
                    "pathway_id": "evidence_pathway_id",
                }
            ).merge(
                affected_keys,
                on=["cancer_id", "evidence_lncrna_id", "evidence_pathway_id"],
                how="left",
                validate="one_to_one",
            )
            expected_fold_keys = counts.loc[
                counts.evidence_leakage_fold.eq(fold), EXACT_KEYS
            ]
            if len(paired) != len(expected_fold_keys) or paired.isna().any().any():
                raise ExperimentEvidenceBridgeError(
                    f"Fold {fold} full/source-removed paired inference is incomplete"
                )
            paired["lncrna_id"] = paired.lncrna_id.map(_canonical_lnc)
            paired["probability_delta_full_minus_removed"] = (
                paired.full_probability - paired.source_removed_probability
            )
            paired["absolute_probability_delta"] = paired[
                "probability_delta_full_minus_removed"
            ].abs()
            paired["same_fold_checkpoint"] = True
            paired["checkpoint_patient_fold"] = fold
            paired["checkpoint_sha256"] = str(declaration["sha256"])
            paired["checkpoint_parameter_sha256"] = parameter_sha
            paired["retrained"] = False
            paired["fold_switched"] = False
            paired["ablation_method"] = ABLATION_METHOD
            paired["absorption_status"] = ABSORPTION_STATUS
            paired["used_for_separate_fusion"] = False
            paired["changes_primary_ranking"] = False
            paired["changes_discovery_ranking"] = False
            fold_frames.append(paired)
            checkpoint_audit[str(fold)] = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": str(declaration["sha256"]),
                "checkpoint_parameter_sha256": parameter_sha,
                "checkpoint_payload_patient_fold": int(payload["patient_fold"]),
                "evaluated_exact_keys": int(len(paired)),
                "same_fold_checkpoint": True,
                "retrained": False,
                "fold_switched": False,
            }
        ablation_private = pd.concat(fold_frames, ignore_index=True)
        if len(ablation_private) != 14_452:
            raise ExperimentEvidenceBridgeError("Source-removal ablation key coverage is incomplete")

        targets = connection.execute(
            f"""
            SELECT p.cancer_id,p.lncrna_id,p.pathway_id,p.fusion_target
            FROM read_parquet({_sql_path(confidence_private_path)}) p
            INNER JOIN affected_keys k
              ON p.cancer_id=k.cancer_id
             AND p.lncrna_id='LNC:' || k.evidence_lncrna_id
             AND p.pathway_id=k.pathway_id
            """
        ).fetchdf().rename(columns={"fusion_target": "fusion_target_private"})
        ablation_private = ablation_private.merge(
            targets, on=EXACT_KEYS, how="left", validate="one_to_one"
        )
        if ablation_private.fusion_target_private.isna().any():
            raise ExperimentEvidenceBridgeError("Private evaluation target is missing")
        metrics = compute_ablation_metrics(ablation_private)
        public_ablation = ablation_private.drop(columns=["fusion_target_private"]).merge(
            counts.drop(columns=["fold_count"]),
            on=EXACT_KEYS + ["evidence_leakage_fold"],
            how="left",
            validate="one_to_one",
        )
        if public_ablation.source_removed_event_count.le(0).any():
            raise ExperimentEvidenceBridgeError(
                "At least one affected key has no non-perturbation Evidence event"
            )
        if not public_ablation.checkpoint_patient_fold.eq(
            public_ablation.evidence_leakage_fold
        ).all():
            raise ExperimentEvidenceBridgeError("Same-fold checkpoint invariant failed")

        native = connection.execute(
            f"""
            SELECT k.cancer_id,k.lncrna_id,k.pathway_id,
                   p.evidence_confidence_probability,
                   p.direction AS evidence_native_direction,
                   p.uncertainty AS evidence_native_uncertainty,
                   p.availability AS evidence_native_available,
                   p.event_count AS evidence_native_event_count,
                   p.evidence_fold
            FROM read_parquet({_sql_path(evidence_predictions_path)}) p
            INNER JOIN affected_keys k
              ON p.cancer_id=k.cancer_id AND p.lncrna_id=k.evidence_lncrna_id
             AND p.pathway_id=k.evidence_pathway_id
            """
        ).fetchdf()
        fusion = connection.execute(
            f"""
            SELECT p.cancer_id,p.lncrna_id,p.pathway_id,p.primary_probability,
                   p.fused_confidence_probability,
                   p.evidence_transformer_native_probability,
                   p.evidence_transformer_native_available,
                   p.confidence_evidence_transformer_logit_contribution,
                   p.confidence_evidence_transformer_fusion_weight,
                   p.primary_ranking_unchanged,
                   p.used_for_primary_release
            FROM read_parquet({_sql_path(fusion_scores_path)}) p
            INNER JOIN affected_keys k
              ON p.cancer_id=k.cancer_id
             AND p.lncrna_id='LNC:' || k.evidence_lncrna_id
             AND p.pathway_id=k.pathway_id
            """
        ).fetchdf()
        if len(native) != 14_452 or len(fusion) != 14_452:
            raise ExperimentEvidenceBridgeError("Evidence/fusion exact-key join is incomplete")
        native_compare = fusion.merge(native, on=EXACT_KEYS, validate="one_to_one")
        probability_mismatch = ~np.isclose(
            native_compare.evidence_transformer_native_probability.to_numpy(float),
            native_compare.evidence_confidence_probability.to_numpy(float),
            rtol=0,
            atol=1e-12,
        )
        if probability_mismatch.any():
            raise ExperimentEvidenceBridgeError(
                f"{int(probability_mismatch.sum())} transparent native probabilities drifted"
            )

        connection.register("bridge_table", bridge)
        summaries = connection.execute(
            """
            SELECT cancer_id,lncrna_id,pathway_id,
                   count(*)::BIGINT AS experiment_event_count,
                   count(DISTINCT experiment_native_event_id)::BIGINT AS experiment_native_event_count,
                   count(DISTINCT experiment_source_record_id)::BIGINT AS source_record_count,
                   count(DISTINCT partner_id)::BIGINT AS partner_count,
                   string_agg(DISTINCT partner_id,';' ORDER BY partner_id) AS partner_ids,
                   string_agg(DISTINCT assay_family,';' ORDER BY assay_family) AS assay_families,
                   count_if(assay_detail_available)::BIGINT AS assay_detail_available_event_count,
                   string_agg(DISTINCT assay_detail,' | ' ORDER BY assay_detail)
                     FILTER (WHERE assay_detail_available AND assay_detail IS NOT NULL
                             AND assay_detail<>'') AS assay_details,
                   string_agg(DISTINCT perturbation_methods,' | ' ORDER BY perturbation_methods)
                     FILTER (WHERE perturbation_method_available
                             AND perturbation_methods IS NOT NULL
                             AND perturbation_methods<>'') AS perturbation_methods,
                   string_agg(DISTINCT readout_assays,' | ' ORDER BY readout_assays)
                     FILTER (WHERE readout_assay_available AND readout_assays IS NOT NULL
                             AND readout_assays<>'') AS readout_assays,
                   string_agg(DISTINCT assay_detail_match_route,';' ORDER BY assay_detail_match_route)
                     AS assay_detail_match_routes,
                   string_agg(DISTINCT assay_detail_evidence_strength,';'
                              ORDER BY assay_detail_evidence_strength)
                     AS assay_detail_evidence_strengths,
                   count_if(assay_detail_manual_review_required)::BIGINT
                     AS assay_detail_manual_review_event_count,
                   string_agg(DISTINCT experiment_type,';' ORDER BY experiment_type) AS experiment_types,
                   string_agg(DISTINCT direction_class,';' ORDER BY direction_class)
                     FILTER (WHERE direction_class IS NOT NULL AND direction_class<>'')
                     AS direction_classes,
                   string_agg(DISTINCT pmid,';' ORDER BY pmid)
                     FILTER (WHERE pmid IS NOT NULL AND pmid<>'' AND pmid<>'unknown') AS pmids,
                   string_agg(DISTINCT cell_line,';' ORDER BY cell_line)
                     FILTER (WHERE cell_line IS NOT NULL AND cell_line<>'' AND cell_line<>'unknown')
                     AS cell_lines,
                   string_agg(DISTINCT tissue,';' ORDER BY tissue)
                     FILTER (WHERE tissue IS NOT NULL AND tissue<>'' AND tissue<>'unknown') AS tissues,
                   count_if(is_physical)::BIGINT AS physical_event_count,
                   min(evidence_leakage_fold)::TINYINT AS evidence_leakage_fold
            FROM bridge_table GROUP BY cancer_id,lncrna_id,pathway_id
            ORDER BY cancer_id,lncrna_id,pathway_id
            """
        ).fetchdf()
        exact_query = (
            summaries.merge(native, on=EXACT_KEYS, validate="one_to_one")
            .merge(fusion, on=EXACT_KEYS, validate="one_to_one")
            .merge(
                public_ablation[
                    EXACT_KEYS
                    + [
                        "full_probability",
                        "source_removed_probability",
                        "probability_delta_full_minus_removed",
                        "absolute_probability_delta",
                        "experiment_selected_in_full_top64_count",
                        "all_evidence_event_count",
                        "source_removed_event_count",
                        "same_fold_checkpoint",
                        "checkpoint_sha256",
                        "checkpoint_parameter_sha256",
                    ]
                ],
                on=EXACT_KEYS,
                validate="one_to_one",
            )
        )
        exact_query["experiment_event_available"] = True
        exact_query["experiment_event_unavailable_reason"] = ""
        exact_query["source_removal_ablation_available"] = True
        exact_query["source_removal_ablation_unavailable_reason"] = ""
        exact_query["absorption_status"] = ABSORPTION_STATUS
        exact_query["confidence_route"] = CONFIDENCE_ROUTE
        exact_query["separate_fusion_forbidden"] = True
        exact_query["used_for_separate_fusion"] = False
        exact_query["used_for_primary_release"] = False
        exact_query["changes_primary_ranking"] = False
        exact_query["changes_discovery_ranking"] = False
        exact_query["scientific_role"] = "DIAGNOSTIC_ATTRIBUTION_AND_NATIVE_FACT_QUERY"
    finally:
        connection.close()

    all_metrics = metrics.loc[metrics.evaluation_group.eq("ALL_PAIR_BLOCKED_OOF")].iloc[0]
    source_removal_status = (
        "MEASURABLE_WITHIN_EXISTING_EVIDENCE_TRANSFORMER"
        if float(all_metrics.source_removal_minus_full_logloss) >= 1.0e-4
        else "NO_INCREMENT_DIAGNOSTIC_ONLY"
    )
    exact_query["source_removal_status"] = source_removal_status
    public_ablation["source_removal_status"] = source_removal_status

    output.mkdir(parents=True, exist_ok=False)
    artifact_frames = {
        "lineage_bridge": bridge,
        "exact_query": exact_query,
        "source_removal_ablation": public_ablation,
        "source_removal_metrics": metrics,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for role, frame in artifact_frames.items():
        path = output / ARTIFACT_FILENAMES[role]
        frame.to_parquet(path, index=False, compression="zstd")
        artifacts[role] = _artifact_metadata(path, frame)

    overlap_counts = {
        "experiment_exact_event_rows": int(len(bridge)),
        "evidence_event_id_matches": int(bridge.evidence_event_id.notna().sum()),
        "full_signature_matches": int(len(bridge)),
        "unmatched_events": 0,
        "unique_exact_keys": int(bridge[EXACT_KEYS].drop_duplicates().shape[0]),
        "keys_with_evidence_native": int(exact_query.evidence_native_available.sum()),
        "physical_event_rows": int(bridge.is_physical.sum()),
        "keys_retaining_nonperturbation_events": int(
            public_ablation.source_removed_event_count.gt(0).sum()
        ),
        "keys_with_perturbation_in_full_top64": int(
            public_ablation.experiment_selected_in_full_top64_count.gt(0).sum()
        ),
    }
    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "bridge_format": BRIDGE_FORMAT,
        "status": "PASS",
        "route_determination": ABSORPTION_STATUS,
        "fully_absorbed_by_evidence_transformer": True,
        "separate_fusion_forbidden": True,
        "confidence_route": CONFIDENCE_ROUTE,
        "source_removal_status": source_removal_status,
        "source_removal_method": ABLATION_METHOD,
        "same_fold_checkpoint_all_keys": True,
        "retraining_performed": False,
        "fold_switching_performed": False,
        "overlap_counts": overlap_counts,
        "checkpoint_audit": checkpoint_audit,
        "aggregate_metrics": {
            key: (
                int(value)
                if isinstance(value, (np.integer,))
                else float(value)
                if isinstance(value, (np.floating,))
                else value
            )
            for key, value in all_metrics.to_dict().items()
            if key != "evidence_leakage_fold"
        },
        "invariants": {
            "experiment_native_event_identity_retained": True,
            "downloadable_event_level_lineage_retained": True,
            "family_broadcast_used": False,
            "independent_experiment_probability_head_created": False,
            "duplicate_fusion_created": False,
            "private_targets_written_to_public_artifacts": False,
            "changes_primary_ranking": False,
            "changes_discovery_ranking": False,
            "production_deployed": False,
        },
    }
    audit_path = output / "EXPERIMENT_EVIDENCE_ABSORPTION_AUDIT.json"
    _json_write(audit_path, audit)

    module_path = Path(__file__).resolve()
    execution_code: dict[str, Any] = {
        "bridge_module": {"path": str(module_path), "sha256": file_sha256(module_path)}
    }
    if runner_path is not None:
        runner = Path(runner_path).resolve()
        execution_code["runner"] = {"path": str(runner), "sha256": file_sha256(runner)}
    authorities = {
        "experiment_binding": {
            "path": str(experiment_binding_path),
            "sha256": EXPERIMENT_BINDING_SHA256,
        },
        "evidence_binding": {
            "path": str(evidence_binding_path),
            "sha256": EVIDENCE_BINDING_SHA256,
        },
        "transparent_fusion_binding": {
            "path": str(fusion_binding_path),
            "sha256": FUSION_BINDING_SHA256,
        },
        "transparent_fusion_independent_audit_binding": {
            "path": str(fusion_post_audit_binding_path),
            "sha256": FUSION_POST_AUDIT_BINDING_SHA256,
        },
        "experiment_exact": {"path": str(exact_path), "sha256": EXPERIMENT_EXACT_SHA256},
        "evidence_event_lineage": {
            "path": str(event_lineage_path),
            "sha256": EVENT_LINEAGE_SHA256,
        },
        "evidence_physical_facts": {
            "path": str(physical_path),
            "sha256": PHYSICAL_FACT_SHA256,
        },
        "evidence_predictions": {
            "path": str(evidence_predictions_path),
            "sha256": evidence_binding["artifacts"]["evidence_predictions"]["sha256"],
        },
        "transparent_fusion_scores": {
            "path": str(fusion_scores_path),
            "sha256": fusion_binding["artifacts"]["secondary_scores"]["sha256"],
        },
        "private_confidence_oof_target": {
            "path": str(confidence_private_path),
            "sha256": fusion_binding["artifacts"]["confidence_oof_private"]["sha256"],
            "public": False,
            "target_values_not_exported": True,
        },
    }
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "bridge_format": BRIDGE_FORMAT,
        "status": "SUCCESS_HASH_BOUND_ABSORPTION_BRIDGE",
        "release_ready": True,
        "production_deployed": False,
        "route_determination": ABSORPTION_STATUS,
        "confidence_route": CONFIDENCE_ROUTE,
        "separate_fusion_forbidden": True,
        "source_removal_status": source_removal_status,
        "scientific_role": "DIAGNOSTIC_ATTRIBUTION_AND_NATIVE_FACT_QUERY",
        "authorities": authorities,
        "checkpoints": checkpoint_audit,
        "artifacts": artifacts,
        "audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
        "execution_code": execution_code,
        "invariants": audit["invariants"],
    }
    manifest_path = output / "EXPERIMENT_PERTURBATION_BRIDGE_MANIFEST.json"
    _json_write(manifest_path, manifest)
    binding = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "binding_format": BINDING_FORMAT,
        "status": "SUCCESS_HASH_BOUND_ABSORPTION_BRIDGE",
        "release_ready": True,
        "production_deployed": False,
        "route_determination": ABSORPTION_STATUS,
        "fully_absorbed_by_evidence_transformer": True,
        "confidence_route": CONFIDENCE_ROUTE,
        "separate_fusion_forbidden": True,
        "source_removal_status": source_removal_status,
        "artifacts": artifacts,
        "audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "authorities": authorities,
        "checkpoints": checkpoint_audit,
        "execution_code": execution_code,
        "invariants": audit["invariants"],
    }
    binding_path = output / "EXPERIMENT_PERTURBATION_BRIDGE_BINDING.json"
    _json_write(binding_path, binding)
    success = {
        "status": binding["status"],
        "binding": str(binding_path),
        "binding_sha256": file_sha256(binding_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "audit": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "route_determination": ABSORPTION_STATUS,
        "separate_fusion_forbidden": True,
        "source_removal_status": source_removal_status,
        "production_deployed": False,
    }
    _json_write(output / "SUCCESS.json", success)
    return success


__all__ = [
    "ABLATION_METHOD",
    "ABSORPTION_STATUS",
    "BINDING_FORMAT",
    "CONFIDENCE_ROUTE",
    "ExperimentEvidenceBridgeError",
    "build_absorption_bridge",
    "compute_ablation_metrics",
    "deterministic_checkpoint_predictions",
    "file_sha256",
    "materialize_experiment_evidence_bridge",
]
