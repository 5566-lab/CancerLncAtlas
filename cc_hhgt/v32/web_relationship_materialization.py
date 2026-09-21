"""Fresh-V3.2-only website relationship materialization.

The legacy website tables mix display semantics with historical V2.9/V3.0
artifacts.  This module deliberately accepts only canonical V3.2 exact-pathway
scores and exact-target evidence.  It keeps relationship classes mutually
exclusive and makes the historical ``significant_*`` compatibility field
explicitly mean "non-exploratory", never hypothesis-test significance.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]
INPUT_MANIFEST_FORMAT = "CancerLncAtlas.v32.web_relationship_inputs.v1"
OUTPUT_MANIFEST_FORMAT = "CancerLncAtlas.v32.web_relationship_staging.v1"
EXPECTED_CANCERS = 33
EXPECTED_FOLDS = tuple(range(5))
TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PUBLIC_COLUMNS = {
    "held_out_proxy_label",
    "label",
    "label_class",
    "proxy_label",
    "association_proxy_label",
    "strong_association_label",
    "sample_weight",
    "fusion_target",
    "ground_truth",
    "held_out_label",
    "is_positive",
    "outcome",
    "target_label",
    "test_label",
    "y_true",
}


@dataclass(frozen=True)
class RelationshipThresholds:
    observed_core_evidence_min: float = 0.70
    model_supported_probability_min: float = 0.80
    predicted_candidate_probability_min: float = 0.90
    predicted_candidate_max_observed_evidence: float = 0.35
    max_prediction_uncertainty: float = 0.25
    required_folds: int = 5

    def validate(self) -> None:
        probabilities = [
            self.observed_core_evidence_min,
            self.model_supported_probability_min,
            self.predicted_candidate_probability_min,
            self.predicted_candidate_max_observed_evidence,
            self.max_prediction_uncertainty,
        ]
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Relationship thresholds must lie in [0, 1]")
        if self.required_folds < 2:
            raise ValueError("At least two independently trained folds are required")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _require_unique(frame: pd.DataFrame, label: str) -> None:
    duplicated = frame.duplicated(KEYS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, KEYS].head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate exact relationship keys: {examples}")


def _require_v32(frame: pd.DataFrame, label: str) -> None:
    _require_columns(frame, ["analysis_version"], label)
    versions = sorted(set(frame["analysis_version"].dropna().astype(str)))
    if versions != [ANALYSIS_VERSION]:
        raise ValueError(
            f"{label} is not exclusively fresh V3.2: observed versions={versions}"
        )
    for column in (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
        "historical_predictions_used",
        "historical_rankings_used",
    ):
        if column in frame and frame[column].fillna(False).astype(bool).any():
            raise ValueError(f"{label} declares historical reuse in {column}")


def _probability(series: pd.Series, label: str, *, allow_null: bool = False) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    if not allow_null and result.isna().any():
        raise ValueError(f"{label} contains null or non-numeric values")
    observed = result.dropna()
    if not np.isfinite(observed).all() or not observed.between(0.0, 1.0).all():
        raise ValueError(f"{label} contains values outside [0, 1]")
    return result


def classify_fresh_v32_relationships(
    primary: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    prediction_uncertainty: pd.Series | None = None,
    thresholds: RelationshipThresholds | None = None,
) -> pd.DataFrame:
    """Return a leakage-safe, exact-pathway V3.2 relationship table.

    ``prediction_uncertainty`` must be derived only from independently trained
    fold/seed predictions.  A caller may instead provide a column named
    ``prediction_uncertainty`` in ``primary``.
    """

    thresholds = thresholds or RelationshipThresholds()
    thresholds.validate()
    primary_required = [
        *KEYS,
        "pathway_family_id",
        "association_membership_probability",
        "association_direction_probability",
        "association_direction",
        "n_folds_available",
        "analysis_version",
    ]
    evidence_required = [
        *KEYS,
        "evidence_confidence_probability",
        "availability",
        "direct_target_evidence",
        "family_to_exact_broadcast",
        "analysis_version",
    ]
    _require_columns(primary, primary_required, "primary scores")
    _require_columns(evidence, evidence_required, "evidence scores")
    forbidden = sorted(FORBIDDEN_PUBLIC_COLUMNS & set(primary.columns))
    if forbidden:
        raise ValueError(f"Held-out/private target columns entered materialization: {forbidden}")
    _require_v32(primary, "primary scores")
    _require_v32(evidence, "evidence scores")
    _require_unique(primary, "primary scores")
    _require_unique(evidence, "evidence scores")
    if evidence["family_to_exact_broadcast"].fillna(False).astype(bool).any():
        raise ValueError("Family-level evidence was broadcast into exact pathways")

    result = primary.copy()
    probability = _probability(
        result["association_membership_probability"], "primary probability"
    )
    direction_probability = _probability(
        result["association_direction_probability"], "direction probability"
    )
    folds = pd.to_numeric(result["n_folds_available"], errors="coerce")
    if folds.isna().any() or folds.lt(thresholds.required_folds).any():
        raise ValueError("Primary scores do not have the required independent folds")

    if prediction_uncertainty is None:
        if "prediction_uncertainty" not in result:
            raise ValueError(
                "Fresh V3.2 website materialization requires fold/seed prediction uncertainty"
            )
        uncertainty = result["prediction_uncertainty"]
    else:
        if len(prediction_uncertainty) != len(result):
            raise ValueError("Prediction uncertainty length does not match primary scores")
        uncertainty = pd.Series(prediction_uncertainty.to_numpy(), index=result.index)
    uncertainty = _probability(uncertainty, "prediction uncertainty")
    result["prediction_uncertainty"] = uncertainty

    evidence_columns = [
        *KEYS,
        "evidence_confidence_probability",
        "availability",
        "direct_target_evidence",
        "family_to_exact_broadcast",
    ]
    for optional in ("direction", "event_count", "uncertainty", "training_run_id"):
        if optional in evidence:
            evidence_columns.append(optional)
    evidence_for_merge = evidence[evidence_columns].rename(
        columns={
            "direction": "evidence_direction",
            "uncertainty": "evidence_uncertainty",
            "training_run_id": "evidence_training_run_id",
        }
    )
    result = result.merge(
        evidence_for_merge,
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_evidence"),
    )
    if result["availability"].isna().any():
        raise ValueError("Evidence table does not cover the complete candidate universe")

    available = (
        result["availability"].astype(bool)
        & result["direct_target_evidence"].fillna(False).astype(bool)
    )
    evidence_probability = _probability(
        result["evidence_confidence_probability"],
        "evidence probability",
        allow_null=True,
    )
    if evidence_probability[available].isna().any():
        raise ValueError("Available exact evidence has null probability")
    if evidence_probability[~available].notna().any():
        raise ValueError("Unavailable exact evidence must remain null")
    observed_evidence = evidence_probability.fillna(0.0)
    stable = uncertainty.le(thresholds.max_prediction_uncertainty)

    observed = available & observed_evidence.ge(
        thresholds.observed_core_evidence_min
    )
    predicted = (
        ~observed
        & probability.ge(thresholds.predicted_candidate_probability_min)
        & observed_evidence.le(
            thresholds.predicted_candidate_max_observed_evidence
        )
        & stable
    )
    supported = (
        ~observed
        & ~predicted
        & available
        & observed_evidence.gt(0.0)
        & probability.ge(thresholds.model_supported_probability_min)
        & stable
    )
    result["relationship_class"] = np.select(
        [observed, predicted, supported],
        ["observed_core", "predicted_candidate", "model_supported"],
        default="exploratory",
    )
    result["display_probability"] = probability
    # ``observed_evidence`` is a private classification helper: filling an
    # unavailable probability with zero is convenient for threshold logic, but
    # zero is a biological statement (observed and unsupported), whereas null
    # means the evidence source was not available.  Preserve that distinction
    # in the published table.
    result["observed_evidence_probability"] = evidence_probability
    result["evidence_available"] = available
    result["pathway_target_level"] = "exact_pathway"
    result["family_to_exact_broadcast"] = False
    result["analysis_version"] = ANALYSIS_VERSION
    evidence_direction = (
        result["evidence_direction"].fillna("").astype(str).str.lower()
        if "evidence_direction" in result
        else pd.Series("", index=result.index)
    )
    model_direction = np.where(direction_probability.ge(0.5), "positive", "negative")
    result["final_direction"] = np.where(
        available & evidence_direction.isin(["positive", "negative"]),
        evidence_direction,
        model_direction,
    )
    _require_unique(result, "materialized relationships")
    return result


def build_cancer_overview(relationships: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unambiguous V3.2 counts for the website cancer overview."""

    _require_columns(
        relationships,
        [*KEYS, "relationship_class", "display_probability", "analysis_version"],
        "relationships",
    )
    _require_v32(relationships, "relationships")
    _require_unique(relationships, "relationships")
    overview = (
        relationships.groupby("cancer_id", observed=True)
        .agg(
            scored_relationship_count=("pathway_id", "size"),
            detectable_lncRNAs=("lncrna_id", "nunique"),
            scored_exact_pathways=("pathway_id", "nunique"),
            observed_core_relations=(
                "relationship_class",
                lambda values: int(pd.Series(values).eq("observed_core").sum()),
            ),
            model_supported_relations=(
                "relationship_class",
                lambda values: int(pd.Series(values).eq("model_supported").sum()),
            ),
            predicted_candidate_relations=(
                "relationship_class",
                lambda values: int(pd.Series(values).eq("predicted_candidate").sum()),
            ),
            non_exploratory_relations=(
                "relationship_class",
                lambda values: int(pd.Series(values).ne("exploratory").sum()),
            ),
            mean_primary_probability=("display_probability", "mean"),
        )
        .reset_index()
    )
    # Compatibility only: callers must display the explicit semantics field.
    overview["significant_lncRNA_pathway_relations"] = overview[
        "non_exploratory_relations"
    ]
    overview["significant_relation_semantics"] = (
        "compatibility alias for non_exploratory_relations; not statistical significance"
    )
    overview["pathway_target_level"] = "exact_pathway"
    overview["family_to_exact_broadcast"] = False
    overview["analysis_version"] = ANALYSIS_VERSION
    overview["model_version"] = "CC-HHGT_V3.2"
    return overview


class WebRelationshipMaterializationError(RuntimeError):
    """Raised when a formal input or staging-output invariant is violated."""


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_sha(value: object, label: str) -> str:
    candidate = str(value or "").lower()
    if not _SHA256.fullmatch(candidate):
        raise WebRelationshipMaterializationError(f"{label} is not a SHA-256")
    return candidate


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WebRelationshipMaterializationError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise WebRelationshipMaterializationError(f"{label} must be a JSON object")
    return payload


def _artifact_path(manifest_path: Path, declaration: Mapping[str, Any], label: str) -> Path:
    raw = declaration.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise WebRelationshipMaterializationError(f"{label} lacks a path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if candidate.is_symlink():
        raise WebRelationshipMaterializationError(f"{label} may not be a symlink")
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise WebRelationshipMaterializationError(
            f"{label} is not a regular non-symlink file: {candidate}"
        )
    if candidate.suffix.lower() not in {".parquet", ".pq"}:
        raise WebRelationshipMaterializationError(f"{label} must be Parquet")
    expected = _require_sha(declaration.get("sha256"), f"{label} declared SHA-256")
    observed = _sha256(candidate)
    if observed != expected:
        raise WebRelationshipMaterializationError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return candidate


def _false_only(payload: Mapping[str, Any], label: str) -> None:
    required_false = (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
        "historical_predictions_used",
        "historical_rankings_used",
    )
    for field in required_false:
        if payload.get(field) is not False:
            raise WebRelationshipMaterializationError(
                f"{label} must explicitly declare {field}=false"
            )


def _validate_input_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], str, dict[str, str]]:
    if path.is_symlink():
        raise WebRelationshipMaterializationError("Input manifest may not be a symlink")
    source = path.resolve()
    if not source.is_file():
        raise WebRelationshipMaterializationError(
            f"Input manifest is not a regular non-symlink file: {source}"
        )
    expected = _require_sha(expected_sha256, "Expected input manifest SHA-256")
    observed = _sha256(source)
    if observed != expected:
        raise WebRelationshipMaterializationError(
            f"Input manifest SHA-256 mismatch: {observed} != {expected}"
        )
    payload = _load_json_object(source, "input manifest")
    required = {
        "manifest_format": INPUT_MANIFEST_FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "family_to_exact_broadcast": False,
        "primary_score_status": "FINAL_SELECTED",
        "selection_status": "PASS",
        "selection_split": "validation_only",
        "test_metrics_used_for_selection": False,
        "expected_cancers": EXPECTED_CANCERS,
        "expected_cancer_ids": list(TCGA_CANCERS),
        "required_folds": list(EXPECTED_FOLDS),
    }
    for field, value in required.items():
        if payload.get(field) != value:
            raise WebRelationshipMaterializationError(
                f"Input manifest has invalid {field}: {payload.get(field)!r}"
            )
    _false_only(payload, "Input manifest")
    primary_run = str(payload.get("primary_training_run_id") or "")
    evidence_run = str(payload.get("evidence_training_run_id") or "")
    selection_sha = _require_sha(payload.get("selection_sha256"), "Selection SHA-256")
    universe_sha = _require_sha(
        payload.get("candidate_universe_sha256"), "Candidate-universe SHA-256"
    )
    if not primary_run or not evidence_run:
        raise WebRelationshipMaterializationError(
            "Input manifest lacks primary/evidence training run IDs"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise WebRelationshipMaterializationError("Input manifest lacks artifacts")
    primary = artifacts.get("primary")
    evidence = artifacts.get("exact_evidence")
    folds = artifacts.get("fold_predictions")
    if not isinstance(primary, dict) or primary.get("role") != "final_selected_primary":
        raise WebRelationshipMaterializationError("Invalid final selected primary declaration")
    if not isinstance(evidence, dict) or evidence.get("role") != "exact_target_evidence":
        raise WebRelationshipMaterializationError("Invalid exact evidence declaration")
    if not isinstance(folds, list) or len(folds) != len(EXPECTED_FOLDS):
        raise WebRelationshipMaterializationError("Exactly five fold declarations are required")
    declarations: list[tuple[str, Mapping[str, Any], str]] = [
        ("primary", primary, primary_run),
        ("exact_evidence", evidence, evidence_run),
    ]
    fold_map: dict[int, Mapping[str, Any]] = {}
    checkpoint_hashes: set[str] = set()
    partition_hashes: set[str] = set()
    for item in folds:
        if not isinstance(item, dict):
            raise WebRelationshipMaterializationError("Fold declaration must be an object")
        try:
            fold_id = int(item.get("fold_id"))
        except (TypeError, ValueError) as exc:
            raise WebRelationshipMaterializationError("Fold ID must be an integer") from exc
        if fold_id in fold_map or fold_id not in EXPECTED_FOLDS:
            raise WebRelationshipMaterializationError(f"Invalid/duplicate fold ID: {fold_id}")
        if item.get("role") != "independent_fold_prediction":
            raise WebRelationshipMaterializationError(f"Fold {fold_id} has invalid role")
        checkpoint_hashes.add(
            _require_sha(item.get("checkpoint_sha256"), f"Fold {fold_id} checkpoint SHA")
        )
        partition_hashes.add(
            _require_sha(
                item.get("training_partition_sha256"),
                f"Fold {fold_id} training-partition SHA",
            )
        )
        fold_map[fold_id] = item
        declarations.append((f"fold_{fold_id}", item, primary_run))
    if sorted(fold_map) != list(EXPECTED_FOLDS):
        raise WebRelationshipMaterializationError("Fold IDs must be exactly 0..4")
    if len(checkpoint_hashes) != 5 or len(partition_hashes) != 5:
        raise WebRelationshipMaterializationError(
            "Five folds must bind distinct checkpoints and training partitions"
        )

    paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    data_hashes: set[str] = set()
    for label, declaration, training_run_id in declarations:
        if declaration.get("analysis_version") != ANALYSIS_VERSION:
            raise WebRelationshipMaterializationError(f"{label} is not fresh V3.2")
        if declaration.get("training_run_id") != training_run_id:
            raise WebRelationshipMaterializationError(f"{label} training lineage mismatch")
        if declaration.get("selection_sha256") != selection_sha:
            raise WebRelationshipMaterializationError(f"{label} selection lineage mismatch")
        if declaration.get("candidate_universe_sha256") != universe_sha:
            raise WebRelationshipMaterializationError(
                f"{label} candidate-universe lineage mismatch"
            )
        _false_only(declaration, label)
        if label == "exact_evidence" and declaration.get("family_to_exact_broadcast") is not False:
            raise WebRelationshipMaterializationError(
                "Exact evidence must explicitly prohibit family broadcast"
            )
        paths[label] = _artifact_path(source, declaration, label)
        artifact_hashes[label] = str(declaration["sha256"]).lower()
        if label.startswith("fold_"):
            artifact_sha = str(declaration["sha256"]).lower()
            if artifact_sha in data_hashes:
                raise WebRelationshipMaterializationError(
                    "Fold prediction files must be distinct artifacts"
                )
            data_hashes.add(artifact_sha)
    return payload, paths, observed, artifact_hashes


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


def _fetch_scalar(connection: Any, query: str) -> Any:
    row = connection.execute(query).fetchone()
    return None if row is None else row[0]


def _columns(connection: Any, view: str) -> list[str]:
    return [str(row[0]) for row in connection.execute(f"DESCRIBE {view}").fetchall()]


def _validate_table_view(
    connection: Any,
    view: str,
    *,
    required: Sequence[str],
    training_run_id: str,
    fold_id: int | None = None,
) -> int:
    columns = _columns(connection, view)
    missing = sorted(set(required) - set(columns))
    if missing:
        raise WebRelationshipMaterializationError(f"{view} missing columns: {missing}")
    lower_columns = {column.lower(): column for column in columns}
    forbidden = sorted(
        lower_columns[column.lower()]
        for column in FORBIDDEN_PUBLIC_COLUMNS
        if column.lower() in lower_columns
    )
    if forbidden:
        raise WebRelationshipMaterializationError(
            f"{view} contains held-out/private columns: {forbidden}"
        )
    collisions = sorted(
        lower_columns[column]
        for column in {"prediction_uncertainty", "fold_ensemble_mean_probability"}
        if column in lower_columns
    )
    if collisions:
        raise WebRelationshipMaterializationError(
            f"{view} contains derived columns that must be recomputed: {collisions}"
        )
    row_count = int(_fetch_scalar(connection, f"SELECT count(*) FROM {view}"))
    if row_count <= 0:
        raise WebRelationshipMaterializationError(f"{view} is empty")
    invalid_keys = int(
        _fetch_scalar(
            connection,
            f"""SELECT count(*) FROM {view}
                 WHERE cancer_id IS NULL OR trim(CAST(cancer_id AS VARCHAR)) = ''
                    OR lncrna_id IS NULL OR trim(CAST(lncrna_id AS VARCHAR)) = ''
                    OR pathway_id IS NULL OR trim(CAST(pathway_id AS VARCHAR)) = ''
                    OR regexp_matches(CAST(cancer_id AS VARCHAR), '[\\t\\r\\n]')
                    OR regexp_matches(CAST(lncrna_id AS VARCHAR), '[\\t\\r\\n]')
                    OR regexp_matches(CAST(pathway_id AS VARCHAR), '[\\t\\r\\n]')""",
        )
    )
    if invalid_keys:
        raise WebRelationshipMaterializationError(f"{view} has {invalid_keys} invalid keys")
    duplicates = int(
        _fetch_scalar(
            connection,
            f"""SELECT count(*) FROM (
                    SELECT {', '.join(KEYS)}, count(*) AS n FROM {view}
                    GROUP BY {', '.join(KEYS)} HAVING count(*) <> 1
                )""",
        )
    )
    if duplicates:
        raise WebRelationshipMaterializationError(f"{view} has duplicate exact keys")
    expected_version = _sql_string(ANALYSIS_VERSION)
    expected_run = _sql_string(training_run_id)
    bad_lineage = int(
        _fetch_scalar(
            connection,
            f"""SELECT count(*) FROM {view}
                 WHERE analysis_version IS NULL
                    OR CAST(analysis_version AS VARCHAR) <> '{expected_version}'
                    OR training_run_id IS NULL
                    OR CAST(training_run_id AS VARCHAR) <> '{expected_run}'""",
        )
    )
    if bad_lineage:
        raise WebRelationshipMaterializationError(f"{view} has row-level lineage drift")
    for column in (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
        "historical_predictions_used",
        "historical_rankings_used",
    ):
        if column in columns:
            count = int(
                _fetch_scalar(
                    connection,
                    f"SELECT count(*) FROM {view} WHERE try_cast({column} AS BOOLEAN) IS DISTINCT FROM FALSE",
                )
            )
            if count:
                raise WebRelationshipMaterializationError(
                    f"{view} row-level {column} is not exclusively false"
                )
    if fold_id is not None:
        bad_fold = int(
            _fetch_scalar(
                connection,
                f"SELECT count(*) FROM {view} WHERE try_cast(fold_id AS INTEGER) IS DISTINCT FROM {fold_id}",
            )
        )
        if bad_fold:
            raise WebRelationshipMaterializationError(f"{view} fold_id mismatch")
    return row_count


def _probability_check(connection: Any, view: str, column: str) -> None:
    invalid = int(
        _fetch_scalar(
            connection,
            f"""SELECT count(*) FROM {view}
                 WHERE try_cast({column} AS DOUBLE) IS NULL
                    OR NOT isfinite(try_cast({column} AS DOUBLE))
                    OR try_cast({column} AS DOUBLE) < 0
                    OR try_cast({column} AS DOUBLE) > 1""",
        )
    )
    if invalid:
        raise WebRelationshipMaterializationError(
            f"{view}.{column} has {invalid} invalid probabilities"
        )


def _assert_same_keys(connection: Any, left: str, right: str) -> None:
    using = ", ".join(KEYS)
    left_only = int(
        _fetch_scalar(
            connection,
            f"SELECT count(*) FROM {left} ANTI JOIN {right} USING ({using})",
        )
    )
    right_only = int(
        _fetch_scalar(
            connection,
            f"SELECT count(*) FROM {right} ANTI JOIN {left} USING ({using})",
        )
    )
    if left_only or right_only:
        raise WebRelationshipMaterializationError(
            f"Candidate-key mismatch {left}<->{right}: left_only={left_only}, right_only={right_only}"
        )


def materialize_fresh_v32_web_relationships(
    *,
    input_manifest_path: str | Path,
    expected_input_manifest_sha256: str,
    output_root: str | Path,
    memory_limit: str = "1GB",
    temp_directory: str | Path | None = None,
    thresholds: RelationshipThresholds | None = None,
) -> dict[str, Any]:
    """Materialize an immutable, low-memory, fresh-V3.2 website staging bundle.

    The only accepted input surface is a SHA-pinned formal manifest.  Five
    independently declared fold predictions are key-aligned in DuckDB and their
    sample standard deviation is calculated without labels.  This function
    never deploys the result or modifies a pre-existing directory.
    """

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency is declared by project
        raise WebRelationshipMaterializationError("duckdb is required") from exc

    if not re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", str(memory_limit).upper()):
        raise WebRelationshipMaterializationError("memory_limit must look like 512MB or 1GB")
    threshold = thresholds or RelationshipThresholds()
    threshold.validate()
    if threshold.required_folds != 5:
        raise WebRelationshipMaterializationError("Formal web materialization requires five folds")
    implementation_path = Path(__file__).resolve()
    implementation_sha256 = _sha256(implementation_path)

    manifest_path = Path(input_manifest_path).resolve()
    input_manifest, paths, input_manifest_sha, input_artifact_hashes = _validate_input_manifest(
        manifest_path, expected_input_manifest_sha256
    )
    destination = Path(output_root).resolve()
    if destination.exists():
        raise WebRelationshipMaterializationError(
            f"Refusing to overwrite or reuse staging output: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    work = destination / f".materializing.{os.getpid()}"
    work.mkdir()
    relationship_path = work / "web_exact_pathway_relationships.parquet"
    selected_path = work / "web_exact_pathway_selected.parquet"
    overview_path = work / "web_cancer_overview.parquet"
    database_path = work / "materialization.duckdb"
    if temp_directory:
        spill_base = Path(temp_directory).resolve()
        spill_base.mkdir(parents=True, exist_ok=True)
        spill = spill_base / f".v32_web_materializer_spill.{os.getpid()}"
    else:
        spill = work / "duckdb_spill"
    spill.mkdir(parents=True, exist_ok=False)

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("SET threads = 1")
        connection.execute(f"SET memory_limit = '{str(memory_limit).upper()}'")
        connection.execute(f"SET temp_directory = '{_sql_path(spill)}'")
        view_names = {
            "primary": "primary_input",
            "exact_evidence": "exact_evidence_input",
            **{f"fold_{fold}": f"fold_{fold}" for fold in EXPECTED_FOLDS},
        }
        for role, path in paths.items():
            view = view_names[role]
            connection.execute(
                f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{_sql_path(path)}')"
            )

        primary_required = [
            *KEYS,
            "pathway_family_id",
            "association_membership_probability",
            "association_direction_probability",
            "association_direction",
            "n_folds_available",
            "analysis_version",
            "training_run_id",
        ]
        evidence_required = [
            *KEYS,
            "evidence_confidence_probability",
            "availability",
            "direct_target_evidence",
            "family_to_exact_broadcast",
            "direction",
            "analysis_version",
            "training_run_id",
        ]
        fold_required = [
            *KEYS,
            "association_membership_probability",
            "fold_id",
            "analysis_version",
            "training_run_id",
        ]
        row_counts: dict[str, int] = {}
        row_counts["primary"] = _validate_table_view(
            connection,
            "primary_input",
            required=primary_required,
            training_run_id=str(input_manifest["primary_training_run_id"]),
        )
        primary_reserved = sorted(
            set(_columns(connection, "primary_input"))
            & {
                "display_probability",
                "evidence_available",
                "evidence_confidence_probability",
                "evidence_training_run_id",
                "family_to_exact_broadcast",
                "final_direction",
                "fold_ensemble_mean_probability",
                "observed_evidence_probability",
                "prediction_uncertainty",
                "relationship_class",
            }
        )
        if primary_reserved:
            raise WebRelationshipMaterializationError(
                f"Primary input contains materializer-owned output columns: {primary_reserved}"
            )
        row_counts["exact_evidence"] = _validate_table_view(
            connection,
            "exact_evidence_input",
            required=evidence_required,
            training_run_id=str(input_manifest["evidence_training_run_id"]),
        )
        for fold_id in EXPECTED_FOLDS:
            view = f"fold_{fold_id}"
            row_counts[view] = _validate_table_view(
                connection,
                view,
                required=fold_required,
                training_run_id=str(input_manifest["primary_training_run_id"]),
                fold_id=fold_id,
            )
        for view in ("primary_input", *[f"fold_{item}" for item in EXPECTED_FOLDS]):
            _probability_check(connection, view, "association_membership_probability")
        _probability_check(connection, "primary_input", "association_direction_probability")
        invalid_folds = int(
            _fetch_scalar(
                connection,
                "SELECT count(*) FROM primary_input WHERE try_cast(n_folds_available AS INTEGER) IS DISTINCT FROM 5",
            )
        )
        if invalid_folds:
            raise WebRelationshipMaterializationError(
                "Primary scores do not declare exactly five available folds"
            )
        family_as_target = int(
            _fetch_scalar(
                connection,
                """SELECT count(*) FROM primary_input
                   WHERE pathway_family_id IS NULL
                      OR trim(CAST(pathway_family_id AS VARCHAR)) = ''
                      OR CAST(pathway_id AS VARCHAR) = CAST(pathway_family_id AS VARCHAR)""",
            )
        )
        if family_as_target:
            raise WebRelationshipMaterializationError(
                "Primary input substituted a pathway family for an exact pathway target"
            )
        invalid_direction = int(
            _fetch_scalar(
                connection,
                """SELECT count(*) FROM primary_input
                   WHERE lower(CAST(association_direction AS VARCHAR)) NOT IN ('positive','negative')
                      OR lower(CAST(association_direction AS VARCHAR)) <>
                         CASE WHEN try_cast(association_direction_probability AS DOUBLE) >= 0.5
                              THEN 'positive' ELSE 'negative' END""",
            )
        )
        if invalid_direction:
            raise WebRelationshipMaterializationError(
                "Primary direction labels disagree with direction probabilities"
            )
        evidence_invalid = int(
            _fetch_scalar(
                connection,
                """SELECT count(*) FROM exact_evidence_input
                   WHERE try_cast(family_to_exact_broadcast AS BOOLEAN) IS DISTINCT FROM FALSE
                      OR try_cast(availability AS BOOLEAN) IS NULL
                      OR try_cast(direct_target_evidence AS BOOLEAN) IS NULL
                      OR try_cast(availability AS BOOLEAN) <> try_cast(direct_target_evidence AS BOOLEAN)
                      OR (try_cast(availability AS BOOLEAN) AND
                          (try_cast(evidence_confidence_probability AS DOUBLE) IS NULL OR
                           NOT isfinite(try_cast(evidence_confidence_probability AS DOUBLE)) OR
                           try_cast(evidence_confidence_probability AS DOUBLE) < 0 OR
                           try_cast(evidence_confidence_probability AS DOUBLE) > 1))
                      OR (NOT try_cast(availability AS BOOLEAN) AND
                          evidence_confidence_probability IS NOT NULL)
                      OR (try_cast(availability AS BOOLEAN) AND
                          lower(CAST(direction AS VARCHAR)) NOT IN ('positive','negative'))
                      OR (NOT try_cast(availability AS BOOLEAN) AND
                          coalesce(trim(CAST(direction AS VARCHAR)), '') <> '')""",
            )
        )
        if evidence_invalid:
            raise WebRelationshipMaterializationError(
                f"Exact evidence has {evidence_invalid} availability/broadcast/value violations"
            )

        observed_cancers = sorted(
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT CAST(cancer_id AS VARCHAR) FROM primary_input ORDER BY 1"
            ).fetchall()
        )
        if observed_cancers != list(TCGA_CANCERS):
            raise WebRelationshipMaterializationError(
                f"Primary input cancer set is not the canonical 33 TCGA cancers: {observed_cancers}"
            )
        for view in ("exact_evidence_input", *[f"fold_{item}" for item in EXPECTED_FOLDS]):
            _assert_same_keys(connection, "primary_input", view)

        values = [
            f"try_cast(fold_{fold}.association_membership_probability AS DOUBLE)"
            for fold in EXPECTED_FOLDS
        ]
        fold_sum = " + ".join(values)
        fold_squares = " + ".join(f"({value} * {value})" for value in values)
        fold_mean = f"(({fold_sum}) / 5.0)"
        fold_std = (
            f"sqrt(greatest(0.0, (({fold_squares}) - "
            f"((({fold_sum}) * ({fold_sum})) / 5.0)) / 4.0))"
        )
        joins = "\n".join(
            f"JOIN fold_{fold} USING ({', '.join(KEYS)})" for fold in EXPECTED_FOLDS
        )
        uncertainty_limit = float(threshold.max_prediction_uncertainty)
        observed_limit = float(threshold.observed_core_evidence_min)
        predicted_limit = float(threshold.predicted_candidate_probability_min)
        predicted_evidence_limit = float(
            threshold.predicted_candidate_max_observed_evidence
        )
        supported_limit = float(threshold.model_supported_probability_min)
        relationship_case = f"""CASE
            WHEN try_cast(exact_evidence_input.availability AS BOOLEAN)
             AND try_cast(exact_evidence_input.evidence_confidence_probability AS DOUBLE) >= {observed_limit}
              THEN 'observed_core'
            WHEN try_cast(primary_input.association_membership_probability AS DOUBLE) >= {predicted_limit}
             AND coalesce(try_cast(exact_evidence_input.evidence_confidence_probability AS DOUBLE), 0.0) <= {predicted_evidence_limit}
             AND ({fold_std}) <= {uncertainty_limit}
              THEN 'predicted_candidate'
            WHEN try_cast(exact_evidence_input.availability AS BOOLEAN)
             AND try_cast(exact_evidence_input.evidence_confidence_probability AS DOUBLE) > 0.0
             AND try_cast(primary_input.association_membership_probability AS DOUBLE) >= {supported_limit}
             AND ({fold_std}) <= {uncertainty_limit}
              THEN 'model_supported'
            ELSE 'exploratory' END"""
        evidence_columns = _columns(connection, "exact_evidence_input")
        optional_selects: list[str] = []
        for column in ("event_count", "uncertainty"):
            if column in evidence_columns:
                optional_selects.append(f"exact_evidence_input.{column} AS evidence_{column}")
        optional_sql = "" if not optional_selects else ",\n" + ",\n".join(optional_selects)
        primary_columns = _columns(connection, "primary_input")
        regenerated_primary_columns = [
            column
            for column in (
                "old_checkpoint_loaded",
                "old_predictions_used_as_features",
                "old_rankings_used_as_outputs",
                "historical_predictions_used",
                "historical_rankings_used",
                "pathway_family_role",
                "pathway_target_level",
            )
            if column in primary_columns
        ]
        primary_projection = "primary_input.*"
        if regenerated_primary_columns:
            primary_projection += " EXCLUDE (" + ", ".join(regenerated_primary_columns) + ")"
        query = f"""SELECT
              {primary_projection},
              try_cast(primary_input.association_membership_probability AS DOUBLE) AS display_probability,
              {fold_mean} AS fold_ensemble_mean_probability,
              {fold_std} AS prediction_uncertainty,
              try_cast(exact_evidence_input.availability AS BOOLEAN) AS evidence_available,
              try_cast(exact_evidence_input.evidence_confidence_probability AS DOUBLE)
                AS evidence_confidence_probability,
              try_cast(exact_evidence_input.evidence_confidence_probability AS DOUBLE)
                AS observed_evidence_probability,
              try_cast(exact_evidence_input.direct_target_evidence AS BOOLEAN) AS direct_target_evidence,
              false AS family_to_exact_broadcast,
              '{_sql_string(str(input_manifest['evidence_training_run_id']))}'
                AS evidence_training_run_id,
              CASE WHEN try_cast(exact_evidence_input.availability AS BOOLEAN)
                   THEN lower(CAST(exact_evidence_input.direction AS VARCHAR))
                   ELSE lower(CAST(primary_input.association_direction AS VARCHAR)) END AS final_direction,
              {relationship_case} AS relationship_class,
              'exact_pathway' AS pathway_target_level,
              'auxiliary_hierarchy_only' AS pathway_family_role,
              false AS old_checkpoint_loaded,
              false AS old_predictions_used_as_features,
              false AS old_rankings_used_as_outputs,
              false AS historical_predictions_used,
              false AS historical_rankings_used
              {optional_sql}
            FROM primary_input
            JOIN exact_evidence_input USING ({', '.join(KEYS)})
            {joins}"""
        connection.execute(
            f"COPY ({query}) TO '{_sql_path(relationship_path)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        connection.execute(
            f"""COPY (
                  SELECT * FROM read_parquet('{_sql_path(relationship_path)}')
                  WHERE relationship_class <> 'exploratory'
                ) TO '{_sql_path(selected_path)}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"""
        )
        semantics = (
            "compatibility alias for non_exploratory_relations; "
            "not statistical significance"
        )
        connection.execute(
            f"""COPY (
                  SELECT cancer_id,
                         count(*)::BIGINT AS scored_relationship_count,
                         count(DISTINCT lncrna_id)::BIGINT AS scored_lncRNAs,
                         count(DISTINCT lncrna_id)::BIGINT AS detectable_lncRNAs,
                         count(DISTINCT pathway_id)::BIGINT AS scored_exact_pathways,
                         sum(CASE WHEN relationship_class='observed_core' THEN 1 ELSE 0 END)::BIGINT
                           AS observed_core_relations,
                         sum(CASE WHEN relationship_class='model_supported' THEN 1 ELSE 0 END)::BIGINT
                           AS model_supported_relations,
                         sum(CASE WHEN relationship_class='predicted_candidate' THEN 1 ELSE 0 END)::BIGINT
                           AS predicted_candidate_relations,
                         sum(CASE WHEN relationship_class<>'exploratory' THEN 1 ELSE 0 END)::BIGINT
                           AS non_exploratory_relations,
                         sum(CASE WHEN relationship_class<>'exploratory' THEN 1 ELSE 0 END)::BIGINT
                           AS significant_lncRNA_pathway_relations,
                         '{_sql_string(semantics)}' AS significant_relation_semantics,
                         'compatibility alias for scored_lncRNAs in the filtered candidate universe; not raw-assay detectability'
                           AS detectable_lncRNA_semantics,
                         avg(display_probability) AS mean_primary_probability,
                         'exact_pathway' AS pathway_target_level,
                         false AS family_to_exact_broadcast,
                         '{_sql_string(ANALYSIS_VERSION)}' AS analysis_version,
                         'CC-HHGT_V3.2' AS model_version
                  FROM read_parquet('{_sql_path(relationship_path)}')
                  GROUP BY cancer_id ORDER BY cancer_id
                ) TO '{_sql_path(overview_path)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        output_rows = {
            "relationships": int(
                _fetch_scalar(
                    connection,
                    f"SELECT count(*) FROM read_parquet('{_sql_path(relationship_path)}')",
                )
            ),
            "selected": int(
                _fetch_scalar(
                    connection,
                    f"SELECT count(*) FROM read_parquet('{_sql_path(selected_path)}')",
                )
            ),
            "overview": int(
                _fetch_scalar(
                    connection,
                    f"SELECT count(*) FROM read_parquet('{_sql_path(overview_path)}')",
                )
            ),
        }
        if output_rows["relationships"] != row_counts["primary"]:
            raise WebRelationshipMaterializationError("Relationship output row count drifted")
        if output_rows["overview"] != EXPECTED_CANCERS:
            raise WebRelationshipMaterializationError("Overview does not cover 33 cancers")
        output_duplicate_keys = int(
            _fetch_scalar(
                connection,
                f"""SELECT count(*) FROM (
                      SELECT {', '.join(KEYS)}, count(*) AS n
                      FROM read_parquet('{_sql_path(relationship_path)}')
                      GROUP BY {', '.join(KEYS)} HAVING count(*) <> 1
                    )""",
            )
        )
        if output_duplicate_keys:
            raise WebRelationshipMaterializationError("Output has duplicate exact keys")
    finally:
        connection.close()

    if _sha256(manifest_path) != input_manifest_sha:
        raise WebRelationshipMaterializationError(
            "Input manifest changed during materialization"
        )
    for role, path in paths.items():
        if _sha256(path) != input_artifact_hashes[role]:
            raise WebRelationshipMaterializationError(
                f"{role} changed during materialization"
            )
    if _sha256(implementation_path) != implementation_sha256:
        raise WebRelationshipMaterializationError(
            "Materializer implementation changed during execution"
        )

    database_path.unlink(missing_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for role, path, rows in (
        ("relationships", relationship_path, output_rows["relationships"]),
        ("selected", selected_path, output_rows["selected"]),
        ("overview", overview_path, output_rows["overview"]),
    ):
        final_path = destination / path.name
        os.replace(path, final_path)
        artifacts[role] = {
            "path": path.name,
            "sha256": _sha256(final_path),
            "rows": rows,
        }
    # DuckDB spill files are scratch artifacts created under a unique directory
    # owned by this invocation.  They are never part of the staging bundle.
    for child in spill.iterdir():
        if child.is_dir() and not child.is_symlink():
            raise WebRelationshipMaterializationError(
                f"Unexpected nested directory in DuckDB spill area: {child}"
            )
        child.unlink()
    spill.rmdir()
    work.rmdir()
    manifest = {
        "manifest_format": OUTPUT_MANIFEST_FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "family_to_exact_broadcast": False,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "production_deployed": False,
        "staging_only": True,
        "cancers": EXPECTED_CANCERS,
        "folds": list(EXPECTED_FOLDS),
        "uncertainty_definition": "sample standard deviation across five independent fold probabilities; ddof=1; no labels",
        "input_manifest": {
            "path": str(manifest_path),
            "sha256": input_manifest_sha,
            "primary_training_run_id": input_manifest["primary_training_run_id"],
            "evidence_training_run_id": input_manifest["evidence_training_run_id"],
            "selection_sha256": input_manifest["selection_sha256"],
            "candidate_universe_sha256": input_manifest["candidate_universe_sha256"],
        },
        "thresholds": asdict(threshold),
        "implementation": {
            "module": "cc_hhgt/v32/web_relationship_materialization.py",
            "sha256": implementation_sha256,
        },
        "input_rows": row_counts,
        "artifacts": artifacts,
    }
    manifest_path_out = destination / "MATERIALIZATION_MANIFEST.json"
    _atomic_json(manifest_path_out, manifest)
    success = {
        "status": "PASS",
        "manifest_format": OUTPUT_MANIFEST_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "manifest": manifest_path_out.name,
        "manifest_sha256": _sha256(manifest_path_out),
        "production_deployed": False,
        "staging_only": True,
        "release_ready": False,
    }
    _atomic_json(destination / "SUCCESS.json", success)
    return {**success, "output_root": str(destination), "artifacts": artifacts}
