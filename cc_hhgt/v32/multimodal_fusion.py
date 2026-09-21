"""Fresh V3.2 availability-aware residual fusion.

The exact-pathway probability remains the immutable primary score.  This
module trains a secondary, pair-blocked score from *current V3.2* expert
outputs.  It deliberately never opens a historical checkpoint, prediction,
ranking, web table, or fitted feature.

The fusion is an offset model::

    logit(adjusted) = logit(primary) + sum_j w_j * available_j
                      * (logit(expert_j) - centre_j)

with non-negative weights.  Consequently a row with no available auxiliary
expert falls back bit-for-bit to the primary probability.  The contribution
of every available expert is explicit and missing evidence is never encoded
as zero probability.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "multimodal_fusion"
TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
PRIMARY_COLUMN = "association_membership_probability"
LABEL_COLUMN = "held_out_proxy_label"
FUSION_FOLD_COLUMN = "fusion_pair_fold"
N_FOLDS = 5
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_RESIDUAL_FUSION_CHECKPOINT_V1"
PREDICTION_FORMAT = "CC_HHGT_V3_2_SECONDARY_FUSION_PREDICTION_V1"

_LEGACY_RESULT = re.compile(
    r"(?:^|[/\\._-])(?:legacy|historical|old)(?:[/\\._-]|$)|"
    r"(?:^|[^a-z0-9])v(?:2(?:[._-]?\d+)*|3[._-]?[01])(?:[^0-9]|$)",
    flags=re.IGNORECASE,
)
_DERIVED_ROLE = re.compile(
    r"checkpoint|prediction|probabilit|ranking|embedding|web.?table|cache|"
    r"processed.?training|fitted.?feature",
    flags=re.IGNORECASE,
)


class MultimodalFusionError(RuntimeError):
    """Raised when a fusion input, split, fit, or output violates the contract."""


@dataclass(frozen=True)
class ExpertSpec:
    """One current-V3.2 expert used by a secondary fusion endpoint."""

    expert_id: str
    probability_column: str
    availability_column: str
    endpoint_roles: tuple[str, ...]
    split_unit: str
    source_role: str = "provenance_audited_standardized_source"
    independent_of_primary_label: bool = True
    direct_target_evidence: bool = False

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.expert_id):
            raise MultimodalFusionError(f"Invalid expert_id: {self.expert_id!r}")
        if not self.probability_column or not self.availability_column:
            raise MultimodalFusionError(f"{self.expert_id} lacks probability/availability columns")
        allowed_roles = {"discovery", "confidence", "actionability"}
        if not self.endpoint_roles or not set(self.endpoint_roles).issubset(allowed_roles):
            raise MultimodalFusionError(
                f"{self.expert_id} has invalid endpoint roles: {self.endpoint_roles}"
            )
        if self.direct_target_evidence and "discovery" in self.endpoint_roles:
            raise MultimodalFusionError(
                f"Direct target evidence {self.expert_id} is forbidden in discovery fusion"
            )
        if not self.independent_of_primary_label:
            raise MultimodalFusionError(
                f"{self.expert_id} is not independent/OOF relative to primary supervision"
            )
        if self.source_role not in {
            "raw_source_data",
            "provenance_audited_standardized_source",
            "identifier_crosswalk",
            "pathway_membership",
            "task_definition",
            "split_definition",
            "current_v32_oof_prediction",
        }:
            raise MultimodalFusionError(
                f"{self.expert_id} has forbidden source role {self.source_role!r}"
            )


@dataclass(frozen=True)
class ResidualFusionConfig:
    seed: int = 20260826
    folds: int = N_FOLDS
    learning_rate: float = 0.05
    l2: float = 1.0e-3
    max_steps: int = 600
    patience: int = 60
    tolerance: float = 1.0e-7
    max_train_rows: int = 2_000_000
    max_validation_rows: int = 500_000
    batch_size: int = 65_536
    evaluation_interval: int = 10
    clip_probability: float = 1.0e-6
    increment_logloss_delta: float = 1.0e-4

    def validate(self) -> None:
        if self.folds != N_FOLDS:
            raise MultimodalFusionError("Formal V3.2 fusion requires exactly five folds")
        if not 0 < self.learning_rate <= 1 or self.l2 < 0:
            raise MultimodalFusionError("Invalid fusion optimiser configuration")
        if (
            self.max_steps < 1
            or self.patience < 1
            or self.max_train_rows < 10
            or self.max_validation_rows < 10
            or self.batch_size < 10
            or self.evaluation_interval < 1
        ):
            raise MultimodalFusionError("Fusion training limits are invalid")
        if not 0 < self.clip_probability < 0.01:
            raise MultimodalFusionError("clip_probability is invalid")


@dataclass(frozen=True)
class ResidualFusionModel:
    endpoint: str
    expert_ids: tuple[str, ...]
    centres: tuple[float, ...]
    weights: tuple[float, ...]
    seed: int
    initial_parameter_sha256: str
    final_parameter_sha256: str
    optimiser_steps: int
    validation_logloss: float
    clip_probability: float = 1.0e-6

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "analysis_version": ANALYSIS_VERSION,
                "module_id": MODULE_ID,
                "primary_probability_offset_fixed": True,
                "missing_expert_fallback": "EXACT_PRIMARY_PROBABILITY",
                "zero_total_contribution_fallback": "EXACT_PRIMARY_PROBABILITY",
                "weights_constrained_nonnegative": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
            }
        )
        return payload


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_sha(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _canonical_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(TARGET_KEYS) - set(frame.columns))
    if missing:
        raise MultimodalFusionError(f"{label} lacks exact target keys: {missing}")
    output = frame.copy()
    for column in TARGET_KEYS:
        output[column] = output[column].astype("string").str.strip()
        if output[column].isna().any() or output[column].eq("").any():
            raise MultimodalFusionError(f"{label}.{column} contains empty identifiers")
    if output.duplicated(list(TARGET_KEYS)).any():
        raise MultimodalFusionError(f"{label} duplicates exact target keys")
    return output


def _probability(values: pd.Series, label: str, *, allow_null: bool) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(numeric)
    if not allow_null and not finite.all():
        raise MultimodalFusionError(f"{label} contains null/non-finite values")
    if ((numeric[finite] < 0) | (numeric[finite] > 1)).any():
        raise MultimodalFusionError(f"{label} is outside [0, 1]")
    return numeric


def _assert_current_v32(frame: pd.DataFrame, label: str) -> None:
    if "analysis_version" not in frame:
        raise MultimodalFusionError(f"{label} lacks analysis_version")
    versions = frame.analysis_version.dropna().astype(str)
    if versions.empty or not versions.eq(ANALYSIS_VERSION).all():
        raise MultimodalFusionError(f"{label} is not uniformly current V3.2")
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized in {
            "oldcheckpointloaded",
            "oldpredictionsused",
            "oldpredictionsusedasfeatures",
            "historicalpredictionsused",
            "historicalrankingsused",
        } and frame[column].fillna(False).astype(bool).any():
            raise MultimodalFusionError(f"{label} declares forbidden historical use: {column}")


def validate_source_declarations(declarations: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed on a legacy/derived input declaration before opening it."""

    if not declarations:
        raise MultimodalFusionError("Fusion requires source declarations")
    for item in declarations:
        path = str(item.get("path", "")).strip()
        role = str(item.get("source_role", "")).strip()
        sha = str(item.get("sha256", "")).lower()
        generation = str(item.get("generation", "")).strip().lower()
        if not path or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise MultimodalFusionError("Fusion source declaration lacks path/SHA256")
        if _LEGACY_RESULT.search(path) and role not in {
            "raw_source_data", "identifier_crosswalk", "pathway_membership", "task_definition"
        }:
            raise MultimodalFusionError(f"Legacy derived fusion input is forbidden: {path}")
        if _DERIVED_ROLE.search(role) and role != "current_v32_oof_prediction":
            raise MultimodalFusionError(f"Forbidden derived fusion source role: {role}")
        if generation not in {"current_v32", "static_annotation", "raw_source"}:
            raise MultimodalFusionError(f"Invalid fusion source generation: {generation}")


def pair_blocked_fold(lncrna_id: object, pathway_id: object, *, seed: int = 20260826) -> int:
    token = f"{str(lncrna_id).strip()}|{str(pathway_id).strip()}|{int(seed)}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % N_FOLDS


def _logit(values: np.ndarray, clip: float) -> np.ndarray:
    clipped = np.clip(values, clip, 1.0 - clip)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _fold_id(value: object) -> int:
    match = re.search(r"(?:PF|FOLD|PATIENT_FOLD)?[_-]?(\d+)$", str(value), re.IGNORECASE)
    if not match:
        raise MultimodalFusionError(f"Cannot parse patient fold identifier: {value!r}")
    fold = int(match.group(1))
    if fold not in range(N_FOLDS):
        raise MultimodalFusionError(f"Patient fold is outside 0..4: {value!r}")
    return fold


def build_pair_aggregated_fusion_frame(
    primary_final: pd.DataFrame,
    primary_fold_predictions: Sequence[pd.DataFrame],
    experts: Mapping[ExpertSpec, pd.DataFrame],
    *,
    seed: int = 20260826,
) -> pd.DataFrame:
    """Build one candidate row with a five-fold held-out soft target.

    Fusion itself is split by the cancer-agnostic lncRNA/exact-pathway pair,
    so all cancer contexts of a pair remain in one outer fold.  The primary
    and every auxiliary probability must already be OOF or independent with
    respect to its own raw-data supervision.
    """

    primary = _canonical_keys(primary_final, "primary_final")
    _assert_current_v32(primary, "primary_final")
    if PRIMARY_COLUMN not in primary:
        raise MultimodalFusionError(f"primary_final lacks {PRIMARY_COLUMN}")
    _probability(primary[PRIMARY_COLUMN], f"primary_final.{PRIMARY_COLUMN}", allow_null=False)
    if len(primary_fold_predictions) != N_FOLDS:
        raise MultimodalFusionError("Fusion target requires all five primary fold predictions")

    target_parts: list[pd.DataFrame] = []
    observed_folds: set[int] = set()
    primary_keys = primary[list(TARGET_KEYS)].sort_values(list(TARGET_KEYS)).reset_index(drop=True)
    for index, source in enumerate(primary_fold_predictions):
        fold_frame = _canonical_keys(source, f"primary_fold[{index}]")
        if LABEL_COLUMN not in fold_frame or "patient_fold_id" not in fold_frame:
            raise MultimodalFusionError(
                f"primary_fold[{index}] lacks patient_fold_id/{LABEL_COLUMN}"
            )
        unique_ids = fold_frame.patient_fold_id.dropna().map(_fold_id).unique()
        if len(unique_ids) != 1:
            raise MultimodalFusionError(
                f"primary_fold[{index}] does not contain exactly one patient fold"
            )
        fold = int(unique_ids[0])
        if fold in observed_folds:
            raise MultimodalFusionError(f"Duplicate primary patient fold {fold}")
        observed_folds.add(fold)
        labels = pd.to_numeric(fold_frame[LABEL_COLUMN], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise MultimodalFusionError(f"primary_fold[{index}] labels are not binary")
        fold_keys = fold_frame[list(TARGET_KEYS)].sort_values(list(TARGET_KEYS)).reset_index(drop=True)
        if len(fold_keys) != len(primary_keys) or not fold_keys.equals(primary_keys):
            raise MultimodalFusionError(
                f"primary_fold[{index}] candidate universe differs from primary_final"
            )
        part = fold_frame[list(TARGET_KEYS)].copy()
        part["patient_fold"] = fold
        part[LABEL_COLUMN] = labels.astype(np.float32)
        target_parts.append(part)
    if observed_folds != set(range(N_FOLDS)):
        raise MultimodalFusionError(f"Primary folds are incomplete: {sorted(observed_folds)}")

    targets = pd.concat(target_parts, ignore_index=True)
    target_summary = targets.groupby(list(TARGET_KEYS), observed=True, as_index=False).agg(
        fusion_target=(LABEL_COLUMN, "mean"),
        fusion_target_positive_folds=(LABEL_COLUMN, "sum"),
        fusion_target_fold_count=("patient_fold", "nunique"),
    )
    if not target_summary.fusion_target_fold_count.eq(N_FOLDS).all():
        raise MultimodalFusionError("At least one candidate lacks five held-out target folds")

    frame = primary[list(TARGET_KEYS) + [PRIMARY_COLUMN]].rename(
        columns={PRIMARY_COLUMN: "primary_probability"}
    )
    frame = frame.merge(target_summary, on=list(TARGET_KEYS), how="left", validate="one_to_one")
    frame[FUSION_FOLD_COLUMN] = [
        pair_blocked_fold(lnc, pathway, seed=seed)
        for lnc, pathway in frame[["lncrna_id", "pathway_id"]].itertuples(index=False, name=None)
    ]

    seen_experts: set[str] = set()
    for spec, raw in experts.items():
        spec.validate()
        if spec.expert_id in seen_experts:
            raise MultimodalFusionError(f"Duplicate fusion expert: {spec.expert_id}")
        seen_experts.add(spec.expert_id)
        expert = _canonical_keys(raw, spec.expert_id)
        _assert_current_v32(expert, spec.expert_id)
        missing = sorted(
            {spec.probability_column, spec.availability_column} - set(expert.columns)
        )
        if missing:
            raise MultimodalFusionError(f"{spec.expert_id} lacks columns: {missing}")
        expert_keys = expert[list(TARGET_KEYS)].sort_values(list(TARGET_KEYS)).reset_index(drop=True)
        if len(expert_keys) != len(primary_keys) or not expert_keys.equals(primary_keys):
            raise MultimodalFusionError(
                f"{spec.expert_id} must provide a typed row for the full exact candidate universe"
            )
        available = expert[spec.availability_column]
        if available.isna().any():
            raise MultimodalFusionError(f"{spec.expert_id} availability contains nulls")
        available = available.astype(bool)
        probability = _probability(
            expert[spec.probability_column],
            f"{spec.expert_id}.{spec.probability_column}",
            allow_null=True,
        )
        if np.isnan(probability[available.to_numpy()]).any():
            raise MultimodalFusionError(
                f"{spec.expert_id} has available rows without a probability"
            )
        if np.isfinite(probability[~available.to_numpy()]).any():
            raise MultimodalFusionError(
                f"{spec.expert_id} fills unavailable rows with a numeric value"
            )
        keep = expert[list(TARGET_KEYS)].copy()
        keep[f"{spec.expert_id}_probability"] = probability
        keep[f"{spec.expert_id}_available"] = available.to_numpy(bool)
        frame = frame.merge(keep, on=list(TARGET_KEYS), how="left", validate="one_to_one")

    if frame[list(TARGET_KEYS)].duplicated().any() or len(frame) != len(primary):
        raise MultimodalFusionError("Fusion frame changed the primary exact candidate universe")
    return frame.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def _duckdb_literal(value: str | Path) -> str:
    return "'" + str(Path(value).resolve()).replace("\\", "/").replace("'", "''") + "'"


def _duckdb_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def build_pair_aggregated_fusion_frame_from_parquet(
    primary_final_path: str | Path,
    primary_fold_paths: Sequence[str | Path],
    experts: Mapping[ExpertSpec, str | Path],
    *,
    seed: int = 20260826,
    expected_rows: int | None = None,
    temp_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the formal fusion frame with one DuckDB scan per bound source.

    The in-memory builder above is convenient for tests and small cohorts, but
    retaining five 3.3-million-row fold tables plus every expert table at once
    is unnecessary.  This path validates and aggregates the same exact-key
    contract inside DuckDB, then materialises only the final fusion frame.
    It returns an explicit input audit so the release runner can bind the
    candidate-universe, fold and typed-null checks it actually performed.
    """

    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
        raise MultimodalFusionError("DuckDB is required for formal fusion input loading") from exc

    primary_path = Path(primary_final_path).resolve()
    fold_paths = tuple(Path(path).resolve() for path in primary_fold_paths)
    if len(fold_paths) != N_FOLDS:
        raise MultimodalFusionError("Formal fusion requires five primary fold sources")
    source_paths = (primary_path, *fold_paths, *(Path(path).resolve() for path in experts.values()))
    for source in source_paths:
        if not source.is_file() or source.is_symlink():
            raise MultimodalFusionError(f"Fusion parquet source is missing or unsafe: {source}")

    specs = tuple(experts)
    if not specs:
        raise MultimodalFusionError("Formal fusion requires auxiliary experts")
    expert_ids: set[str] = set()
    for spec in specs:
        spec.validate()
        if spec.expert_id in expert_ids:
            raise MultimodalFusionError(f"Duplicate fusion expert: {spec.expert_id}")
        expert_ids.add(spec.expert_id)

    connection = duckdb.connect(database=":memory:")
    try:
        if temp_directory is not None:
            temporary = Path(temp_directory).resolve()
            temporary.mkdir(parents=True, exist_ok=True)
            connection.execute(f"SET temp_directory={_duckdb_literal(temporary)}")

        key_projection = ", ".join(
            f"trim(CAST({_duckdb_identifier(column)} AS VARCHAR)) AS {_duckdb_identifier(column)}"
            for column in TARGET_KEYS
        )
        key_tuple = ", ".join(_duckdb_identifier(column) for column in TARGET_KEYS)
        key_bad = " OR ".join(
            f"{_duckdb_identifier(column)} IS NULL OR trim(CAST({_duckdb_identifier(column)} AS VARCHAR)) = ''"
            for column in TARGET_KEYS
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE primary_norm AS
            SELECT {key_projection},
                   TRY_CAST({_duckdb_identifier(PRIMARY_COLUMN)} AS DOUBLE) AS primary_probability,
                   CAST(analysis_version AS VARCHAR) AS analysis_version
            FROM read_parquet({_duckdb_literal(primary_path)})
            """
        )
        primary_summary = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT ({key_tuple})) AS distinct_keys,
                   count_if({key_bad}) AS bad_keys,
                   count_if(primary_probability IS NULL OR NOT isfinite(primary_probability)
                            OR primary_probability < 0 OR primary_probability > 1) AS bad_probability,
                   count(DISTINCT analysis_version) AS version_count,
                   min(analysis_version) AS version
            FROM primary_norm
            """
        ).fetchone()
        primary_rows = int(primary_summary[0])
        if expected_rows is not None and primary_rows != int(expected_rows):
            raise MultimodalFusionError(
                f"Primary candidate rows differ from formal authority: {primary_rows} != {expected_rows}"
            )
        if (
            primary_rows < 1
            or int(primary_summary[1]) != primary_rows
            or int(primary_summary[2]) != 0
            or int(primary_summary[3]) != 0
            or int(primary_summary[4]) != 1
            or str(primary_summary[5]) != ANALYSIS_VERSION
        ):
            raise MultimodalFusionError(f"Primary parquet failed formal audit: {primary_summary}")

        fold_selects: list[str] = []
        observed_folds: set[int] = set()
        fold_audit: list[dict[str, Any]] = []
        for index, path in enumerate(fold_paths):
            relation = f"fold_{index}_norm"
            connection.execute(
                f"""
                CREATE TEMP VIEW {relation} AS
                SELECT {key_projection},
                       CAST(patient_fold_id AS VARCHAR) AS patient_fold_id,
                       TRY_CAST({_duckdb_identifier(LABEL_COLUMN)} AS DOUBLE) AS held_out_proxy_label
                FROM read_parquet({_duckdb_literal(path)})
                """
            )
            summary = connection.execute(
                f"""
                SELECT count(*) AS rows,
                       count(DISTINCT ({key_tuple})) AS distinct_keys,
                       count_if({key_bad}) AS bad_keys,
                       count_if(held_out_proxy_label IS NULL OR NOT isfinite(held_out_proxy_label)
                                OR held_out_proxy_label NOT IN (0, 1)) AS bad_labels,
                       count(DISTINCT patient_fold_id) AS patient_fold_count,
                       min(patient_fold_id) AS patient_fold_id
                FROM {relation}
                """
            ).fetchone()
            if (
                int(summary[0]) != primary_rows
                or int(summary[1]) != primary_rows
                or int(summary[2]) != 0
                or int(summary[3]) != 0
                or int(summary[4]) != 1
            ):
                raise MultimodalFusionError(f"Primary fold {index} failed formal audit: {summary}")
            fold_id = _fold_id(summary[5])
            if fold_id in observed_folds:
                raise MultimodalFusionError(f"Duplicate primary patient fold {fold_id}")
            observed_folds.add(fold_id)
            difference = connection.execute(
                f"""
                SELECT
                  (SELECT count(*) FROM (
                    SELECT {key_tuple} FROM {relation}
                    EXCEPT SELECT {key_tuple} FROM primary_norm
                  )) AS fold_only,
                  (SELECT count(*) FROM (
                    SELECT {key_tuple} FROM primary_norm
                    EXCEPT SELECT {key_tuple} FROM {relation}
                  )) AS primary_only
                """
            ).fetchone()
            if tuple(map(int, difference)) != (0, 0):
                raise MultimodalFusionError(
                    f"Primary fold {fold_id} candidate universe differs: {difference}"
                )
            fold_selects.append(
                f"SELECT {key_tuple}, {fold_id} AS patient_fold, held_out_proxy_label FROM {relation}"
            )
            fold_audit.append(
                {
                    "patient_fold": fold_id,
                    "rows": primary_rows,
                    "distinct_keys": primary_rows,
                    "fold_only_keys": 0,
                    "primary_only_keys": 0,
                    "bad_labels": 0,
                }
            )
        if observed_folds != set(range(N_FOLDS)):
            raise MultimodalFusionError(f"Primary folds are incomplete: {sorted(observed_folds)}")

        connection.execute(
            f"""
            CREATE TEMP TABLE target_summary AS
            SELECT {key_tuple},
                   avg(held_out_proxy_label) AS fusion_target,
                   sum(held_out_proxy_label) AS fusion_target_positive_folds,
                   count(DISTINCT patient_fold) AS fusion_target_fold_count
            FROM ({' UNION ALL '.join(fold_selects)})
            GROUP BY {key_tuple}
            """
        )
        target_summary = connection.execute(
            """
            SELECT count(*) AS rows,
                   count_if(fusion_target_fold_count <> 5) AS incomplete,
                   count_if(fusion_target IS NULL OR NOT isfinite(fusion_target)
                            OR fusion_target < 0 OR fusion_target > 1) AS bad_target
            FROM target_summary
            """
        ).fetchone()
        if tuple(map(int, target_summary)) != (primary_rows, 0, 0):
            raise MultimodalFusionError(f"Aggregated fusion target failed audit: {target_summary}")

        expert_audit: dict[str, dict[str, Any]] = {}
        expert_joins: list[str] = []
        expert_projection: list[str] = []
        for index, (spec, raw_path) in enumerate(experts.items()):
            path = Path(raw_path).resolve()
            relation = f"expert_{index}_norm"
            probability_column = _duckdb_identifier(spec.probability_column)
            availability_column = _duckdb_identifier(spec.availability_column)
            connection.execute(
                f"""
                CREATE TEMP TABLE {relation} AS
                SELECT {key_projection},
                       TRY_CAST({probability_column} AS DOUBLE) AS expert_probability,
                       TRY_CAST({availability_column} AS BOOLEAN) AS expert_available,
                       CAST(analysis_version AS VARCHAR) AS analysis_version
                FROM read_parquet({_duckdb_literal(path)})
                """
            )
            summary = connection.execute(
                f"""
                SELECT count(*) AS rows,
                       count(DISTINCT ({key_tuple})) AS distinct_keys,
                       count_if({key_bad}) AS bad_keys,
                       count_if(expert_available IS NULL) AS null_availability,
                       count_if(expert_available AND
                                (expert_probability IS NULL OR NOT isfinite(expert_probability)
                                 OR expert_probability < 0 OR expert_probability > 1)) AS bad_available,
                       count_if(NOT expert_available AND expert_probability IS NOT NULL)
                         AS filled_unavailable,
                       count(DISTINCT analysis_version) AS version_count,
                       min(analysis_version) AS version,
                       count_if(expert_available) AS available_rows
                FROM {relation}
                """
            ).fetchone()
            if (
                int(summary[0]) != primary_rows
                or int(summary[1]) != primary_rows
                or any(int(summary[position]) != 0 for position in range(2, 6))
                or int(summary[6]) != 1
                or str(summary[7]) != ANALYSIS_VERSION
            ):
                raise MultimodalFusionError(
                    f"Expert {spec.expert_id} failed typed full-universe audit: {summary}"
                )
            difference = connection.execute(
                f"""
                SELECT
                  (SELECT count(*) FROM (
                    SELECT {key_tuple} FROM {relation}
                    EXCEPT SELECT {key_tuple} FROM primary_norm
                  )) AS expert_only,
                  (SELECT count(*) FROM (
                    SELECT {key_tuple} FROM primary_norm
                    EXCEPT SELECT {key_tuple} FROM {relation}
                  )) AS primary_only
                """
            ).fetchone()
            if tuple(map(int, difference)) != (0, 0):
                raise MultimodalFusionError(
                    f"Expert {spec.expert_id} candidate universe differs: {difference}"
                )
            alias = f"e{index}"
            expert_joins.append(f"JOIN {relation} {alias} USING ({key_tuple})")
            expert_projection.extend(
                (
                    f"{alias}.expert_probability AS {_duckdb_identifier(spec.expert_id + '_probability')}",
                    f"{alias}.expert_available AS {_duckdb_identifier(spec.expert_id + '_available')}",
                )
            )
            expert_audit[spec.expert_id] = {
                "rows": primary_rows,
                "distinct_keys": primary_rows,
                "available_rows": int(summary[8]),
                "unavailable_rows": primary_rows - int(summary[8]),
                "expert_only_keys": 0,
                "primary_only_keys": 0,
                "typed_null_violations": 0,
                "analysis_version": ANALYSIS_VERSION,
            }

        final_projection = ",\n                   ".join(expert_projection)
        frame = connection.execute(
            f"""
            SELECT p.{_duckdb_identifier(TARGET_KEYS[0])},
                   p.{_duckdb_identifier(TARGET_KEYS[1])},
                   p.{_duckdb_identifier(TARGET_KEYS[2])},
                   p.primary_probability,
                   t.fusion_target,
                   t.fusion_target_positive_folds,
                   t.fusion_target_fold_count,
                   {final_projection}
            FROM primary_norm p
            JOIN target_summary t USING ({key_tuple})
            {' '.join(expert_joins)}
            ORDER BY {key_tuple}
            """
        ).fetchdf()
    finally:
        connection.close()

    pair_folds = np.fromiter(
        (
            pair_blocked_fold(lncrna, pathway, seed=seed)
            for lncrna, pathway in frame[["lncrna_id", "pathway_id"]].itertuples(
                index=False, name=None
            )
        ),
        dtype=np.int8,
        count=len(frame),
    )
    frame.insert(7, FUSION_FOLD_COLUMN, pair_folds)
    if (
        len(frame) != primary_rows
        or frame[list(TARGET_KEYS)].duplicated().any()
        or set(pd.unique(frame[FUSION_FOLD_COLUMN])) != set(range(N_FOLDS))
    ):
        raise MultimodalFusionError("Formal parquet builder changed the exact candidate universe")
    audit = {
        "status": "PASS",
        "builder": "DUCKDB_STREAMED_EXACT_KEY_JOIN",
        "analysis_version": ANALYSIS_VERSION,
        "candidate_rows": primary_rows,
        "candidate_distinct_keys": primary_rows,
        "primary_probability_violations": 0,
        "primary_folds": sorted(fold_audit, key=lambda item: item["patient_fold"]),
        "all_five_primary_folds_present": True,
        "target_rows": primary_rows,
        "target_incomplete_rows": 0,
        "experts": expert_audit,
        "all_experts_full_candidate_universe": True,
        "all_unavailable_probabilities_null": True,
        "pair_blocked_folds": N_FOLDS,
    }
    return frame, audit


def _subsample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(frame.fusion_target.to_numpy(float) >= 0.5)
    negative = np.flatnonzero(frame.fusion_target.to_numpy(float) < 0.5)
    positive_budget = min(len(positive), max(maximum // 2, 1))
    negative_budget = min(len(negative), maximum - positive_budget)
    if positive_budget + negative_budget < maximum:
        positive_budget = min(len(positive), maximum - negative_budget)
    selected = np.concatenate(
        [
            rng.choice(positive, positive_budget, replace=False),
            rng.choice(negative, negative_budget, replace=False),
        ]
    )
    rng.shuffle(selected)
    return frame.iloc[selected].reset_index(drop=True)


def _expert_matrix(
    frame: pd.DataFrame,
    expert_ids: Sequence[str],
    centres: np.ndarray | None,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for expert in expert_ids:
        probability = pd.to_numeric(frame[f"{expert}_probability"], errors="coerce").to_numpy(float)
        # Pandas copy-on-write may expose a read-only ndarray.  The mask is
        # intentionally refined in place below, so request an owned buffer.
        available = frame[f"{expert}_available"].fillna(False).astype(bool).to_numpy(copy=True)
        available &= np.isfinite(probability)
        values = np.zeros(len(frame), dtype=float)
        values[available] = _logit(probability[available], clip)
        logits.append(values)
        masks.append(available.astype(float))
    logit_matrix = np.column_stack(logits) if logits else np.empty((len(frame), 0), float)
    mask_matrix = np.column_stack(masks) if masks else np.empty((len(frame), 0), float)
    if centres is None:
        centre_values = np.zeros(len(expert_ids), dtype=float)
        for column in range(len(expert_ids)):
            valid = mask_matrix[:, column].astype(bool)
            centre_values[column] = (
                float(np.median(logit_matrix[valid, column])) if valid.any() else 0.0
            )
    else:
        centre_values = np.asarray(centres, dtype=float)
        if centre_values.shape != (len(expert_ids),):
            raise MultimodalFusionError("Fusion centre dimension mismatch")
    matrix = mask_matrix * (logit_matrix - centre_values[None, :])
    return matrix, mask_matrix, centre_values


def _logloss(target: np.ndarray, probability: np.ndarray, clip: float) -> float:
    value = np.clip(probability, clip, 1.0 - clip)
    return float(-np.mean(target * np.log(value) + (1.0 - target) * np.log1p(-value)))


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(probability - target)))


def fit_residual_fusion_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    expert_ids: Sequence[str],
    *,
    endpoint: str,
    config: ResidualFusionConfig,
    seed: int,
) -> ResidualFusionModel:
    """Fit one fresh non-negative residual model with a fixed primary offset."""

    config.validate()
    if endpoint not in {"discovery", "confidence", "actionability"}:
        raise MultimodalFusionError(f"Invalid fusion endpoint: {endpoint}")
    experts = tuple(expert_ids)
    if not experts or len(experts) != len(set(experts)):
        raise MultimodalFusionError("Fusion requires distinct non-empty expert IDs")
    train = _subsample(train, config.max_train_rows, seed)
    validation = _subsample(validation, config.max_validation_rows, seed + 1)
    if train.empty or validation.empty:
        raise MultimodalFusionError("Fusion train/validation split is empty")
    target_train = pd.to_numeric(train.fusion_target, errors="coerce").to_numpy(float)
    target_validation = pd.to_numeric(
        validation.fusion_target, errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(target_train).all()
        or not np.isfinite(target_validation).all()
        or ((target_train < 0) | (target_train > 1)).any()
        or ((target_validation < 0) | (target_validation > 1)).any()
    ):
        raise MultimodalFusionError("Fusion target is not finite within [0, 1]")
    primary_train = _probability(
        train.primary_probability, "fusion train primary_probability", allow_null=False
    )
    primary_validation = _probability(
        validation.primary_probability,
        "fusion validation primary_probability",
        allow_null=False,
    )
    train_matrix, train_mask, centres = _expert_matrix(
        train, experts, None, config.clip_probability
    )
    validation_matrix, _, _ = _expert_matrix(
        validation, experts, centres, config.clip_probability
    )
    if not train_mask.any():
        raise MultimodalFusionError("No auxiliary expert is available in fusion training")
    offset_train = _logit(primary_train, config.clip_probability)
    offset_validation = _logit(primary_validation, config.clip_probability)

    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.0025, 0.0125, size=len(experts)).astype(float)
    initial_sha = _parameter_sha(weights)
    moment = np.zeros_like(weights)
    velocity = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    best_weights = weights.copy()
    best_loss = math.inf
    stale = 0
    steps = 0
    for step in range(1, config.max_steps + 1):
        if len(train_matrix) <= config.batch_size:
            indices = np.arange(len(train_matrix))
        else:
            indices = rng.choice(len(train_matrix), config.batch_size, replace=False)
        matrix = train_matrix[indices]
        target = target_train[indices]
        probability = _sigmoid(offset_train[indices] + matrix @ weights)
        gradient = matrix.T @ (probability - target) / max(len(indices), 1)
        gradient += 2.0 * config.l2 * weights
        moment = beta1 * moment + (1.0 - beta1) * gradient
        velocity = beta2 * velocity + (1.0 - beta2) * np.square(gradient)
        corrected_moment = moment / (1.0 - beta1**step)
        corrected_velocity = velocity / (1.0 - beta2**step)
        weights -= config.learning_rate * corrected_moment / (
            np.sqrt(corrected_velocity) + 1.0e-8
        )
        weights = np.maximum(weights, 0.0)
        steps = step
        if step % config.evaluation_interval != 0 and step != config.max_steps:
            continue
        validation_probability = _sigmoid(offset_validation + validation_matrix @ weights)
        validation_loss = _logloss(
            target_validation, validation_probability, config.clip_probability
        ) + config.l2 * float(np.square(weights).sum())
        if validation_loss + config.tolerance < best_loss:
            best_loss = validation_loss
            best_weights = weights.copy()
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    final_sha = _parameter_sha(best_weights)
    if steps < 1:
        raise MultimodalFusionError("Fusion optimiser performed no steps")
    return ResidualFusionModel(
        endpoint=endpoint,
        expert_ids=experts,
        centres=tuple(map(float, centres)),
        weights=tuple(map(float, best_weights)),
        seed=int(seed),
        initial_parameter_sha256=initial_sha,
        final_parameter_sha256=final_sha,
        optimiser_steps=int(steps),
        validation_logloss=float(best_loss),
        clip_probability=float(config.clip_probability),
    )


def apply_residual_fusion_model(
    model: ResidualFusionModel,
    frame: pd.DataFrame,
    *,
    include_private_target: bool = False,
) -> pd.DataFrame:
    """Apply one model and emit explicit per-expert logit contributions."""

    experts = tuple(model.expert_ids)
    centres = np.asarray(model.centres, dtype=float)
    weights = np.asarray(model.weights, dtype=float)
    if len(experts) != len(centres) or len(experts) != len(weights):
        raise MultimodalFusionError("Fusion checkpoint dimensions are inconsistent")
    if (weights < 0).any() or not np.isfinite(weights).all():
        raise MultimodalFusionError("Fusion checkpoint has invalid weights")
    primary = _probability(
        frame.primary_probability, "fusion apply primary_probability", allow_null=False
    )
    matrix, masks, _ = _expert_matrix(
        frame, experts, centres, model.clip_probability
    )
    contributions = matrix * weights[None, :]
    total_contribution = contributions.sum(axis=1)
    adjusted = _sigmoid(
        _logit(primary, model.clip_probability) + total_contribution
    )
    auxiliary_count = masks.sum(axis=1).astype(np.int16)
    no_auxiliary = auxiliary_count == 0
    zero_total_contribution = total_contribution == 0.0
    # Avoid a numerically unnecessary sigmoid(logit(primary)) round trip.
    # This is also the correct fail-closed behaviour when an available expert
    # has learned a zero weight: the head remains visible, but it cannot alter
    # the secondary score by ~1e-16 merely because it was available.
    adjusted[zero_total_contribution] = primary[zero_total_contribution]
    score_column = {
        "discovery": "discovery_adjusted_probability",
        "confidence": "fused_confidence_probability",
        "actionability": "drug_actionability_probability",
    }[model.endpoint]
    output_columns = list(TARGET_KEYS)
    if include_private_target:
        output_columns += [
            column
            for column in ("fusion_target", FUSION_FOLD_COLUMN)
            if column in frame
        ]
    output = frame[output_columns].copy()
    output["primary_probability"] = primary
    output[score_column] = adjusted
    output["available_auxiliary_expert_count"] = auxiliary_count
    output["no_available_auxiliary_expert"] = no_auxiliary
    output["zero_total_contribution"] = zero_total_contribution
    output["primary_fallback"] = zero_total_contribution
    for index, expert in enumerate(experts):
        available = masks[:, index].astype(bool)
        output[f"{expert}_available"] = available
        contribution = contributions[:, index]
        output[f"{expert}_logit_contribution"] = np.where(
            available, contribution, np.nan
        )
        output[f"{expert}_fusion_weight"] = float(weights[index])
    output["analysis_version"] = ANALYSIS_VERSION
    output["module_id"] = MODULE_ID
    output["prediction_format"] = PREDICTION_FORMAT
    output["primary_ranking_unchanged"] = True
    output["adjusted_ranking_is_secondary"] = True
    output["used_for_primary_release"] = False
    output["old_checkpoint_loaded"] = False
    output["old_predictions_used_as_features"] = False
    if not np.isfinite(output[score_column].to_numpy(float)).all():
        raise MultimodalFusionError("Fusion produced a non-finite adjusted probability")
    if not output[score_column].between(0, 1).all():
        raise MultimodalFusionError("Fusion produced an adjusted probability outside [0, 1]")
    if not np.array_equal(
        output.loc[zero_total_contribution, score_column].to_numpy(float),
        output.loc[zero_total_contribution, "primary_probability"].to_numpy(float),
    ):
        raise MultimodalFusionError(
            "Zero-total-contribution rows do not exactly fall back to primary"
        )
    return output


@dataclass
class CrossfitFusionResult:
    endpoint: str
    expert_specs: tuple[ExpertSpec, ...]
    oof_prediction: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_models: tuple[ResidualFusionModel, ...]
    final_model: ResidualFusionModel
    performance_outcome: str
    scientific_status: str


def train_pair_blocked_crossfit_fusion(
    frame: pd.DataFrame,
    specs: Sequence[ExpertSpec],
    *,
    endpoint: str,
    config: ResidualFusionConfig | None = None,
) -> CrossfitFusionResult:
    """Train five pair-blocked outer models plus one final fresh model."""

    settings = config or ResidualFusionConfig()
    settings.validate()
    selected = tuple(spec for spec in specs if endpoint in spec.endpoint_roles)
    for spec in selected:
        spec.validate()
    if not selected:
        raise MultimodalFusionError(f"No experts are declared for {endpoint} fusion")
    expert_ids = tuple(spec.expert_id for spec in selected)
    for expert in expert_ids:
        required = {f"{expert}_probability", f"{expert}_available"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise MultimodalFusionError(f"Fusion frame lacks {expert}: {missing}")
    if set(pd.unique(frame[FUSION_FOLD_COLUMN])) != set(range(N_FOLDS)):
        raise MultimodalFusionError("Fusion pair folds do not cover 0..4")

    oof_parts: list[pd.DataFrame] = []
    models: list[ResidualFusionModel] = []
    metric_rows: list[dict[str, Any]] = []
    for heldout in range(N_FOLDS):
        validation_fold = (heldout + 1) % N_FOLDS
        train = frame.loc[
            ~frame[FUSION_FOLD_COLUMN].isin([heldout, validation_fold])
        ]
        validation = frame.loc[frame[FUSION_FOLD_COLUMN].eq(validation_fold)]
        test = frame.loc[frame[FUSION_FOLD_COLUMN].eq(heldout)]
        model = fit_residual_fusion_model(
            train,
            validation,
            expert_ids,
            endpoint=endpoint,
            config=settings,
            seed=settings.seed + heldout,
        )
        prediction = apply_residual_fusion_model(
            model, test, include_private_target=True
        )
        score_column = {
            "discovery": "discovery_adjusted_probability",
            "confidence": "fused_confidence_probability",
            "actionability": "drug_actionability_probability",
        }[endpoint]
        target = prediction.fusion_target.to_numpy(float)
        primary = prediction.primary_probability.to_numpy(float)
        adjusted = prediction[score_column].to_numpy(float)
        metric_rows.append(
            {
                "endpoint": endpoint,
                "heldout_pair_fold": f"PAIR_FOLD_{heldout}",
                "validation_pair_fold": f"PAIR_FOLD_{validation_fold}",
                "test_rows": int(len(prediction)),
                "primary_logloss": _logloss(target, primary, settings.clip_probability),
                "adjusted_logloss": _logloss(target, adjusted, settings.clip_probability),
                "primary_brier": _brier(target, primary),
                "adjusted_brier": _brier(target, adjusted),
                "initial_parameter_sha256": model.initial_parameter_sha256,
                "final_parameter_sha256": model.final_parameter_sha256,
                "optimiser_steps": model.optimiser_steps,
            }
        )
        oof_parts.append(prediction)
        models.append(model)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        list(TARGET_KEYS), kind="stable"
    ).reset_index(drop=True)
    if len(oof) != len(frame) or oof[list(TARGET_KEYS)].duplicated().any():
        raise MultimodalFusionError("Crossfit fusion did not predict every exact candidate once")
    target = oof.fusion_target.to_numpy(float)
    score_column = {
        "discovery": "discovery_adjusted_probability",
        "confidence": "fused_confidence_probability",
        "actionability": "drug_actionability_probability",
    }[endpoint]
    primary_loss = _logloss(
        target, oof.primary_probability.to_numpy(float), settings.clip_probability
    )
    adjusted_loss = _logloss(
        target, oof[score_column].to_numpy(float), settings.clip_probability
    )
    outcome = (
        "INCREMENT"
        if primary_loss - adjusted_loss >= settings.increment_logloss_delta
        else "NO_INCREMENT"
    )
    status = "SECONDARY_VALIDATED" if outcome == "INCREMENT" else "DIAGNOSTIC_ONLY"

    final_validation_fold = N_FOLDS - 1
    final_model = fit_residual_fusion_model(
        frame.loc[~frame[FUSION_FOLD_COLUMN].eq(final_validation_fold)],
        frame.loc[frame[FUSION_FOLD_COLUMN].eq(final_validation_fold)],
        expert_ids,
        endpoint=endpoint,
        config=settings,
        seed=settings.seed + 10_000,
    )
    metrics = pd.DataFrame(metric_rows)
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {
                        "endpoint": endpoint,
                        "heldout_pair_fold": "ALL_OOF",
                        "validation_pair_fold": "NOT_APPLICABLE",
                        "test_rows": int(len(oof)),
                        "primary_logloss": primary_loss,
                        "adjusted_logloss": adjusted_loss,
                        "primary_brier": _brier(
                            target, oof.primary_probability.to_numpy(float)
                        ),
                        "adjusted_brier": _brier(
                            target, oof[score_column].to_numpy(float)
                        ),
                        "performance_outcome": outcome,
                        "scientific_status": status,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return CrossfitFusionResult(
        endpoint=endpoint,
        expert_specs=selected,
        oof_prediction=oof,
        fold_metrics=metrics,
        fold_models=tuple(models),
        final_model=final_model,
        performance_outcome=outcome,
        scientific_status=status,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _artifact_declaration(path: Path, *, public: bool) -> dict[str, Any]:
    rows = None
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            rows = int(pq.ParquetFile(path).metadata.num_rows)
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            raise MultimodalFusionError(f"Cannot inspect fusion parquet: {path}") from exc
    return {
        "path": str(path.resolve()),
        "sha256": artifact_sha256(path),
        "bytes": int(path.stat().st_size),
        "rows": rows,
        "public": bool(public),
    }


def _endpoint_public(
    result: CrossfitFusionResult,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    applied = apply_residual_fusion_model(result.final_model, frame)
    score = {
        "discovery": "discovery_adjusted_probability",
        "confidence": "fused_confidence_probability",
        "actionability": "drug_actionability_probability",
    }[result.endpoint]
    keep = list(TARGET_KEYS) + [score]
    rename: dict[str, str] = {}
    for column in applied.columns:
        if column in keep or column in TARGET_KEYS or column == "primary_probability":
            continue
        if column in {
            "analysis_version",
            "module_id",
            "prediction_format",
            "primary_ranking_unchanged",
            "adjusted_ranking_is_secondary",
            "used_for_primary_release",
            "old_checkpoint_loaded",
            "old_predictions_used_as_features",
        }:
            continue
        keep.append(column)
        rename[column] = f"{result.endpoint}_{column}"
    return applied[list(TARGET_KEYS) + ["primary_probability"] + keep[len(TARGET_KEYS):]].rename(
        columns=rename
    )


def materialize_multimodal_fusion_release(
    output_dir: str | Path,
    fusion_frame: pd.DataFrame,
    discovery: CrossfitFusionResult,
    confidence: CrossfitFusionResult,
    *,
    source_declarations: Sequence[Mapping[str, Any]],
    training_run_id: str,
    input_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an immutable, hash-bound secondary fusion release.

    OOF labels and OOF predictions are explicitly private.  Only final
    candidate scores, weights, contributions, availability, metrics and
    lineage are public; the primary exact-pathway probability is preserved.
    """

    if discovery.endpoint != "discovery" or confidence.endpoint != "confidence":
        raise MultimodalFusionError("Release requires discovery and confidence fits")
    if not re.fullmatch(r"V32-FUSION-[A-Z0-9_-]{8,96}", str(training_run_id)):
        raise MultimodalFusionError("Fusion training_run_id is invalid")
    validate_source_declarations(source_declarations)
    if input_audit is not None:
        if (
            input_audit.get("status") != "PASS"
            or input_audit.get("analysis_version") != ANALYSIS_VERSION
            or int(input_audit.get("candidate_rows", -1)) != len(fusion_frame)
            or input_audit.get("all_five_primary_folds_present") is not True
            or input_audit.get("all_experts_full_candidate_universe") is not True
            or input_audit.get("all_unavailable_probabilities_null") is not True
        ):
            raise MultimodalFusionError("Formal fusion input audit is incomplete")
    verified_sources: list[dict[str, Any]] = []
    for declaration in source_declarations:
        item = dict(declaration)
        source = Path(str(item["path"])).resolve()
        if not source.is_file() or source.is_symlink():
            raise MultimodalFusionError(f"Fusion source is missing or unsafe: {source}")
        observed = artifact_sha256(source)
        if observed != str(item["sha256"]).lower():
            raise MultimodalFusionError(f"Fusion source SHA drift: {source}")
        item["path"] = str(source)
        verified_sources.append(item)

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise MultimodalFusionError(f"Refusing to overwrite non-empty fusion output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    discovery_public = _endpoint_public(discovery, fusion_frame)
    confidence_public = _endpoint_public(confidence, fusion_frame)
    confidence_columns = [
        column
        for column in confidence_public.columns
        if column not in {*TARGET_KEYS, "primary_probability"}
    ]
    public = discovery_public.merge(
        confidence_public[list(TARGET_KEYS) + confidence_columns],
        on=list(TARGET_KEYS),
        how="inner",
        validate="one_to_one",
    )
    if len(public) != len(fusion_frame):
        raise MultimodalFusionError("Fusion public output lost exact candidates")
    native_experts = tuple(
        dict.fromkeys(
            (*discovery.final_model.expert_ids, *confidence.final_model.expert_ids)
        )
    )
    native = fusion_frame[list(TARGET_KEYS)].copy()
    for expert in native_experts:
        probability_column = f"{expert}_probability"
        availability_column = f"{expert}_available"
        if probability_column not in fusion_frame or availability_column not in fusion_frame:
            raise MultimodalFusionError(
                f"Fusion frame lacks native public columns for {expert}"
            )
        available = fusion_frame[availability_column].astype(bool).to_numpy()
        probability = pd.to_numeric(
            fusion_frame[probability_column], errors="coerce"
        ).to_numpy(float)
        if (
            (available & (~np.isfinite(probability))).any()
            or ((~available) & np.isfinite(probability)).any()
        ):
            raise MultimodalFusionError(
                f"Native public expert {expert} violates typed-null semantics"
            )
        native[f"{expert}_native_probability"] = np.where(
            available, probability, np.nan
        )
        native[f"{expert}_native_available"] = available
    public = public.merge(
        native,
        on=list(TARGET_KEYS),
        how="inner",
        validate="one_to_one",
    )
    if len(public) != len(fusion_frame):
        raise MultimodalFusionError("Native expert publication lost exact candidates")
    public["discovery_rank_within_cancer_lncrna"] = (
        public.groupby(["cancer_id", "lncrna_id"], observed=True)
        .discovery_adjusted_probability.rank(method="first", ascending=False)
        .astype("int32")
    )
    public["confidence_rank_within_cancer_lncrna"] = (
        public.groupby(["cancer_id", "lncrna_id"], observed=True)
        .fused_confidence_probability.rank(method="first", ascending=False)
        .astype("int32")
    )
    public["analysis_version"] = ANALYSIS_VERSION
    public["training_run_id"] = str(training_run_id)
    public["module_id"] = MODULE_ID
    public["prediction_format"] = PREDICTION_FORMAT
    public["primary_ranking_unchanged"] = True
    public["adjusted_ranking_is_secondary"] = True
    public["used_for_primary_release"] = False
    public["historical_predictions_used"] = False
    public["historical_rankings_used"] = False
    forbidden_public = {"fusion_target", LABEL_COLUMN, "test_membership_label"}
    if forbidden_public.intersection(public.columns):
        raise MultimodalFusionError("Fusion public output leaks private target labels")

    prediction_path = output / "multimodal_secondary_scores.parquet"
    metrics_path = output / "crossfit_metrics.parquet"
    discovery_oof_path = output / "discovery_oof.PRIVATE.parquet"
    confidence_oof_path = output / "confidence_oof.PRIVATE.parquet"
    checkpoint_path = output / "FUSION_CHECKPOINT.json"
    sources_path = output / "SOURCE_INPUTS.json"
    input_audit_path = output / "FUSION_INPUT_AUDIT.json"
    lineage_path = output / "MODULE_LINEAGE.json"
    _atomic_parquet(public, prediction_path)
    metrics = pd.concat(
        [discovery.fold_metrics, confidence.fold_metrics], ignore_index=True
    )
    _atomic_parquet(metrics, metrics_path)
    _atomic_parquet(discovery.oof_prediction, discovery_oof_path)
    _atomic_parquet(confidence.oof_prediction, confidence_oof_path)
    checkpoint = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": str(training_run_id),
        "module_id": MODULE_ID,
        "discovery": discovery.final_model.to_dict(),
        "confidence": confidence.final_model.to_dict(),
        "fold_models": {
            "discovery": [model.to_dict() for model in discovery.fold_models],
            "confidence": [model.to_dict() for model in confidence.fold_models],
        },
        "all_parameters_newly_trained_v32": True,
        "old_checkpoint_loaded": False,
        "primary_probability_offset_fixed": True,
    }
    _atomic_json(checkpoint_path, checkpoint)
    _atomic_json(
        sources_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "sources": verified_sources,
            "historical_derived_inputs": [],
        },
    )
    if input_audit is not None:
        _atomic_json(input_audit_path, input_audit)
    lineage = {
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": str(training_run_id),
        "module_id": MODULE_ID,
        "status": "SUCCESS_SECONDARY_FUSION",
        "target_level": "exact_pathway",
        "split_unit": "CANCER_AGNOSTIC_LNCRNA_EXACT_PATHWAY_PAIR",
        "folds": N_FOLDS,
        "discovery_direct_target_evidence_used": False,
        "confidence_direct_target_evidence_allowed": True,
        "primary_score_preserved": True,
        "primary_ranking_unchanged": True,
        "adjusted_ranking_is_secondary": True,
        "discovery_performance_outcome": discovery.performance_outcome,
        "discovery_scientific_status": discovery.scientific_status,
        "confidence_performance_outcome": confidence.performance_outcome,
        "confidence_scientific_status": confidence.scientific_status,
        "no_increment_capability_retained": True,
        "missing_expert_policy": "AVAILABILITY_MASK_AND_EXACT_PRIMARY_FALLBACK",
        "zero_total_contribution_exact_primary_fallback": True,
        "native_expert_probabilities_public": True,
        "public_native_experts": list(native_experts),
        "unavailable_probability_fill": None,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "all_output_rows_generated_current_run": True,
        "release_ready": False,
        "production_deployed": False,
        "code_sha256": artifact_sha256(Path(__file__)),
    }
    if input_audit is not None:
        lineage["fusion_input_audit_sha256"] = artifact_sha256(input_audit_path)
    _atomic_json(lineage_path, lineage)

    artifacts = {
        "secondary_scores": _artifact_declaration(prediction_path, public=True),
        "crossfit_metrics": _artifact_declaration(metrics_path, public=True),
        "discovery_oof_private": _artifact_declaration(discovery_oof_path, public=False),
        "confidence_oof_private": _artifact_declaration(confidence_oof_path, public=False),
        "checkpoint": _artifact_declaration(checkpoint_path, public=False),
        "source_inputs": _artifact_declaration(sources_path, public=True),
        "module_lineage": _artifact_declaration(lineage_path, public=True),
    }
    if input_audit is not None:
        artifacts["fusion_input_audit"] = _artifact_declaration(
            input_audit_path, public=True
        )
    binding_path = output / "MULTIMODAL_FUSION_BINDING.json"
    binding = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": str(training_run_id),
        "module_id": MODULE_ID,
        "status": "SUCCESS_SECONDARY_FUSION",
        "primary_score_preserved": True,
        "primary_ranking_unchanged": True,
        "adjusted_ranking_is_secondary": True,
        "zero_total_contribution_exact_primary_fallback": True,
        "native_expert_probabilities_public": True,
        "public_native_experts": list(native_experts),
        "public_rows": int(len(public)),
        "artifacts": artifacts,
        "source_inputs": verified_sources,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(binding_path, binding)
    binding_sha = artifact_sha256(binding_path)
    success_path = output / "SUCCESS.json"
    _atomic_json(
        success_path,
        {
            "status": "SUCCESS_SECONDARY_FUSION",
            "analysis_version": ANALYSIS_VERSION,
            "training_run_id": str(training_run_id),
            "binding": binding_path.name,
            "binding_sha256": binding_sha,
            "public_rows": int(len(public)),
            "primary_ranking_unchanged": True,
            "release_ready": False,
            "production_deployed": False,
        },
    )
    return {
        "output_dir": str(output),
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "prediction_path": str(prediction_path),
        "public_rows": int(len(public)),
        "discovery_performance_outcome": discovery.performance_outcome,
        "confidence_performance_outcome": confidence.performance_outcome,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "CHECKPOINT_FORMAT",
    "CrossfitFusionResult",
    "ExpertSpec",
    "MODULE_ID",
    "MultimodalFusionError",
    "PREDICTION_FORMAT",
    "ResidualFusionConfig",
    "ResidualFusionModel",
    "TARGET_KEYS",
    "apply_residual_fusion_model",
    "artifact_sha256",
    "build_pair_aggregated_fusion_frame",
    "build_pair_aggregated_fusion_frame_from_parquet",
    "fit_residual_fusion_model",
    "materialize_multimodal_fusion_release",
    "pair_blocked_fold",
    "train_pair_blocked_crossfit_fusion",
    "validate_source_declarations",
]
