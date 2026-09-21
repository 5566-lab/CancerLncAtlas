"""Fresh V3.2 structural lncRNA--pathway--target--drug mechanisms.

This companion binds current V3.2 Drug predictions to curated drug targets
through current exact-pathway membership.  A path is a transparent structural
hypothesis, not a learned attribution and not a causal mechanism claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MECHANISM_FORMAT = "CC_HHGT_V3_2_LNCRNA_EXACT_PATHWAY_TARGET_DRUG_MECHANISM_V1"
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_MEMBERSHIP_SHA256 = (
    "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
)
PROBABILITY_COLUMN = "drug_response_association_probability"
_TARGET_KEYS = ["cancer_id", "lncrna_id", "drug_id"]
_FORBIDDEN_PUBLIC_COLUMNS = {
    "label", "association_proxy_label", "held_out_rho", "held_out_n_cell_lines",
    "sample_weight", "replication_status", "fdr", "p_value",
}
_FORBIDDEN_SOURCE_TOKENS = (
    "v2_", "v3_0", "v3_1", "legacy", "historical", "old_prediction",
    "drug_response_replication", "web_table", "checkpoint",
)


class DrugMechanismError(RuntimeError):
    """Raised when a structural mechanism release is not trustworthy."""


def _stable_gene(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"^(?:GENE|PROTEIN):", "", text)
    return re.sub(r"\.\d+$", "", text)


def _path_id(cancer: str, lnc: str, pathway: str, gene: str, drug: str) -> str:
    digest = hashlib.sha256(
        "\0".join((cancer, lnc, pathway, gene, drug)).encode("utf-8")
    ).hexdigest()
    return "DRUGMECH32:" + digest[:24]


def _assert_source(path: str | Path, role: str, *, current_prediction: bool = False) -> Path:
    source = Path(path).resolve()
    if not source.exists() or (not source.is_file() and not source.is_dir()):
        raise FileNotFoundError(source)
    token = re.sub(r"[^a-z0-9]+", "_", source.name.lower())
    found = [item for item in _FORBIDDEN_SOURCE_TOKENS if item in token]
    if found:
        raise DrugMechanismError(f"{role} path looks historical/derived: {source}; {found}")
    if source.is_file() and source.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl"}:
        raise DrugMechanismError(f"{role} cannot be a checkpoint: {source}")
    if current_prediction and "v32" not in token and source.is_file():
        # Formal directory inputs are validated by their contents and hashes;
        # a single ad-hoc file must also self-identify as V3.2 in its name.
        raise DrugMechanismError("A standalone Drug prediction file must identify V3.2")
    return source


def _normalise_targets(targets: pd.DataFrame) -> pd.DataFrame:
    required = {"drug_id", "gene_id"}
    if missing := sorted(required - set(targets.columns)):
        raise DrugMechanismError(f"Drug targets lack columns: {missing}")
    result = targets.copy()
    result["drug_id"] = result.drug_id.astype(str).str.strip()
    result["target_gene_id"] = result.gene_id.map(_stable_gene)
    for source, target in (
        ("target_gene_symbol", "target_gene_symbol"),
        ("drug_name", "drug_name"),
        ("mapping_method", "target_mapping_method"),
    ):
        result[target] = (
            result[source].fillna("").astype(str).str.strip()
            if source in result
            else ""
        )
    result = result.loc[result.drug_id.ne("") & result.target_gene_id.ne("")]
    result = result[
        ["drug_id", "target_gene_id", "target_gene_symbol", "drug_name", "target_mapping_method"]
    ].drop_duplicates()
    if result.empty:
        raise DrugMechanismError("No curated drug targets retain canonical IDs")

    # A single DrugCentral target can arrive through several equivalent HGNC
    # aliases/mapping routes.  Collapse those routes deterministically so that
    # a biological drug--target pair cannot create several mechanism rows whose
    # only difference is annotation order.
    def first_nonempty(values: pd.Series) -> str:
        unique = sorted({str(value).strip() for value in values if str(value).strip()})
        return unique[0] if unique else ""

    def joined_nonempty(values: pd.Series) -> str:
        return "|".join(sorted({str(value).strip() for value in values if str(value).strip()}))

    return (
        result.groupby(["drug_id", "target_gene_id"], as_index=False, observed=True)
        .agg(
            target_gene_symbol=("target_gene_symbol", first_nonempty),
            drug_name=("drug_name", first_nonempty),
            target_mapping_method=("target_mapping_method", joined_nonempty),
        )
        .sort_values(["drug_id", "target_gene_id"], kind="stable")
        .reset_index(drop=True)
    )


def _normalise_membership(membership: pd.DataFrame) -> pd.DataFrame:
    required = {"pathway_id", "gene_id"}
    if missing := sorted(required - set(membership.columns)):
        raise DrugMechanismError(f"Exact membership lacks columns: {missing}")
    result = membership[["pathway_id", "gene_id"]].dropna().copy()
    result["pathway_id"] = result.pathway_id.astype(str).str.strip()
    result["target_gene_id"] = result.gene_id.map(_stable_gene)
    result = result.loc[result.pathway_id.ne("") & result.target_gene_id.ne("")]
    return result[["pathway_id", "target_gene_id"]].drop_duplicates()


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {*_TARGET_KEYS, PROBABILITY_COLUMN, "availability", "analysis_version", "training_run_id"}
    if missing := sorted(required - set(predictions.columns)):
        raise DrugMechanismError(f"Current Drug predictions lack columns: {missing}")
    forbidden = sorted(_FORBIDDEN_PUBLIC_COLUMNS & set(predictions.columns))
    if forbidden:
        raise DrugMechanismError(f"Private labels/statistics entered public predictions: {forbidden}")
    result = predictions.copy()
    result = result.loc[result.availability.fillna(False).astype(bool)].copy()
    if result.empty:
        raise DrugMechanismError("Current Drug predictions contain no available rows")
    if set(result.analysis_version.astype(str)) != {ANALYSIS_VERSION}:
        raise DrugMechanismError("Drug predictions are not current V3.2")
    if result[_TARGET_KEYS].isna().any().any() or result.duplicated(_TARGET_KEYS).any():
        raise DrugMechanismError("Available Drug prediction keys are null or duplicated")
    result[PROBABILITY_COLUMN] = pd.to_numeric(result[PROBABILITY_COLUMN], errors="coerce")
    if result[PROBABILITY_COLUMN].isna().any() or not result[PROBABILITY_COLUMN].between(0, 1).all():
        raise DrugMechanismError("Drug probabilities must be finite within 0..1")
    if result.training_run_id.astype(str).str.strip().eq("").any():
        raise DrugMechanismError("Drug prediction lacks training_run_id")
    return result


def build_structural_mechanisms(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    membership: pd.DataFrame,
    drug_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Build exact target-mediated paths in memory for fixtures/small releases."""

    prediction = _validate_predictions(predictions)
    required_candidate = {"cancer_id", "lncrna_id", "pathway_id"}
    if missing := sorted(required_candidate - set(candidates.columns)):
        raise DrugMechanismError(f"Exact candidates lack columns: {missing}")
    candidate = candidates[list(required_candidate)].dropna().drop_duplicates().copy()
    member = _normalise_membership(membership)
    target = _normalise_targets(drug_targets)
    paths = (
        prediction.merge(candidate, on=["cancer_id", "lncrna_id"], how="inner", validate="many_to_many")
        .merge(target, on="drug_id", how="inner", validate="many_to_many")
        .merge(member, on=["pathway_id", "target_gene_id"], how="inner", validate="many_to_many")
    )
    if paths.empty:
        raise DrugMechanismError("No V3.2 Drug prediction has a curated exact-pathway target route")
    paths = paths.drop_duplicates(
        ["cancer_id", "lncrna_id", "pathway_id", "target_gene_id", "drug_id"]
    )
    paths["mechanism_path_id"] = [
        _path_id(*row)
        for row in paths[
            ["cancer_id", "lncrna_id", "pathway_id", "target_gene_id", "drug_id"]
        ].itertuples(index=False, name=None)
    ]
    paths["mechanism_type"] = "CURATED_DRUG_TARGET_MEMBER_OF_CURRENT_EXACT_PATHWAY"
    paths["mechanism_semantics"] = "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION"
    paths["causal_mechanism_claimed"] = False
    paths["target_contribution_claimed"] = False
    paths["family_to_exact_broadcast"] = False
    paths["analysis_version"] = ANALYSIS_VERSION
    paths["generation"] = "V3.2_FRESH_STRUCTURAL_MECHANISM_BINDING"
    columns = [
        "mechanism_path_id", "cancer_id", "lncrna_id", "pathway_id",
        "target_gene_id", "target_gene_symbol", "drug_id", "drug_name",
        PROBABILITY_COLUMN, "training_run_id", "target_mapping_method",
        "mechanism_type", "mechanism_semantics", "causal_mechanism_claimed",
        "target_contribution_claimed", "family_to_exact_broadcast",
        "analysis_version", "generation",
    ]
    optional = [
        column for column in ("prediction_fold_mask", "cell_line_folds_with_prediction")
        if column in paths
    ]
    return paths[columns + optional].sort_values(
        ["cancer_id", "lncrna_id", "drug_id", "pathway_id", "target_gene_id"],
        kind="stable",
    ).reset_index(drop=True)


def _parquet_relation(path: Path) -> str:
    value = path.resolve().as_posix().replace("'", "''")
    if path.is_dir():
        value = value.rstrip("/") + "/**/*.parquet"
    return f"read_parquet('{value}', union_by_name=true)"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise DrugMechanismError(f"Formal Drug bundle lacks {role}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DrugMechanismError(f"Cannot parse formal Drug {role}: {path}") from exc
    if not isinstance(payload, dict):
        raise DrugMechanismError(f"Formal Drug {role} is not a JSON object")
    return payload


def _validate_formal_drug_bundle(
    prediction_source: Path,
    target_source: Path,
    *,
    prediction_sha256: str,
    target_sha256: str,
) -> dict[str, Any]:
    """Bind mechanisms to one successful fresh streaming Drug run."""

    if not prediction_source.is_dir():
        raise DrugMechanismError("Formal Drug predictions must be the sparse prediction directory")
    bundle = prediction_source.parent.resolve()
    success_path = bundle / "SUCCESS.json"
    lineage_path = bundle / "MODULE_LINEAGE.json"
    success = _read_json_object(success_path, "SUCCESS.json")
    lineage = _read_json_object(lineage_path, "MODULE_LINEAGE.json")
    for name, payload in (("SUCCESS", success), ("MODULE_LINEAGE", lineage)):
        if payload.get("analysis_version") != ANALYSIS_VERSION:
            raise DrugMechanismError(f"Formal Drug {name} is not current V3.2")
        if payload.get("module_id") != "drug":
            raise DrugMechanismError(f"Formal Drug {name} has the wrong module_id")
        if str(payload.get("training_run_id", "")).strip() == "":
            raise DrugMechanismError(f"Formal Drug {name} lacks training_run_id")
    if success.get("status") != "SUCCESS" or success.get("release_ready") is not True:
        raise DrugMechanismError("Formal Drug SUCCESS is not release-ready")
    if lineage.get("training_status") != "SUCCESS" or lineage.get("release_ready") is not True:
        raise DrugMechanismError("Formal Drug lineage is not release-ready")
    if success["training_run_id"] != lineage["training_run_id"]:
        raise DrugMechanismError("Formal Drug SUCCESS/lineage training_run_id mismatch")
    if Path(str(success.get("prediction_path", ""))).resolve() != prediction_source:
        raise DrugMechanismError("Formal Drug SUCCESS does not bind the supplied prediction root")
    if success.get("prediction_sha256") != prediction_sha256:
        raise DrugMechanismError("Formal Drug SUCCESS prediction SHA mismatch")
    if lineage.get("prediction_sha256") != prediction_sha256:
        raise DrugMechanismError("Formal Drug lineage prediction SHA mismatch")
    if success.get("lineage_sha256") != artifact_sha256(lineage_path):
        raise DrugMechanismError("Formal Drug lineage SHA mismatch")
    required_truths = (
        "private_head_trained_from_scratch",
        "all_five_folds_have_optimizer_updates",
        "all_non_null_predictions_newly_trained_v32",
        "available_keys_unique",
        "release_ready",
    )
    if any(lineage.get(key) is not True for key in required_truths):
        raise DrugMechanismError("Formal Drug lineage lacks fresh-training/release assertions")
    required_falses = (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
        "old_gdsc_prism_association_tables_used",
        "partial_not_publishable",
    )
    if any(lineage.get(key) is not False for key in required_falses):
        raise DrugMechanismError("Formal Drug lineage permits historical or partial results")

    inputs = lineage.get("input_artifacts")
    if not isinstance(inputs, list):
        raise DrugMechanismError("Formal Drug lineage lacks input_artifacts")
    target_matches = [
        row for row in inputs
        if isinstance(row, dict)
        and Path(str(row.get("path", ""))).resolve() == target_source
    ]
    if len(target_matches) != 1 or target_matches[0].get("sha256") != target_sha256:
        raise DrugMechanismError("Curated drug-target input is not hash-bound by the formal Drug run")
    return {
        "bundle_root": str(bundle),
        "success_path": str(success_path),
        "success_sha256": artifact_sha256(success_path),
        "lineage_path": str(lineage_path),
        "lineage_sha256": artifact_sha256(lineage_path),
        "training_run_id": str(success["training_run_id"]),
    }


def materialize_drug_mechanisms(
    *,
    predictions_path: str | Path,
    candidates_path: str | Path,
    membership_path: str | Path,
    drug_targets_path: str | Path,
    output_root: str | Path,
    strict_formal_authority: bool = True,
) -> dict[str, Any]:
    """Out-of-core materialisation for a current formal V3.2 Drug run."""

    prediction_source = _assert_source(predictions_path, "current Drug predictions")
    candidate_source = _assert_source(candidates_path, "current exact candidates")
    membership_source = _assert_source(membership_path, "current exact membership")
    target_source = _assert_source(drug_targets_path, "curated drug targets")
    source_hashes = {
        "predictions": artifact_sha256(prediction_source),
        "candidates": artifact_sha256(candidate_source),
        "membership": artifact_sha256(membership_source),
        "targets": artifact_sha256(target_source),
    }
    formal_binding: dict[str, Any] | None = None
    if strict_formal_authority:
        if source_hashes["candidates"] != FORMAL_CANDIDATE_SHA256:
            raise DrugMechanismError("Candidate SHA is not current formal V3.2 authority")
        if source_hashes["membership"] != FORMAL_MEMBERSHIP_SHA256:
            raise DrugMechanismError("Membership SHA is not current formal V3.2 authority")
        formal_binding = _validate_formal_drug_bundle(
            prediction_source,
            target_source,
            prediction_sha256=source_hashes["predictions"],
            target_sha256=source_hashes["targets"],
        )
    output = Path(output_root).resolve()
    if output.exists():
        raise DrugMechanismError(f"Drug mechanism output reuse is forbidden: {output}")
    output.mkdir(parents=True)
    try:
        # Small/static annotations are canonicalised once; candidates and
        # predictions stay in DuckDB-backed parquet scans.
        membership = _normalise_membership(
            pd.read_parquet(membership_source, columns=["pathway_id", "gene_id"])
        )
        targets = _normalise_targets(pd.read_parquet(target_source))
        membership_cache = output / ".membership.normalized.parquet"
        target_cache = output / ".targets.normalized.parquet"
        membership.to_parquet(membership_cache, index=False, compression="zstd")
        targets.to_parquet(target_cache, index=False, compression="zstd")
        try:
            import duckdb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise DrugMechanismError("duckdb is required for mechanism materialisation") from exc
        con = duckdb.connect(":memory:")
        mechanism_path = output / "lncrna_exact_pathway_target_drug_mechanisms.parquet"
        try:
            prediction = _parquet_relation(prediction_source)
            candidate = _parquet_relation(candidate_source)
            member = _parquet_relation(membership_cache)
            target = _parquet_relation(target_cache)
            columns = {
                row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {prediction}").fetchall()
            }
            required = {*_TARGET_KEYS, PROBABILITY_COLUMN, "availability", "analysis_version", "training_run_id"}
            if missing := sorted(required - columns):
                raise DrugMechanismError(f"Current Drug predictions lack columns: {missing}")
            if forbidden := sorted(_FORBIDDEN_PUBLIC_COLUMNS & columns):
                raise DrugMechanismError(f"Private columns entered public predictions: {forbidden}")
            candidate_columns = {
                row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {candidate}").fetchall()
            }
            required_candidate = {"cancer_id", "lncrna_id", "pathway_id"}
            if missing := sorted(required_candidate - candidate_columns):
                raise DrugMechanismError(f"Exact candidates lack columns: {missing}")

            prediction_audit = con.execute(
                f"""
                SELECT
                  count(*) AS source_rows,
                  sum(CASE WHEN availability IS NOT TRUE THEN 1 ELSE 0 END) AS unavailable_rows,
                  sum(CASE WHEN analysis_version IS NULL OR analysis_version <> ? THEN 1 ELSE 0 END) AS wrong_version_rows,
                  sum(CASE WHEN training_run_id IS NULL OR trim(cast(training_run_id AS VARCHAR)) = '' THEN 1 ELSE 0 END) AS blank_run_rows,
                  sum(CASE WHEN cancer_id IS NULL OR trim(cast(cancer_id AS VARCHAR)) = ''
                                 OR lncrna_id IS NULL OR trim(cast(lncrna_id AS VARCHAR)) = ''
                                 OR drug_id IS NULL OR trim(cast(drug_id AS VARCHAR)) = ''
                           THEN 1 ELSE 0 END) AS bad_key_rows,
                  sum(CASE WHEN try_cast({PROBABILITY_COLUMN} AS DOUBLE) IS NULL
                                 OR NOT isfinite(try_cast({PROBABILITY_COLUMN} AS DOUBLE))
                                 OR try_cast({PROBABILITY_COLUMN} AS DOUBLE) NOT BETWEEN 0 AND 1
                           THEN 1 ELSE 0 END) AS bad_probability_rows,
                  count(DISTINCT training_run_id) AS training_runs
                FROM {prediction}
                """,
                [ANALYSIS_VERSION],
            ).fetchone()
            if not prediction_audit or int(prediction_audit[0]) <= 0:
                raise DrugMechanismError("Current Drug prediction root is empty")
            audit_names = (
                "unavailable_rows", "wrong_version_rows", "blank_run_rows",
                "bad_key_rows", "bad_probability_rows",
            )
            bad_prediction_counts = {
                name: int(prediction_audit[index + 1] or 0)
                for index, name in enumerate(audit_names)
            }
            if any(bad_prediction_counts.values()) or int(prediction_audit[6]) != 1:
                raise DrugMechanismError(
                    "Current Drug prediction rows failed the fresh sparse-public contract: "
                    f"{bad_prediction_counts}, training_runs={int(prediction_audit[6])}"
                )
            duplicate_predictions = int(con.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT cancer_id, lncrna_id, drug_id
                  FROM {prediction}
                  GROUP BY ALL HAVING count(*) <> 1
                )
                """
            ).fetchone()[0])
            if duplicate_predictions:
                raise DrugMechanismError(
                    f"Current Drug prediction keys are duplicated: {duplicate_predictions}"
                )
            candidate_audit = con.execute(
                f"""
                SELECT
                  count(*),
                  sum(CASE WHEN cancer_id IS NULL OR trim(cast(cancer_id AS VARCHAR)) = ''
                                 OR lncrna_id IS NULL OR trim(cast(lncrna_id AS VARCHAR)) = ''
                                 OR pathway_id IS NULL OR trim(cast(pathway_id AS VARCHAR)) = ''
                           THEN 1 ELSE 0 END)
                FROM {candidate}
                """
            ).fetchone()
            if not candidate_audit or int(candidate_audit[0]) <= 0 or int(candidate_audit[1] or 0):
                raise DrugMechanismError("Current exact candidates are empty or contain null/blank keys")
            duplicate_candidates = int(con.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT cancer_id, lncrna_id, pathway_id
                  FROM {candidate}
                  GROUP BY ALL HAVING count(*) <> 1
                )
                """
            ).fetchone()[0])
            if duplicate_candidates:
                raise DrugMechanismError(
                    f"Current exact candidate keys are duplicated: {duplicate_candidates}"
                )
            source_training_run_id = str(con.execute(
                f"SELECT min(cast(training_run_id AS VARCHAR)) FROM {prediction}"
            ).fetchone()[0])
            if formal_binding is not None and source_training_run_id != formal_binding["training_run_id"]:
                raise DrugMechanismError("Drug prediction rows do not match the formal bundle training_run_id")
            fold_mask = "p.prediction_fold_mask" if "prediction_fold_mask" in columns else "NULL::UTINYINT"
            fold_count = (
                "p.cell_line_folds_with_prediction"
                if "cell_line_folds_with_prediction" in columns
                else "NULL::INTEGER"
            )
            output_sql = mechanism_path.as_posix().replace("'", "''")
            con.execute(
                f"""
                COPY (
                  SELECT DISTINCT
                    'DRUGMECH32:' || substr(sha256(
                      concat_ws(chr(0), p.cancer_id, p.lncrna_id, c.pathway_id,
                                      t.target_gene_id, p.drug_id)
                    ), 1, 24) AS mechanism_path_id,
                    p.cancer_id, p.lncrna_id, c.pathway_id,
                    t.target_gene_id, t.target_gene_symbol,
                    p.drug_id, t.drug_name,
                    try_cast(p.{PROBABILITY_COLUMN} AS DOUBLE) AS {PROBABILITY_COLUMN},
                    p.training_run_id,
                    {fold_mask} AS prediction_fold_mask,
                    {fold_count} AS cell_line_folds_with_prediction,
                    t.target_mapping_method,
                    'CURATED_DRUG_TARGET_MEMBER_OF_CURRENT_EXACT_PATHWAY' AS mechanism_type,
                    'STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION' AS mechanism_semantics,
                    false AS causal_mechanism_claimed,
                    false AS target_contribution_claimed,
                    false AS family_to_exact_broadcast,
                    '{ANALYSIS_VERSION}' AS analysis_version,
                    'V3.2_FRESH_STRUCTURAL_MECHANISM_BINDING' AS generation
                  FROM {prediction} p
                  JOIN {candidate} c USING (cancer_id, lncrna_id)
                  JOIN {target} t USING (drug_id)
                  JOIN {member} m
                    ON m.pathway_id = c.pathway_id
                   AND m.target_gene_id = t.target_gene_id
                  WHERE p.availability
                    AND p.{PROBABILITY_COLUMN} BETWEEN 0 AND 1
                    AND p.analysis_version = '{ANALYSIS_VERSION}'
                ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            rows = int(con.execute(
                f"SELECT count(*) FROM {_parquet_relation(mechanism_path)}"
            ).fetchone()[0])
            invalid = int(con.execute(
                f"""
                SELECT count(*) FROM {_parquet_relation(mechanism_path)}
                WHERE causal_mechanism_claimed OR target_contribution_claimed
                   OR family_to_exact_broadcast OR analysis_version <> ?
                """,
                [ANALYSIS_VERSION],
            ).fetchone()[0])
            duplicate_paths = int(con.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT cancer_id, lncrna_id, pathway_id, target_gene_id, drug_id
                  FROM {_parquet_relation(mechanism_path)}
                  GROUP BY ALL HAVING count(*) <> 1
                )
                """
            ).fetchone()[0])
        finally:
            con.close()
        if rows <= 0 or invalid or duplicate_paths:
            raise DrugMechanismError(
                "Drug mechanism materialisation is empty/invalid: "
                f"rows={rows}, invalid={invalid}, duplicate_paths={duplicate_paths}"
            )
        membership_cache.unlink(missing_ok=True)
        target_cache.unlink(missing_ok=True)
        manifest = {
            "format": MECHANISM_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "drug",
            "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
            "release_ready": False,
            "production_deployed": False,
            "native_target_keys": list(_TARGET_KEYS),
            "actionability_separate_from_exact_pathway": True,
            "does_not_change_primary_pathway_ranking": True,
            "drug_response_association_probability_not_efficacy_or_direction": True,
            "signed_rho_private_only": True,
            "mechanism_semantics": "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION",
            "causal_mechanism_claimed": False,
            "target_contribution_claimed": False,
            "family_to_exact_broadcast": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "current_v32_drug_predictions_used_as_binding": True,
            "formal_drug_bundle_binding": formal_binding,
            "inputs": {
                "current_v32_drug_predictions": {"path": str(prediction_source), "sha256": source_hashes["predictions"]},
                "current_exact_candidates": {"path": str(candidate_source), "sha256": source_hashes["candidates"]},
                "current_exact_membership": {"path": str(membership_source), "sha256": source_hashes["membership"]},
                "curated_drug_targets": {"path": str(target_source), "sha256": source_hashes["targets"]},
            },
            "counts": {
                "source_available_predictions": int(prediction_audit[0]),
                "source_exact_candidates": int(candidate_audit[0]),
                "mechanism_paths": rows,
            },
            "artifact": {
                "path": mechanism_path.name,
                "sha256": artifact_sha256(mechanism_path),
                "rows": rows,
            },
        }
        _atomic_json(output / "DRUG_MECHANISM_MANIFEST.json", manifest)
        _atomic_json(
            output / "SUCCESS.json",
            {
                "status": manifest["status"],
                "release_ready": False,
                "manifest": "DRUG_MECHANISM_MANIFEST.json",
                "manifest_sha256": artifact_sha256(output / "DRUG_MECHANISM_MANIFEST.json"),
            },
        )
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = [
    "ANALYSIS_VERSION",
    "DrugMechanismError",
    "MECHANISM_FORMAT",
    "PROBABILITY_COLUMN",
    "build_structural_mechanisms",
    "materialize_drug_mechanisms",
]
