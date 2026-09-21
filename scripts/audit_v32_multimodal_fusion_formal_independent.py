"""Independent, read-only post-audit of the formal V3.2 multimodal fusion.

This auditor deliberately does not import the fusion implementation.  It
reconstructs keys, targets, folds, contributions, scores, and metrics from the
hash-bound source artifacts and the serialized checkpoint using DuckDB/NumPy.
It writes only to a new audit directory and never edits the audited release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
EXPECTED_BINDING_SHA256 = "11278387ff40f89e48458f4aec9520c664984f153f7d67a6c5a4cef2a9a728e1"
EXPECTED_ROWS = 3_300_000
EXPECTED_EXPERT_COUNTS = {
    "genomic": (747_408, 2_552_592),
    "single_cell": (954_541, 2_345_459),
    "evidence_transformer": (825_753, 2_474_247),
}
KEYS = ("cancer_id", "lncrna_id", "pathway_id")
CLIP = 1.0e-6
FORMULA_TOLERANCE = 2.0e-12
METRIC_TOLERANCE = 2.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parameter_sha(values: Iterable[float]) -> str:
    array = np.ascontiguousarray(list(values), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def initial_parameter_sha(seed: int, dimensions: int) -> str:
    rng = np.random.default_rng(int(seed))
    weights = rng.uniform(0.0025, 0.0125, size=int(dimensions)).astype(float)
    return parameter_sha(weights)


def deterministic_pair_fold(lncrna_id: object, pathway_id: object) -> int:
    token = f"{str(lncrna_id).strip()}|{str(pathway_id).strip()}|20260826"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % 5


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        passed: bool,
        requirement: str,
        observed: Any,
        expected: Any,
        *,
        severity: str = "ERROR",
    ) -> None:
        self.checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if bool(passed) else "FAIL",
                "severity": severity,
                "requirement": requirement,
                "observed": observed,
                "expected": expected,
            }
        )

    @property
    def failed(self) -> list[str]:
        return [str(item["check_id"]) for item in self.checks if item["status"] == "FAIL"]

    @property
    def passed_count(self) -> int:
        return sum(item["status"] == "PASS" for item in self.checks)


def parquet_rows(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(connection.execute(f"SELECT count(*) FROM read_parquet('{sql_path(path)}')").fetchone()[0])


def parquet_columns(connection: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    frame = connection.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql_path(path)}')").fetchdf()
    return [str(value) for value in frame["column_name"].tolist()]


def referenced_files(payloads: Iterable[tuple[Path, Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect file path/SHA pairs without trusting producer-specific schemas."""
    records: dict[str, dict[str, Any]] = {}
    skipped_directories: set[str] = set()

    def add(path_value: Any, sha_value: Any, owner: Path) -> None:
        if not isinstance(path_value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", str(sha_value or "")):
            return
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (owner.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.is_dir():
            skipped_directories.add(str(candidate))
            return
        records.setdefault(
            str(candidate),
            {"path": str(candidate), "declared_sha256": str(sha_value).lower(), "owners": []},
        )["owners"].append(str(owner))

    def walk(node: Any, owner: Path) -> None:
        if isinstance(node, Mapping):
            if "path" in node and "sha256" in node:
                add(node.get("path"), node.get("sha256"), owner)
            if "local_path" in node and "sha256" in node:
                add(node.get("local_path"), node.get("sha256"), owner)
            for key, value in node.items():
                if key.endswith("_path"):
                    sibling = f"{key[:-5]}_sha256"
                    if sibling in node:
                        add(value, node.get(sibling), owner)
                walk(value, owner)
        elif isinstance(node, list):
            for value in node:
                walk(value, owner)

    for owner, payload in payloads:
        walk(payload, owner)
    return list(records.values()), sorted(skipped_directories)


def logit_expression(expression: str) -> str:
    clipped = f"greatest(least(cast({expression} AS double), {1.0 - CLIP}), {CLIP})"
    return f"(ln({clipped}) - ln(1.0 - {clipped}))"


def case_by_fold(models: list[Mapping[str, Any]], values_key: str, expert_index: int) -> str:
    pieces = [
        f"WHEN {fold} THEN {float(model[values_key][expert_index]):.17g}"
        for fold, model in enumerate(models)
    ]
    return "CASE o.fusion_pair_fold " + " ".join(pieces) + " END"


def endpoint_formula_audit(
    connection: duckdb.DuckDBPyConnection,
    audit: Audit,
    checkpoint: Mapping[str, Any],
    endpoint: str,
    *,
    oof: bool,
) -> dict[str, Any]:
    model = checkpoint["fold_models"][endpoint] if oof else [checkpoint[endpoint]]
    experts = list(model[0]["expert_ids"])
    output_view = "doof" if endpoint == "discovery" else "coof"
    score_column = "discovery_adjusted_probability" if endpoint == "discovery" else "fused_confidence_probability"
    prefix = "" if oof else f"{endpoint}_"
    output_view = output_view if oof else "pub"
    source = {
        "genomic": ("g", "genomic_probability", "genomic_available"),
        "single_cell": ("s", "single_cell_probability", "single_cell_available"),
        "evidence_transformer": ("e", "evidence_probability", "evidence_available"),
    }
    joins = ["JOIN p USING(cancer_id, lncrna_id, pathway_id)"]
    for expert in experts:
        alias = source[expert][0]
        clause = f"JOIN {alias} USING(cancer_id, lncrna_id, pathway_id)"
        if clause not in joins:
            joins.append(clause)
    contribution_terms: list[str] = []
    availability_terms: list[str] = []
    expert_details: dict[str, Any] = {}
    conditions: list[str] = []
    for expert_index, expert in enumerate(experts):
        alias, probability, availability = source[expert]
        if oof:
            weight = case_by_fold(model, "weights", expert_index)
            centre = case_by_fold(model, "centres", expert_index)
        else:
            weight = f"{float(model[0]['weights'][expert_index]):.17g}"
            centre = f"{float(model[0]['centres'][expert_index]):.17g}"
        contribution = f"(({weight}) * ({logit_expression(f'{alias}.{probability}')} - ({centre})))"
        availability_output = f"o.{prefix}{expert}_available"
        contribution_output = f"o.{prefix}{expert}_logit_contribution"
        weight_output = f"o.{prefix}{expert}_fusion_weight"
        expected_available = f"{alias}.{availability}"
        conditions.extend(
            [
                f"count_if({availability_output} IS DISTINCT FROM {expected_available}) AS {expert}_availability_mismatch",
                f"count_if(NOT {expected_available} AND {contribution_output} IS NOT NULL) AS {expert}_typed_null_violation",
                f"count_if({expected_available} AND abs({contribution_output} - {contribution}) > {FORMULA_TOLERANCE}) AS {expert}_contribution_violation",
                f"max(CASE WHEN {expected_available} THEN abs({contribution_output} - {contribution}) ELSE NULL END) AS {expert}_max_contribution_difference",
                f"count_if(abs({weight_output} - ({weight})) > {FORMULA_TOLERANCE}) AS {expert}_weight_violation",
            ]
        )
        contribution_terms.append(f"CASE WHEN {expected_available} THEN {contribution} ELSE 0.0 END")
        availability_terms.append(f"CASE WHEN {expected_available} THEN 1 ELSE 0 END")
    total = " + ".join(contribution_terms)
    available_count = " + ".join(availability_terms)
    primary_logit = logit_expression("p.primary_probability")
    expected_score = (
        f"CASE WHEN ({available_count}) = 0 THEN p.primary_probability "
        f"ELSE 1.0 / (1.0 + exp(-(({primary_logit}) + ({total})))) END"
    )
    output_score = f"o.{score_column}"
    count_column = f"o.{prefix}available_auxiliary_expert_count"
    fallback_column = f"o.{prefix}primary_fallback"
    conditions.extend(
        [
            f"count_if({count_column} <> ({available_count})) AS auxiliary_count_violation",
            f"count_if({fallback_column} IS DISTINCT FROM (({available_count}) = 0)) AS fallback_flag_violation",
            f"count_if(({available_count}) = 0 AND {output_score} IS DISTINCT FROM p.primary_probability) AS no_expert_exact_fallback_violation",
            f"count_if(abs({output_score} - ({expected_score})) > {FORMULA_TOLERANCE}) AS score_formula_violation",
            f"max(abs({output_score} - ({expected_score}))) AS max_score_formula_difference",
            f"count_if(({total}) = 0.0 AND {output_score} IS DISTINCT FROM p.primary_probability) AS zero_total_contribution_nonexact_rows",
            f"count_if(({total}) = 0.0) AS zero_total_contribution_rows",
        ]
    )
    query = f"""
        SELECT {', '.join(conditions)}
        FROM {output_view} o
        {' '.join(joins)}
    """
    row = connection.execute(query).fetchdf().iloc[0].to_dict()
    for expert in experts:
        for suffix, requirement in (
            ("availability_mismatch", "Output expert availability equals its bound expert source"),
            ("typed_null_violation", "Unavailable expert contribution remains typed null"),
            ("contribution_violation", "Per-expert contribution equals w*(logit(p)-centre)"),
            ("weight_violation", "Serialized row weight equals the applicable checkpoint weight"),
        ):
            observed = int(row[f"{expert}_{suffix}"])
            audit.add(
                f"{endpoint}_{'oof' if oof else 'final'}_{expert}_{suffix}",
                observed == 0,
                requirement,
                observed,
                0,
            )
        expert_details[expert] = {
            "max_contribution_difference": (
                None
                if pd.isna(row[f"{expert}_max_contribution_difference"])
                else float(row[f"{expert}_max_contribution_difference"])
            )
        }
    for name, requirement in (
        ("auxiliary_count_violation", "Auxiliary expert count equals source availability masks"),
        ("fallback_flag_violation", "Primary fallback flag is true iff no expert is available"),
        ("no_expert_exact_fallback_violation", "Rows with no available expert fall back bit-for-bit to primary"),
        ("score_formula_violation", "Fused score independently reconstructs from primary offset and contributions"),
    ):
        observed = int(row[name])
        audit.add(
            f"{endpoint}_{'oof' if oof else 'final'}_{name}",
            observed == 0,
            requirement,
            observed,
            0,
        )
    return {
        "experts": experts,
        "expert_details": expert_details,
        "max_score_formula_difference": float(row["max_score_formula_difference"] or 0.0),
        "zero_total_contribution_rows": int(row["zero_total_contribution_rows"]),
        "zero_total_contribution_nonexact_rows": int(row["zero_total_contribution_nonexact_rows"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--binding-sha256", default=EXPECTED_BINDING_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal = args.formal_dir.resolve()
    output = args.output.resolve()
    if not formal.is_dir():
        raise RuntimeError(f"Formal fusion directory is missing: {formal}")
    if output.exists():
        raise RuntimeError(f"Refusing to reuse audit output: {output}")
    output.mkdir(parents=True)

    audit = Audit()
    binding_path = formal / "MULTIMODAL_FUSION_BINDING.json"
    observed_binding_sha = sha256(binding_path)
    audit.add(
        "formal_binding_sha256",
        observed_binding_sha == args.binding_sha256.lower(),
        "The audited release is exactly the requested hash-bound R1",
        observed_binding_sha,
        args.binding_sha256.lower(),
    )
    binding = read_json(binding_path)
    checkpoint_path = formal / "FUSION_CHECKPOINT.json"
    checkpoint = read_json(checkpoint_path)
    lineage_path = formal / "MODULE_LINEAGE.json"
    lineage = read_json(lineage_path)
    input_audit_path = formal / "FUSION_INPUT_AUDIT.json"
    input_manifest = read_json(input_audit_path)
    source_inputs_path = formal / "SOURCE_INPUTS.json"
    source_inputs = read_json(source_inputs_path)
    receipt_path = formal / "FORMAL_RUN_RECEIPT.json"
    receipt = read_json(receipt_path)
    success_path = formal / "SUCCESS.json"
    success = read_json(success_path)

    audit.add(
        "formal_identity_contract",
        binding.get("analysis_version") == ANALYSIS_VERSION
        and binding.get("training_run_id") == "V32-FUSION-FORMAL-20260826-R1"
        and binding.get("status") == "SUCCESS_SECONDARY_FUSION",
        "Binding identifies the formal fresh V3.2 R1 secondary fusion",
        {key: binding.get(key) for key in ("analysis_version", "training_run_id", "status")},
        {
            "analysis_version": ANALYSIS_VERSION,
            "training_run_id": "V32-FUSION-FORMAL-20260826-R1",
            "status": "SUCCESS_SECONDARY_FUSION",
        },
    )

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute(f"PRAGMA temp_directory='{sql_path(output / 'duckdb_tmp')}'")

    artifact_hash_results: list[dict[str, Any]] = []
    artifact_ok = True
    for artifact_id, record in binding["artifacts"].items():
        path = Path(record["path"]).resolve()
        exists = path.is_file() and not path.is_symlink()
        observed_sha = sha256(path) if exists else None
        rows = parquet_rows(connection, path) if exists and path.suffix.lower() == ".parquet" else None
        ok = (
            exists
            and observed_sha == record["sha256"]
            and path.stat().st_size == int(record["bytes"])
            and (record.get("rows") is None or rows == int(record["rows"]))
        )
        artifact_ok &= ok
        artifact_hash_results.append(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "sha256": observed_sha,
                "rows": rows,
                "passed": bool(ok),
                "public": bool(record.get("public")),
            }
        )
    audit.add(
        "formal_artifact_hash_chain",
        artifact_ok,
        "Every formal output matches binding SHA, byte count, and declared row count",
        artifact_hash_results,
        "all bound artifacts exact",
    )

    source_hash_results: list[dict[str, Any]] = []
    source_ok = True
    for record in binding["source_inputs"]:
        path = Path(record["path"]).resolve()
        observed_sha = sha256(path) if path.is_file() and not path.is_symlink() else None
        ok = observed_sha == record["sha256"]
        source_ok &= ok
        source_hash_results.append(
            {
                "path": str(path),
                "declared_sha256": record["sha256"],
                "observed_sha256": observed_sha,
                "generation": record.get("generation"),
                "source_role": record.get("source_role"),
                "passed": bool(ok),
            }
        )
    audit.add(
        "direct_source_sha_chain",
        source_ok,
        "Every directly bound source file matches its declared SHA",
        {"checked": len(source_hash_results), "failed": sum(not item["passed"] for item in source_hash_results)},
        {"failed": 0},
    )
    audit.add(
        "source_manifest_binding_identity",
        source_inputs.get("sources") == binding.get("source_inputs")
        and source_inputs.get("historical_derived_inputs") == [],
        "Public source manifest is identical to binding and declares no historical derived input",
        {
            "same_sources": source_inputs.get("sources") == binding.get("source_inputs"),
            "historical_derived_inputs": source_inputs.get("historical_derived_inputs"),
        },
        {"same_sources": True, "historical_derived_inputs": []},
    )

    nested_paths = [
        Path(record["path"]).resolve()
        for record in binding["source_inputs"]
        if str(record["path"]).lower().endswith(".json")
    ]
    nested_payloads = [(path, read_json(path)) for path in nested_paths]
    refs, skipped_directories = referenced_files(nested_payloads)
    nested_failed: list[dict[str, Any]] = []
    for record in refs:
        path = Path(record["path"])
        observed = sha256(path) if path.is_file() and not path.is_symlink() else None
        record["observed_sha256"] = observed
        record["passed"] = observed == record["declared_sha256"]
        if not record["passed"]:
            nested_failed.append(record)
    audit.add(
        "nested_source_sha_chain",
        not nested_failed,
        "File references inside directly bound provenance JSONs retain their declared SHA chain",
        {
            "file_references_checked": len(refs),
            "failed": nested_failed,
            "directory_hash_records_not_reinterpreted": skipped_directories,
        },
        {"failed": []},
    )

    primary_path = Path(binding["source_inputs"][0]["path"]).resolve()
    fold_paths = [Path(binding["source_inputs"][index]["path"]).resolve() for index in range(1, 6)]
    genomic_path = next(
        Path(item["path"]).resolve()
        for item in binding["source_inputs"]
        if item["sha256"] == input_manifest["genomic_sha256"]
    )
    single_path = next(
        Path(item["path"]).resolve()
        for item in binding["source_inputs"]
        if str(item["path"]).endswith("single_cell_exact_fusion_expert.parquet")
    )
    evidence_path = next(
        Path(item["path"]).resolve()
        for item in binding["source_inputs"]
        if str(item["path"]).endswith("evidence_exact_fusion_expert.parquet")
    )
    public_path = Path(binding["artifacts"]["secondary_scores"]["path"]).resolve()
    discovery_oof_path = Path(binding["artifacts"]["discovery_oof_private"]["path"]).resolve()
    confidence_oof_path = Path(binding["artifacts"]["confidence_oof_private"]["path"]).resolve()
    metrics_path = Path(binding["artifacts"]["crossfit_metrics"]["path"]).resolve()

    connection.execute(
        f"CREATE VIEW p AS SELECT cancer_id, lncrna_id, pathway_id, cast(association_membership_probability AS double) AS primary_probability, analysis_version, n_folds_available FROM read_parquet('{sql_path(primary_path)}')"
    )
    connection.execute(
        f"CREATE VIEW g AS SELECT cancer_id, lncrna_id, pathway_id, mutation_cnv_context_probability AS genomic_probability, genomic_available, genomic_unavailable_reason, mutation_available, cnv_available, analysis_version FROM read_parquet('{sql_path(genomic_path)}')"
    )
    connection.execute(
        f"CREATE VIEW s AS SELECT cancer_id, lncrna_id, pathway_id, single_cell_replication_probability AS single_cell_probability, single_cell_available, single_cell_unavailable_reason, analysis_version FROM read_parquet('{sql_path(single_path)}')"
    )
    connection.execute(
        f"CREATE VIEW e AS SELECT cancer_id, lncrna_id, pathway_id, evidence_confidence_probability AS evidence_probability, availability AS evidence_available, unavailable_reason AS evidence_unavailable_reason, analysis_version FROM read_parquet('{sql_path(evidence_path)}')"
    )
    connection.execute(f"CREATE VIEW pub AS SELECT * FROM read_parquet('{sql_path(public_path)}')")
    connection.execute(f"CREATE VIEW doof AS SELECT * FROM read_parquet('{sql_path(discovery_oof_path)}')")
    connection.execute(f"CREATE VIEW coof AS SELECT * FROM read_parquet('{sql_path(confidence_oof_path)}')")
    connection.execute(f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{sql_path(metrics_path)}')")

    universe_stats: dict[str, Any] = {}
    for view in ("p", "g", "s", "e", "pub", "doof", "coof"):
        row = connection.execute(
            f"SELECT count(*) AS row_count, count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS distinct_keys, count_if(cancer_id IS NULL OR lncrna_id IS NULL OR pathway_id IS NULL) AS null_keys FROM {view}"
        ).fetchone()
        universe_stats[view] = {"rows": int(row[0]), "distinct_keys": int(row[1]), "null_keys": int(row[2])}
    audit.add(
        "exact_candidate_universe_3300000",
        all(
            item["rows"] == EXPECTED_ROWS
            and item["distinct_keys"] == EXPECTED_ROWS
            and item["null_keys"] == 0
            for item in universe_stats.values()
        ),
        "Primary, each expert, public output, and both OOF outputs contain one row for every exact key",
        universe_stats,
        {"rows": EXPECTED_ROWS, "distinct_keys": EXPECTED_ROWS, "null_keys": 0},
    )
    key_differences: dict[str, int] = {}
    for view in ("g", "s", "e", "pub", "doof", "coof"):
        missing = connection.execute(
            f"SELECT count(*) FROM (SELECT cancer_id,lncrna_id,pathway_id FROM p EXCEPT SELECT cancer_id,lncrna_id,pathway_id FROM {view})"
        ).fetchone()[0]
        extra = connection.execute(
            f"SELECT count(*) FROM (SELECT cancer_id,lncrna_id,pathway_id FROM {view} EXCEPT SELECT cancer_id,lncrna_id,pathway_id FROM p)"
        ).fetchone()[0]
        key_differences[view] = int(missing) + int(extra)
    audit.add(
        "exact_key_set_closure",
        not any(key_differences.values()),
        "All sources and outputs have exactly the primary exact-key set",
        key_differences,
        {key: 0 for key in key_differences},
    )

    primary_contract = connection.execute(
        f"""
        SELECT
          count_if(pub.primary_probability IS DISTINCT FROM p.primary_probability) public_primary_mismatch,
          count_if(doof.primary_probability IS DISTINCT FROM p.primary_probability) discovery_oof_primary_mismatch,
          count_if(coof.primary_probability IS DISTINCT FROM p.primary_probability) confidence_oof_primary_mismatch,
          count_if(p.analysis_version <> '{ANALYSIS_VERSION}' OR p.n_folds_available <> 5) primary_contract_mismatch
        FROM p JOIN pub USING(cancer_id,lncrna_id,pathway_id)
        JOIN doof USING(cancer_id,lncrna_id,pathway_id)
        JOIN coof USING(cancer_id,lncrna_id,pathway_id)
        """
    ).fetchdf().iloc[0].to_dict()
    audit.add(
        "primary_probability_rowwise_immutable",
        all(int(value) == 0 for value in primary_contract.values()),
        "Primary probability is preserved row-for-row in public and private fusion outputs",
        {key: int(value) for key, value in primary_contract.items()},
        {key: 0 for key in primary_contract},
    )
    audit.add(
        "primary_source_sha_pinned",
        sha256(primary_path) == "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1",
        "Immutable primary source is the pinned formal V3.2 SHA",
        sha256(primary_path),
        "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1",
    )

    expert_stats: dict[str, Any] = {}
    for expert, (view, probability, availability, reason) in {
        "genomic": ("g", "genomic_probability", "genomic_available", "genomic_unavailable_reason"),
        "single_cell": ("s", "single_cell_probability", "single_cell_available", "single_cell_unavailable_reason"),
        "evidence_transformer": ("e", "evidence_probability", "evidence_available", "evidence_unavailable_reason"),
    }.items():
        row = connection.execute(
            f"""
            SELECT count_if({availability}) available,
                   count_if(NOT {availability}) unavailable,
                   count_if({availability} AND ({probability} IS NULL OR NOT isfinite({probability}) OR {probability}<0 OR {probability}>1)) bad_available,
                   count_if(NOT {availability} AND {probability} IS NOT NULL) bad_unavailable_probability,
                   count_if(NOT {availability} AND ({reason} IS NULL OR trim({reason})='')) missing_unavailable_reason,
                   count_if(analysis_version <> '{ANALYSIS_VERSION}') version_mismatch
            FROM {view}
            """
        ).fetchone()
        expert_stats[expert] = {
            "available": int(row[0]),
            "unavailable": int(row[1]),
            "bad_available": int(row[2]),
            "bad_unavailable_probability": int(row[3]),
            "missing_unavailable_reason": int(row[4]),
            "version_mismatch": int(row[5]),
        }
        expected_available, expected_unavailable = EXPECTED_EXPERT_COUNTS[expert]
        audit.add(
            f"{expert}_availability_and_typed_null",
            expert_stats[expert]
            == {
                "available": expected_available,
                "unavailable": expected_unavailable,
                "bad_available": 0,
                "bad_unavailable_probability": 0,
                "missing_unavailable_reason": 0,
                "version_mismatch": 0,
            },
            "Expert covers 3.3M keys with exact availability counts and null-with-reason semantics",
            expert_stats[expert],
            {
                "available": expected_available,
                "unavailable": expected_unavailable,
                "bad_available": 0,
                "bad_unavailable_probability": 0,
                "missing_unavailable_reason": 0,
                "version_mismatch": 0,
            },
        )

    fold_union = " UNION ALL ".join(
        f"SELECT cancer_id,lncrna_id,pathway_id,cast(patient_fold_id AS integer) AS patient_fold,cast(held_out_proxy_label AS double) AS proxy_label FROM read_parquet('{sql_path(path)}')"
        for path in fold_paths
    )
    connection.execute(f"CREATE TEMP TABLE label_rows AS {fold_union}")
    label_stats = connection.execute(
        """
        SELECT count(*) AS row_count,
               count(DISTINCT (cancer_id,lncrna_id,pathway_id)) AS distinct_keys,
               count(DISTINCT patient_fold) AS fold_count_stat,
               count_if(patient_fold NOT BETWEEN 0 AND 4 OR proxy_label NOT IN (0,1) OR proxy_label IS NULL) AS bad_rows
        FROM label_rows
        """
    ).fetchone()
    connection.execute(
        """
        CREATE TEMP TABLE target_summary AS
        SELECT cancer_id,lncrna_id,pathway_id,avg(proxy_label) AS fusion_target,
               count(DISTINCT patient_fold) fold_count
        FROM label_rows GROUP BY ALL
        """
    )
    target_mismatch = connection.execute(
        """
        SELECT count_if(t.fold_count<>5),
               count_if(d.fusion_target IS DISTINCT FROM t.fusion_target),
               count_if(c.fusion_target IS DISTINCT FROM t.fusion_target),
               count_if(d.fusion_target IS DISTINCT FROM c.fusion_target)
        FROM target_summary t
        JOIN doof d USING(cancer_id,lncrna_id,pathway_id)
        JOIN coof c USING(cancer_id,lncrna_id,pathway_id)
        """
    ).fetchone()
    target_evidence = {
        "label_rows": int(label_stats[0]),
        "exact_keys": int(label_stats[1]),
        "folds": int(label_stats[2]),
        "bad_label_or_fold_rows": int(label_stats[3]),
        "incomplete_targets": int(target_mismatch[0]),
        "discovery_target_mismatch": int(target_mismatch[1]),
        "confidence_target_mismatch": int(target_mismatch[2]),
        "endpoint_target_mismatch": int(target_mismatch[3]),
    }
    audit.add(
        "five_fold_target_reconstruction",
        target_evidence
        == {
            "label_rows": 16_500_000,
            "exact_keys": EXPECTED_ROWS,
            "folds": 5,
            "bad_label_or_fold_rows": 0,
            "incomplete_targets": 0,
            "discovery_target_mismatch": 0,
            "confidence_target_mismatch": 0,
            "endpoint_target_mismatch": 0,
        },
        "Private fusion target is independently reconstructed as the mean of all five held-out primary labels",
        target_evidence,
        {
            "label_rows": 16_500_000,
            "exact_keys": EXPECTED_ROWS,
            "folds": 5,
            "bad_label_or_fold_rows": 0,
            "incomplete_targets": 0,
            "discovery_target_mismatch": 0,
            "confidence_target_mismatch": 0,
            "endpoint_target_mismatch": 0,
        },
    )

    pair_frame = connection.execute(
        """
        SELECT lncrna_id,pathway_id,min(fusion_pair_fold) min_fold,max(fusion_pair_fold) max_fold,
               count(DISTINCT fusion_pair_fold) fold_count
        FROM doof GROUP BY lncrna_id,pathway_id
        """
    ).fetchdf()
    calculated = np.fromiter(
        (
            deterministic_pair_fold(lnc, pathway)
            for lnc, pathway in pair_frame[["lncrna_id", "pathway_id"]].itertuples(index=False, name=None)
        ),
        dtype=np.int8,
        count=len(pair_frame),
    )
    pair_fold_errors = int(
        np.count_nonzero(calculated != pair_frame["min_fold"].to_numpy(dtype=np.int8))
        + np.count_nonzero(pair_frame["min_fold"].to_numpy() != pair_frame["max_fold"].to_numpy())
        + np.count_nonzero(pair_frame["fold_count"].to_numpy() != 1)
    )
    endpoint_fold_mismatch = int(
        connection.execute(
            "SELECT count_if(d.fusion_pair_fold IS DISTINCT FROM c.fusion_pair_fold) FROM doof d JOIN coof c USING(cancer_id,lncrna_id,pathway_id)"
        ).fetchone()[0]
    )
    fold_distribution = {
        int(row[0]): int(row[1])
        for row in connection.execute("SELECT fusion_pair_fold,count(*) FROM doof GROUP BY 1 ORDER BY 1").fetchall()
    }
    audit.add(
        "pair_blocked_five_fold_assignment",
        len(pair_frame) > 0
        and pair_fold_errors == 0
        and endpoint_fold_mismatch == 0
        and set(fold_distribution) == set(range(5)),
        "Cancer-agnostic lncRNA/exact-pathway pairs use one deterministic outer fold across endpoints and all cancers",
        {
            "distinct_pairs": len(pair_frame),
            "deterministic_or_pair_consistency_errors": pair_fold_errors,
            "endpoint_fold_mismatch": endpoint_fold_mismatch,
            "fold_distribution": fold_distribution,
        },
        {"deterministic_or_pair_consistency_errors": 0, "endpoint_fold_mismatch": 0, "folds": [0, 1, 2, 3, 4]},
    )

    discovery_experts = list(checkpoint["discovery"]["expert_ids"])
    confidence_experts = list(checkpoint["confidence"]["expert_ids"])
    audit.add(
        "endpoint_expert_policy",
        discovery_experts == ["genomic", "single_cell"]
        and confidence_experts == ["genomic", "single_cell", "evidence_transformer"],
        "Discovery uses only genomic+single-cell; confidence additionally uses Evidence Transformer",
        {"discovery": discovery_experts, "confidence": confidence_experts},
        {
            "discovery": ["genomic", "single_cell"],
            "confidence": ["genomic", "single_cell", "evidence_transformer"],
        },
    )
    all_column_names = set(parquet_columns(connection, public_path))
    all_checkpoint_experts = set(discovery_experts + confidence_experts)
    source_text = json.dumps(source_inputs, ensure_ascii=False).lower()
    audit.add(
        "drug_excluded_from_exact_pathway_fusion",
        "drug" not in all_checkpoint_experts
        and not any("drug" in name.lower() for name in all_column_names)
        and "drug" not in source_text
        and input_manifest.get("drug_in_exact_pathway_fusion") is False
        and receipt.get("drug_actionability_separate") is True,
        "Drug remains a separate native actionability head and never enters exact-pathway fusion",
        {
            "checkpoint_experts": sorted(all_checkpoint_experts),
            "drug_columns": sorted(name for name in all_column_names if "drug" in name.lower()),
            "drug_in_source_manifest": "drug" in source_text,
            "input_audit_flag": input_manifest.get("drug_in_exact_pathway_fusion"),
            "receipt_separate_flag": receipt.get("drug_actionability_separate"),
        },
        {
            "drug_columns": [],
            "drug_in_source_manifest": False,
            "input_audit_flag": False,
            "receipt_separate_flag": True,
        },
    )

    model_records = [checkpoint["discovery"], checkpoint["confidence"]]
    for endpoint in ("discovery", "confidence"):
        model_records.extend(checkpoint["fold_models"][endpoint])
    model_audit_rows: list[dict[str, Any]] = []
    models_ok = True
    for model in model_records:
        weights = [float(value) for value in model["weights"]]
        observed_final_sha = parameter_sha(weights)
        observed_initial_sha = initial_parameter_sha(int(model["seed"]), len(weights))
        ok = (
            all(math.isfinite(value) and value >= 0 for value in weights)
            and observed_final_sha == model["final_parameter_sha256"]
            and observed_initial_sha == model["initial_parameter_sha256"]
            and int(model["optimiser_steps"]) > 0
            and model.get("weights_constrained_nonnegative") is True
            and model.get("primary_probability_offset_fixed") is True
            and model.get("old_checkpoint_loaded") is False
            and model.get("old_predictions_used_as_features") is False
        )
        models_ok &= ok
        model_audit_rows.append(
            {
                "endpoint": model["endpoint"],
                "seed": int(model["seed"]),
                "experts": model["expert_ids"],
                "weights": weights,
                "final_sha_reconstructed": observed_final_sha,
                "initial_sha_reconstructed": observed_initial_sha,
                "optimiser_steps": int(model["optimiser_steps"]),
                "passed": bool(ok),
            }
        )
    audit.add(
        "fresh_nonnegative_model_parameters",
        models_ok,
        "All 10 outer-fold and 2 final models have reproducible fresh random initialization, nonnegative weights, and bound final parameter SHA",
        model_audit_rows,
        "12/12 models exact and nonnegative",
    )

    formula_details: dict[str, Any] = {}
    for endpoint in ("discovery", "confidence"):
        formula_details[f"{endpoint}_oof"] = endpoint_formula_audit(
            connection, audit, checkpoint, endpoint, oof=True
        )
        formula_details[f"{endpoint}_final"] = endpoint_formula_audit(
            connection, audit, checkpoint, endpoint, oof=False
        )

    # The formal contract promises bitwise fallback only for *no available
    # expert*.  This additional release-quality invariant catches a subtle R1
    # issue: available experts with zero learned weight still trigger a
    # sigmoid(logit(primary)) round trip instead of exact primary assignment.
    zero_contribution_failures = {
        endpoint: details["zero_total_contribution_nonexact_rows"]
        for endpoint, details in formula_details.items()
        if endpoint.endswith("_final")
    }
    audit.add(
        "zero_total_contribution_exact_primary_fallback",
        all(value == 0 for value in zero_contribution_failures.values()),
        "Any row whose total learned auxiliary contribution is exactly zero must preserve the primary score bit-for-bit",
        zero_contribution_failures,
        {key: 0 for key in zero_contribution_failures},
        severity="ERROR",
    )

    metrics_frame = connection.execute("SELECT * FROM metrics").fetchdf()
    recomputed_metrics: dict[str, list[dict[str, Any]]] = {}
    metrics_ok = len(metrics_frame) == 12
    outcomes: dict[str, Any] = {}
    for endpoint, view, score in (
        ("discovery", "doof", "discovery_adjusted_probability"),
        ("confidence", "coof", "fused_confidence_probability"),
    ):
        rows = connection.execute(
            f"""
            SELECT cast(fusion_pair_fold AS varchar) fold, count(*) test_rows,
                   avg(-(fusion_target*ln(greatest(least(primary_probability,{1-CLIP}),{CLIP}))+(1-fusion_target)*ln(1-greatest(least(primary_probability,{1-CLIP}),{CLIP})))) primary_logloss,
                   avg(-(fusion_target*ln(greatest(least({score},{1-CLIP}),{CLIP}))+(1-fusion_target)*ln(1-greatest(least({score},{1-CLIP}),{CLIP})))) adjusted_logloss,
                   avg(pow(primary_probability-fusion_target,2)) primary_brier,
                   avg(pow({score}-fusion_target,2)) adjusted_brier
            FROM {view} GROUP BY fusion_pair_fold
            UNION ALL
            SELECT 'ALL', count(*),
                   avg(-(fusion_target*ln(greatest(least(primary_probability,{1-CLIP}),{CLIP}))+(1-fusion_target)*ln(1-greatest(least(primary_probability,{1-CLIP}),{CLIP})))),
                   avg(-(fusion_target*ln(greatest(least({score},{1-CLIP}),{CLIP}))+(1-fusion_target)*ln(1-greatest(least({score},{1-CLIP}),{CLIP})))),
                   avg(pow(primary_probability-fusion_target,2)), avg(pow({score}-fusion_target,2))
            FROM {view}
            ORDER BY fold
            """
        ).fetchdf()
        endpoint_metrics: list[dict[str, Any]] = []
        for record in rows.to_dict("records"):
            fold = record["fold"]
            heldout = "ALL_OOF" if fold == "ALL" else f"PAIR_FOLD_{fold}"
            declared = metrics_frame.loc[
                metrics_frame.endpoint.eq(endpoint) & metrics_frame.heldout_pair_fold.eq(heldout)
            ]
            match = len(declared) == 1
            if match:
                declared_row = declared.iloc[0]
                for column in (
                    "primary_logloss",
                    "adjusted_logloss",
                    "primary_brier",
                    "adjusted_brier",
                ):
                    match &= abs(float(declared_row[column]) - float(record[column])) <= METRIC_TOLERANCE
                match &= int(declared_row["test_rows"]) == int(record["test_rows"])
            metrics_ok &= bool(match)
            endpoint_metrics.append({**record, "declared_match": bool(match)})
        recomputed_metrics[endpoint] = endpoint_metrics
        total = next(item for item in endpoint_metrics if item["fold"] == "ALL")
        outcome = (
            "INCREMENT"
            if float(total["primary_logloss"]) - float(total["adjusted_logloss"]) >= 1.0e-4
            else "NO_INCREMENT"
        )
        aggregate = metrics_frame.loc[
            metrics_frame.endpoint.eq(endpoint) & metrics_frame.heldout_pair_fold.eq("ALL_OOF")
        ].iloc[0]
        outcomes[endpoint] = {
            "primary_logloss": float(total["primary_logloss"]),
            "adjusted_logloss": float(total["adjusted_logloss"]),
            "delta": float(total["primary_logloss"]) - float(total["adjusted_logloss"]),
            "recomputed_outcome": outcome,
            "declared_outcome": aggregate["performance_outcome"],
            "declared_status": aggregate["scientific_status"],
        }
    audit.add(
        "oof_metrics_independent_reconstruction",
        metrics_ok,
        "All 5-fold and aggregate OOF logloss/Brier metrics independently reproduce",
        recomputed_metrics,
        "12/12 metric rows reproduce within tolerance",
    )
    audit.add(
        "no_increment_retained_as_diagnostic_only",
        all(
            item["recomputed_outcome"] == "NO_INCREMENT"
            and item["declared_outcome"] == "NO_INCREMENT"
            and item["declared_status"] == "DIAGNOSTIC_ONLY"
            for item in outcomes.values()
        )
        and lineage.get("no_increment_capability_retained") is True,
        "Both negative performance outcomes remain DIAGNOSTIC_ONLY and are not removed",
        {"outcomes": outcomes, "lineage_retained": lineage.get("no_increment_capability_retained")},
        {"discovery": "NO_INCREMENT/DIAGNOSTIC_ONLY", "confidence": "NO_INCREMENT/DIAGNOSTIC_ONLY", "retained": True},
    )

    mutation_rows = connection.execute("SELECT count_if(mutation_available),count_if(cnv_available) FROM g").fetchone()
    retained_columns = {
        "genomic": all(
            name in all_column_names
            for name in (
                "discovery_genomic_available",
                "discovery_genomic_logit_contribution",
                "confidence_genomic_available",
                "confidence_genomic_logit_contribution",
            )
        ),
        "single_cell": all(
            name in all_column_names
            for name in (
                "discovery_single_cell_available",
                "discovery_single_cell_logit_contribution",
                "confidence_single_cell_available",
                "confidence_single_cell_logit_contribution",
            )
        ),
        "evidence_transformer": all(
            name in all_column_names
            for name in (
                "confidence_evidence_transformer_available",
                "confidence_evidence_transformer_logit_contribution",
            )
        ),
    }
    audit.add(
        "negative_result_capabilities_not_deleted",
        int(mutation_rows[0]) == 698_302
        and int(mutation_rows[1]) == 86_461
        and all(retained_columns.values())
        and all(EXPECTED_EXPERT_COUNTS[name][0] > 0 for name in EXPECTED_EXPERT_COUNTS),
        "Mutation/CNV, single-cell, and Evidence remain bound and queryable as diagnostic experts despite NO_INCREMENT or zero weights",
        {
            "mutation_available_rows": int(mutation_rows[0]),
            "cnv_available_rows": int(mutation_rows[1]),
            "expert_available_rows": {key: value[0] for key, value in EXPECTED_EXPERT_COUNTS.items()},
            "public_diagnostic_columns_retained": retained_columns,
        },
        {
            "mutation_available_rows": 698_302,
            "cnv_available_rows": 86_461,
            "public_diagnostic_columns_retained": {key: True for key in retained_columns},
        },
    )

    public_artifacts = {
        name: record for name, record in binding["artifacts"].items() if record.get("public") is True
    }
    private_artifacts = {
        name: record for name, record in binding["artifacts"].items() if record.get("public") is False
    }
    forbidden = {"fusion_target", "held_out_proxy_label", "test_membership_label"}
    public_schema_leaks: dict[str, list[str]] = {}
    public_keyed_metric_rows = 0
    for name, record in public_artifacts.items():
        path = Path(record["path"])
        if path.suffix.lower() == ".parquet":
            columns = set(parquet_columns(connection, path))
            public_schema_leaks[name] = sorted(columns & forbidden)
            if name == "crossfit_metrics":
                public_keyed_metric_rows = int(bool(columns & set(KEYS)))
    private_flags_ok = (
        set(private_artifacts) == {"checkpoint", "confidence_oof_private", "discovery_oof_private"}
        and "fusion_target" in parquet_columns(connection, discovery_oof_path)
        and "fusion_target" in parquet_columns(connection, confidence_oof_path)
    )
    audit.add(
        "private_targets_not_public",
        all(not values for values in public_schema_leaks.values())
        and public_keyed_metric_rows == 0
        and private_flags_ok,
        "Row-level fusion targets/OOF membership remain only in artifacts explicitly marked private",
        {
            "public_schema_leaks": public_schema_leaks,
            "metrics_contains_exact_keys": bool(public_keyed_metric_rows),
            "private_artifacts": sorted(private_artifacts),
            "private_flags_ok": private_flags_ok,
        },
        {
            "public_schema_leaks": {},
            "metrics_contains_exact_keys": False,
            "private_artifacts": ["checkpoint", "confidence_oof_private", "discovery_oof_private"],
        },
    )

    nested_by_name = {path.name: payload for path, payload in nested_payloads}
    genomic_lineage = nested_by_name.get("LINEAGE.json", {})
    single_binding = nested_by_name.get("SINGLE_CELL_FUSION_BINDING.json", {})
    evidence_binding = nested_by_name.get("EVIDENCE_FUSION_BINDING.json", {})
    evidence_output = nested_by_name.get("EVIDENCE_OUTPUT_BINDING.json", {})
    fresh_evidence = {
        "checkpoint_all_parameters_new": checkpoint.get("all_parameters_newly_trained_v32"),
        "checkpoint_old_loaded": checkpoint.get("old_checkpoint_loaded"),
        "source_generations": sorted({item.get("generation") for item in binding["source_inputs"]}),
        "source_historical_derived_inputs": source_inputs.get("historical_derived_inputs"),
        "genomic_private_head_from_scratch": genomic_lineage.get("private_head_trained_from_scratch"),
        "genomic_old_loaded": genomic_lineage.get("old_checkpoint_loaded"),
        "single_cell_old_loaded": single_binding.get("old_checkpoint_loaded"),
        "single_cell_old_predictions": single_binding.get("old_predictions_used_as_features"),
        "evidence_old_loaded": evidence_binding.get("old_checkpoint_loaded"),
        "evidence_old_predictions": evidence_binding.get("old_predictions_used_as_features"),
        "evidence_five_fresh_heads": evidence_output.get("five_fresh_private_heads_verified"),
        "evidence_historical_checkpoints": evidence_output.get("historical_checkpoints_used"),
        "evidence_historical_predictions": evidence_output.get("historical_predictions_used"),
        "lineage_historical_predictions": lineage.get("historical_predictions_used"),
        "lineage_historical_rankings": lineage.get("historical_rankings_used"),
    }
    audit.add(
        "fresh_v32_no_old_result_reuse",
        fresh_evidence
        == {
            "checkpoint_all_parameters_new": True,
            "checkpoint_old_loaded": False,
            "source_generations": ["current_v32"],
            "source_historical_derived_inputs": [],
            "genomic_private_head_from_scratch": True,
            "genomic_old_loaded": False,
            "single_cell_old_loaded": False,
            "single_cell_old_predictions": False,
            "evidence_old_loaded": False,
            "evidence_old_predictions": False,
            "evidence_five_fresh_heads": True,
            "evidence_historical_checkpoints": False,
            "evidence_historical_predictions": False,
            "lineage_historical_predictions": False,
            "lineage_historical_rankings": False,
        },
        "Fusion and all expert heads are current V3.2 outputs with no historical prediction/ranking/checkpoint reuse",
        fresh_evidence,
        "all fresh/no-old flags exact",
    )

    code_paths = {
        "fusion": formal.parents[1] / "cc_hhgt" / "v32" / "multimodal_fusion.py",
        "policy": formal.parents[1] / "cc_hhgt" / "v32" / "fusion_expert_policy.py",
        "runner": formal.parents[1] / "scripts" / "run_v32_multimodal_fusion_formal.py",
        "auditor": Path(__file__).resolve(),
    }
    code_hashes = {name: sha256(path) for name, path in code_paths.items()}
    code_chain_ok = (
        code_hashes["fusion"] == lineage.get("code_sha256")
        and code_hashes["runner"] == receipt.get("runner_sha256")
    )
    audit.add(
        "formal_code_sha_chain",
        code_chain_ok,
        "Current formal fusion implementation and runner match the code hashes recorded at R1 materialization",
        code_hashes,
        {"fusion": lineage.get("code_sha256"), "runner": receipt.get("runner_sha256")},
    )
    audit.add(
        "success_receipt_binding_chain",
        success.get("binding_sha256") == observed_binding_sha
        and receipt.get("binding_sha256") == observed_binding_sha
        and success.get("public_rows") == EXPECTED_ROWS
        and receipt.get("public_rows") == EXPECTED_ROWS,
        "SUCCESS and formal receipt both point to the exact audited binding and 3.3M public rows",
        {
            "success_binding_sha256": success.get("binding_sha256"),
            "receipt_binding_sha256": receipt.get("binding_sha256"),
            "success_rows": success.get("public_rows"),
            "receipt_rows": receipt.get("public_rows"),
        },
        {"binding_sha256": observed_binding_sha, "public_rows": EXPECTED_ROWS},
    )

    report = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_INDEPENDENT_POST_AUDIT_REPORT_V1",
        "analysis_version": ANALYSIS_VERSION,
        "audit_scope": "formal_multimodal_fusion_r1_read_only_post_audit",
        "audited_directory": str(formal),
        "audited_binding_path": str(binding_path),
        "audited_binding_sha256": observed_binding_sha,
        "status": "PASS" if not audit.failed else "FAIL",
        "fail_closed": True,
        "pass_count": audit.passed_count,
        "fail_count": len(audit.failed),
        "failed_checks": audit.failed,
        "checks": audit.checks,
        "universe_statistics": universe_stats,
        "expert_statistics": expert_stats,
        "formula_reconstruction": formula_details,
        "oof_outcomes": outcomes,
        "artifact_hash_results": artifact_hash_results,
        "direct_source_hash_results": source_hash_results,
        "nested_source_file_references_checked": len(refs),
        "private_target_policy": "OOF_TARGETS_AND_OOF_PREDICTIONS_PRIVATE",
        "primary_score_preserved": all(int(value) == 0 for value in primary_contract.values()),
        "drug_actionability_separate": True,
        "negative_result_capabilities_retained": True,
        "release_decision": {
            "r1_accepted_for_unified_release": not audit.failed,
            "r1_preserved_as_problem_evidence": bool(audit.failed),
            "production_deployed": False,
        },
        "critical_finding": {
            "check_id": "zero_total_contribution_exact_primary_fallback",
            "description": "Available zero-weight experts cause sigmoid(logit(primary)) round-trip differences instead of bitwise primary fallback.",
            "observed_final_nonexact_rows": zero_contribution_failures,
            "primary_probability_column_changed": False,
            "requires_new_materialization_not_in_place_edit": bool(any(zero_contribution_failures.values())),
        },
        "code_sha256": code_hashes,
    }
    report_path = output / "INDEPENDENT_POST_AUDIT_REPORT.json"
    atomic_json(report_path, report)
    report_sha = sha256(report_path)
    audit_binding = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_INDEPENDENT_POST_AUDIT_BINDING_V1",
        "analysis_version": ANALYSIS_VERSION,
        "status": report["status"],
        "fail_closed": True,
        "formal_binding": {"path": str(binding_path), "sha256": observed_binding_sha},
        "report": {"path": str(report_path), "sha256": report_sha},
        "audit_code": {"path": str(code_paths["auditor"]), "sha256": code_hashes["auditor"]},
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "failed_checks": report["failed_checks"],
        "primary_score_preserved": report["primary_score_preserved"],
        "exact_candidate_rows": EXPECTED_ROWS,
        "pair_blocked_folds": 5,
        "discovery_performance_outcome": outcomes["discovery"]["recomputed_outcome"],
        "confidence_performance_outcome": outcomes["confidence"]["recomputed_outcome"],
        "no_increment_capabilities_retained": True,
        "r1_accepted_for_unified_release": report["release_decision"]["r1_accepted_for_unified_release"],
        "production_deployed": False,
    }
    audit_binding_path = output / "INDEPENDENT_POST_AUDIT_BINDING.json"
    atomic_json(audit_binding_path, audit_binding)
    audit_binding_sha = sha256(audit_binding_path)
    completion = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_INDEPENDENT_POST_AUDIT_COMPLETION_V1",
        "status": report["status"],
        "binding_path": str(audit_binding_path),
        "binding_sha256": audit_binding_sha,
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "failed_checks": report["failed_checks"],
    }
    atomic_json(output / "AUDIT_COMPLETE.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
