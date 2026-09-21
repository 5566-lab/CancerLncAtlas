"""Independent post-materialisation audit of the V3.2 single-cell expert.

The materialiser and adapter are never imported.  The expected 3.3-million-row
table is reconstructed with an independent SQL left join from the frozen
candidate authority and exact single-cell source.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import duckdb


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
AUDIT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FUSION_POST_MATERIALIZATION_AUDIT_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FUSION_POST_MATERIALIZATION_BINDING_V1"
FUSION_BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FUSION_INPUT_BINDING_V1"
ADAPTER_AUDIT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FUSION_ADAPTER_AUDIT_V1"
ADAPTER_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_EXACT_FUSION_EXPERT_V1"
ORIGINAL_AUDIT_BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_INDEPENDENT_AUDIT_BINDING_V1"
EXPECTED_ORIGINAL_AUDIT_BINDING_SHA256 = (
    "5309b2eaf8e63d0daba241d6a9bca2870a382f711896916b4eb364cbfbc84829"
)
EXPECTED_ORIGINAL_AUDIT_REPORT_SHA256 = (
    "157143178d20e953d845e185a3b90796c346cea90951413082bb0703bd27674f"
)
EXPECTED_CANDIDATE_SHA256 = "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
EXPECTED_EXACT_SHA256 = "83e25646ef604b9d1a76f5872c73421ed0041c2fe097549e7c8838848ede7112"
EXPECTED_ROWS = 3_300_000
EXPECTED_AVAILABLE = 954_541
EXPECTED_UNAVAILABLE = 2_345_459
EXPECTED_EXACT_ROWS = 2_554_541


class FusionPostAuditError(RuntimeError):
    """Raised when post-materialisation evidence fails closed."""


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise FusionPostAuditError(f"unsafe or missing directory: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise FusionPostAuditError(f"empty directory: {source}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise FusionPostAuditError(f"symlink is forbidden: {item}")
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FusionPostAuditError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


@dataclass
class Checks:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self, name: str, condition: bool, *, observed: Any = None,
        expected: Any = None,
    ) -> None:
        row: dict[str, Any] = {
            "check": name,
            "status": "PASS" if bool(condition) else "FAIL",
        }
        if observed is not None:
            row["observed"] = observed
        if expected is not None:
            row["expected"] = expected
        self.rows.append(row)

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(row["status"] == "PASS" for row in self.rows)


def verify_post_binding(report_path: str | Path, binding: Mapping[str, Any]) -> bool:
    source = Path(report_path)
    return (
        source.is_file()
        and binding.get("format") == BINDING_FORMAT
        and binding.get("status") == "PASS"
        and binding.get("post_materialization_audit_sha256") == file_sha256(source)
    )


def run_post_materialization_audit(
    *, repo_root: str | Path, materialization_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    material = Path(materialization_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FusionPostAuditError(f"post audit refuses output reuse: {output}")
    if not material.is_dir() or material.is_symlink():
        raise FusionPostAuditError(f"unsafe materialization directory: {material}")
    if output == material or material in output.parents:
        raise FusionPostAuditError("post-audit output must be outside materialization directory")
    output.mkdir(parents=True)
    checks = Checks()
    started = datetime.now(timezone.utc).isoformat()

    prediction_path = material / "single_cell_exact_fusion_expert.parquet"
    adapter_audit_path = material / "ADAPTER_AUDIT.json"
    fusion_binding_path = material / "SINGLE_CELL_FUSION_BINDING.json"
    material_success_path = material / "SUCCESS.json"
    material_files = {
        "prediction": prediction_path,
        "adapter_audit": adapter_audit_path,
        "fusion_binding": fusion_binding_path,
        "success": material_success_path,
    }
    for name, path in material_files.items():
        checks.add(f"material_file_exists::{name}", path.is_file() and not path.is_symlink(),
                   observed=str(path))
    if not all(path.is_file() and not path.is_symlink() for path in material_files.values()):
        raise FusionPostAuditError("materialization is incomplete or unsafe")
    material_hash_before = directory_sha256(material)
    material_file_hashes_before = {name: file_sha256(path) for name, path in material_files.items()}

    fusion_binding = _json(fusion_binding_path)
    adapter_audit = _json(adapter_audit_path)
    material_success = _json(material_success_path)
    checks.add("fusion_binding_format_status",
               fusion_binding.get("format") == FUSION_BINDING_FORMAT
               and fusion_binding.get("status") == "PASS_HASH_BOUND_FUSION_INPUT"
               and fusion_binding.get("analysis_version") == ANALYSIS_VERSION)
    checks.add("adapter_audit_format_status",
               adapter_audit.get("format") == ADAPTER_AUDIT_FORMAT
               and adapter_audit.get("status") == "PASS"
               and adapter_audit.get("analysis_version") == ANALYSIS_VERSION)
    checks.add("material_success_status",
               material_success.get("status") == "PASS_HASH_BOUND_FUSION_INPUT"
               and material_success.get("fusion_input_eligible") is True
               and material_success.get("primary_ranking_unchanged") is True
               and material_success.get("release_ready") is False
               and material_success.get("production_deployed") is False)

    prediction_sha = file_sha256(prediction_path)
    adapter_audit_sha = file_sha256(adapter_audit_path)
    fusion_binding_sha = file_sha256(fusion_binding_path)
    success_sha = file_sha256(material_success_path)
    checks.add("prediction_sha_recomputed",
               fusion_binding.get("prediction", {}).get("sha256") == prediction_sha,
               observed=prediction_sha,
               expected=fusion_binding.get("prediction", {}).get("sha256"))
    checks.add("adapter_audit_sha_recomputed",
               fusion_binding.get("adapter_audit", {}).get("sha256") == adapter_audit_sha,
               observed=adapter_audit_sha,
               expected=fusion_binding.get("adapter_audit", {}).get("sha256"))
    checks.add("fusion_binding_sha_recomputed",
               material_success.get("binding_sha256") == fusion_binding_sha,
               observed=fusion_binding_sha,
               expected=material_success.get("binding_sha256"))
    checks.add("material_success_names_binding",
               material_success.get("binding") == fusion_binding_path.name)

    candidate_path = Path(str(fusion_binding.get("candidate_authority", {}).get("path", ""))).resolve()
    exact_path = Path(str(fusion_binding.get("single_cell_exact_source", {}).get("path", ""))).resolve()
    original_binding_path = Path(
        str(fusion_binding.get("independent_audit_binding", {}).get("path", ""))
    ).resolve()
    original_report_path = Path(
        str(fusion_binding.get("independent_audit_report", {}).get("path", ""))
    ).resolve()
    dependencies = {
        "candidate_authority": candidate_path,
        "single_cell_exact_source": exact_path,
        "original_audit_binding": original_binding_path,
        "original_audit_report": original_report_path,
    }
    for name, path in dependencies.items():
        checks.add(f"dependency_exists::{name}", path.is_file() and not path.is_symlink(),
                   observed=str(path))
    if not all(path.is_file() and not path.is_symlink() for path in dependencies.values()):
        raise FusionPostAuditError("bound dependency is missing or unsafe")

    candidate_sha = file_sha256(candidate_path)
    exact_sha = file_sha256(exact_path)
    original_binding_sha = file_sha256(original_binding_path)
    original_report_sha = file_sha256(original_report_path)
    checks.add("candidate_hash_recomputed",
               candidate_sha == EXPECTED_CANDIDATE_SHA256
               and candidate_sha == fusion_binding.get("candidate_authority", {}).get("sha256"))
    checks.add("exact_source_hash_recomputed",
               exact_sha == EXPECTED_EXACT_SHA256
               and exact_sha == fusion_binding.get("single_cell_exact_source", {}).get("sha256"))
    checks.add("original_audit_binding_sha_is_5309_authority",
               original_binding_sha == EXPECTED_ORIGINAL_AUDIT_BINDING_SHA256
               and original_binding_sha
               == fusion_binding.get("independent_audit_binding", {}).get("sha256"),
               observed=original_binding_sha,
               expected=EXPECTED_ORIGINAL_AUDIT_BINDING_SHA256)
    checks.add("original_audit_report_sha_recomputed",
               original_report_sha == EXPECTED_ORIGINAL_AUDIT_REPORT_SHA256
               and original_report_sha
               == fusion_binding.get("independent_audit_report", {}).get("sha256"))
    original_binding = _json(original_binding_path)
    original_report = _json(original_report_path)
    checks.add("original_audit_binding_internal_contract",
               original_binding.get("format") == ORIGINAL_AUDIT_BINDING_FORMAT
               and original_binding.get("status") == "PASS"
               and original_binding.get("audit_report_sha256") == original_report_sha
               and original_binding.get("release_decision", {}).get(
                   "full_universe_fusion_adapter_materialization"
               ) == "AUTHORIZED")
    checks.add("original_audit_report_internal_contract",
               original_report.get("status") == "PASS"
               and original_report.get("failed_checks") == []
               and all(row.get("status") == "PASS" for row in original_report.get("checks", [])))

    adapter_code_path = repo / "cc_hhgt" / "v32" / "single_cell_fusion_adapter.py"
    binding_code_path = repo / "cc_hhgt" / "v32" / "single_cell_fusion_binding.py"
    materializer_path = repo / "scripts" / "materialize_v32_single_cell_fusion_adapter.py"
    checks.add("adapter_code_hash_recomputed",
               file_sha256(adapter_code_path) == fusion_binding.get("adapter_code_sha256"))
    checks.add("binding_code_hash_recomputed",
               file_sha256(binding_code_path) == fusion_binding.get("binding_code_sha256"))

    binding_semantics = {
        "adapter_format": ADAPTER_FORMAT,
        "target_level": "cancer_x_lncrna_x_exact_pathway",
        "source_target_level": "dataset_x_celltype_x_lncrna_x_exact_pathway",
        "candidate_rows": EXPECTED_ROWS,
        "available_rows": EXPECTED_AVAILABLE,
        "unavailable_rows": EXPECTED_UNAVAILABLE,
        "fusion_input_eligible": True,
        "direct_target_evidence": False,
        "affects_discovery": True,
        "affects_confidence": True,
        "changes_primary_ranking": False,
        "family_to_exact_broadcast": False,
        "unavailable_encoding": "null_with_reason",
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "release_ready": False,
        "production_deployed": False,
    }
    binding_drift = {
        key: {"observed": fusion_binding.get(key), "expected": value}
        for key, value in binding_semantics.items() if fusion_binding.get(key) != value
    }
    checks.add("fusion_binding_semantics", not binding_drift,
               observed=binding_drift or "no drift", expected="no drift")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    pred = _sql_path(prediction_path)
    cand = _sql_path(candidate_path)
    exact = _sql_path(exact_path)
    schema_rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pred}')").fetchall()
    columns = [row[0] for row in schema_rows]
    expected_columns = [
        "cancer_id", "lncrna_id", "pathway_id",
        "single_cell_replication_probability", "single_cell_available",
        "single_cell_unavailable_reason", "analysis_version", "module_id",
        "adapter_format", "target_level", "source_target_level", "aggregation",
        "direct_target_evidence", "family_to_exact_broadcast",
        "changes_primary_ranking", "availability_encoding",
    ]
    checks.add("prediction_schema_exact", columns == expected_columns,
               observed=columns, expected=expected_columns)
    stats = con.execute(f"""
        SELECT count(*) AS rows,
               count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS unique_keys,
               count(DISTINCT cancer_id) AS cancers,
               count_if(cancer_id IS NULL OR trim(cancer_id)='' OR
                        lncrna_id IS NULL OR trim(lncrna_id)='' OR
                        pathway_id IS NULL OR trim(pathway_id)='') AS null_keys,
               count_if(single_cell_available) AS available_rows,
               count_if(NOT single_cell_available) AS unavailable_rows,
               count_if(single_cell_available AND
                        (single_cell_replication_probability IS NULL OR
                         NOT isfinite(single_cell_replication_probability) OR
                         single_cell_replication_probability < 0 OR
                         single_cell_replication_probability > 1)) AS bad_available,
               count_if(NOT single_cell_available AND
                        single_cell_replication_probability IS NOT NULL) AS filled_unavailable,
               count_if(NOT single_cell_available AND
                        (single_cell_unavailable_reason IS NULL OR
                         trim(single_cell_unavailable_reason)='')) AS missing_reason,
               count_if(single_cell_available AND
                        single_cell_unavailable_reason IS NOT NULL) AS available_with_reason,
               count_if(direct_target_evidence) AS direct_evidence,
               count_if(family_to_exact_broadcast) AS family_broadcast,
               count_if(changes_primary_ranking) AS primary_change,
               stddev_pop(single_cell_replication_probability) AS probability_sd
        FROM read_parquet('{pred}')
    """).fetchone()
    checks.add("prediction_3_3m_unique_33c",
               int(stats[0]) == EXPECTED_ROWS and int(stats[1]) == EXPECTED_ROWS
               and int(stats[2]) == 33 and int(stats[3]) == 0,
               observed={"rows": int(stats[0]), "unique_keys": int(stats[1]),
                         "cancers": int(stats[2]), "null_keys": int(stats[3])})
    checks.add("prediction_954541_available_2345459_unavailable",
               int(stats[4]) == EXPECTED_AVAILABLE and int(stats[5]) == EXPECTED_UNAVAILABLE,
               observed={"available": int(stats[4]), "unavailable": int(stats[5])})
    checks.add("prediction_probability_null_reason_semantics",
               int(stats[6]) == 0 and int(stats[7]) == 0 and int(stats[8]) == 0
               and int(stats[9]) == 0 and float(stats[13]) > 0,
               observed={"bad_available": int(stats[6]),
                         "filled_unavailable": int(stats[7]),
                         "missing_reason": int(stats[8]),
                         "available_with_reason": int(stats[9]),
                         "probability_sd": float(stats[13])})
    checks.add("prediction_no_direct_family_primary_effect",
               int(stats[10]) == 0 and int(stats[11]) == 0 and int(stats[12]) == 0,
               observed={"direct": int(stats[10]), "family": int(stats[11]),
                         "primary": int(stats[12])})

    metadata = con.execute(f"""
        SELECT count_if(analysis_version <> '{ANALYSIS_VERSION}') AS bad_version,
               count_if(module_id <> 'single_cell') AS bad_module,
               count_if(adapter_format <> '{ADAPTER_FORMAT}') AS bad_adapter,
               count_if(target_level <> 'cancer_x_lncrna_x_exact_pathway') AS bad_target,
               count_if(source_target_level <>
                        'dataset_x_celltype_x_lncrna_x_exact_pathway') AS bad_source_target,
               count_if(aggregation <>
                        'MEAN_OF_DONOR_OR_DATASET_BLOCKED_OOF_PREDICTIONS') AS bad_aggregation,
               count_if(availability_encoding <> 'null_with_reason') AS bad_encoding
        FROM read_parquet('{pred}')
    """).fetchone()
    checks.add("prediction_metadata_uniform", all(int(value) == 0 for value in metadata),
               observed=list(map(int, metadata)), expected=[0] * len(metadata))

    prediction_minus_candidate = int(con.execute(f"""
        SELECT count(*) FROM (
          SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet('{pred}')
          EXCEPT
          SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet('{cand}')
        )
    """).fetchone()[0])
    candidate_minus_prediction = int(con.execute(f"""
        SELECT count(*) FROM (
          SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet('{cand}')
          EXCEPT
          SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet('{pred}')
        )
    """).fetchone()[0])
    checks.add("prediction_candidate_bidirectional_exact",
               prediction_minus_candidate == 0 and candidate_minus_prediction == 0,
               observed={"prediction_minus_candidate": prediction_minus_candidate,
                         "candidate_minus_prediction": candidate_minus_prediction})

    exact_stats = con.execute(f"""
        SELECT count(*) AS rows,
               count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS unique_keys,
               count_if(NOT exact_pathway_only) AS non_exact
        FROM read_parquet('{exact}')
    """).fetchone()
    checks.add("exact_source_unique_literal_pathway",
               int(exact_stats[0]) == EXPECTED_EXACT_ROWS
               and int(exact_stats[1]) == EXPECTED_EXACT_ROWS
               and int(exact_stats[2]) == 0,
               observed=tuple(map(int, exact_stats)))
    reconstructed_difference = int(con.execute(f"""
        WITH sparse AS (
          SELECT cancer_id, lncrna_id, pathway_id,
                 single_cell_replication_probability AS probability,
                 single_cell_available AS available,
                 single_cell_unavailable_reason AS reason
          FROM read_parquet('{exact}')
        ), expected AS (
          SELECT c.cancer_id, c.lncrna_id, c.pathway_id,
                 s.probability,
                 coalesce(s.available, false) AS available,
                 CASE WHEN s.cancer_id IS NULL
                      THEN 'NO_SINGLE_CELL_EXACT_PAIR_MEASUREMENT'
                      ELSE s.reason END AS reason
          FROM read_parquet('{cand}') AS c
          LEFT JOIN sparse AS s USING (cancer_id, lncrna_id, pathway_id)
        ), observed AS (
          SELECT cancer_id, lncrna_id, pathway_id,
                 single_cell_replication_probability AS probability,
                 single_cell_available AS available,
                 single_cell_unavailable_reason AS reason
          FROM read_parquet('{pred}')
        )
        SELECT count(*) FROM expected FULL OUTER JOIN observed
        USING (cancer_id, lncrna_id, pathway_id)
        WHERE expected.cancer_id IS NULL OR observed.cancer_id IS NULL
           OR NOT (expected.probability IS NOT DISTINCT FROM observed.probability)
           OR expected.available != observed.available
           OR NOT (expected.reason IS NOT DISTINCT FROM observed.reason)
    """).fetchone()[0])
    checks.add("prediction_independently_reconstructed_rowwise",
               reconstructed_difference == 0,
               observed=reconstructed_difference, expected=0)

    recomputed_adapter_observed = {
        "rows": int(stats[0]),
        "distinct_keys": int(stats[1]),
        "available_rows": int(stats[4]),
        "unavailable_rows": int(stats[5]),
        "bad_available_rows": int(stats[6]),
        "filled_unavailable_rows": int(stats[7]),
        "missing_unavailable_reason_rows": int(stats[8]),
        "analysis_version_mismatch_rows": int(metadata[0]),
        "changes_primary_ranking_rows": int(stats[12]),
        "family_broadcast_rows": int(stats[11]),
        "direct_target_evidence_rows": int(stats[10]),
    }
    checks.add("adapter_audit_recomputed_exactly",
               adapter_audit.get("observed") == recomputed_adapter_observed,
               observed=recomputed_adapter_observed,
               expected=adapter_audit.get("observed"))
    checks.add("adapter_audit_expected_counts",
               adapter_audit.get("expected_from_independent_audit") == {
                   "candidate_rows": EXPECTED_ROWS,
                   "exact_source_rows": EXPECTED_EXACT_ROWS,
                   "available_rows": EXPECTED_AVAILABLE,
                   "unavailable_rows": EXPECTED_UNAVAILABLE,
               })
    con.close()

    material_hash_after = directory_sha256(material)
    material_file_hashes_after = {name: file_sha256(path) for name, path in material_files.items()}
    checks.add("materialization_unchanged_during_post_audit",
               material_hash_before == material_hash_after
               and material_file_hashes_before == material_file_hashes_after,
               observed={"before": material_hash_before, "after": material_hash_after})

    status = "PASS" if checks.passed else "FAIL"
    report = {
        "format": AUDIT_FORMAT,
        "status": status,
        "fail_closed": True,
        "analysis_version": ANALYSIS_VERSION,
        "audit_id": "V32-SINGLE-CELL-FUSION-ADAPTER-20260826-R1-POST-AUDIT",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "independence": {
            "adapter_or_materializer_imported": False,
            "materialization_modified": False,
            "training_invoked": False,
            "remote_read_or_write": False,
            "production_deployed": False,
            "expected_table_reconstructed_from_candidate_and_exact_source": True,
        },
        "materialization": {
            "path": str(material),
            "directory_sha256_before": material_hash_before,
            "directory_sha256_after": material_hash_after,
            "file_hashes": {
                "prediction_sha256": prediction_sha,
                "adapter_audit_sha256": adapter_audit_sha,
                "fusion_binding_sha256": fusion_binding_sha,
                "success_sha256": success_sha,
            },
        },
        "dependencies": {
            "candidate_authority": {"path": str(candidate_path), "sha256": candidate_sha},
            "single_cell_exact_source": {"path": str(exact_path), "sha256": exact_sha},
            "original_audit_binding": {
                "path": str(original_binding_path), "sha256": original_binding_sha,
            },
            "original_audit_report": {
                "path": str(original_report_path), "sha256": original_report_sha,
            },
        },
        "code_hashes": {
            "adapter": file_sha256(adapter_code_path),
            "binding_materializer": file_sha256(binding_code_path),
            "materializer_runner": file_sha256(materializer_path),
        },
        "prediction": {
            "rows": int(stats[0]), "unique_keys": int(stats[1]),
            "cancers": int(stats[2]), "available_rows": int(stats[4]),
            "unavailable_rows": int(stats[5]),
            "probability_sd": float(stats[13]),
            "prediction_minus_candidate": prediction_minus_candidate,
            "candidate_minus_prediction": candidate_minus_prediction,
            "rowwise_reconstruction_difference": reconstructed_difference,
            "bad_available_rows": int(stats[6]),
            "filled_unavailable_rows": int(stats[7]),
            "missing_unavailable_reason_rows": int(stats[8]),
            "direct_target_evidence_rows": int(stats[10]),
            "family_to_exact_broadcast_rows": int(stats[11]),
            "changes_primary_ranking_rows": int(stats[12]),
        },
        "release_decision": {
            "fusion_input_accepted": status == "PASS",
            "eligible_for_multimodal_fusion_input": status == "PASS",
            "primary_ranking_may_be_changed": False,
            "direct_target_evidence": False,
            "family_to_exact_broadcast": False,
            "production_deployed": False,
        },
        "checks": checks.rows,
        "failed_checks": [row["check"] for row in checks.rows if row["status"] != "PASS"],
    }
    report_path = output / "POST_MATERIALIZATION_AUDIT.json"
    _atomic_json(report_path, report)
    report_sha = file_sha256(report_path)
    auditor_path = Path(__file__).resolve()
    runner_path = repo / "scripts" / "audit_v32_single_cell_fusion_post_materialization.py"
    test_path = repo / "tests" / "test_v32_single_cell_fusion_post_audit.py"
    post_binding = {
        "format": BINDING_FORMAT,
        "status": status,
        "fail_closed": True,
        "analysis_version": ANALYSIS_VERSION,
        "post_materialization_audit_path": str(report_path),
        "post_materialization_audit_sha256": report_sha,
        "materialization_path": str(material),
        "materialization_directory_sha256": material_hash_after,
        "prediction_sha256": prediction_sha,
        "adapter_audit_sha256": adapter_audit_sha,
        "fusion_binding_sha256": fusion_binding_sha,
        "original_independent_audit_binding_sha256": original_binding_sha,
        "audit_code_hashes": {
            "single_cell_fusion_post_audit.py": file_sha256(auditor_path),
            "audit_v32_single_cell_fusion_post_materialization.py": (
                file_sha256(runner_path) if runner_path.is_file() else None
            ),
            "test_v32_single_cell_fusion_post_audit.py": (
                file_sha256(test_path) if test_path.is_file() else None
            ),
        },
        "release_decision": report["release_decision"],
    }
    post_binding_path = output / "POST_MATERIALIZATION_AUDIT_BINDING.json"
    _atomic_json(post_binding_path, post_binding)
    post_binding_sha = file_sha256(post_binding_path)
    completion = {
        "status": status,
        "post_materialization_audit_path": str(report_path),
        "post_materialization_audit_sha256": report_sha,
        "post_materialization_binding_path": str(post_binding_path),
        "post_materialization_binding_sha256": post_binding_sha,
        "prediction_sha256": prediction_sha,
        "adapter_audit_sha256": adapter_audit_sha,
        "fusion_binding_sha256": fusion_binding_sha,
        "original_independent_audit_binding_sha256": original_binding_sha,
        "failed_checks": report["failed_checks"],
        "eligible_for_multimodal_fusion_input": status == "PASS",
    }
    _atomic_json(output / ("SUCCESS.json" if status == "PASS" else "FAILURE.json"), completion)
    if status != "PASS":
        raise FusionPostAuditError(f"post-materialization audit failed: {report['failed_checks']}")
    return completion


__all__ = [
    "AUDIT_FORMAT", "BINDING_FORMAT", "FusionPostAuditError",
    "directory_sha256", "file_sha256", "run_post_materialization_audit",
    "verify_post_binding",
]
