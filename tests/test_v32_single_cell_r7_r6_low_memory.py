from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import pytest

from cc_hhgt.v32.single_cell_r7_streaming import estimate_streaming_resources
from scripts.transfer_v32_payloads_to_local_gpu import (
    build_plan,
    load_copy_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "scripts" / "run_v32_single_cell_r7_streaming.py"
    spec = importlib.util.spec_from_file_location("r7_r6_low_memory_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sarc_gate():
    path = ROOT / "scripts" / "run_v32_single_cell_r7_r6_sarc_gate.py"
    spec = importlib.util.spec_from_file_location("r7_r6_sarc_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sarc_r3_gate():
    path = ROOT / "scripts" / "run_v32_single_cell_r7_r7_sarc_gate.py"
    spec = importlib.util.spec_from_file_location("r7_r7_sarc_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sarc_r4_gate():
    path = ROOT / "scripts" / "run_v32_single_cell_r7_r8_sarc_gate.py"
    spec = importlib.util.spec_from_file_location("r7_r8_sarc_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r6_bridge_records_superseded_code_and_immutable_memory_audit() -> None:
    artifact = ROOT / "artifacts" / "single_cell_r7_streaming_tool_bridge_20260829_r6"
    manifest_path = artifact / "COPY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    validation = json.loads((artifact / "VALIDATION.json").read_text(encoding="utf-8"))
    assert manifest_sha == (
        "910c52e615df271672a6bf4360c5acfbcdc2ba4d66604e768b62e68fafbf913a"
    )
    assert manifest["entry_count"] == 8
    assert manifest["total_bytes"] == 128_030
    runner_record = next(
        row
        for row in manifest["entries"]
        if row["target_path"].endswith("run_v32_single_cell_r7_streaming.py")
    )
    assert runner_record["sha256"] == (
        "a271a19c1282741f3c6406de6c707ee06201ab30b53cad5e2486dd2f85cec967"
    )
    assert validation["immutable_memory_audit_included"] is True
    assert validation["memory_limit_bytes"] == 512 * 1024**2
    assert validation["memory_limit_increased"] is False
    assert validation["association_evidence_accumulated_in_python"] is False
    assert validation["hnsc_overwrite_permitted"] is False


def test_memory_audit_distinguishes_snapshot_from_formal_gate() -> None:
    path = ROOT / "artifacts" / (
        "single_cell_r7_sarc_memory_recovery_audit_20260829_r1"
    ) / "MEMORY_ROOT_CAUSE_AND_REMEDIATION.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    earlier = audit["hnsc_memory_measurement_reconciliation"]["earlier_pilot_snapshot"]
    formal = audit["hnsc_memory_measurement_reconciliation"]["formal_r3_supervised_run"]
    assert earlier["reported_child_vmhwm_kib"] == 348_788
    assert earlier["full_lifecycle_maximum_proven"] is False
    assert formal["max_observed_child_vmhwm_bytes"] == 532_205_568
    assert formal["process_group_rss_sum"] is False
    assert audit["gate"]["comparison"] == (
        "max(child_current_VmRSS, child_VmHWM) > gate"
    )


def test_sarc_r6_plan_accounts_for_external_sort_and_remains_below_gate() -> None:
    estimate = estimate_streaming_resources(
        cells=6_228,
        features=35_446,
        protein_genes=19_232,
        lncrnas=15_502,
        pathways=2_135,
        donor_celltype_groups=35,
        nnz=20_377_019,
        chunk_cells=64,
        association_pathway_block=32,
    )
    assert estimate["estimated_peak_ram_bytes"] == 471_580_560
    assert estimate["estimated_peak_ram_bytes"] < 512 * 1024**2
    assert estimate["external_sort_phase_bytes"] == 93_165_280
    assert estimate["association_evidence_accumulated_in_python"] is False
    runner = _load_runner()
    assert runner.build_parser().parse_args(
        [
            "--mode",
            "plan",
            "--r7-root",
            "r7",
            "--server-preflight-json",
            "preflight.json",
            "--expected-server-preflight-sha256",
            "a" * 64,
            "--expected-run-status-sha256",
            "b" * 64,
            "--cancer-id",
            "SARC",
        ]
    ).association_pathway_block == 32


def test_sarc_gate_is_single_cancer_and_keeps_512_mib() -> None:
    gate = _load_sarc_gate()
    assert gate.CANCER == "SARC"
    assert gate.MEMORY_LIMIT_BYTES == 512 * 1024**2
    assert gate.POLL_SECONDS == 0.25
    assert gate.TOOL_MANIFEST_SHA256 == (
        "910c52e615df271672a6bf4360c5acfbcdc2ba4d66604e768b62e68fafbf913a"
    )
    argv = gate.runner_args(
        mode="run",
        plan_path=None,
        output_root=Path("./data/CancerLncAtlas/results/model/new_sarc"),
    )
    assert "--cancer-id" in argv and argv[argv.index("--cancer-id") + 1] == "SARC"
    assert argv[argv.index("--association-pathway-block") + 1] == "32"
    assert "HNSC" not in argv


def test_sarc_gate_rejects_plan_without_low_memory_contract() -> None:
    gate = _load_sarc_gate()
    valid = {
        "cancer_id": "SARC",
        "source_generation": "V3.2_R7_FRESH_FROM_RAW_H5",
        "raw_h5_full_sha256_verified": True,
        "donor_is_biological_replicate": True,
        "no_cell_level_pathway_output": True,
        "resource_estimate": {
            "association_bh_external_spill_enabled": True,
            "association_evidence_accumulated_in_python": False,
            "runtime_baseline_budget_bytes": 352 * 1024**2,
            "estimated_peak_ram_bytes": 471_580_560,
        },
    }
    gate.validate_plan(valid)
    valid["resource_estimate"]["association_evidence_accumulated_in_python"] = True
    try:
        gate.validate_plan(valid)
    except gate.SarcGateError as exc:
        assert "no_python_evidence" in str(exc)
    else:
        raise AssertionError("SARC gate accepted Python evidence accumulation")


def test_parquet_writer_uses_explicit_schema_across_null_and_text_batches(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    schema = pa.schema(
        [
            ("unavailable_reason", pa.string()),
            ("score", pa.float32()),
        ]
    )
    empty = pd.DataFrame(
        {
            "unavailable_reason": pd.Series(dtype="object"),
            "score": pd.Series(dtype="float32"),
        }
    )
    cases = (
        (
            "null_then_text",
            [
                pd.DataFrame(
                    {"unavailable_reason": [None, pd.NA], "score": [None, None]}
                ),
                empty,
                pd.DataFrame(
                    {"unavailable_reason": ["中文", ""], "score": [1.0, 2.0]}
                ),
            ],
            [None, None, "中文", ""],
        ),
        (
            "text_then_null",
            [
                pd.DataFrame(
                    {"unavailable_reason": ["中文", ""], "score": [1.0, 2.0]}
                ),
                empty,
                pd.DataFrame(
                    {"unavailable_reason": [None, pd.NA], "score": [None, None]}
                ),
            ],
            ["中文", "", None, None],
        ),
    )
    for name, batches, expected in cases:
        path = tmp_path / f"{name}.parquet"
        writer = runner._ParquetFrameWriter(path, schema=schema)
        for batch in batches:
            writer.write(batch)
        assert writer.close() == 4
        assert pq.read_schema(path).equals(schema, check_metadata=False)
        table = pq.read_table(path)
        assert table.schema.equals(schema, check_metadata=False)
        assert table.column("unavailable_reason").to_pylist() == expected


def test_parquet_writer_emits_schema_valid_file_with_only_empty_batches(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    schema = pa.schema([("reason", pa.string()), ("available", pa.bool_())])
    path = tmp_path / "empty.parquet"
    writer = runner._ParquetFrameWriter(path, schema=schema)
    writer.write(pd.DataFrame({"reason": [], "available": []}))
    assert writer.close() == 0
    assert pq.read_schema(path).equals(schema, check_metadata=False)
    assert pq.read_table(path).num_rows == 0


def test_r7_tool_bridge_records_explicit_schema_runner_and_both_audits() -> None:
    artifact = ROOT / "artifacts" / "single_cell_r7_streaming_tool_bridge_20260829_r7"
    manifest_path = artifact / "COPY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    validation = json.loads((artifact / "VALIDATION.json").read_text(encoding="utf-8"))
    assert manifest_sha == (
        "2216f5208e5f293bd899621911461e6ee71cf693c11d443350817989761c0e79"
    )
    assert manifest["entry_count"] == 10
    assert manifest["total_bytes"] == 136_130
    assert validation["explicit_arrow_schema_before_first_batch"] is True
    assert validation["first_batch_schema_inference"] is False
    assert validation["immutable_memory_audit_included"] is True
    assert validation["immutable_schema_audit_included"] is True
    assert validation["local_selected_tests"] == "33_PASSED"


def test_sarc_r3_supervisor_binds_r7_tool_and_keeps_gate() -> None:
    gate = _load_sarc_r3_gate()
    assert gate.CANCER == "SARC"
    assert gate.MEMORY_LIMIT_BYTES == 512 * 1024**2
    assert gate.POLL_SECONDS == 0.25
    assert gate.TOOL_ROOT == Path(
        "./data/CancerLncAtlas/runtime/tools/"
        "single_cell_r7_streaming_20260829_r7"
    )
    assert gate.TOOL_MANIFEST_SHA256 == (
        "2216f5208e5f293bd899621911461e6ee71cf693c11d443350817989761c0e79"
    )
    assert gate.TOOL_SHAS["scripts/run_v32_single_cell_r7_streaming.py"] == (
        "dc2dd3049a3da065488c1f1278c8549316896277a918ed916e565d02d73839ef"
    )


def test_serial_compartment_bh_matches_original_query_by_exact_key_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    raw = pd.DataFrame(
        {
            "dataset_id": ["数据集"] * 12,
            "cancer_id": ["SARC"] * 12,
            "compartment_order": np.asarray(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int8
            ),
            "compartment": ["malignant"] * 5 + ["immune"] * 4 + ["stromal"] * 3,
            "lncrna_id": [
                "L2", "长链一", "长链一", "L3", "L4",
                "L2", "免疫一", "L3", "L4",
                "基质二", "基质一", "L9",
            ],
            "lncrna_symbol": [
                "S2", "符号一", "符号一", "S3", "S4",
                "S2", "免疫符号", "S3", "S4",
                "基质符号二", "基质符号一", "S9",
            ],
            "pathway_id": [
                "P1", "通路二", "P1", "P1", "P2",
                "P1", "通路二", "P1", "P2",
                "P2", "通路一", "P3",
            ],
            "n_donors": np.asarray([8] * 5 + [7] * 4 + [6] * 3, dtype=np.int64),
            "spearman_rho": [
                0.8, -0.7, 0.6, 0.5, -0.5,
                0.9, 0.7, -0.6, 0.5,
                0.75, -0.65, 0.55,
            ],
            "nominal_p": [
                0.01, 0.01, 0.02, 0.2, 0.04,
                0.005, 0.05, 0.05, 0.3,
                0.02, 0.02, 0.4,
            ],
            "total_tests": np.asarray(
                [100] * 5 + [20] * 4 + [30] * 3, dtype=np.int64
            ),
        }
    )
    exact_key = ["compartment_order", "lncrna_id", "pathway_id"]
    assert not raw.duplicated(exact_key).any()
    assert (raw.groupby("compartment_order").total_tests.first() > raw.groupby("compartment_order").size()).all()
    reference_path = tmp_path / "reference.parquet"
    raw.to_parquet(reference_path, index=False)
    reference = duckdb.connect().execute(
        f"""
        WITH ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY compartment_order
                ORDER BY nominal_p, lncrna_id, pathway_id
            ) AS retained_rank
            FROM read_parquet('{reference_path.as_posix()}')
        ), adjusted AS (
            SELECT *, LEAST(1.0, MIN(
                nominal_p * total_tests / retained_rank
            ) OVER (
                PARTITION BY compartment_order
                ORDER BY retained_rank DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )) AS bh_q_global_tests
            FROM ranked
        )
        SELECT compartment_order, lncrna_id, pathway_id, retained_rank,
               bh_q_global_tests,
               bh_q_global_tests <= 0.10 AS fdr_0_10_pass
        FROM adjusted
        """
    ).fetchdf()

    first_raw = tmp_path / "first_raw.parquet"
    first_output = tmp_path / "first_evidence.parquet"
    first_scratch = tmp_path / "first_scratch"
    raw.sample(frac=1.0, random_state=17).to_parquet(first_raw, index=False)
    monkeypatch.setattr(runner.shutil, "rmtree", lambda _path: None)
    assert runner._finalize_association_evidence(
        raw_path=first_raw,
        output_path=first_output,
        scratch=first_scratch,
        raw_rows=len(raw),
    ) == len(raw)
    staged_parts = []
    for order in (0, 1, 2):
        staged = pd.read_parquet(
            first_scratch / f"compartment_order={order}.adjusted.parquet"
        )
        staged.insert(0, "compartment_order", order)
        staged_parts.append(staged)
    staged = pd.concat(staged_parts, ignore_index=True)
    comparison_columns = exact_key + [
        "retained_rank", "bh_q_global_tests"
    ]
    pd.testing.assert_frame_equal(
        staged[comparison_columns].sort_values(exact_key).reset_index(drop=True),
        reference[comparison_columns].sort_values(exact_key).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=1e-15,
    )
    observed = pd.read_parquet(first_output)
    observed_keyed = observed.merge(
        raw[exact_key + ["compartment"]],
        on=["lncrna_id", "pathway_id", "compartment"],
        validate="one_to_one",
    )
    final_reference = reference[exact_key + ["bh_q_global_tests", "fdr_0_10_pass"]]
    pd.testing.assert_frame_equal(
        observed_keyed[exact_key + ["bh_q_global_tests", "fdr_0_10_pass"]]
        .sort_values(exact_key)
        .reset_index(drop=True),
        final_reference.sort_values(exact_key).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=1e-15,
    )

    monkeypatch.undo()
    second_raw = tmp_path / "second_raw.parquet"
    second_output = tmp_path / "second_evidence.parquet"
    raw.sample(frac=1.0, random_state=29).to_parquet(second_raw, index=False)
    runner._finalize_association_evidence(
        raw_path=second_raw,
        output_path=second_output,
        scratch=tmp_path / "second_scratch",
        raw_rows=len(raw),
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first_output),
        pd.read_parquet(second_output),
        check_dtype=False,
        check_exact=True,
    )
    assert runner.ASSOCIATION_EXECUTION_STRATEGY == (
        "SERIAL_PER_COMPARTMENT_EXTERNAL_SORT_V1"
    )
    assert runner.ASSOCIATION_BH_FAMILY == "COMPARTMENT_ORDER"
    assert runner.ASSOCIATION_STAGE_KEY_POLICY.endswith("NO_PARQUET_ROWID_JOIN")


def test_bh_handles_zero_rows_and_an_empty_formal_family(tmp_path: Path) -> None:
    runner = _load_runner()
    empty_raw = tmp_path / "empty_raw.parquet"
    runner._ParquetFrameWriter(
        empty_raw, schema=runner._raw_evidence_schema()
    ).close()
    empty_output = tmp_path / "empty_evidence.parquet"
    assert runner._finalize_association_evidence(
        raw_path=empty_raw,
        output_path=empty_output,
        scratch=tmp_path / "empty_scratch",
        raw_rows=0,
    ) == 0
    assert pq.read_table(empty_output).num_rows == 0
    assert pq.read_schema(empty_output).equals(
        runner._evidence_schema(), check_metadata=False
    )

    raw = pd.DataFrame(
        {
            "dataset_id": ["D", "D"],
            "cancer_id": ["SARC", "SARC"],
            "compartment_order": np.asarray([0, 2], dtype=np.int8),
            "compartment": ["malignant", "stromal"],
            "lncrna_id": ["L1", "基质一"],
            "lncrna_symbol": ["S1", "基质符号"],
            "pathway_id": ["P1", "通路一"],
            "n_donors": np.asarray([8, 6], dtype=np.int64),
            "spearman_rho": [0.7, -0.8],
            "nominal_p": [0.01, 0.02],
            "total_tests": np.asarray([100, 30], dtype=np.int64),
        }
    )
    missing_family_raw = tmp_path / "missing_family_raw.parquet"
    missing_family_output = tmp_path / "missing_family_evidence.parquet"
    raw.to_parquet(missing_family_raw, index=False)
    assert runner._finalize_association_evidence(
        raw_path=missing_family_raw,
        output_path=missing_family_output,
        scratch=tmp_path / "missing_family_scratch",
        raw_rows=2,
    ) == 2
    assert pd.read_parquet(missing_family_output).compartment.tolist() == [
        "malignant",
        "stromal",
    ]


def test_bh_rejects_nonunique_raw_exact_key(tmp_path: Path) -> None:
    runner = _load_runner()
    raw = pd.DataFrame(
        {
            "dataset_id": ["D", "D"],
            "cancer_id": ["SARC", "SARC"],
            "compartment_order": np.asarray([0, 0], dtype=np.int8),
            "compartment": ["malignant", "malignant"],
            "lncrna_id": ["L1", "L1"],
            "lncrna_symbol": ["S1", "S1"],
            "pathway_id": ["P1", "P1"],
            "n_donors": np.asarray([8, 8], dtype=np.int64),
            "spearman_rho": [0.7, 0.6],
            "nominal_p": [0.01, 0.02],
            "total_tests": np.asarray([100, 100], dtype=np.int64),
        }
    )
    raw_path = tmp_path / "duplicate_raw.parquet"
    raw.to_parquet(raw_path, index=False)
    with pytest.raises(
        runner.R7StreamingRunError,
        match="raw exact key is not unique",
    ):
        runner._finalize_association_evidence(
            raw_path=raw_path,
            output_path=tmp_path / "duplicate_evidence.parquet",
            scratch=tmp_path / "duplicate_scratch",
            raw_rows=2,
        )


def test_r8_tool_bridge_binds_equivalent_bh_runner_and_three_audits() -> None:
    artifact = ROOT / "artifacts" / "single_cell_r7_streaming_tool_bridge_20260829_r8"
    manifest, manifest_sha = load_copy_manifest(artifact / "COPY_MANIFEST.json")
    plan = build_plan(manifest, manifest_sha256=manifest_sha)
    validation = json.loads((artifact / "VALIDATION.json").read_text(encoding="utf-8"))
    assert manifest_sha == (
        "89360f86e1c96f36c69fcf347ed752ba657bb388ebfe62fdd0500a5a852ab55f"
    )
    assert plan["entry_count"] == 12
    assert plan["total_bytes"] == 154_263
    assert validation["association_bh_family"] == "COMPARTMENT_ORDER"
    assert validation["association_family_scope_changed"] is False
    assert validation["association_total_tests_denominator_changed"] is False
    assert validation["parquet_physical_rowid_join_used"] is False
    assert validation["duckdb_memory_limit"] == "64MB"
    assert validation["large_spool_dual_gate_smoke"] == "PASS"
    assert validation["immutable_duckdb_audit_included"] is True


def test_sarc_r4_supervisor_binds_r8_tool_and_unchanged_gate() -> None:
    gate = _load_sarc_r4_gate()
    assert gate.CANCER == "SARC"
    assert gate.MEMORY_LIMIT_BYTES == 512 * 1024**2
    assert gate.POLL_SECONDS == 0.25
    assert gate.TOOL_ROOT == Path(
        "./data/CancerLncAtlas/runtime/tools/"
        "single_cell_r7_streaming_20260829_r8"
    )
    assert gate.TOOL_MANIFEST_SHA256 == (
        "89360f86e1c96f36c69fcf347ed752ba657bb388ebfe62fdd0500a5a852ab55f"
    )
    assert gate.TOOL_SHAS["scripts/run_v32_single_cell_r7_streaming.py"] == (
        "1d91fac0de8246c57c5d5c2a7299d178abc964a9360af99f397d48eb7ddcddfa"
    )
