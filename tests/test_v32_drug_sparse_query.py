from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.drug_sparse_query import (
    CURRENT_V32_CORE_UNAVAILABLE,
    DrugSparseAssetError,
    DrugSparseQueryBundle,
    MODEL_OUTPUT_MISSING_FAIL_CLOSED,
    MODEL_RUN_AUDITED_UNAVAILABLE,
    NO_HELD_OUT_NATIVE_ASSAY,
    NO_MATCHED_LNCRNA_EXPRESSION,
    OUTSIDE_CONCEPTUAL_UNIVERSE,
    PROBABILITY_COLUMN,
    DUCKDB_TEMP_ENV,
    _bounded_duckdb,
    _parquet_relation,
    _validate_available_bindings,
    factor_core_entity_availability,
    factor_expression_fold_coverage,
    load_drug_sparse_query_bundle,
    write_drug_sparse_query_sidecars,
)
from cc_hhgt.v32.input_lineage import artifact_sha256


L_OK = "LNC:ENSG00000000001"
L_NO_ASSAY = "LNC:ENSG00000000002"
L_NO_EXPRESSION = "LNC:ENSG00000000003"
L_NO_CORE = "LNC:ENSG00000000004"
L_OUTPUT_MISSING = "LNC:ENSG00000000005"
D_OK = "DRUG:ok"
D_NO_ASSAY = "DRUG:no-assay"
D_NO_EXPRESSION = "DRUG:no-expression"
D_NO_CORE = "DRUG:no-core"
D_OUTPUT_MISSING = "DRUG:output-missing"


def _factor_frames() -> dict[str, pd.DataFrame]:
    exact = pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 6,
            "lncrna_id": [
                L_OK,
                L_NO_ASSAY,
                L_NO_EXPRESSION,
                L_NO_CORE,
                L_OUTPUT_MISSING,
                # A second pathway reaches the same Drug key; the conceptual
                # count must use DISTINCT cancer × lncRNA × drug keys.
                L_OK,
            ],
            "pathway_id": ["PATH:1"] * 5 + ["PATH:2"],
        }
    )
    edges = pd.DataFrame(
        {
            "pathway_id": ["PATH:1"] * 5 + ["PATH:2"],
            "drug_id": [
                D_OK,
                D_NO_ASSAY,
                D_NO_EXPRESSION,
                D_NO_CORE,
                D_OUTPUT_MISSING,
                D_OK,
            ],
        }
    )
    assay = pd.DataFrame(
        [
            {"cancer_id": "BRCA", "drug_id": D_OK, "cell_line_fold_id": 0},
            {
                "cancer_id": "BRCA",
                "drug_id": D_NO_EXPRESSION,
                "cell_line_fold_id": 1,
            },
            {"cancer_id": "BRCA", "drug_id": D_NO_CORE, "cell_line_fold_id": 2},
            {
                "cancer_id": "BRCA",
                "drug_id": D_OUTPUT_MISSING,
                "cell_line_fold_id": 3,
            },
        ]
    )
    expression = pd.DataFrame(
        [
            {"cancer_id": "BRCA", "lncrna_id": L_OK, "fold_id": 0},
            # Expression exists, but not in the drug's held-out assay fold.
            {"cancer_id": "BRCA", "lncrna_id": L_NO_EXPRESSION, "fold_id": 0},
            {"cancer_id": "BRCA", "lncrna_id": L_NO_CORE, "fold_id": 2},
            {"cancer_id": "BRCA", "lncrna_id": L_OUTPUT_MISSING, "fold_id": 3},
        ]
    )
    core = factor_core_entity_availability(
        lncrna_ids_by_fold={
            0: [L_OK],
            1: [L_NO_EXPRESSION],
            2: [L_NO_CORE],
            3: [L_OUTPUT_MISSING],
            4: [],
        },
        cancer_ids_by_fold={fold: ["BRCA"] for fold in range(5)},
        drug_target_ids_by_fold={
            0: [D_OK],
            1: [D_NO_EXPRESSION],
            # Deliberately absent in fold 2: CURRENT_V32_CORE_UNAVAILABLE.
            2: [],
            3: [D_OUTPUT_MISSING],
            4: [],
        },
    )
    return {
        "exact": exact,
        "edges": edges,
        "assay": assay,
        "expression": expression,
        "core": core,
    }


def _available_frame(*, empty: bool = False) -> pd.DataFrame:
    rows = [] if empty else [
        {
            "cancer_id": "BRCA",
            "lncrna_id": L_OK,
            "drug_id": D_OK,
            PROBABILITY_COLUMN: 0.83,
            "availability": True,
            "prediction_fold_mask": 1,
            "cell_line_folds_with_prediction": 1,
        }
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "cancer_id",
            "lncrna_id",
            "drug_id",
            PROBABILITY_COLUMN,
            "availability",
            "prediction_fold_mask",
            "cell_line_folds_with_prediction",
        ],
    ).astype(
        {
            "cancer_id": "string",
            "lncrna_id": "string",
            "drug_id": "string",
            PROBABILITY_COLUMN: "float64",
            "availability": "bool",
            "prediction_fold_mask": "int64",
            "cell_line_folds_with_prediction": "int64",
        }
    )


def _write_bundle(
    root: Path,
    *,
    status: str = "SUCCESS",
    conceptual_candidate_rows: int = 25,
    available_frame: pd.DataFrame | None = None,
    factor_overrides: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    factors = _factor_frames()
    if factor_overrides:
        factors.update(factor_overrides)
    available_root = root / "drug_response_association" / "cancer-BRCA"
    available_root.mkdir(parents=True)
    if available_frame is None:
        available_frame = _available_frame(empty=status != "SUCCESS")
    available_frame.to_parquet(
        available_root / "part-00000.parquet", index=False
    )
    return write_drug_sparse_query_sidecars(
        bundle_root=root,
        exact_candidates=factors["exact"],
        pathway_drug_edges=factors["edges"],
        assay_fold_eligibility=factors["assay"],
        expression_fold_coverage=factors["expression"],
        core_entity_availability=factors["core"],
        available_predictions_path=root / "drug_response_association",
        conceptual_candidate_rows=conceptual_candidate_rows,
        training_run_id="v32-drug-sparse-test",
        model_run_status=status,
        model_run_unavailable_reason=(
            "NATIVE_ASSAYED_CANDIDATE_UNIVERSE_EXCEEDS_VALIDATED_RUNTIME"
            if status == "AUDITED_UNAVAILABLE"
            else None
        ),
    )


def _separate_manifest_authority(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, Path]:
    artifact_root = tmp_path / "artifacts"
    result = _write_bundle(artifact_root)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    manifest_path = authority_root / Path(str(result["manifest_path"])).name
    audit_path = authority_root / Path(str(result["validation_audit_path"])).name
    shutil.copy2(result["manifest_path"], manifest_path)
    shutil.copy2(result["validation_audit_path"], audit_path)
    return result, manifest_path, artifact_root


def test_loader_supports_separately_pinned_manifest_and_artifact_root(
    tmp_path: Path,
) -> None:
    result, manifest_path, artifact_root = _separate_manifest_authority(tmp_path)
    bundle = load_drug_sparse_query_bundle(
        manifest_path,
        expected_manifest_sha256=result["manifest_sha256"],
        artifact_root=artifact_root,
    )
    assert bundle.artifact_root == artifact_root.resolve()
    assert bundle.resolve("BRCA", L_OK, D_OK)["availability"] is True


def test_loader_artifact_root_fails_closed_on_wrong_root_escape_hash_and_missing(
    tmp_path: Path,
) -> None:
    result, manifest_path, artifact_root = _separate_manifest_authority(tmp_path)

    with pytest.raises(DrugSparseAssetError, match="missing or unhashable"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=result["manifest_sha256"],
            artifact_root=manifest_path.parent,
        )

    escaped_manifest = manifest_path.parent / "ESCAPED.json"
    escaped = json.loads(manifest_path.read_text(encoding="utf-8"))
    escaped["artifacts"]["exact_candidates"]["path"] = (
        "../artifacts/drug_sparse_query_factors/exact_candidates.parquet"
    )
    escaped_manifest.write_text(json.dumps(escaped), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="must be relative"):
        load_drug_sparse_query_bundle(
            escaped_manifest,
            expected_manifest_sha256=artifact_sha256(escaped_manifest),
            artifact_root=artifact_root,
        )

    wrong_hash_root = tmp_path / "same-name-wrong-hash"
    shutil.copytree(artifact_root, wrong_hash_root)
    wrong_exact = (
        wrong_hash_root
        / "drug_sparse_query_factors"
        / "exact_candidates.parquet"
    )
    wrong_exact.write_bytes(b"same-name-but-not-the-authorized-content")
    with pytest.raises(DrugSparseAssetError, match="SHA256 mismatch"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=result["manifest_sha256"],
            artifact_root=wrong_hash_root,
        )

    missing_root = tmp_path / "missing-file"
    shutil.copytree(artifact_root, missing_root)
    (
        missing_root
        / "drug_sparse_query_factors"
        / "expression_fold_coverage.parquet"
    ).unlink()
    with pytest.raises(DrugSparseAssetError, match="missing or unhashable"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=result["manifest_sha256"],
            artifact_root=missing_root,
        )
def test_sparse_resolver_distinguishes_available_and_each_absence_reason(
    tmp_path: Path,
) -> None:
    result = _write_bundle(tmp_path / "bundle")
    resolver = load_drug_sparse_query_bundle(
        result["manifest_path"],
        expected_manifest_sha256=result["manifest_sha256"],
    )

    available = resolver.resolve("brca", "ENSG00000000001.9", D_OK)
    assert available["availability"] is True
    assert available[PROBABILITY_COLUMN] == pytest.approx(0.83)
    assert available["failure_reason"] is None

    cases = [
        (("BRCA", L_OK, "DRUG:not-in-edge"), OUTSIDE_CONCEPTUAL_UNIVERSE),
        (("BRCA", L_NO_ASSAY, D_NO_ASSAY), NO_HELD_OUT_NATIVE_ASSAY),
        (("BRCA", L_NO_EXPRESSION, D_NO_EXPRESSION), NO_MATCHED_LNCRNA_EXPRESSION),
        (("BRCA", L_NO_CORE, D_NO_CORE), CURRENT_V32_CORE_UNAVAILABLE),
        (("BRCA", L_OUTPUT_MISSING, D_OUTPUT_MISSING), MODEL_OUTPUT_MISSING_FAIL_CLOSED),
    ]
    for key, reason in cases:
        observed = resolver.resolve(*key)
        assert observed["availability"] is False
        assert observed[PROBABILITY_COLUMN] is None
        assert observed["failure_reason"] == reason
        assert observed["tcga_patient_response_claimed"] is False

    manifest = resolver.manifest
    assert manifest["dense_candidate_table_materialized"] is False
    assert set(manifest["artifacts"]) == {
        "exact_candidates",
        "pathway_drug_edges",
        "assay_fold_eligibility",
        "expression_fold_coverage",
        "core_entity_availability",
        "available_predictions",
    }
    assert manifest["typed_absence_resolver"] is True
    assert manifest["available_fold_provenance_required"] is True
    audit_path = Path(result["validation_audit_path"])
    assert artifact_sha256(audit_path) == result["validation_audit_sha256"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["identities"]["fold_mask_popcount_equals_declared_fold_count"] is True


def test_audited_unavailable_is_global_but_outside_universe_has_precedence(
    tmp_path: Path,
) -> None:
    result = _write_bundle(tmp_path / "audited", status="AUDITED_UNAVAILABLE")
    resolver = load_drug_sparse_query_bundle(
        result["manifest_path"],
        expected_manifest_sha256=result["manifest_sha256"],
    )
    inside = resolver.resolve("BRCA", L_OK, D_OK)
    assert inside["failure_reason"] == MODEL_RUN_AUDITED_UNAVAILABLE
    assert inside[PROBABILITY_COLUMN] is None
    assert inside["failure_detail"].startswith("NATIVE_ASSAYED")
    outside = resolver.resolve("BRCA", L_OK, "DRUG:not-in-edge")
    assert outside["failure_reason"] == OUTSIDE_CONCEPTUAL_UNIVERSE


def test_expression_and_core_factor_builders_are_compact() -> None:
    mapping = pd.DataFrame(
        [
            {
                "dataset_id": "GDSC",
                "cell_line_id": f"C{index}",
                "canonical_model_id": f"M{index}",
                "cancer_id": "BRCA",
                "cell_line_fold_id": 0,
            }
            for index in range(5)
        ]
    )
    expression = pd.DataFrame(
        [
            {
                "dataset_id": "GDSC",
                "cell_line_id": f"C{index}",
                "lncrna_id": lncrna,
                "expression_value": float(index),
            }
            for lncrna, count in ((L_OK, 5), (L_NO_EXPRESSION, 2))
            for index in range(count)
        ]
    )
    coverage = factor_expression_fold_coverage(
        expression,
        mapping,
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=4,
    )
    assert coverage[["cancer_id", "lncrna_id", "fold_id"]].to_dict("records") == [
        {"cancer_id": "BRCA", "lncrna_id": L_OK, "fold_id": 0}
    ]
    core = factor_core_entity_availability(
        lncrna_ids_by_fold={fold: ([L_OK] if fold == 0 else []) for fold in range(5)},
        cancer_ids_by_fold={fold: ["BRCA"] for fold in range(5)},
        drug_target_ids_by_fold={fold: ([D_OK] if fold == 0 else []) for fold in range(5)},
    )
    assert len(core) == 7  # five cancers + one lncRNA + one drug-target record
    assert set(core.entity_type) == {"cancer", "lncRNA", "drug_target"}


def test_loader_rejects_hash_drift_missing_sidecar_and_wrong_manifest_hash(
    tmp_path: Path,
) -> None:
    result = _write_bundle(tmp_path / "hash")
    with pytest.raises(DrugSparseAssetError, match="manifest SHA256 mismatch"):
        load_drug_sparse_query_bundle(
            result["manifest_path"], expected_manifest_sha256="0" * 64
        )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    expression_path = (
        Path(result["manifest_path"]).parent
        / manifest["artifacts"]["expression_fold_coverage"]["path"]
    )
    pd.DataFrame(
        [{"cancer_id": "BRCA", "lncrna_id": L_OK, "fold_id": 4}]
    ).to_parquet(expression_path, index=False)
    with pytest.raises(DrugSparseAssetError, match="SHA256 mismatch"):
        load_drug_sparse_query_bundle(
            result["manifest_path"],
            expected_manifest_sha256=result["manifest_sha256"],
        )

    result = _write_bundle(tmp_path / "missing")
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    missing_path = (
        Path(result["manifest_path"]).parent
        / manifest["artifacts"]["core_entity_availability"]["path"]
    )
    missing_path.unlink()
    with pytest.raises(DrugSparseAssetError, match="missing or unhashable"):
        load_drug_sparse_query_bundle(
            result["manifest_path"],
            expected_manifest_sha256=result["manifest_sha256"],
        )


def test_loader_rejects_manifest_path_escape(tmp_path: Path) -> None:
    result = _write_bundle(tmp_path / "confined")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = (
        manifest_path.parent / manifest["artifacts"]["exact_candidates"]["path"]
    )
    escape = tmp_path / "escaped-exact.parquet"
    pd.read_parquet(source).to_parquet(escape, index=False)
    declaration = manifest["artifacts"]["exact_candidates"]
    declaration["path"] = "../escaped-exact.parquet"
    declaration["sha256"] = artifact_sha256(escape)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="must be relative"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )


def test_writer_recomputes_distinct_conceptual_count_and_available_bound(
    tmp_path: Path,
) -> None:
    # The factors contain two pathway routes to (BRCA, L_OK, D_OK), so this
    # specifically verifies DISTINCT keys rather than raw join rows.
    with pytest.raises(DrugSparseAssetError, match="declared=24, observed=25"):
        _write_bundle(
            tmp_path / "wrong-conceptual-count", conceptual_candidate_rows=24
        )

    too_many = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": f"LNC:ENSG{index:011d}",
                "drug_id": f"DRUG:outside-{index}",
                PROBABILITY_COLUMN: 0.5,
                "availability": True,
                "prediction_fold_mask": 1,
                "cell_line_folds_with_prediction": 1,
            }
            for index in range(100, 126)
        ]
    )
    with pytest.raises(DrugSparseAssetError, match="available_rows exceeds"):
        _write_bundle(tmp_path / "too-many-available", available_frame=too_many)


def test_writer_rejects_available_key_outside_or_without_same_fold_support(
    tmp_path: Path,
) -> None:
    outside = _available_frame()
    outside.loc[0, "drug_id"] = "DRUG:not-in-edge"
    with pytest.raises(DrugSparseAssetError, match="outside the factored conceptual"):
        _write_bundle(tmp_path / "outside-available", available_frame=outside)

    no_core = _available_frame()
    no_core.loc[0, "lncrna_id"] = L_NO_CORE
    no_core.loc[0, "drug_id"] = D_NO_CORE
    with pytest.raises(DrugSparseAssetError, match="per-mask-fold assay/expression/current-core"):
        _write_bundle(tmp_path / "unsupported-available", available_frame=no_core)


def test_loader_recomputes_conceptual_count_and_rejects_hashed_semantic_drift(
    tmp_path: Path,
) -> None:
    result = _write_bundle(tmp_path / "manifest-count")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["conceptual_candidate_rows"] = 24
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="does not match DISTINCT"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )

    # Now construct a bundle whose artifact hash and row declaration are both
    # updated after removing the available key's assay fold.  Hash validation
    # alone would accept it; the relational semantic gate must not.
    result = _write_bundle(tmp_path / "hashed-assay-contradiction")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = manifest["artifacts"]["assay_fold_eligibility"]
    assay_path = manifest_path.parent / declaration["path"]
    assay = pd.read_parquet(assay_path)
    assay = assay.loc[assay.drug_id.ne(D_OK)].reset_index(drop=True)
    assay.to_parquet(assay_path, index=False)
    declaration["sha256"] = artifact_sha256(assay_path)
    declaration["rows"] = len(assay)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="per-mask-fold assay/expression/current-core"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )


def test_loader_rejects_hash_consistent_available_key_outside_universe(
    tmp_path: Path,
) -> None:
    result = _write_bundle(tmp_path / "hashed-outside-available")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = manifest["artifacts"]["available_predictions"]
    available_root = manifest_path.parent / declaration["path"]
    part = next(available_root.rglob("*.parquet"))
    available = pd.read_parquet(part)
    available.loc[0, "drug_id"] = "DRUG:not-in-edge"
    available.to_parquet(part, index=False)
    declaration["sha256"] = artifact_sha256(available_root)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="outside the factored conceptual"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )


def test_dataset_expression_coverage_uses_canonical_model_union() -> None:
    mapping_rows: list[dict[str, object]] = []
    expression_rows: list[dict[str, object]] = []
    for dataset in ("GDSC", "PRISM"):
        for index in range(3):
            mapping_rows.append(
                {
                    "dataset_id": dataset,
                    "cell_line_id": f"{dataset}-C{index}",
                    "canonical_model_id": f"M{index}",
                    "cancer_id": "BRCA",
                    "cell_line_fold_id": 0,
                }
            )
            expression_rows.append(
                {
                    "dataset_id": dataset,
                    "cell_line_id": f"{dataset}-C{index}",
                    "lncrna_id": L_OK,
                    "expression_value": float(index),
                }
            )
    coverage = factor_expression_fold_coverage(
        pd.DataFrame(expression_rows),
        pd.DataFrame(mapping_rows),
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=4,
    )
    # Each dataset has three rows, but they are the same three canonical models;
    # 3 + 3 must not be reported as six independent models.
    assert coverage.empty

    mapping_rows.append(
        {
            "dataset_id": "PRISM",
            "cell_line_id": "PRISM-C3",
            "canonical_model_id": "M3",
            "cancer_id": "BRCA",
            "cell_line_fold_id": 0,
        }
    )
    expression_rows.append(
        {
            "dataset_id": "PRISM",
            "cell_line_id": "PRISM-C3",
            "lncrna_id": L_OK,
            "expression_value": 3.0,
        }
    )
    coverage = factor_expression_fold_coverage(
        pd.DataFrame(expression_rows),
        pd.DataFrame(mapping_rows),
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=4,
    )
    assert coverage.matched_expression_models.tolist() == [4]


@pytest.mark.parametrize(
    ("bad_fold", "message"),
    [
        (256, "folds outside 0..4"),
        (1.5, "non-integer fold IDs"),
        (None, "null fold IDs"),
    ],
)
def test_writer_rejects_wide_overflow_fractional_and_null_folds(
    tmp_path: Path, bad_fold: object, message: str
) -> None:
    assay = _factor_frames()["assay"].copy()
    assay["cell_line_fold_id"] = assay["cell_line_fold_id"].astype(object)
    assay.loc[0, "cell_line_fold_id"] = bad_fold
    with pytest.raises(DrugSparseAssetError, match=message):
        _write_bundle(
            tmp_path / f"bad-fold-{str(bad_fold).replace('.', '-')}",
            factor_overrides={"assay": assay},
        )


@pytest.mark.parametrize("role", ["exact", "edges", "assay", "expression", "core"])
def test_writer_rejects_duplicate_factor_keys(tmp_path: Path, role: str) -> None:
    frame = _factor_frames()[role]
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DrugSparseAssetError, match="duplicate keys"):
        _write_bundle(
            tmp_path / f"writer-duplicate-{role}",
            factor_overrides={role: duplicated},
        )


def test_writer_rejects_null_before_identifier_string_coercion(tmp_path: Path) -> None:
    exact = _factor_frames()["exact"].copy()
    exact.loc[0, "lncrna_id"] = pd.NA
    with pytest.raises(DrugSparseAssetError, match="null key"):
        _write_bundle(
            tmp_path / "writer-null-key", factor_overrides={"exact": exact}
        )


@pytest.mark.parametrize(
    ("mask", "fold_count", "message"),
    [
        (0, 1, "invalid prediction_fold_mask"),
        (32, 1, "invalid prediction_fold_mask"),
        (3, 1, "fold count does not equal fold-mask popcount"),
        (1, 2, "fold count does not equal fold-mask popcount"),
    ],
)
def test_writer_rejects_invalid_fold_masks_and_counts(
    tmp_path: Path, mask: int, fold_count: int, message: str
) -> None:
    available = _available_frame()
    available.loc[0, "prediction_fold_mask"] = mask
    available.loc[0, "cell_line_folds_with_prediction"] = fold_count
    with pytest.raises(DrugSparseAssetError, match=message):
        _write_bundle(
            tmp_path / f"invalid-mask-{mask}-{fold_count}",
            available_frame=available,
        )


def test_writer_rejects_mask_fold_without_full_support(tmp_path: Path) -> None:
    # Fold zero is supported for this key.  Adding fold one to the prediction
    # mask must fail even though at least one valid fold still exists.
    available = _available_frame()
    available.loc[0, "prediction_fold_mask"] = 3
    available.loc[0, "cell_line_folds_with_prediction"] = 2
    with pytest.raises(
        DrugSparseAssetError,
        match="per-mask-fold assay/expression/current-core",
    ):
        _write_bundle(
            tmp_path / "mask-points-to-unsupported-fold",
            available_frame=available,
        )


def _legacy_full_binding_counts(paths: dict[str, Path]) -> dict[str, int]:
    """Execute the former all-cancer SQL as an exact regression oracle."""

    import duckdb

    available = _parquet_relation(paths["available_predictions"])
    exact = _parquet_relation(paths["exact_candidates"])
    edges = _parquet_relation(paths["pathway_drug_edges"])
    assay = _parquet_relation(paths["assay_fold_eligibility"])
    expression = _parquet_relation(paths["expression_fold_coverage"])
    core = _parquet_relation(paths["core_entity_availability"])
    with duckdb.connect(database=":memory:") as con:
        counts = tuple(
            map(
                int,
                con.execute(
                    f"""
                    WITH
                    available_keys AS (
                      SELECT cancer_id, lncrna_id, drug_id,
                             CAST(prediction_fold_mask AS INTEGER)
                               AS prediction_fold_mask,
                             CAST(cell_line_folds_with_prediction AS INTEGER)
                               AS cell_line_folds_with_prediction
                      FROM {available}
                    ),
                    conceptual_available AS (
                      SELECT DISTINCT a.cancer_id, a.lncrna_id, a.drug_id
                      FROM available_keys a
                      JOIN {exact} c
                        ON c.cancer_id = a.cancer_id
                       AND c.lncrna_id = a.lncrna_id
                      JOIN {edges} d
                        ON d.pathway_id = c.pathway_id
                       AND d.drug_id = a.drug_id
                    ),
                    fold_contributions AS (
                      SELECT a.cancer_id, a.lncrna_id, a.drug_id, f.fold_id
                      FROM available_keys a
                      CROSS JOIN range(0, 5) AS f(fold_id)
                      WHERE (a.prediction_fold_mask & (1 << f.fold_id)) != 0
                    ),
                    same_fold_eligible AS (
                      SELECT DISTINCT a.cancer_id, a.lncrna_id, a.drug_id, a.fold_id
                      FROM fold_contributions a
                      JOIN {assay} s
                        ON s.cancer_id = a.cancer_id
                       AND s.drug_id = a.drug_id
                       AND s.fold_id = a.fold_id
                      JOIN {expression} x
                        ON x.cancer_id = a.cancer_id
                       AND x.lncrna_id = a.lncrna_id
                       AND x.fold_id = a.fold_id
                      JOIN {core} cc
                        ON cc.fold_id = a.fold_id
                       AND cc.entity_type = 'cancer'
                       AND cc.entity_id = a.cancer_id
                      JOIN {core} cl
                        ON cl.fold_id = a.fold_id
                       AND cl.entity_type = 'lncRNA'
                       AND cl.entity_id = a.lncrna_id
                      JOIN {core} cd
                        ON cd.fold_id = a.fold_id
                       AND cd.entity_type = 'drug_target'
                       AND cd.entity_id = a.drug_id
                    )
                    SELECT
                      (SELECT count(*) FROM available_keys),
                      (SELECT count(*) FROM conceptual_available),
                      (SELECT coalesce(sum(cell_line_folds_with_prediction), 0)
                         FROM available_keys),
                      (SELECT count(*) FROM fold_contributions),
                      (SELECT count(*) FROM same_fold_eligible)
                    """
                ).fetchone(),
            )
        )
    return dict(
        zip(
            (
                "available_rows",
                "conceptual_available_rows",
                "declared_fold_contributions",
                "expanded_fold_contributions",
                "supported_fold_contributions",
            ),
            counts,
            strict=True,
        )
    )


def test_partitioned_binding_equals_legacy_full_sql_across_cancers_and_folds(
    tmp_path: Path,
) -> None:
    factors = _factor_frames()
    coad_exact = factors["exact"].assign(cancer_id="COAD")
    coad_assay = factors["assay"].assign(cancer_id="COAD")
    coad_expression = factors["expression"].assign(cancer_id="COAD")
    coad_core_cancer = factors["core"].loc[
        factors["core"].entity_type.eq("cancer")
    ].assign(entity_id="COAD")
    extra_core = pd.DataFrame(
        [
            {"fold_id": 1, "entity_type": "lncRNA", "entity_id": L_OK},
            {"fold_id": 1, "entity_type": "drug_target", "entity_id": D_OK},
        ]
    )
    factors["exact"] = pd.concat(
        [factors["exact"], coad_exact], ignore_index=True
    )
    factors["assay"] = pd.concat(
        [
            factors["assay"],
            coad_assay,
            pd.DataFrame(
                [{"cancer_id": "COAD", "drug_id": D_OK, "cell_line_fold_id": 1}]
            ),
        ],
        ignore_index=True,
    )
    factors["expression"] = pd.concat(
        [
            factors["expression"],
            coad_expression,
            pd.DataFrame(
                [{"cancer_id": "COAD", "lncrna_id": L_OK, "fold_id": 1}]
            ),
        ],
        ignore_index=True,
    )
    factors["core"] = pd.concat(
        [factors["core"], coad_core_cancer, extra_core], ignore_index=True
    )
    available = pd.concat(
        [
            _available_frame(),
            _available_frame().assign(
                cancer_id="COAD",
                prediction_fold_mask=3,
                cell_line_folds_with_prediction=2,
            ),
        ],
        ignore_index=True,
    )
    root = tmp_path / "partitioned-equals-legacy"
    _write_bundle(
        root,
        conceptual_candidate_rows=50,
        available_frame=available,
        factor_overrides=factors,
    )
    paths = {
        "available_predictions": root / "drug_response_association",
        **{
            role: root / "drug_sparse_query_factors" / f"{role}.parquet"
            for role in (
                "exact_candidates",
                "pathway_drug_edges",
                "assay_fold_eligibility",
                "expression_fold_coverage",
                "core_entity_availability",
            )
        },
    }
    expected = _legacy_full_binding_counts(paths)
    observed = _validate_available_bindings(paths)
    assert expected == {
        "available_rows": 2,
        "conceptual_available_rows": 2,
        "declared_fold_contributions": 3,
        "expanded_fold_contributions": 3,
        "supported_fold_contributions": 3,
    }
    assert observed == expected


@pytest.mark.parametrize(
    ("role", "factor_key"),
    [
        ("exact_candidates", "exact_candidates"),
        ("pathway_drug_edges", "pathway_drug_edges"),
        ("assay_fold_eligibility", "assay_fold_eligibility"),
        ("expression_fold_coverage", "expression_fold_coverage"),
        ("core_entity_availability", "core_entity_availability"),
        ("available_predictions", "available_predictions"),
    ],
)
def test_loader_independently_rejects_hash_consistent_duplicate_keys(
    tmp_path: Path, role: str, factor_key: str
) -> None:
    result = _write_bundle(tmp_path / f"loader-duplicate-{role}")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = manifest["artifacts"][factor_key]
    artifact_path = manifest_path.parent / declaration["path"]
    part = (
        next(artifact_path.rglob("*.parquet"))
        if artifact_path.is_dir()
        else artifact_path
    )
    frame = pd.read_parquet(part)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(part, index=False)
    declaration["sha256"] = artifact_sha256(artifact_path)
    declaration["rows"] = int(declaration["rows"]) + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="duplicate keys"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )


def test_loader_independently_rejects_hash_consistent_fold_256(tmp_path: Path) -> None:
    result = _write_bundle(tmp_path / "loader-fold-256")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = manifest["artifacts"]["assay_fold_eligibility"]
    path = manifest_path.parent / declaration["path"]
    frame = pd.read_parquet(path)
    frame.loc[0, "fold_id"] = 256
    frame.to_parquet(path, index=False)
    declaration["sha256"] = artifact_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="integer domain 0..4"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )


def test_loader_rejects_hash_consistent_null_available_key(tmp_path: Path) -> None:
    result = _write_bundle(tmp_path / "loader-null-available")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = manifest["artifacts"]["available_predictions"]
    root = manifest_path.parent / declaration["path"]
    part = next(root.rglob("*.parquet"))
    frame = pd.read_parquet(part)
    frame.loc[0, "lncrna_id"] = pd.NA
    frame.to_parquet(part, index=False)
    declaration["sha256"] = artifact_sha256(root)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="null keys"):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )


def test_loader_rejects_hash_consistent_mask_with_unsupported_fold(
    tmp_path: Path,
) -> None:
    result = _write_bundle(tmp_path / "loader-mask-unsupported-fold")
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = manifest["artifacts"]["available_predictions"]
    root = manifest_path.parent / declaration["path"]
    part = next(root.rglob("*.parquet"))
    frame = pd.read_parquet(part)
    frame.loc[0, "prediction_fold_mask"] = 3
    frame.loc[0, "cell_line_folds_with_prediction"] = 2
    frame.to_parquet(part, index=False)
    declaration["sha256"] = artifact_sha256(root)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        DrugSparseAssetError,
        match="per-mask-fold assay/expression/current-core",
    ):
        load_drug_sparse_query_bundle(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
        )


def test_loader_rejects_unpinned_formal_load_and_validation_audit_drift(
    tmp_path: Path,
) -> None:
    result = _write_bundle(tmp_path / "formal-hash-and-audit")
    with pytest.raises(DrugSparseAssetError, match="requires expected_manifest_sha256"):
        load_drug_sparse_query_bundle(result["manifest_path"])

    audit_path = Path(result["validation_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["status"] = "FAIL"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(DrugSparseAssetError, match="validation audit SHA256 mismatch"):
        load_drug_sparse_query_bundle(
            result["manifest_path"],
            expected_manifest_sha256=result["manifest_sha256"],
        )


@pytest.mark.parametrize("entrypoint", ["function", "constructor"])
def test_public_loader_has_no_legacy_expected_hash_opt_out(
    tmp_path: Path, entrypoint: str
) -> None:
    result = _write_bundle(tmp_path / f"no-public-hash-opt-out-{entrypoint}")
    loader = (
        load_drug_sparse_query_bundle
        if entrypoint == "function"
        else DrugSparseQueryBundle
    )
    with pytest.raises(TypeError, match="require_expected_manifest_sha256"):
        loader(
            result["manifest_path"],
            require_expected_manifest_sha256=False,  # type: ignore[call-arg]
        )
    with pytest.raises(DrugSparseAssetError, match="requires expected_manifest_sha256"):
        loader(
            result["manifest_path"],
            expected_manifest_sha256=None,
        )


@pytest.mark.parametrize(
    ("max_temp_size", "message"),
    [
        ("", "cannot be empty"),
        ("unbounded", "must be a byte size"),
        ("1MB", "must be at least 256MB"),
    ],
)
def test_bounded_duckdb_rejects_invalid_or_too_low_temp_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_temp_size: str,
    message: str,
) -> None:
    monkeypatch.setenv(
        "CC_HHGT_DRUG_SPARSE_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", max_temp_size
    )
    with pytest.raises(DrugSparseAssetError, match=message):
        _write_bundle(tmp_path / f"bad-temp-cap-{max_temp_size or 'empty'}")


def test_bounded_duckdb_cleans_private_spill_directory_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spill_parent = tmp_path / "spill-parent"
    monkeypatch.setenv(DUCKDB_TEMP_ENV, str(spill_parent))
    observed_spill: Path | None = None
    with pytest.raises(RuntimeError, match="deliberate validator failure"):
        with _bounded_duckdb() as connection:
            observed_spill = Path(
                connection.execute(
                    "SELECT current_setting('temp_directory')"
                ).fetchone()[0]
            ).resolve()
            assert observed_spill.parent == spill_parent.resolve()
            (observed_spill / "failure-path-sentinel").write_text(
                "spill", encoding="utf-8"
            )
            raise RuntimeError("deliberate validator failure")
    assert observed_spill is not None
    assert not observed_spill.exists()
    assert list(spill_parent.iterdir()) == []
