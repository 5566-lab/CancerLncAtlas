from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.single_cell_diagnostic_rerun import (
    CONFIG_FORMAT,
    EXPECTED_CANCERS,
    FORMAL_CANCERS,
    PREFLIGHT_FORMAT,
    ROOT_PROVENANCE,
    SingleCellDiagnosticError,
    _filter_exact_tsv,
    audit_cancer_inputs,
    build_preflight,
    prepare_workspace,
    sha256_file,
    validate_config,
    validate_source_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config"
    / "v32_single_cell_diagnostic_rerun_20260827_r4_exact_signed_intersection.json"
)
SNAPSHOT = (
    ROOT
    / "artifacts"
    / "v32_single_cell_pipeline_source_snapshot_20260827_r3_exact_signed_intersection"
)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_formal_config_is_33_coverage_17_execution_and_diagnostic_only() -> None:
    config = load_config()
    validate_config(config)
    assert config["format"] == CONFIG_FORMAT
    assert tuple(config["coverage_cancers"]) == EXPECTED_CANCERS
    assert tuple(config["formal_cancers"]) == FORMAL_CANCERS
    assert config["policy"]["root_provenance"] == ROOT_PROVENANCE
    assert config["policy"]["root_is_explicit"] is False
    assert config["policy"]["historical_result_reuse_allowed"] is False
    assert config["policy"]["processed_seurat_fallback_allowed"] is False
    assert config["policy"]["model_fusion_permitted"] is False
    assert config["policy"]["primary_score_weight"] == 0
    assert config["policy"]["secondary_score_weight"] == 0
    assert len(config["runtime"]["r_library_paths"]) == 2
    assert config["workspace_links"]["processed/pathway_gene_member_weighted.tsv"] == (
        config["workspace_links"]["processed/pathway_gene_member.tsv"]
    )


def test_config_rejects_explicit_root_and_mutable_old_stage() -> None:
    config = load_config()
    explicit = copy.deepcopy(config)
    explicit["policy"]["root_is_explicit"] = True
    with pytest.raises(SingleCellDiagnosticError, match="root_is_explicit"):
        validate_config(explicit)
    old = copy.deepcopy(config)
    old["compute_output_root"] = (
        "./data/CancerLncAtlas/results/sc_trajectory_staging"
    )
    with pytest.raises(SingleCellDiagnosticError, match="isolated V3.2 diagnostic"):
        validate_config(old)


def test_recovered_source_snapshot_is_sha_bound_and_detects_drift(tmp_path: Path) -> None:
    records = validate_source_snapshot(SNAPSHOT)
    assert len(records) == 7
    assert all(record["sha256"] for record in records.values())
    copied = tmp_path / "snapshot"
    shutil.copytree(SNAPSHOT, copied)
    target = copied / "13_sc_malignant_trajectory.R"
    target.write_text(target.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(SingleCellDiagnosticError, match="SHA drift"):
        validate_source_snapshot(copied)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def _fresh_cancer_fixture(tmp_path: Path, cancer: str = "ACC") -> dict:
    source = tmp_path / cancer
    source.mkdir(parents=True)
    barcodes = [f"raw-{index}" for index in range(6)]
    h5 = source / "raw_feature_bc_matrix.h5"
    h5.write_bytes(b"10x-h5-fixture-read-is-monkeypatched")
    annotation = pd.DataFrame(
        {
            "cell_id": [f"cell-{index}" for index in range(6)],
            "raw_cell_id": barcodes,
            "dataset_id": ["DS"] * 6,
            "patient_id": ["P1", "P2", "P3", "P1", "P2", "P3"],
            "sample_id": [f"S{index}" for index in range(6)],
            "sample_class": ["tumor"] * 6,
            "cell_type_major": ["Malignant"] * 3 + ["T cell"] * 3,
        }
    )
    qc = pd.DataFrame(
        {"cell_id": annotation.cell_id, "pass_qc": True, "input_expression_scale": "raw_counts"}
    )
    malignant = pd.DataFrame(
        {
            "cell_id": annotation.cell_id,
            "malignant_status": ["malignant"] * 3 + ["non_malignant"] * 3,
            "primary_analysis_flag": [True] * 3 + [False] * 3,
        }
    )
    paths = {
        "raw_h5": h5,
        "annotation": source / "sc_cell_annotation.tsv.gz",
        "qc": source / "sc_cell_qc.tsv.gz",
        "malignant": source / "sc_malignant_call.tsv.gz",
    }
    _write_tsv(annotation, paths["annotation"])
    _write_tsv(qc, paths["qc"])
    _write_tsv(malignant, paths["malignant"])
    config = load_config()
    config["input_templates"] = {key: str(value) for key, value in paths.items()}
    return config


def test_raw_h5_and_bound_metadata_are_sufficient_for_diagnostic_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cc_hhgt.v32.single_cell_diagnostic_rerun as diagnostic

    config = _fresh_cancer_fixture(tmp_path)
    monkeypatch.setattr(
        diagnostic,
        "_h5_barcodes",
        lambda _path: ([f"raw-{index}" for index in range(6)], (1, 6)),
    )
    record = audit_cancer_inputs(cancer="ACC", config=config, require_available=True)
    assert record["diagnostic_runnable"] is True
    assert record["h5_cells"] == 6
    assert record["primary_malignant_donors"] == 3
    assert record["immune_candidate_cells"] == 3
    assert record["trajectory_numeric_precondition"] is False
    assert record["donor_association_precondition"] is False


def test_processed_seurat_expression_source_is_rejected(tmp_path: Path) -> None:
    config = _fresh_cancer_fixture(tmp_path)
    config["input_templates"]["raw_h5"] = str(
        tmp_path / "sc_reanalysis" / "01_ACC_processed_seurat.rds"
    )
    with pytest.raises(SingleCellDiagnosticError, match="raw 10x H5"):
        audit_cancer_inputs(cancer="ACC", config=config, require_available=False)


def test_local_deferred_preflight_does_not_claim_remote_inputs_or_start_compute() -> None:
    payload = build_preflight(CONFIG, scope="coverage33", verify_inputs=False)
    assert payload["format"] == PREFLIGHT_FORMAT
    assert payload["status"] == "PASS"
    assert payload["computation_started"] is False
    assert payload["training_started"] is False
    assert payload["release_ready"] is False
    assert payload["exact_pathway_authority"]["status"] == (
        "DEFERRED_TO_SERVER_FILE_PREFLIGHT"
    )
    assert payload["full_exact_pathway_cell_level_ucell_in_seven_file_snapshot"] is False


def _workspace_config(tmp_path: Path) -> tuple[Path, dict]:
    config = load_config()
    output = tmp_path / "v32_full_multitask" / "single_cell_diagnostic_fresh_test"
    config["compute_output_root"] = str(output)
    config["source_snapshot"]["path"] = str(SNAPSHOT)
    link_sources = tmp_path / "links"
    link_sources.mkdir()
    links: dict[str, str] = {}
    for index, relative in enumerate(config["workspace_links"]):
        source = link_sources / f"source-{index}"
        if "." in Path(relative).name:
            source.write_text("fixture\n", encoding="utf-8")
        else:
            source.mkdir()
        links[relative] = str(source)
    config["workspace_links"] = links
    pd.DataFrame(
        {"ensembl_gene_id": ["ENSG1"], "gene_symbol": ["GENE1"]}
    ).to_csv(
        Path(links["processed/dimensions"]) / "dim_gene.tsv",
        sep="\t",
        index=False,
    )
    config["input_templates"] = {
        "raw_h5": links["processed/sc_tool_input"].rstrip("/")
        + "/{cancer}/raw_feature_bc_matrix.h5",
        "annotation": links["results/sc"].rstrip("/")
        + "/{cancer}/sc_cell_annotation.tsv.gz",
        "qc": links["results/sc"].rstrip("/") + "/{cancer}/sc_cell_qc.tsv.gz",
        "malignant": links["results/sc"].rstrip("/")
        + "/{cancer}/sc_malignant_call.tsv.gz",
    }
    candidate = tmp_path / "candidate.parquet"
    membership = tmp_path / "membership.parquet"
    pd.DataFrame({"pathway_id": ["EXACT:A"]}).to_parquet(candidate, index=False)
    pd.DataFrame(
        {"pathway_id": ["EXACT:A"], "gene_id": ["ENSG1"]}
    ).to_parquet(membership, index=False)
    config["exact_pathway_authority"]["candidate_path"] = str(candidate)
    config["exact_pathway_authority"]["membership_path"] = str(membership)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, config


def test_workspace_is_new_isolated_and_never_links_old_result_trees(tmp_path: Path) -> None:
    config_path, config = _workspace_config(tmp_path)
    preflight = {
        "format": PREFLIGHT_FORMAT,
        "status": "PASS",
        "computation_started": False,
        "config_sha256": sha256_file(config_path),
        "run_cancers": list(FORMAL_CANCERS),
        "root_provenance": ROOT_PROVENANCE,
        "root_is_explicit": False,
        "historical_result_reuse_allowed": False,
        "processed_seurat_fallback_allowed": False,
        "all_expression_from_raw_h5": True,
        "model_fusion_permitted": False,
    }
    preflight_path = tmp_path / "PREFLIGHT.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    result = prepare_workspace(
        config_path,
        preflight_path,
        expected_preflight_sha256=sha256_file(preflight_path),
    )
    workspace = Path(result["workspace"])
    assert result["historical_output_tree_linked"] is False
    assert result["processed_seurat_tree_linked"] is False
    assert (workspace / "python" / "common.py").is_file()
    assert (workspace / "R" / "13_sc_malignant_trajectory.R").is_file()
    assert (workspace / "processed" / "pathway_gene_member_weighted.tsv").is_symlink()
    exact_members = pd.read_csv(
        workspace / "processed" / "pathway_gene_member.tsv", sep="\t"
    )
    assert exact_members.to_dict("records") == [
        {"pathway_id": "EXACT:A", "gene_symbol": "GENE1"}
    ]
    assert result["exact_symbol_membership_edges"] == 1
    assert not (workspace / "results" / "sc_trajectory_staging").exists()
    with pytest.raises(SingleCellDiagnosticError, match="output reuse"):
        prepare_workspace(
            config_path,
            preflight_path,
            expected_preflight_sha256=sha256_file(preflight_path),
        )


def test_exact_postfilter_adds_diagnostic_inferred_root_contract(tmp_path: Path) -> None:
    source = tmp_path / "input.tsv.gz"
    destination = tmp_path / "output.tsv.gz"
    pd.DataFrame(
        {
            "pathway_id": ["EXACT:A", "REFERENCE_ONLY:B"],
            "activity_score": [0.2, 0.8],
        }
    ).to_csv(source, sep="\t", index=False, compression="gzip")
    rows = _filter_exact_tsv(source, destination, {"EXACT:A"})
    result = pd.read_csv(destination, sep="\t")
    assert rows == 1
    assert result.pathway_id.tolist() == ["EXACT:A"]
    assert result.evidence_tier.tolist() == ["diagnostic_inferred_root"]
    assert result.root_provenance.tolist() == [ROOT_PROVENANCE]
    assert result.root_is_explicit.tolist() == [False]


def test_source_audit_does_not_misrepresent_full_ucell_or_model_readiness() -> None:
    report = json.loads(
        (
            ROOT
            / "artifacts"
            / "v32_single_cell_pipeline_source_audit_20260826_r1"
            / "AUDIT_REPORT.json"
        ).read_text(encoding="utf-8")
    )
    assert report["ucell_scope"]["all_exact_pathway_cell_level_ucell"] is False
    assert report["leakage_audit"]["model_fusion_permitted"] is False
    assert report["remote_compute_started"] is False
    assert report["production_deployed"] is False
