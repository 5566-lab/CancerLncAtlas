from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cc_hhgt.v32.formal_graph import variant_edges
from cc_hhgt.v32.formal_graph_authority import (
    GRAPH_INPUT_RECEIPT_FORMAT,
    GRAPH_INPUT_RECEIPT_STATUS,
    FormalGraphAuthorityError,
    bind_formal_graph_variant,
    build_bound_formal_graph,
    load_formal_graph_input_authority,
    train_identity_hashes,
    validate_formal_graph_payload_binding,
)
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    validate_frozen_v32_patient_fold_binding,
)
from cc_hhgt.v32.patient_folds import assign_outer_split
from scripts.prepare_v32_formal import build_graph


ROOT = Path(__file__).resolve().parents[1]
PATIENT_ROOT = ROOT / "artifacts/v32_patient_fold_authority_20260829_r1"
PATIENT_MAP = PATIENT_ROOT / "SAMPLE_PATIENT_FOLD_MAP.tsv"
PATIENT_RECEIPT = PATIENT_ROOT / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"


def _declaration(
    path: Path,
    *,
    source_role: str,
    outer_fold: int | None = None,
    identity: dict | None = None,
) -> dict:
    result = {
        "path": str(path.resolve()),
        "sha256": artifact_sha256(path),
        "source_role": source_role,
        "outcome_derived": False,
        "historical_model_output": False,
        "toy_or_synthetic": False,
    }
    if outer_fold is not None:
        result.update(
            {
                "outer_fold": int(outer_fold),
                "source_split": "train",
                "patient_first": True,
                "train_sample_patient_sha256": identity[
                    "train_sample_patient_sha256"
                ],
                "train_patient_sha256": identity["train_patient_sha256"],
            }
        )
    return result


def _fixture(tmp_path: Path):
    pytest.importorskip("pyarrow")
    patient_audit = validate_frozen_v32_patient_fold_binding(
        PATIENT_MAP, PATIENT_RECEIPT
    )
    manifest = pd.read_csv(PATIENT_MAP, sep="\t", dtype=str)
    manifest["patient_fold_id"] = pd.to_numeric(
        manifest.patient_fold_id, errors="raise"
    ).astype(int)
    static_frames = {
        "detection": pd.DataFrame(
            {"cancer_id": ["BRCA"], "lncrna_id": ["L1"], "detection_rate": [0.8]}
        ),
        "signed_membership": pd.DataFrame(
            {"gene_id": ["G1", "G2"], "pathway_id": ["P1", "P1"], "weight": [1.0, -0.5]}
        ),
        "pathway_hierarchy": pd.DataFrame(
            {"pathway_id": ["P1"], "pathway_family_id": ["F1"], "weight": [1.0]}
        ),
        "lnc_protein_binding": pd.DataFrame(
            {
                "lncrna_id": ["L1"],
                "protein_id": ["PR1"],
                "weight": [0.9],
                "cancer_id": pd.Series([pd.NA], dtype="string"),
                "is_context_specific": [False],
            }
        ),
        "protein_gene_encoding": pd.DataFrame(
            {"protein_id": ["PR1", "PR2"], "gene_id": ["G1", "G2"], "weight": [1.0, 1.0]}
        ),
        "ppi": pd.DataFrame(
            {"protein_id_a": ["PR1"], "protein_id_b": ["PR2"], "weight": [0.7]}
        ),
    }
    static_paths = {}
    for artifact_id, frame in static_frames.items():
        path = tmp_path / f"{artifact_id}.parquet"
        frame.to_parquet(path, index=False)
        static_paths[artifact_id] = path

    folds = {}
    splits = {}
    for fold in range(5):
        split = assign_outer_split(manifest, fold, n_folds=5, validation_offset=1)
        splits[fold] = split
        identity = train_identity_hashes(split, cancer_scope=["BRCA"])
        expression = split.loc[
            split.cancer_id.astype(str).eq("BRCA")
            & split.split.astype(str).eq("train"),
            ["cancer_id", "sample_id", "patient_id"],
        ].drop_duplicates()
        expression_path = tmp_path / f"fold_{fold}_expression.parquet"
        expression.to_parquet(expression_path, index=False)
        coexpression_path = tmp_path / f"fold_{fold}_coexpression.parquet"
        pd.DataFrame(
            {
                "cancer_id": ["BRCA"],
                "lncrna_id": ["L1"],
                "gene_id": ["G2"],
                "rho": [0.6 if fold % 2 == 0 else -0.6],
                "source_split": ["train"],
                "edge_outer_fold": [fold],
            }
        ).to_parquet(coexpression_path, index=False)
        folds[str(fold)] = {
            "outer_fold": fold,
            "train_expression": _declaration(
                expression_path,
                source_role="outer_train_expression",
                outer_fold=fold,
                identity=identity,
            ),
            "train_coexpression": _declaration(
                coexpression_path,
                source_role="outer_train_coexpression",
                outer_fold=fold,
                identity=identity,
            ),
        }

    artifacts = {}
    for artifact_id, path in static_paths.items():
        artifacts[artifact_id] = _declaration(
            path,
            source_role=(
                "outcome_free_expression_detection"
                if artifact_id == "detection"
                else "static_annotation"
            ),
        )
    receipt = {
        "format": GRAPH_INPUT_RECEIPT_FORMAT,
        "status": GRAPH_INPUT_RECEIPT_STATUS,
        "cancer_scope": ["BRCA"],
        "patient_fold_authority": {
            "manifest_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        },
        "artifacts": artifacts,
        "fold_artifacts": folds,
        "gates": {
            "historical_graph_rows_used": False,
            "legacy_graph_root_fallback_allowed": False,
            "toy_or_synthetic_fallback_allowed": False,
            "outer_train_expression_only": True,
            "outer_train_coexpression_only": True,
            "static_evidence_outcome_free": True,
            "same_node_and_relation_schema_all_variants": True,
        },
    }
    receipt_path = tmp_path / "FRESH_G012_INPUT_AUTHORITY.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs = load_formal_graph_input_authority(
        receipt_path=receipt_path,
        receipt_sha256=artifact_sha256(receipt_path),
        patient_authority_audit=patient_audit,
        static_inputs={
            key: (path, artifact_sha256(path)) for key, path in static_paths.items()
        },
        fold_expression_pattern=str(tmp_path / "fold_{fold}_expression.parquet"),
        fold_coexpression_pattern=str(tmp_path / "fold_{fold}_coexpression.parquet"),
    )
    candidates = pd.DataFrame(
        {"cancer_id": ["BRCA"], "lncrna_id": ["L1"], "pathway_id": ["P1"]}
    )
    return inputs, candidates, splits


def test_safe_minimal_authority_builds_three_true_masks(tmp_path: Path) -> None:
    inputs, candidates, splits = _fixture(tmp_path)
    bound = build_graph(
        inputs,
        outer_fold=0,
        split_manifest=splits[0],
        candidate_pairs=candidates,
    )
    assert bound.binding["gates"]["historical_graph_rows_used"] is False
    assert bound.binding["gates"]["outer_train_coexpression_only"] is True
    counts = {
        arm: len(variant_edges(bound.authority, arm)) for arm in ("G0", "G1", "G2")
    }
    assert counts["G0"] < counts["G1"] < counts["G2"]
    assert bound.authority.manifest["same_relation_schema_all_variants"] is True
    for arm in ("G0", "G1", "G2"):
        binding = bind_formal_graph_variant(bound.binding, bound.authority, arm)
        bundle = SimpleNamespace(
            nodes=bound.authority.nodes,
            edges=variant_edges(bound.authority, arm),
        )
        assert validate_formal_graph_payload_binding(
            binding, outer_fold=0, variant=arm, bundle=bundle
        )["status"] == "PASS_FRESH_G012_PAYLOAD_BINDING"
    g0_binding = bind_formal_graph_variant(bound.binding, bound.authority, "G0")
    substituted_g2 = SimpleNamespace(
        nodes=bound.authority.nodes,
        edges=variant_edges(bound.authority, "G2"),
    )
    with pytest.raises(FormalGraphAuthorityError, match="declared G0/G1/G2 arm"):
        validate_formal_graph_payload_binding(
            g0_binding, outer_fold=0, variant="G0", bundle=substituted_g2
        )
    missing_provenance = json.loads(json.dumps(g0_binding))
    missing_provenance["static_artifacts"]["detection"].pop("sha256")
    with pytest.raises(FormalGraphAuthorityError, match="binding failed"):
        validate_formal_graph_payload_binding(
            missing_provenance,
            outer_fold=0,
            variant="G0",
            bundle=SimpleNamespace(
                nodes=bound.authority.nodes,
                edges=variant_edges(bound.authority, "G0"),
            ),
        )


def test_fold_expression_leakage_or_sha_drift_fails_closed(tmp_path: Path) -> None:
    inputs, candidates, splits = _fixture(tmp_path)
    expression_path = tmp_path / "fold_0_expression.parquet"
    expression = pd.read_parquet(expression_path)
    leaked = splits[0].loc[
        splits[0].cancer_id.astype(str).eq("BRCA")
        & splits[0].split.astype(str).eq("validation"),
        ["cancer_id", "sample_id", "patient_id"],
    ].head(1)
    pd.concat([expression, leaked], ignore_index=True).to_parquet(
        expression_path, index=False
    )
    with pytest.raises(FormalGraphAuthorityError, match="SHA256 drift"):
        build_bound_formal_graph(
            inputs,
            outer_fold=0,
            split_manifest=splits[0],
            candidate_pairs=candidates,
        )


def test_legacy_build_call_and_unbound_old_payload_are_rejected() -> None:
    with pytest.raises(TypeError):
        build_graph(["L1"], ["P1"], ["BRCA"])
    with pytest.raises(FormalGraphAuthorityError, match="binding failed"):
        validate_formal_graph_payload_binding({}, outer_fold=0, variant="G0")
    source = (ROOT / "scripts/prepare_v32_formal.py").read_text(encoding="utf-8")
    assert "GRAPH_ROOT" not in source
    assert "Formal prepared output reuse is forbidden" in source


def test_server_launcher_requires_every_fresh_authority_and_variant_roots() -> None:
    source = (ROOT / "scripts/server_prepare_v32_g012_patient_first_r2.sh").read_text(
        encoding="utf-8"
    )
    for token in (
        "V32_G012_GRAPH_AUTHORITY_RECEIPT",
        "V32_G012_GRAPH_AUTHORITY_RECEIPT_SHA256",
        "V32_G012_GRAPH_DETECTION_SHA256",
        "V32_G012_GRAPH_SIGNED_MEMBERSHIP_SHA256",
        "V32_G012_GRAPH_PATHWAY_HIERARCHY_SHA256",
        "V32_G012_GRAPH_LNC_PROTEIN_BINDING_SHA256",
        "V32_G012_GRAPH_PROTEIN_GENE_ENCODING_SHA256",
        "V32_G012_GRAPH_PPI_SHA256",
        "V32_G012_GRAPH_FOLD_EXPRESSION_PATTERN",
        "V32_G012_GRAPH_FOLD_COEXPRESSION_PATTERN",
    ):
        assert token in source
    assert 'test ! -e "$output_root/PATIENT_FOLD_4.pt"' in source
    for arm in ("G0", "G1", "G2"):
        assert f'$output_root/{arm}/PATIENT_FOLD_4.pt' in source
