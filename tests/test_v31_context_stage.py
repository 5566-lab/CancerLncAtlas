from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.common import file_sha256
from cc_hhgt.v31_context_stage import run_multimodal_context_stage
from cc_hhgt.v31_pilot_gate import PILOT_CANCERS, PILOT_SEEDS


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows: list[dict] = []
    graph_rows: list[dict] = []
    context_rows: list[dict] = []
    for heldout in PILOT_CANCERS:
        train_cancer = f"TRAIN_{heldout}"
        validation_cancer = f"VAL_{heldout}"
        for cancer in (train_cancer, validation_cancer, heldout):
            for index in range(4):
                context_rows.append(
                    {
                        "cancer_id": cancer,
                        "lncrna_id": f"L{index}",
                        "signal": float(index),
                        "signal__available": True,
                    }
                )
        for subtype in ("Pathway", "RNAss", "DNAss"):
            task = "pathway" if subtype == "Pathway" else "state"
            for index, label in enumerate((0, 0, 1, 1)):
                common = {
                    "candidate_id": f"train:{heldout}:{subtype}:{index}",
                    "cancer_id": train_cancer,
                    "lncrna_id": f"L{index}",
                    "loco_cancer": heldout,
                    "task": task,
                    "target_subtype": subtype,
                    "split": "train",
                    "label": label,
                    "z_base": 0.0,
                    "best_simple_score": 0.5,
                    "metric_scope": "cancer_crossfit_OOF",
                }
                if subtype == "Pathway":
                    common["pathway_family_id"] = f"P{index}"
                else:
                    common["state_id"] = subtype
                train_rows.append(common)
            for split, cancer in (("val", validation_cancer), ("test", heldout)):
                for seed in PILOT_SEEDS:
                    for index, label in enumerate((0, 0, 1, 1)):
                        probability = 0.45 if label == 0 else 0.55
                        row = {
                            "candidate_id": f"{split}:{heldout}:{subtype}:{index}",
                            "cancer_id": cancer,
                            "lncrna_id": f"L{index}",
                            "loco_cancer": heldout,
                            "seed": seed,
                            "task": task,
                            "target_subtype": subtype,
                            "split": split,
                            "proxy_label": label,
                            "raw_logit": float(np.log(probability / (1 - probability))),
                            "proxy_positive_probability": probability,
                            "prediction_scale": "raw_probability",
                        }
                        if subtype == "Pathway":
                            row["pathway_family_id"] = f"P{index}"
                        else:
                            row["state_id"] = subtype
                        graph_rows.append(row)
    return pd.DataFrame(train_rows), pd.DataFrame(graph_rows), pd.DataFrame(context_rows)


def test_multimodal_stage_uses_validation_and_exact_unavailable_fallback(tmp_path: Path) -> None:
    residual, graph, context = _synthetic_inputs()
    residual_path = tmp_path / "residual.parquet"
    graph_path = tmp_path / "graph.parquet"
    residual.to_parquet(residual_path, index=False)
    graph.to_parquet(graph_path, index=False)
    best_gate = tmp_path / "BEST.json"
    graph_gate = tmp_path / "GRAPH.json"
    best_gate.write_text(json.dumps({"status": "PASS", "failures": []}))
    graph_gate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "failures": [],
                "selection_used_test_labels": False,
            }
        )
    )
    hard_report = tmp_path / "B2_HARD_REPORT.json"
    hard_report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "failures": [],
                "stage": "B2_HARD_REPORT",
                "decision": "HARD_GO_B3",
                "b3_authorized": True,
                "full_cancer_training_authorized": False,
                "b2_gate_sha256": file_sha256(graph_gate),
                "gates": {
                    "delta_auprc": True,
                    "cluster_aware_ci": True,
                    "cancer_direction": True,
                    "calibration": True,
                    "exact_fallback": True,
                    "integrity": True,
                },
            }
        )
    )
    output = tmp_path / "output"
    result = run_multimodal_context_stage(
        residual_base_path=residual_path,
        best_simple_gate_path=best_gate,
        graph_predictions_path=graph_path,
        graph_gate_path=graph_gate,
        b2_hard_report_path=hard_report,
        modules={
            "RNA": (context, None),
            "GENOMIC": None,
            "ATAC": None,
            "SINGLECELL": None,
        },
        module_lineage={"RNA": {"synthetic": True}},
        output_dir=output,
        shrinkage_grid=[0.1],
        aggregation_git_commit="test",
    )
    assert result["status"] == "PASS"
    admission = pd.read_csv(output / "MODULE_ADMISSION_MATRIX.tsv", sep="\t")
    assert not admission.test_metric_used_for_selection.astype(bool).any()
    unavailable = admission.loc[admission.module.eq("SINGLECELL")]
    assert unavailable.decision.eq("OFF").all()
    fallback = pd.read_csv(output / "CONTEXT_FALLBACK_AUDIT.tsv", sep="\t")
    assert fallback.status.eq("PASS").all()
    assert (output / "CONTEXT_RESIDUAL_GATE.json").is_file()


def test_multimodal_stage_refuses_b3_without_hard_go(tmp_path: Path) -> None:
    residual, graph, context = _synthetic_inputs()
    residual_path = tmp_path / "residual.parquet"
    graph_path = tmp_path / "graph.parquet"
    residual.to_parquet(residual_path, index=False)
    graph.to_parquet(graph_path, index=False)
    best_gate = tmp_path / "BEST.json"
    graph_gate = tmp_path / "GRAPH.json"
    hard_report = tmp_path / "B2_HARD_REPORT.json"
    best_gate.write_text(json.dumps({"status": "PASS", "failures": []}))
    graph_gate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "failures": [],
                "selection_used_test_labels": False,
            }
        )
    )
    hard_report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "failures": [],
                "stage": "B2_HARD_REPORT",
                "decision": "HARD_STOP_AFTER_B2",
                "b3_authorized": False,
            }
        )
    )
    with pytest.raises(RuntimeError, match="did not issue HARD_GO_B3"):
        run_multimodal_context_stage(
            residual_base_path=residual_path,
            best_simple_gate_path=best_gate,
            graph_predictions_path=graph_path,
            graph_gate_path=graph_gate,
            b2_hard_report_path=hard_report,
            modules={"RNA": (context, None)},
            module_lineage={"RNA": {"synthetic": True}},
            output_dir=tmp_path / "output",
            shrinkage_grid=[0.1],
            aggregation_git_commit="test",
        )
