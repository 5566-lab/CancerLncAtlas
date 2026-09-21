from pathlib import Path

import numpy as np
import pandas as pd
import torch
import importlib.util

from cc_hhgt.gnn import GraphBundle, export_node_embeddings
from cc_hhgt.multiseed import (
    CALIBRATION_ARTIFACTS,
    GNN_MODELS,
    model_root_for_seed,
    required_artifacts,
    seed_plan,
    summarize_multiseed,
)


def _load_gpu_runner():
    path = Path(__file__).parents[1] / "scripts" / "22_train_gpu.py"
    spec = importlib.util.spec_from_file_location("train_gpu", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_return_packager():
    path = Path(__file__).parents[1] / "scripts" / "24_package_gpu_results.py"
    spec = importlib.util.spec_from_file_location("package_gpu_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path):
    return {
        "analysis_version": "test",
        "random_seed": 10,
        "_results": tmp_path,
        "training": {"export_node_embeddings": True},
        "multiseed": {
            "enabled": True,
            "primary_seed": 10,
            "replicate_seeds": [20, 30],
            "confidence_level": 0.95,
            "metric_columns": ["auroc", "auprc", "brier", "ece", "log_loss"],
        },
    }


def test_seed_plan_keeps_primary_and_replicates_separate(tmp_path):
    cfg = _config(tmp_path)
    assert seed_plan(cfg) == (10, [20, 30], [10, 20, 30])
    assert model_root_for_seed(cfg, 10) == tmp_path / "models"
    assert model_root_for_seed(cfg, 20) == tmp_path / "models_multiseed/seed_20"
    artifacts = required_artifacts(cfg, include_calibration=True)
    assert "node_embeddings.pt" in artifacts
    assert set(CALIBRATION_ARTIFACTS) <= set(artifacts)


def test_gpu_runner_offsets_fold_seeds_without_changing_fold_layout(tmp_path):
    runner = _load_gpu_runner()
    folds = pd.DataFrame(
        {
            "fold_id": ["LOCO_A", "LOCO_B"],
            "split_seed": [20260726, 20260727],
        }
    )
    adjusted = runner.adjust_fold_seeds(folds, 20260726, 20261726)
    assert adjusted.fold_id.tolist() == folds.fold_id.tolist()
    assert adjusted.split_seed.tolist() == [20261726, 20261727]
    primary_root, _ = runner.experiment_output_paths(
        tmp_path, 20260726, 20260726
    )
    repeat_root, _ = runner.experiment_output_paths(
        tmp_path, 20260726, 20261726
    )
    assert primary_root == tmp_path / "models"
    assert repeat_root == tmp_path / "models_multiseed/seed_20261726"


def test_multiseed_summary_uses_three_seed_level_means(tmp_path):
    cfg = _config(tmp_path)
    tables = tmp_path / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "fold_id": "LOCO_BRCA",
                "test_cancer": "BRCA",
                "validation_cancer": "CESC",
                "train_cancers": "LUAD",
                "reference_cancers": "",
                "n_train_cancers": 1,
                "split_seed": 10,
            }
        ]
    ).to_csv(tables / "fold_manifest.tsv", sep="\t", index=False)
    values = {
        10: {"rgcn": 0.60, "hgt": 0.65, "cc_hhgt": 0.70},
        20: {"rgcn": 0.62, "hgt": 0.67, "cc_hhgt": 0.72},
        30: {"rgcn": 0.64, "hgt": 0.69, "cc_hhgt": 0.74},
    }
    artifacts = required_artifacts(cfg, include_calibration=True)
    for seed, model_values in values.items():
        root = model_root_for_seed(cfg, seed)
        for model, auprc in model_values.items():
            directory = root / model / "LOCO_BRCA"
            directory.mkdir(parents=True)
            for artifact in artifacts:
                (directory / artifact).write_bytes(b"x")
            pd.DataFrame(
                [
                    {
                        "fold_id": "LOCO_BRCA",
                        "model_name": model,
                        "split": "test",
                        "auroc": auprc + 0.1,
                        "auprc": auprc,
                        "brier": 1 - auprc,
                        "ece": (1 - auprc) / 2,
                        "log_loss": 1.2 - auprc,
                    }
                ]
            ).to_csv(directory / "metrics_calibrated.tsv", sep="\t", index=False)
    result = summarize_multiseed(cfg, calibrated=True)
    assert result["complete_model_fold_runs"] == 9
    summary = pd.read_csv(
        tmp_path / "reports/multiseed/model_multiseed_summary.tsv",
        sep="\t",
    )
    row = summary.loc[
        summary.model_name.eq("cc_hhgt") & summary.metric.eq("auprc")
    ].iloc[0]
    assert row.n_seeds == 3
    assert abs(row.mean_across_seed_fold_means - 0.72) < 1e-12
    assert row.ci_low < 0.72 < row.ci_high
    deltas = pd.read_csv(
        tmp_path / "reports/multiseed/architecture_ablation_deltas.tsv",
        sep="\t",
    )
    assert np.allclose(
        deltas.loc[deltas.metric.eq("auprc"), "cc_hhgt_minus_hgt"],
        0.05,
    )


class _FakeModel:
    def __init__(self, encoded):
        self.encoded = encoded

    def eval(self):
        return self

    def encode(self, graph):
        return self.encoded


def test_embedding_export_aligns_rows_by_node_type_index(tmp_path):
    nodes = pd.DataFrame(
        {
            "node_type": ["lncRNA", "gene", "lncRNA"],
            "node_index_within_type": [1, 0, 0],
        }
    )
    bundle = GraphBundle(
        nodes=nodes,
        edges=pd.DataFrame(),
        node_maps={},
        hetero_data=None,
        homogeneous={},
    )
    cfg = {"analysis_version": "test", "random_seed": 10}
    fold = pd.Series({"fold_id": "LOCO_BRCA", "split_seed": 10})
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoded = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    metadata = export_node_embeddings(
        cfg,
        _FakeModel(encoded),
        graph=None,
        bundle=bundle,
        kind="rgcn",
        fold_row=fold,
        model_dir=model_dir,
    )
    payload = torch.load(
        model_dir / "node_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["node_embeddings"]["lncRNA"].float().tolist() == [
        [3.0, 3.0],
        [1.0, 1.0],
    ]
    assert metadata["embedding_dimensions"]["gene"] == [1, 2]


def test_return_packager_requires_all_three_seeds(tmp_path):
    packager = _load_return_packager()
    results = tmp_path / "results"
    (results / "tables").mkdir(parents=True)
    pd.DataFrame([{"fold_id": "LOCO_BRCA"}]).to_csv(
        results / "tables/fold_manifest.tsv",
        sep="\t",
        index=False,
    )
    config = tmp_path / "gpu.yaml"
    config.write_text(
        "multiseed:\n"
        "  primary_seed: 10\n"
        "  replicate_seeds: [20, 30]\n",
        encoding="utf-8",
    )
    for seed in [10, 20, 30]:
        root = (
            results / "models"
            if seed == 10
            else results / "models_multiseed" / f"seed_{seed}"
        )
        for model in packager.MODELS:
            directory = root / model / "LOCO_BRCA"
            directory.mkdir(parents=True)
            for artifact in packager.EXPECTED:
                (directory / artifact).write_bytes(b"x")
    complete, missing, seeds = packager.complete_model_dirs(results, config)
    assert seeds == [10, 20, 30]
    assert len(complete) == 9
    assert missing == []
    (results / "models_multiseed/seed_30/cc_hhgt/LOCO_BRCA/node_embeddings.pt").unlink()
    _, missing, _ = packager.complete_model_dirs(results, config)
    assert missing == ["seed=30/cc_hhgt/LOCO_BRCA: node_embeddings.pt"]
