#!/usr/bin/env python3
"""Verify the frozen GPU bundle, model inputs, and optional CUDA runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


EXPECTED_FOLDS = 31
EXPECTED_CANDIDATE_PARTITIONS = 33
EXPECTED_GRAPH_NODES = 142_812
EXPECTED_GRAPH_EDGES = 23_465_113
ISOLATED_EXTERNAL_SOURCES = {
    "Lnc2Cancer",
    "LncRNADisease",
    "RNADisease",
    "GSE85011",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_column_values(path: Path, column: str) -> set[str]:
    values: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=[column], batch_size=1_000_000):
        values.update(str(value) for value in pc.unique(batch.column(0)).to_pylist())
    return values


def verify_checksums(root: Path) -> int:
    manifest = root / "BUNDLE_SHA256SUMS"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Checksum target is missing: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results = root / "results" / "model" / "cc_hhgt_v2_1_gpu"
    tables = results / "tables"
    external = tables / "external_validation"
    standardized = results / "standardized"
    required = [
        root / "config" / "model_v2_1_gpu_4070tis.yaml",
        root / "交接.md",
        root / "04_train_multiseed_linux.sh",
        root / "04_train_multiseed_windows.ps1",
        root / "05_package_results_linux.sh",
        root / "05_package_results_windows.ps1",
        root / "scripts" / "25_train_multiseed_gpu.py",
        root / "scripts" / "26_summarize_multiseed.py",
        tables / "graph_node.parquet",
        tables / "graph_edge.parquet",
        tables / "fold_manifest.tsv",
        tables / "candidate_universe_manifest.tsv",
        tables / "candidate_universe",
        standardized / "external_validation_evidence.parquet",
        standardized / "external_lncRNA_disease_evidence.parquet",
        standardized / "gse85011_growth_modifier_evidence.parquet",
        external / "external_validation_manifest.json",
        external / "external_validation_source_summary.tsv",
        external / "external_validation_leakage_audit.tsv",
        external / "gse85011_external_database_overlap.tsv",
        external / "gse85011_study_summary.tsv",
        root / "input" / "lnc2cancer3_human_lncRNA.tsv",
        root / "input" / "lncrnadisease3_human_lncRNA.tsv",
        root / "input" / "rnadisease4_human_lncRNA_experimental.tsv",
        root / "input" / "rnadisease4_lncRNA_predicted.tsv",
        root / "input" / "gse85011_sample_metadata.tsv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    checksum_count = None if args.skip_checksums else verify_checksums(root)
    folds = pd.read_csv(tables / "fold_manifest.tsv", sep="\t")
    if len(folds) != EXPECTED_FOLDS or folds.fold_id.nunique() != EXPECTED_FOLDS:
        raise RuntimeError(f"Expected {EXPECTED_FOLDS} unique folds, observed {len(folds)} rows")
    candidate_files = sorted((tables / "candidate_universe").glob("cancer_id=*/part-0.parquet"))
    if len(candidate_files) != EXPECTED_CANDIDATE_PARTITIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATE_PARTITIONS} candidate partitions, observed {len(candidate_files)}"
        )
    node_rows = parquet_rows(tables / "graph_node.parquet")
    edge_rows = parquet_rows(tables / "graph_edge.parquet")
    if node_rows != EXPECTED_GRAPH_NODES:
        raise RuntimeError(f"Expected {EXPECTED_GRAPH_NODES} graph nodes, observed {node_rows}")
    if edge_rows != EXPECTED_GRAPH_EDGES:
        raise RuntimeError(f"Expected {EXPECTED_GRAPH_EDGES} graph edges, observed {edge_rows}")
    candidate_rows = sum(parquet_rows(path) for path in candidate_files)
    gpu_config = yaml.safe_load(
        (root / "config" / "model_v2_1_gpu_4070tis.yaml").read_text(
            encoding="utf-8"
        )
    )
    multiseed = gpu_config.get("multiseed", {})
    primary_seed = int(multiseed.get("primary_seed", -1))
    replicate_seeds = [int(seed) for seed in multiseed.get("replicate_seeds", [])]
    all_seeds = [primary_seed, *replicate_seeds]
    if (
        not multiseed.get("enabled")
        or all_seeds != [20260726, 20261726, 20262726]
    ):
        raise RuntimeError(
            "Expected primary seed 20260726 and replicate seeds "
            f"20261726/20262726: {all_seeds}"
        )
    if not gpu_config.get("training", {}).get("export_node_embeddings", False):
        raise RuntimeError("training.export_node_embeddings must be enabled")
    graph_sources = parquet_column_values(tables / "graph_edge.parquet", "source_database")
    leaked_sources = graph_sources & ISOLATED_EXTERNAL_SOURCES
    if leaked_sources:
        raise RuntimeError(
            f"Isolated external sources leaked into graph_edge: {sorted(leaked_sources)}"
        )
    external_rows = parquet_rows(
        standardized / "external_validation_evidence.parquet"
    )
    if external_rows < 600_000:
        raise RuntimeError(
            f"External evidence snapshot is unexpectedly small: {external_rows}"
        )
    external_sources = parquet_column_values(
        standardized / "external_validation_evidence.parquet",
        "source_database",
    )
    if external_sources != ISOLATED_EXTERNAL_SOURCES:
        raise RuntimeError(
            f"External source set mismatch: {sorted(external_sources)}"
        )
    gse = pd.read_parquet(
        standardized / "gse85011_growth_modifier_evidence.parquet",
        columns=["growth_modifier_hit", "growth_effect_direction"],
    )
    if (
        gse.empty
        or not gse.growth_modifier_hit.fillna(False).all()
        or not gse.growth_effect_direction.eq("unknown").all()
    ):
        raise RuntimeError("GSE85011 web evidence must be hit=true and direction=unknown")
    leakage = pd.read_csv(
        external / "external_validation_leakage_audit.tsv", sep="\t"
    )
    if leakage.empty or not leakage.status.eq("PASS").all():
        raise RuntimeError("External-validation leakage audit is not all PASS")

    runtime: dict[str, object] = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "checksum_files_verified": checksum_count,
        "folds": len(folds),
        "candidate_partitions": len(candidate_files),
        "candidate_rows": candidate_rows,
        "graph_nodes": node_rows,
        "graph_edges": edge_rows,
        "external_evidence_rows": external_rows,
        "external_sources": sorted(external_sources),
        "external_sources_in_training_graph": sorted(leaked_sources),
        "gse85011_web_rows": len(gse),
        "multiseed_plan": all_seeds,
        "planned_model_fold_runs": len(all_seeds) * EXPECTED_FOLDS * 3,
        "export_node_embeddings": True,
    }
    try:
        import torch
        import torch_geometric

        runtime.update(
            {
                "torch": torch.__version__,
                "torch_geometric": torch_geometric.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_runtime": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            runtime["gpu_name"] = props.name
            runtime["gpu_total_vram_gb"] = round(props.total_memory / 1024**3, 3)
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            runtime["cuda_smoke_sum"] = float(x.sum().cpu())
    except ImportError as exc:
        runtime["gpu_import_error"] = str(exc)
        if args.require_gpu:
            raise RuntimeError("PyTorch/PyG is not installed") from exc

    if args.require_gpu and not runtime.get("cuda_available", False):
        raise RuntimeError("CUDA verification failed: torch.cuda.is_available() is False")

    if args.require_gpu:
        status_dir = results / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / "gpu_environment.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(runtime, ensure_ascii=False, indent=2))
    print("GPU bundle verification: PASS")


if __name__ == "__main__":
    main()
