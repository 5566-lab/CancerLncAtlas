"""Resume V3.2 subtype materialization with deterministic pathway sharding.

This script consumes already materialized exact-pathway Gene Set member parts.
Each pathway is handled by the unchanged V3.2 subtype implementation with its
own stable seed, so sharding changes scheduling only.  Subtype labels remain a
downstream analysis and never feed back into CC-HHGT probabilities or ranks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.subtypes import (  # noqa: E402
    RankedSubtypeConfig,
    classify_cancer_program_subtypes,
    classify_pathway_context_subtypes,
)


MEMBER_COLUMNS = (
    "geneset_id",
    "geneset_rank",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "association_membership_probability",
    "association_direction",
    "shared_or_local_scope",
)


def _frozen_config() -> RankedSubtypeConfig:
    return RankedSubtypeConfig(
        top_n=200,
        rbo_p=0.98,
        rbo_weight=0.5,
        weighted_jaccard_weight=0.5,
        k_min=1,
        k_max=4,
        program_k_max=6,
        silhouette_threshold=0.25,
        bootstrap_ari_threshold=0.75,
        bootstrap_replicates=200,
        min_cancers=6,
        min_members=10,
        min_cluster_size=3,
        random_seed=20260726,
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _run_shard(
    shard_id: int,
    input_path: str,
    result_root: str,
    config_values: dict[str, Any],
) -> dict[str, Any]:
    """Run one disjoint pathway shard in a spawned worker."""

    result_dir = Path(result_root)
    success_path = result_dir / f"shard_{shard_id:04d}.SUCCESS.json"
    context_path = result_dir / f"shard_{shard_id:04d}.context.parquet"
    conservation_path = result_dir / f"shard_{shard_id:04d}.conservation.parquet"
    pairwise_path = result_dir / f"shard_{shard_id:04d}.pairwise.parquet"
    if all(path.is_file() for path in (success_path, context_path, conservation_path, pairwise_path)):
        return json.loads(success_path.read_text(encoding="utf-8"))

    members = pd.read_parquet(input_path, columns=list(MEMBER_COLUMNS))
    config = RankedSubtypeConfig(**config_values)
    context, conservation, pairwise = classify_pathway_context_subtypes(members, config)
    _write_frame(context, context_path)
    _write_frame(conservation, conservation_path)
    _write_frame(pairwise, pairwise_path)
    result = {
        "status": "SUCCESS",
        "shard_id": int(shard_id),
        "pid": int(os.getpid()),
        "pathways": int(members.pathway_id.nunique()),
        "member_rows": int(len(members)),
        "context_rows": int(len(context)),
        "conservation_rows": int(len(conservation)),
        "pairwise_rows": int(len(pairwise)),
    }
    success_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _prepare_shards(
    *, member_paths: list[Path], shard_root: Path, n_shards: int
) -> tuple[list[Path], int]:
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_root / "SHARD_MANIFEST.json"
    existing = sorted(shard_root.glob("shard_*.members.parquet"))
    if manifest_path.is_file() and len(existing) == n_shards:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("n_shards", -1)) != n_shards:
            raise RuntimeError("existing subtype shard count disagrees with request")
        return existing, int(manifest["member_rows"])

    members = pd.concat(
        [pd.read_parquet(path, columns=list(MEMBER_COLUMNS)) for path in member_paths],
        ignore_index=True,
    )
    members["pathway_id"] = members.pathway_id.astype(str)
    cost = (
        members.groupby("pathway_id", observed=True)
        .agg(member_rows=("lncrna_id", "size"), cancers=("cancer_id", "nunique"))
        .reset_index()
    )
    cost["estimated_cost"] = cost.member_rows.astype(float) * cost.cancers.astype(float)
    bins: list[list[str]] = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    ordered = cost.sort_values(
        ["estimated_cost", "pathway_id"], ascending=[False, True], kind="stable"
    )
    for row in ordered.itertuples(index=False):
        target = min(range(n_shards), key=lambda index: (loads[index], index))
        bins[target].append(str(row.pathway_id))
        loads[target] += float(row.estimated_cost)

    paths: list[Path] = []
    for shard_id, pathway_ids in enumerate(bins):
        if not pathway_ids:
            raise RuntimeError("more shards requested than exact pathways")
        path = shard_root / f"shard_{shard_id:04d}.members.parquet"
        subset = members.loc[members.pathway_id.isin(pathway_ids)].sort_values(
            ["pathway_id", "cancer_id", "association_direction", "geneset_rank", "lncrna_id"],
            kind="stable",
        )
        _write_frame(subset, path)
        paths.append(path)
    manifest = {
        "status": "READY",
        "n_shards": n_shards,
        "member_rows": int(len(members)),
        "pathways": int(members.pathway_id.nunique()),
        "loads": loads,
        "pathway_seed_isolation": True,
        "bootstrap_replicates": 200,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths, int(len(members))


def _concat_sorted(paths: list[Path], sort_columns: list[str]) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-root", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--shards", type=int, default=24)
    parser.add_argument("--run-name", default="bootstrap200_seed20260726_parallel_v1")
    args = parser.parse_args()
    if args.workers < 1 or args.shards < args.workers:
        raise ValueError("workers must be positive and shards must be at least workers")

    root = Path(args.materialization_root)
    member_paths = sorted((root / "geneset_member_parts").glob("*.parquet"))
    master_paths = sorted((root / "geneset_master_parts").glob("*.parquet"))
    if len(member_paths) != 33 or len(master_paths) != 33:
        raise RuntimeError("parallel subtype resume requires all 33 cancer Gene Set parts")
    run_root = root / "subtype_parallel" / args.run_name
    input_root = run_root / "inputs"
    result_root = run_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    shard_paths, member_rows = _prepare_shards(
        member_paths=member_paths, shard_root=input_root, n_shards=args.shards
    )
    config = _frozen_config()
    config_values = asdict(config)
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_shard, shard_id, str(path), str(result_root), config_values
            ): shard_id
            for shard_id, path in enumerate(shard_paths)
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(json.dumps({"status": "SUBTYPE_SHARD_DONE", **result}, sort_keys=True), flush=True)

    if len(completed) != args.shards or any(item.get("status") != "SUCCESS" for item in completed):
        raise RuntimeError("not all pathway subtype shards completed")
    context = _concat_sorted(
        sorted(result_root.glob("shard_*.context.parquet")),
        ["pathway_id", "cancer_id"],
    )
    conservation = _concat_sorted(
        sorted(result_root.glob("shard_*.conservation.parquet")), ["pathway_id"]
    )
    pairwise = _concat_sorted(
        sorted(result_root.glob("shard_*.pairwise.parquet")),
        ["pathway_id", "cancer_a", "cancer_b"],
    )
    program, program_similarity = classify_cancer_program_subtypes(
        context, conservation, pairwise, config
    )
    _write_frame(context, root / "PATHWAY_CONTEXT_SUBTYPE.parquet")
    _write_frame(conservation, root / "PATHWAY_CONSERVATION.parquet")
    _write_frame(pairwise, root / "PATHWAY_PAIRWISE_SIMILARITY.parquet")
    _write_frame(program, root / "CANCER_PROGRAM_SUBTYPE.parquet")
    _write_frame(program_similarity, root / "CANCER_PROGRAM_SIMILARITY.parquet")

    master = pd.concat([pd.read_parquet(path) for path in master_paths], ignore_index=True)
    summary = {
        "status": "SUCCESS",
        "analysis_version": "V3_2_ONESEED_FORMAL_RELEASE",
        "cancers": 33,
        "genesets": int(len(master)),
        "subtype_member_rows": int(member_rows),
        "pathway_context_rows": int(len(context)),
        "pathway_conservation_rows": int(len(conservation)),
        "pathway_pairwise_rows": int(len(pairwise)),
        "cancer_program_rows": int(len(program)),
        "parallel_workers": int(args.workers),
        "parallel_shards": int(args.shards),
        "bootstrap_replicates": int(config.bootstrap_replicates),
        "pathway_seed_isolation": True,
        "family_used_as_target": False,
        "subtype_feedback_to_model": False,
        "regulatory_evidence_used_for_ranking": False,
    }
    (root / "MATERIALIZATION_SUCCESS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
