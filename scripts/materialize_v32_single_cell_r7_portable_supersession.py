#!/usr/bin/env python3
"""Build a fail-closed, portable r7 control supersession for cell-level work.

The r6 control layer points its annotation, exact-membership and derived asset
families at ``/public0``.  This materializer copies only immutable authority
inputs (GENCODE source/cache and exact membership), rewrites raw-input bindings
to the authorized portable root, and deliberately leaves all derived asset
families unbound.  It never relabels an r5/r6 derived result as fresh.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
R6_HASHES = {
    "RUN_STATUS.json": "3a9ec28a6168f1e38d65521d9e2d7fff818becef6f6d74a3dd9846c1715c0423",
    "TRAINING_HANDOFF.json": "627fbe52df38e58692b4091d9d1ab870da6629c6eec56f9c6eb7aa5c11c03ad4",
    "dataset_manifest_33c.parquet": "bea2d20a261063dacdd05b3b18865f8de0fec3504086e08e6295b63fe0a4ffad",
}
EXACT_MEMBERSHIP_SHA256 = (
    "22b6215920f58b13d5e80f8fe93f6964ec0ea0fc14288c64aa758509e4ace975"
)
FORMAL_17 = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LGG",
    "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM", "UCEC",
)
DERIVED_ASSET_FAMILIES = (
    "activity", "association", "exact_pathway_availability",
    "expression_facts", "lnc_celltype",
)
AUTHORIZED_PREFIXES = (
    "./data/CancerLncAtlas/",
    "./data/CancerLncAtlas/",
)


class PortableSupersessionError(RuntimeError):
    """Raised when r7 cannot prove portable, non-historical lineage."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attach_contract(value: dict[str, Any], key: str) -> None:
    value.pop(key, None)
    value[key] = _canonical_sha256(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise PortableSupersessionError(f"Output reuse is forbidden: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _assert_server_path(value: str, role: str) -> str:
    normalized = str(value).replace("\\", "/")
    if not normalized.startswith(AUTHORIZED_PREFIXES):
        raise PortableSupersessionError(f"{role} is outside authorized roots: {value}")
    if normalized.startswith("${DATA_ROOT}/"):
        raise PortableSupersessionError(f"{role} still points at /public0: {value}")
    return normalized


def _copy_new(source: Path, target: Path) -> None:
    if target.exists():
        raise PortableSupersessionError(f"Output reuse is forbidden: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def materialize(
    *,
    r6_root: Path,
    output_root: Path,
    portable_server_root: str,
    gtf_source: Path,
    local_annotation: Path,
    local_annotation_provenance: Path,
    exact_membership: Path,
    server_readability_tsv: Path,
    run_id: str,
) -> dict[str, Any]:
    portable = _assert_server_path(portable_server_root.rstrip("/") + "/", "r7 root").rstrip("/")
    for name, expected in R6_HASHES.items():
        path = r6_root / name
        if artifact_sha256(path) != expected:
            raise PortableSupersessionError(f"r6 control hash drift: {name}")
    if artifact_sha256(exact_membership) != EXACT_MEMBERSHIP_SHA256:
        raise PortableSupersessionError("Exact 2,135-pathway membership hash drift")

    run6 = json.loads((r6_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    handoff6 = json.loads(
        (r6_root / "TRAINING_HANDOFF.json").read_text(encoding="utf-8")
    )
    if run6.get("blocked_fresh_direct_id_counts_typed_unavailable") is not True:
        raise PortableSupersessionError("r6 semantic correction is absent")
    if handoff6.get("blocked_fresh_direct_id_counts_typed_unavailable") is not True:
        raise PortableSupersessionError("r6 handoff does not bind semantic correction")
    if tuple(sorted(run6.get("formal_eligible_cancers", []))) != tuple(sorted(FORMAL_17)):
        raise PortableSupersessionError("r6 formal-eligible scope is not the pinned 17 cancers")

    annotation_payload = json.loads(local_annotation_provenance.read_text(encoding="utf-8"))
    annotation_sha = artifact_sha256(local_annotation)
    if annotation_payload.get("annotation_sha256") != annotation_sha:
        raise PortableSupersessionError("Locally rebuilt annotation provenance SHA mismatch")
    if annotation_payload.get("source_gtf_sha256") != artifact_sha256(gtf_source):
        raise PortableSupersessionError("Locally rebuilt annotation GTF SHA mismatch")

    authority = output_root / "authority"
    authority.mkdir(parents=True, exist_ok=True)
    gtf_target = authority / "source" / "gencode.v50.annotation.gtf.gz"
    membership_target = authority / "exact_pathway_membership_2135.parquet"
    _copy_new(gtf_source, gtf_target)
    _copy_new(exact_membership, membership_target)
    if artifact_sha256(gtf_target) != artifact_sha256(gtf_source):
        raise PortableSupersessionError("Portable GTF copy verification failed")
    if artifact_sha256(membership_target) != EXACT_MEMBERSHIP_SHA256:
        raise PortableSupersessionError("Portable membership copy verification failed")

    annotation_server = f"{portable}/authority/gencode_v50_gene_annotation.parquet"
    provenance_server = f"{portable}/authority/GENCODE_ANNOTATION_PROVENANCE.json"
    gtf_server = f"{portable}/authority/source/gencode.v50.annotation.gtf.gz"
    membership_server = f"{portable}/authority/exact_pathway_membership_2135.parquet"
    portable_annotation_provenance = copy.deepcopy(annotation_payload)
    portable_annotation_provenance.update(
        {
            "source_gtf_path": gtf_server,
            "annotation_path": annotation_server,
            "portable_supersession_run_id": run_id,
            "local_build_provenance_sha256": artifact_sha256(local_annotation_provenance),
        }
    )
    _attach_contract(portable_annotation_provenance, "contract_sha256")
    # Replace the local-path provenance in-place only because this r7 output is
    # new and has not yet been published; the immutable r6 source is untouched.
    local_annotation_provenance.write_text(
        json.dumps(portable_annotation_provenance, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    raw_check = pd.read_csv(server_readability_tsv, sep="\t")
    if set(raw_check.columns) != {"cancer_id", "h5_bytes", "metadata_bytes"}:
        raise PortableSupersessionError("Server readability table has unexpected columns")
    if tuple(sorted(raw_check.cancer_id.astype(str))) != tuple(sorted(FORMAL_17)):
        raise PortableSupersessionError("Server readability table does not cover formal 17")
    if raw_check[["h5_bytes", "metadata_bytes"]].le(0).any().any():
        raise PortableSupersessionError("A formal raw input is missing or empty")

    manifest = pd.read_parquet(r6_root / "dataset_manifest_33c.parquet")
    manifest["portable_supersession_run_id"] = run_id
    manifest["portable_raw_input_bound"] = manifest.cancer_id.astype(str).isin(FORMAL_17)
    manifest["derived_assets_bound"] = False
    manifest["fresh_cell_level_compute_status"] = "PENDING_FRESH_RECOMPUTE"
    manifest_path = output_root / "dataset_manifest_33c.parquet"
    if manifest_path.exists():
        raise PortableSupersessionError(f"Output reuse is forbidden: {manifest_path}")
    manifest.to_parquet(manifest_path, index=False, compression="zstd")

    records: dict[str, Any] = {}
    for cancer in map(str, run6["scope"]):
        old = run6["per_cancer"][cancer]
        formal = cancer in FORMAL_17
        h5_path = _assert_server_path(old["h5_path"], f"{cancer} H5")
        metadata_path = _assert_server_path(old["metadata_path"], f"{cancer} metadata")
        records[cancer] = {
            "analysis_version": ANALYSIS_VERSION,
            "cancer_id": cancer,
            "dataset_id": old.get("dataset_id"),
            "status": old.get("status"),
            "formal_eligible": formal,
            "blocking_reason": old.get("blocking_reason"),
            "h5_path": h5_path,
            "h5_sha256": old.get("h5_sha256"),
            "metadata_path": metadata_path,
            "metadata_sha256": old.get("metadata_sha256"),
            "annotation_path": annotation_server,
            "annotation_sha256": annotation_sha,
            "annotation_provenance_path": provenance_server,
            "annotation_provenance_sha256": artifact_sha256(local_annotation_provenance),
            "exact_membership_path": membership_server,
            "exact_membership_sha256": EXACT_MEMBERSHIP_SHA256,
            "raw_input_readability_verified": formal,
            "derived_assets_bound": False,
            "source_partition_reused": False,
            "fresh_direct_id_lncrna_feature_count": old.get(
                "fresh_direct_id_lncrna_feature_count"
            ),
            "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "historical_sc_trajectory_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "release_ready": False,
            "production_deployed": False,
        }
        _attach_contract(records[cancer], "contract_sha256")

    run7 = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_RUN_STATUS_V1",
        "analysis_version": ANALYSIS_VERSION,
        "run_id": run_id,
        "scope": list(map(str, run6["scope"])),
        "scope_count": 33,
        "formal_eligible_cancers": list(FORMAL_17),
        "formal_eligible_count": len(FORMAL_17),
        "formal_raw_inputs_readable_count": len(raw_check),
        "portable_authority_bound": True,
        "derived_assets_bound": False,
        "derived_assets_recompute_required": True,
        "blocked_fresh_direct_id_counts_typed_unavailable": True,
        "per_cancer": records,
        "supersedes_run_id": run6.get("run_id"),
        "supersedes_run_status_sha256": R6_HASHES["RUN_STATUS.json"],
        "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "single_cell_module_complete": False,
        "training_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _attach_contract(run7, "status_contract_sha256")
    _write_json(output_root / "RUN_STATUS.json", run7)

    assets = {
        name: {
            "path": None,
            "status": "UNBOUND_FRESH_RECOMPUTE_REQUIRED",
            "historical_asset_relabelled_fresh": False,
        }
        for name in DERIVED_ASSET_FAMILIES
    }
    handoff7 = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_TRAINING_HANDOFF_V1",
        "analysis_version": ANALYSIS_VERSION,
        "run_id": run_id,
        "dataset_manifest_path": f"{portable}/dataset_manifest_33c.parquet",
        "dataset_manifest_sha256": artifact_sha256(manifest_path),
        "annotation_path": annotation_server,
        "annotation_sha256": annotation_sha,
        "annotation_provenance_path": provenance_server,
        "annotation_provenance_sha256": artifact_sha256(local_annotation_provenance),
        "exact_membership_path": membership_server,
        "exact_membership_sha256": EXACT_MEMBERSHIP_SHA256,
        "assets": assets,
        "assets_reused_by_reference_from_fresh_r5_partitions": False,
        "historical_assets_relabelled_fresh": False,
        "fresh_cell_level_preflight_ready": True,
        "private_head_training_ready_for_available_partitions": False,
        "blocked_fresh_direct_id_counts_typed_unavailable": True,
        "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "single_cell_module_complete": False,
        "training_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _attach_contract(handoff7, "handoff_contract_sha256")
    _write_json(output_root / "TRAINING_HANDOFF.json", handoff7)

    preflight = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_RAW_INPUT_READABILITY_PREFLIGHT_V1",
        "analysis_version": ANALYSIS_VERSION,
        "run_id": run_id,
        "method": "REMOTE_TEST_READABLE_AND_STAT_SIZE_ONLY",
        "formal_cancers": raw_check.to_dict(orient="records"),
        "formal_cancer_count": len(raw_check),
        "all_formal_h5_and_metadata_readable_nonempty": True,
        "content_sha_revalidation_pending": True,
        "h5_schema_and_metadata_schema_preflight_pending": True,
        "heavy_recompute_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _attach_contract(preflight, "preflight_contract_sha256")
    _write_json(output_root / "RAW_INPUT_READABILITY_PREFLIGHT.json", preflight)

    authority_control = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_AUTHORITY_V1",
        "analysis_version": ANALYSIS_VERSION,
        "run_id": run_id,
        "portable_server_root": portable,
        "gencode_gtf": {"path": gtf_server, "sha256": artifact_sha256(gtf_target)},
        "gencode_annotation": {"path": annotation_server, "sha256": annotation_sha},
        "gencode_provenance": {
            "path": provenance_server,
            "sha256": artifact_sha256(local_annotation_provenance),
        },
        "exact_membership": {
            "path": membership_server,
            "sha256": EXACT_MEMBERSHIP_SHA256,
            "exact_pathways": 2135,
        },
        "public0_consumable_references": 0,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "historical_sc_trajectory_used": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _attach_contract(authority_control, "authority_contract_sha256")
    _write_json(output_root / "PORTABLE_AUTHORITY.json", authority_control)

    supersession = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_SUPERSESSION_V1",
        "analysis_version": ANALYSIS_VERSION,
        "run_id": run_id,
        "reason": "R6_CONSUMABLE_PATHS_BOUND_TO_UNAUTHORIZED_PUBLIC0",
        "supersedes_run_id": run6.get("run_id"),
        "superseded_control_sha256": R6_HASHES,
        "annotation_rebuilt_from_gtf": True,
        "exact_membership_copied_and_hash_bound": True,
        "raw_h5_metadata_copied": False,
        "raw_h5_metadata_referenced_from_authorized_public8": True,
        "derived_assets_copied": False,
        "derived_assets_relabelled_fresh": False,
        "derived_assets_recompute_required": True,
        "heavy_recompute_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _attach_contract(supersession, "supersession_contract_sha256")
    _write_json(output_root / "SUPERSESSION.json", supersession)

    paths = sorted(path for path in output_root.rglob("*") if path.is_file())
    sha_table = pd.DataFrame(
        {
            "artifact": [str(path.relative_to(output_root)).replace("\\", "/") for path in paths],
            "sha256": [artifact_sha256(path) for path in paths],
        }
    )
    sha_path = output_root / "CONTROL_SHA256.tsv"
    if sha_path.exists():
        raise PortableSupersessionError(f"Output reuse is forbidden: {sha_path}")
    sha_table.to_csv(sha_path, sep="\t", index=False)
    return supersession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--portable-server-root", required=True)
    parser.add_argument("--gencode-gtf", required=True, type=Path)
    parser.add_argument("--local-annotation", required=True, type=Path)
    parser.add_argument("--local-annotation-provenance", required=True, type=Path)
    parser.add_argument("--exact-membership", required=True, type=Path)
    parser.add_argument("--server-readability-tsv", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = materialize(
        r6_root=args.r6_root.resolve(),
        output_root=args.output_root.resolve(),
        portable_server_root=args.portable_server_root,
        gtf_source=args.gencode_gtf.resolve(),
        local_annotation=args.local_annotation.resolve(),
        local_annotation_provenance=args.local_annotation_provenance.resolve(),
        exact_membership=args.exact_membership.resolve(),
        server_readability_tsv=args.server_readability_tsv.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
