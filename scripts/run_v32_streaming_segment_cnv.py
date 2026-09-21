#!/usr/bin/env python3
"""Materialize resumable full33 compact CNV partitions from verified segments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--download-complete", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--entity-intervals", required=True)
    parser.add_argument("--pathway-membership", required=True)
    parser.add_argument("--sample-gene-mutation", required=True)
    parser.add_argument("--sample-lncrna-mutation", required=True)
    parser.add_argument("--mc3", required=True)
    parser.add_argument("--core-embedding-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cancers", nargs="*")
    parser.add_argument("--event-threshold", type=float, default=0.30)
    parser.add_argument("--max-memory-bytes", type=int, default=536_870_912)
    parser.add_argument("--max-segment-rows", type=int, default=2_000_000)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.gdc_segment_cnv import TCGA_CANCERS
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )
    from cc_hhgt.v32.segment_cnv_streaming import materialize_streaming_store

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    patient_folds = resolve(args.patient_folds)
    patient_fold_receipt = resolve(args.patient_fold_authority_receipt)
    validate_frozen_v32_patient_fold_binding(patient_folds, patient_fold_receipt)

    result = materialize_streaming_store(
        download_complete_path=resolve(args.download_complete),
        staging_root=resolve(args.staging_root), candidates_path=resolve(args.candidates),
        patient_folds_path=patient_folds, entity_intervals_path=resolve(args.entity_intervals),
        membership_path=resolve(args.pathway_membership), mutation_gene_path=resolve(args.sample_gene_mutation),
        mutation_lncrna_path=resolve(args.sample_lncrna_mutation), mc3_path=resolve(args.mc3),
        core_manifest_path=resolve(args.core_embedding_manifest), output_root=resolve(args.output_root),
        cancers=args.cancers or TCGA_CANCERS, event_threshold=args.event_threshold,
        max_memory_bytes=args.max_memory_bytes, max_segment_rows=args.max_segment_rows,
        repo_root=root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
