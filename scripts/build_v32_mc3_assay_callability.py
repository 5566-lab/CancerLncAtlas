#!/usr/bin/env python3
"""Build V3.2 mutation inputs with explicit MC3 sample-assay callability.

The historical parquet inputs used here are standardized *event tables* made
from MC3, not model predictions.  MC3 does not provide per-locus depth in these
assets, so this builder adds a reserved sentinel for patients whose exome assay
was observed.  The downstream V3.2 module can then distinguish an assayed
sample with no observed event from a patient with no mutation assay.  The
result is deliberately marked diagnostic-only and must not be described as
per-locus wild-type callability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cc_hhgt.v32.genomic_training import ASSAY_CALLABLE_SENTINEL
from cc_hhgt.v32.input_lineage import artifact_sha256


def _boolean(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    mapped = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    observed = set(mapped.loc[values.notna()].unique())
    if not observed.issubset(allowed):
        raise ValueError(f"{name} is not an explicit boolean: {sorted(observed)}")
    return mapped.isin({"true", "1", "yes"})


def _patient(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(".", "-", regex=False)
    return text.where(~text.str.upper().str.startswith("TCGA-"), text.str.slice(0, 12))


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_inputs(
    sample_gene_path: Path,
    sample_lncrna_path: Path,
    sample_crosswalk_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    gene_raw = pd.read_parquet(sample_gene_path)
    lnc_raw = pd.read_parquet(sample_lncrna_path)
    crosswalk = pd.read_parquet(sample_crosswalk_path)
    _require(
        gene_raw,
        {"cancer_id", "patient_id", "gene_id", "is_mutated"},
        "sample-gene MC3 event table",
    )
    _require(
        lnc_raw,
        {"cancer_id", "patient_id", "lncrna_id", "lncrna_any_mutation"},
        "sample-lncRNA MC3 event table",
    )
    _require(
        crosswalk,
        {"cancer_id", "patient_id", "mutation_assay_available"},
        "MC3 sample crosswalk",
    )

    gene_event = _boolean(gene_raw["is_mutated"], "is_mutated")
    gene = pd.DataFrame(
        {
            "cancer_id": gene_raw.cancer_id.astype(str).str.upper(),
            "patient_id": _patient(gene_raw.patient_id),
            "gene_id": gene_raw.gene_id.astype(str),
            "is_mutated": gene_event,
            "mutation_callable": gene_event,
            "mutation_count": pd.to_numeric(
                gene_raw["mutation_count"] if "mutation_count" in gene_raw else gene_event,
                errors="coerce",
            ).fillna(gene_event.astype(float)),
            "absence_is_wildtype": False,
        }
    )
    gene = gene.loc[gene.is_mutated].drop_duplicates(
        ["cancer_id", "patient_id", "gene_id"], keep="first"
    )

    lnc_event = _boolean(lnc_raw["lncrna_any_mutation"], "lncrna_any_mutation")
    lnc = pd.DataFrame(
        {
            "cancer_id": lnc_raw.cancer_id.astype(str).str.upper(),
            "patient_id": _patient(lnc_raw.patient_id),
            "lncrna_id": lnc_raw.lncrna_id.astype(str),
            "lncrna_any_mutation": lnc_event,
            "lncrna_mutation_available": lnc_event,
            "exonic_variant_count": pd.to_numeric(
                lnc_raw["exonic_variant_count"]
                if "exonic_variant_count" in lnc_raw
                else lnc_event,
                errors="coerce",
            ).fillna(lnc_event.astype(float)),
            "absence_is_wildtype": False,
        }
    )
    lnc = lnc.loc[lnc.lncrna_any_mutation].drop_duplicates(
        ["cancer_id", "patient_id", "lncrna_id"], keep="first"
    )

    assayed = crosswalk.loc[
        _boolean(crosswalk.mutation_assay_available, "mutation_assay_available"),
        ["cancer_id", "patient_id"],
    ].copy()
    assayed["cancer_id"] = assayed.cancer_id.astype(str).str.upper()
    assayed["patient_id"] = _patient(assayed.patient_id)
    assayed = assayed.drop_duplicates(["cancer_id", "patient_id"])
    if assayed.empty:
        raise ValueError("No explicitly assayed MC3 patients were found")

    gene_sentinel = assayed.assign(
        gene_id=ASSAY_CALLABLE_SENTINEL,
        is_mutated=False,
        mutation_callable=True,
        mutation_count=0.0,
        absence_is_wildtype=False,
    )
    lnc_sentinel = assayed.assign(
        lncrna_id=ASSAY_CALLABLE_SENTINEL,
        lncrna_any_mutation=False,
        lncrna_mutation_available=True,
        exonic_variant_count=0.0,
        absence_is_wildtype=False,
    )
    gene = pd.concat([gene, gene_sentinel], ignore_index=True)
    lnc = pd.concat([lnc, lnc_sentinel], ignore_index=True)

    output_root.mkdir(parents=True, exist_ok=True)
    gene_output = output_root / "sample_gene_mutation_mc3_assay_callable.parquet"
    lnc_output = output_root / "sample_lncrna_mutation_mc3_assay_callable.parquet"
    _atomic_parquet(gene, gene_output)
    _atomic_parquet(lnc, lnc_output)
    sources = [sample_gene_path, sample_lncrna_path, sample_crosswalk_path]
    source_records = [
        {
            "path": str(path.resolve()),
            "sha256": artifact_sha256(path),
            "source_role": "standardized_raw_event_or_assay_metadata",
            "model_result": False,
        }
        for path in sources
    ]
    lineage: dict[str, Any] = {
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "artifact_role": "V3.2_MUTATION_STANDARDIZED_INPUT",
        "diagnostic_only": True,
        "callability_resolution": "MC3_WES_SAMPLE_ASSAY_LEVEL_NOT_LOCUS_DEPTH",
        "absence_semantics": "NO_EVENT_OBSERVED_IN_EXPLICITLY_ASSAYED_SAMPLE",
        "per_locus_wildtype_claimed": False,
        "missing_patient_assumed_wildtype": False,
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "sentinel": ASSAY_CALLABLE_SENTINEL,
        "assayed_patients": int(len(assayed)),
        "cancers": sorted(assayed.cancer_id.unique().tolist()),
        "gene_event_rows": int(gene.is_mutated.sum()),
        "lncrna_event_rows": int(lnc.lncrna_any_mutation.sum()),
        "source_artifacts": source_records,
        "outputs": {
            "sample_gene_mutation": {
                "path": str(gene_output.resolve()),
                "sha256": artifact_sha256(gene_output),
                "rows": int(len(gene)),
            },
            "sample_lncrna_mutation": {
                "path": str(lnc_output.resolve()),
                "sha256": artifact_sha256(lnc_output),
                "rows": int(len(lnc)),
            },
        },
    }
    lineage["lineage_sha256"] = _canonical_sha(lineage)
    lineage_path = output_root / "MC3_ASSAY_CALLABILITY_LINEAGE.json"
    _atomic_json(lineage, lineage_path)
    return {
        "status": "SUCCESS",
        "gene_output": str(gene_output.resolve()),
        "lncrna_output": str(lnc_output.resolve()),
        "lineage": str(lineage_path.resolve()),
        "assayed_patients": int(len(assayed)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-gene-mutation", type=Path, required=True)
    parser.add_argument("--sample-lncrna-mutation", type=Path, required=True)
    parser.add_argument("--sample-crosswalk", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_inputs(
        args.sample_gene_mutation,
        args.sample_lncrna_mutation,
        args.sample_crosswalk,
        args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
