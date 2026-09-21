#!/usr/bin/env python3
"""Assemble patient-level inputs for the V3.2 distal-regulatory mutation head."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORMAL_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_filtered(path: Path, cancer: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, filters=[("cancer_id", "==", cancer)])
    except (TypeError, ValueError):
        frame = pd.read_parquet(path)
        return frame.loc[frame.cancer_id.astype(str).str.upper().eq(cancer)].copy()


def _lncrna_core(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.removeprefix("LNC:")
        .str.split(".", regex=False)
        .str[0]
    )


def _atac_long(root: Path, cancer: str, allowed_lncrnas: set[str]) -> pd.DataFrame:
    path = root / f"{cancer}.distal_accessibility.tsv.gz"
    columns = ["cancer_id", "patient_id", "lncrna_id", "atac_distal_accessibility"]
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    wide = pd.read_csv(path, sep="\t", compression="gzip")
    if wide.empty or "gene_id" not in wide:
        return pd.DataFrame(columns=columns)
    allowed = pd.DataFrame({"lncrna_id": sorted(map(str, allowed_lncrnas))})
    allowed["lncrna_core"] = _lncrna_core(allowed.lncrna_id)
    if allowed.lncrna_core.duplicated().any():
        raise RuntimeError("Allowed lncRNA identifiers collide after namespace normalization")
    core_to_id = allowed.set_index("lncrna_core").lncrna_id.to_dict()
    wide["lncrna_core"] = _lncrna_core(wide.gene_id)
    wide = wide.loc[wide.lncrna_core.isin(core_to_id)].copy()
    if wide.empty:
        return pd.DataFrame(columns=columns)
    wide["lncrna_id"] = wide.lncrna_core.map(core_to_id)
    long = wide.melt(
        id_vars=["lncrna_id"],
        value_vars=[
            column
            for column in wide.columns
            if column not in {"gene_id", "lncrna_core", "lncrna_id"}
        ],
        var_name="patient_id",
        value_name="atac_distal_accessibility",
    )
    long["atac_distal_accessibility"] = pd.to_numeric(
        long.atac_distal_accessibility, errors="coerce"
    )
    long = long.loc[np.isfinite(long.atac_distal_accessibility.to_numpy(float))]
    long["patient_id"] = long.patient_id.astype(str).str[:12]
    long.insert(0, "cancer_id", cancer)
    return long[columns].drop_duplicates(
        ["cancer_id", "patient_id", "lncrna_id"], keep="last"
    )


def _purity_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"cancer_id", "patient_id"}
    if missing := sorted(required - set(frame.columns)):
        raise RuntimeError(f"Purity covariates lack {missing}")
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    sources = pd.Series(None, index=frame.index, dtype=object)
    for column in ("purity", "absolute_purity", "paper_purity"):
        if column not in frame:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce")
        take = values.isna() & candidate.notna()
        values.loc[take] = candidate.loc[take]
        sources.loc[take] = column
    local = pd.DataFrame(
        {
            "cancer_id": frame.cancer_id.astype(str).str.upper(),
            "patient_id": frame.patient_id.astype(str).str[:12],
            "tumor_purity": values,
            "tumor_purity_source": sources,
        }
    )
    local = local.loc[np.isfinite(local.tumor_purity.to_numpy(float))]
    # Equal patient weight; multiple tumour aliquots are averaged first.
    purity = local.groupby(
        ["cancer_id", "patient_id"], observed=True, sort=False
    ).tumor_purity.mean().reset_index()
    source = local.groupby(
        ["cancer_id", "patient_id"], observed=True, sort=False
    ).tumor_purity_source.agg(lambda x: ";".join(sorted(set(map(str, x))))).reset_index()
    return purity.merge(source, on=["cancer_id", "patient_id"], validate="one_to_one")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _typed_feature(
    frame: pd.DataFrame,
    feature: str,
    availability: str | None = None,
) -> None:
    flag = availability or f"{feature}__available"
    frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    if flag not in frame:
        frame[flag] = frame[feature].notna()
    frame[flag] = frame[flag].fillna(False).astype(bool)
    valid = frame[flag] & np.isfinite(frame[feature].to_numpy(float))
    frame[flag] = valid
    frame.loc[~valid, feature] = np.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-expression-root", type=Path, required=True)
    parser.add_argument("--distal-mutation", type=Path, required=True)
    parser.add_argument("--distal-atac-root", type=Path, required=True)
    parser.add_argument("--methylation", type=Path, required=True)
    parser.add_argument("--local-cnv", type=Path, required=True)
    parser.add_argument("--purity-covariates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancers", default=",".join(FORMAL_CANCERS))
    args = parser.parse_args()

    sources = {
        "distal_mutation": args.distal_mutation.resolve(),
        "methylation": args.methylation.resolve(),
        "local_cnv": args.local_cnv.resolve(),
        "purity_covariates": args.purity_covariates.resolve(),
    }
    for label, path in sources.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label}: {path}")
    expression_root = args.formal_expression_root.resolve()
    atac_root = args.distal_atac_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Head input output reuse refused: {output}")
    output.mkdir(parents=True)
    purity_all = _purity_table(sources["purity_covariates"])
    cancers = tuple(
        item.strip().upper() for item in args.cancers.split(",") if item.strip()
    )
    if not cancers or not set(cancers).issubset(FORMAL_CANCERS):
        raise RuntimeError("Requested cancers are outside the formal 33-cancer contract")

    cancer_audits: list[dict[str, Any]] = []
    output_files: list[dict[str, Any]] = []
    for cancer in cancers:
        expression_path = expression_root / f"cancer_id={cancer}" / "part-0.parquet"
        if not expression_path.is_file():
            raise FileNotFoundError(expression_path)
        mutation = _read_filtered(sources["distal_mutation"], cancer)
        if mutation.empty:
            raise RuntimeError(f"Distal mutation table lacks formal cancer {cancer}")
        mutation["patient_id"] = mutation.patient_id.astype(str).str[:12]
        mutation["lncrna_id"] = mutation.lncrna_id.astype(str)
        allowed_lncrnas = set(mutation.lncrna_id)
        expression = pd.read_parquet(
            expression_path,
            columns=["cancer_id", "patient_id", "lncrna_id", "logcpm"],
        )
        expression["patient_id"] = expression.patient_id.astype(str).str[:12]
        expression["lncrna_id"] = expression.lncrna_id.astype(str)
        expression = expression.loc[expression.lncrna_id.isin(allowed_lncrnas)]
        expression["logcpm"] = pd.to_numeric(expression.logcpm, errors="coerce")
        expression = (
            expression.groupby(
                ["cancer_id", "patient_id", "lncrna_id"], observed=True, sort=False
            ).logcpm.mean().reset_index().rename(columns={"logcpm": "lncrna_expression"})
        )
        base = expression.merge(
            mutation[
                [
                    "cancer_id", "patient_id", "lncrna_id", "patient_fold_id",
                    "distal_mutation_burden", "distal_mutation_burden__available",
                ]
            ],
            on=["cancer_id", "patient_id", "lncrna_id"],
            how="left",
            validate="one_to_one",
        )
        if base.patient_fold_id.isna().any():
            raise RuntimeError(f"Expression rows lack frozen patient fold in {cancer}")

        atac = _atac_long(atac_root, cancer, allowed_lncrnas)
        base = base.merge(
            atac, on=["cancer_id", "patient_id", "lncrna_id"],
            how="left", validate="one_to_one",
        )
        methylation = _read_filtered(sources["methylation"], cancer)
        methylation["patient_id"] = methylation.patient_id.astype(str).str[:12]
        methylation["lncrna_id"] = methylation.lncrna_id.astype(str)
        methylation_columns = [
            "cancer_id", "patient_id", "lncrna_id",
            "promoter_methylation_beta", "promoter_methylation_beta__available",
            "distal_methylation_beta", "distal_methylation_beta__available",
        ]
        base = base.merge(
            methylation[methylation_columns],
            on=["cancer_id", "patient_id", "lncrna_id"],
            how="left", validate="one_to_one",
        )
        cnv = _read_filtered(sources["local_cnv"], cancer)
        cnv["patient_id"] = cnv.patient_id.astype(str).str[:12]
        cnv["lncrna_id"] = cnv.lncrna_id.astype(str)
        cnv = cnv.rename(
            columns={"cnv_value": "local_cnv_log2", "cnv_callable": "local_cnv_log2__available"}
        )
        base = base.merge(
            cnv[
                ["cancer_id", "patient_id", "lncrna_id", "local_cnv_log2", "local_cnv_log2__available"]
            ],
            on=["cancer_id", "patient_id", "lncrna_id"],
            how="left", validate="one_to_one",
        )
        purity = purity_all.loc[purity_all.cancer_id.eq(cancer)].copy()
        purity = purity.rename(columns={"tumor_purity": "tumor_purity"})
        base = base.merge(
            purity, on=["cancer_id", "patient_id"], how="left", validate="many_to_one"
        )
        base["atac_distal_accessibility__available"] = base[
            "atac_distal_accessibility"
        ].notna()
        base["tumor_purity__available"] = base.tumor_purity.notna()
        for feature in (
            "distal_mutation_burden",
            "atac_distal_accessibility",
            "promoter_methylation_beta",
            "distal_methylation_beta",
            "local_cnv_log2",
            "tumor_purity",
        ):
            _typed_feature(base, feature)
        if base.duplicated(["cancer_id", "patient_id", "lncrna_id"]).any():
            raise RuntimeError(f"Assembled input has duplicate keys in {cancer}")
        destination = output / f"cancer_id={cancer}"
        destination.mkdir()
        path = destination / "part-0.parquet"
        base = base.sort_values(["patient_id", "lncrna_id"], kind="stable")
        _atomic_parquet(base, path)
        coverage = {
            feature: int(base[f"{feature}__available"].sum())
            for feature in (
                "distal_mutation_burden", "atac_distal_accessibility",
                "promoter_methylation_beta", "distal_methylation_beta",
                "local_cnv_log2", "tumor_purity",
            )
        }
        cancer_audits.append(
            {
                "cancer_id": cancer,
                "rows": int(len(base)),
                "patients": int(base.patient_id.nunique()),
                "lncrnas": int(base.lncrna_id.nunique()),
                "feature_available_rows": coverage,
            }
        )
        output_files.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )

    audit = {
        "format": "CANCERLNCATLAS_V32_DISTAL_REGULATORY_HEAD_INPUT_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cancers": list(cancers),
        "mutation_feature_for_formal_head": "distal_mutation_burden",
        "recurrence_weighted_feature_excluded_from_formal_head": True,
        "recurrence_exclusion_reason": "WITHIN_CANCER_RECURRENCE_COUNT_NOT_OUTER_FOLD_RECOMPUTED",
        "missing_assay_assumed_zero": False,
        "multiple_tumour_aliquots_equal_patient_weight": True,
        "sources": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in sources.items()
        },
        "formal_expression_root": str(expression_root),
        "distal_atac_root": str(atac_root),
        "atac_lncrna_namespace_normalized_to_formal_ids": True,
        "cancer_audits": cancer_audits,
        "outputs": output_files,
    }
    audit_path = output / "AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "audit": str(audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
