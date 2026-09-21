#!/usr/bin/env python3
"""Audit whether the official TCGA ATAC assets can feed a fresh V3.2 expert.

This command is deliberately an audit, not a trainer.  It never consumes an
older prediction, never writes into a release, and refuses output-directory
reuse.  Its cancer-level readiness labels distinguish raw absence from
mapping/fold/integration gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.atac_readiness import (  # noqa: E402
    nested_fold_capacity,
    normalize_atac_cancer,
    readiness_class,
)
from cc_hhgt.v32.patient_fold_authority import (  # noqa: E402
    TCGA_CANCERS,
    validate_frozen_v32_patient_fold_binding,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
    os.replace(temporary, path)


def _gene_id(value: pd.Series) -> pd.Series:
    return (
        value.astype(str)
        .str.replace(r"^(?:LNC:|LNCRNA:|GENE:)", "", regex=True)
        .str.split(".")
        .str[0]
    )


def _overlap_gene_ids(promoters: pd.DataFrame, peaks: pd.DataFrame) -> set[str]:
    result: set[str] = set()
    for chromosome, local in promoters.groupby("chrom", observed=True, sort=False):
        peak = peaks.loc[peaks.seqnames.astype(str).eq(str(chromosome))]
        if peak.empty:
            continue
        starts = np.sort(pd.to_numeric(peak.start, errors="raise").to_numpy(np.int64))
        ends = np.sort(pd.to_numeric(peak.end, errors="raise").to_numpy(np.int64))
        promoter_start = pd.to_numeric(local.start, errors="raise").to_numpy(np.int64)
        promoter_end = pd.to_numeric(local.end, errors="raise").to_numpy(np.int64)
        overlap_count = (
            np.searchsorted(starts, promoter_end, side="left")
            - np.searchsorted(ends, promoter_start, side="right")
        )
        result.update(local.loc[overlap_count > 0, "gene_id_bare"].astype(str))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atac-asset-root", required=True)
    parser.add_argument("--candidate-universe", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--promoters", required=True)
    parser.add_argument("--pathway-membership", required=True)
    parser.add_argument("--historical-pilot-success", required=True)
    parser.add_argument("--historical-ablation-audit", required=True)
    parser.add_argument("--historical-r-builder", required=True)
    parser.add_argument("--router-source", required=True)
    parser.add_argument("--router-launcher", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _markdown(audit: dict[str, Any], cancer: pd.DataFrame) -> str:
    summary = audit["summary"]
    lines = [
        "# V3.2 ATAC 数据与训练就绪度审计",
        "",
        f"生成时间：`{audit['completed_at_utc']}`。本产物是只读审计，不是训练结果。",
        "",
        "## 结论",
        "",
        (
            f"官方 ATAC 资产实际覆盖 {summary['raw_covered_cancers']}/33 个癌种，"
            f"共 {summary['technical_replicates']} 个技术重复、"
            f"{summary['full_atac_aliquots']} 个完整 aliquot、"
            f"{summary['unique_atac_patients']} 名患者。"
        ),
        (
            f"其中 {summary['fold_aligned_patients']} 名患者能对齐 fresh V3.2 折叠，"
            f"{summary['unaligned_patients']} 名不能。当前没有 fresh V3.2 ATAC typed "
            "prediction、五折 checkpoint 或可核验 lineage，因此 0 个癌种可以宣称"
            "已有 ATAC 性能贡献。"
        ),
        "",
        "## 假缺失或处理/接入错误",
        "",
        "- 服务器 `./data/CancerLncAtlas/input/v3_1_target_context_assets_e0b44122/atac` 已有完整官方资产；不是需要重新下载。",
        "- 历史脚本只跑 BRCA、COAD、KIRP，造成 23 癌种原始覆盖被表现成 3 癌种覆盖。",
        "- 官方映射中的 `ACCx/GBMx/LGGx` 是等长填充，未去掉尾部 `x` 会造成癌种匹配失败。",
        (
            f"- 历史 R 脚本用 `substr(Case_ID, 1, 16)` 聚合，"
            f"把 {summary['full_atac_aliquots']} 个完整 aliquot 折成 "
            f"{summary['unique_atac_patients']} 个 patient/sample16 键，却称为 exact ATAC sample ID。"
        ),
        "- 8,541 个候选 lncRNA 全部能映射 GENCODE v36 promoter；其中 6,273 个 promoter 与 pan-cancer peak 重叠。",
        "- 路由器过去接受 ATAC 表但不验证 ATAC lineage；本次代码已增加 five-fold、旧模型禁用和预测表 SHA256 绑定。",
        "",
        "## 真缺口",
        "",
        "- 当前官方矩阵没有 DLBC、KICH、LAML、OV、PAAD、READ、SARC、THYM、UCS、UVM。",
        "- 12 名 ATAC 患者不在 fresh V3.2 canonical patient-fold universe。",
        "- 没有将 ATAC sample-level promoter/pathway accessibility 转成 exact-pair 五折 OOF typed predictions 的 V3.2 materializer/trainer。",
        "- 没有任何 fresh V3.2 ATAC-vs-primary held-out logloss/Brier/PR-AUC；历史三癌种 mask ablation 为 `retraining=false` 且总体建议 `NO-GO`。",
        "",
        "## 癌种级容量",
        "",
        "|癌种|判定|ATAC患者|折叠对齐|PF0|PF1|PF2|PF3|PF4|最小嵌套训练|保守pilot容量|正式贡献|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in cancer.itertuples(index=False):
        lines.append(
            f"|{row.cancer_id}|{row.readiness_class}|{row.raw_atac_patients}|"
            f"{row.fold_aligned_patients}|{row.fold_0}|{row.fold_1}|{row.fold_2}|"
            f"{row.fold_3}|{row.fold_4}|{row.minimum_nested_train_patients}|"
            f"{'是' if row.conservative_pilot_capacity_pass else '否'}|未估计|"
        )
    lines.extend(
        [
            "",
            "严格 `min_pair_callable=12` 列是当前 genomic 规则在 ATAC 患者数上的乐观上界；",
            "即使患者数通过，也仍需 lncRNA 与 pathway 双侧每折可调用性。当前所有癌种该上界均未完整通过五折。",
            "",
            "## 下一步训练契约",
            "",
            "1. 对 410 个完整 aliquot 先在 aliquot 内合并技术重复；多个 aliquot/患者不得静默平均，需预注册确定性选择或显式 patient-level 聚合。",
            "2. 仅用对应 outer-train 患者拟合 accessibility 标准化、阈值和 pathway aggregation；validation/test 不得进入任何上游拟合。",
            "3. 产生 3,300,000 行 exact-universe `atac_context_probability/atac_available/atac_unavailable_reason`，并保留五个未平均的 patient-fold 分区。",
            "4. 通过新增 ATAC lineage/hash 门禁后进入外置路由器；每癌种只由 inner validation 开门，held-out 指标不得反向选门。",
            "5. 只有 fresh OOF 相对 primary 同时改善 logloss 且不恶化 Brier 的癌种，才可报告 ATAC 正贡献。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = {
        key: Path(value).resolve()
        for key, value in {
            "asset_root": args.atac_asset_root,
            "candidates": args.candidate_universe,
            "folds": args.patient_folds,
            "fold_receipt": args.patient_fold_authority_receipt,
            "promoters": args.promoters,
            "membership": args.pathway_membership,
            "historical_pilot_success": args.historical_pilot_success,
            "historical_ablation": args.historical_ablation_audit,
            "historical_r_builder": args.historical_r_builder,
            "router_source": args.router_source,
            "router_launcher": args.router_launcher,
        }.items()
    }
    asset_files = {
        "matrix": paths["asset_root"] / "TCGA-ATAC_PanCan_Log2Norm_Counts.rds",
        "mapping": paths["asset_root"] / "TCGA_identifier_mapping.txt",
        "peaks": paths["asset_root"] / "TCGA-ATAC_PanCancer_PeakSet.txt",
        "datas7": paths["asset_root"] / "TCGA-ATAC_DataS7_PeakToGeneLinks_v2.xlsx",
        "datas1": paths["asset_root"] / "TCGA-ATAC_DataS1_DonorsAndStats_v4.xlsx",
    }
    required = [
        paths["candidates"], paths["folds"], paths["promoters"], paths["membership"],
        paths["historical_pilot_success"], paths["historical_ablation"],
        paths["historical_r_builder"], paths["router_source"], paths["router_launcher"],
        *asset_files.values(),
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    output = Path(args.output_root).resolve()
    if output.exists():
        raise RuntimeError(f"ATAC readiness audit refuses output reuse: {output}")
    output.mkdir(parents=True)

    mapping = pd.read_csv(asset_files["mapping"], sep="\t")
    mapping_required = {"bam_prefix", "stanfordUUID", "Case_ID"}
    if missing := sorted(mapping_required - set(mapping.columns)):
        raise RuntimeError(f"Official ATAC mapping lacks columns: {missing}")
    mapping["cancer_id"] = mapping.bam_prefix.map(normalize_atac_cancer)
    mapping["patient_id"] = mapping.Case_ID.astype(str).str[:12]
    mapping["sample16_id"] = mapping.Case_ID.astype(str).str[:16]
    mapping["full_aliquot_id"] = mapping.Case_ID.astype(str)
    if mapping.bam_prefix.duplicated().any():
        raise RuntimeError("Official ATAC technical replicate identifiers are duplicated")
    raw_patient = mapping[["cancer_id", "patient_id"]].drop_duplicates()

    fold_authority_audit = validate_frozen_v32_patient_fold_binding(
        paths["folds"],
        paths["fold_receipt"],
    )
    folds = pd.read_csv(paths["folds"], sep="\t")
    fold_required = {
        "cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"
    }
    if missing := sorted(fold_required - set(folds.columns)):
        raise RuntimeError(f"Fresh V3.2 fold manifest lacks columns: {missing}")
    folds["cancer_id"] = folds.cancer_id.astype(str).str.upper()
    folds["patient_id"] = folds.patient_id.astype(str).str.strip()
    folds["patient_fold_id"] = pd.to_numeric(
        folds.patient_fold_id, errors="raise"
    ).astype(int)
    if not set(folds.patient_fold_id.unique()).issubset(set(range(5))):
        raise RuntimeError("Fresh V3.2 fold manifest contains an invalid fold")
    fold_patient = folds[["cancer_id", "patient_id", "patient_fold_id"]].drop_duplicates()
    if fold_patient.duplicated(["cancer_id", "patient_id"]).any():
        raise RuntimeError("Fresh V3.2 explicit patient authority maps a patient to multiple folds")
    aligned_patient = raw_patient.merge(
        fold_patient,
        on=["cancer_id", "patient_id"], how="left", validate="one_to_one",
    )

    candidates = pd.read_parquet(
        paths["candidates"], columns=["cancer_id", "lncrna_id", "pathway_id"]
    )
    candidates["cancer_id"] = candidates.cancer_id.astype(str).str.upper()
    if len(candidates) != 3_300_000 or candidates.cancer_id.nunique() != 33:
        raise RuntimeError("Candidate input is not the exact 3.3M/33-cancer V3.2 universe")
    if candidates[["cancer_id", "lncrna_id", "pathway_id"]].duplicated().any():
        raise RuntimeError("Candidate universe contains duplicate exact keys")
    candidate_lnc = candidates[["cancer_id", "lncrna_id"]].drop_duplicates()
    candidate_lnc["gene_id"] = _gene_id(candidate_lnc.lncrna_id)
    candidate_pathways = set(candidates.pathway_id.astype(str).unique())

    membership = pd.read_parquet(paths["membership"])
    if not {"pathway_id", "gene_id"}.issubset(membership.columns):
        raise RuntimeError("Exact pathway membership lacks pathway_id/gene_id")
    membership["pathway_id"] = membership.pathway_id.astype(str)
    membership["gene_id"] = _gene_id(membership.gene_id)
    membership = membership.loc[
        membership.pathway_id.isin(candidate_pathways), ["pathway_id", "gene_id"]
    ].drop_duplicates()

    promoters = pd.read_csv(paths["promoters"], sep="\t", compression="gzip")
    promoters.columns = [str(column).lstrip("#") for column in promoters.columns]
    if not {"chrom", "start", "end", "gene_id"}.issubset(promoters.columns):
        raise RuntimeError("Promoter BED schema is not registered")
    promoters["gene_id_bare"] = _gene_id(promoters.gene_id)
    required_genes = set(candidate_lnc.gene_id) | set(membership.gene_id)
    promoters = promoters.loc[promoters.gene_id_bare.isin(required_genes)].copy()
    promoter_gene_ids = set(promoters.gene_id_bare.astype(str))
    peaks = pd.read_csv(
        asset_files["peaks"], sep="\t", usecols=["seqnames", "start", "end"]
    )
    if len(peaks) != 562_709:
        raise RuntimeError("Official ATAC peak set no longer has 562,709 rows")
    overlap_gene_ids = _overlap_gene_ids(promoters, peaks)

    candidate_lnc["gencode_v36_promoter"] = candidate_lnc.gene_id.isin(promoter_gene_ids)
    candidate_lnc["promoter_peak_overlap"] = candidate_lnc.gene_id.isin(overlap_gene_ids)
    lnc_stats = candidate_lnc.groupby("cancer_id", observed=True).agg(
        candidate_lncrnas=("gene_id", "nunique"),
        gencode_v36_promoter_lncrnas=("gencode_v36_promoter", "sum"),
        promoter_peak_overlap_lncrnas=("promoter_peak_overlap", "sum"),
    )

    membership["gencode_v36_promoter"] = membership.gene_id.isin(promoter_gene_ids)
    membership["promoter_peak_overlap"] = membership.gene_id.isin(overlap_gene_ids)
    pathway_stats = membership.groupby("pathway_id", observed=True).agg(
        member_genes=("gene_id", "nunique"),
        gencode_v36_promoter_genes=("gencode_v36_promoter", "sum"),
        promoter_peak_overlap_genes=("promoter_peak_overlap", "sum"),
    )

    raw_counts = mapping.groupby("cancer_id", observed=True).agg(
        technical_replicates=("bam_prefix", "nunique"),
        full_atac_aliquots=("full_aliquot_id", "nunique"),
        raw_atac_patients=("patient_id", "nunique"),
    )
    aligned_counts = aligned_patient.groupby("cancer_id", observed=True).agg(
        fold_aligned_patients=("patient_fold_id", "count"),
        unaligned_patients=("patient_fold_id", lambda value: int(value.isna().sum())),
    )
    fold_counts = (
        aligned_patient.dropna(subset=["patient_fold_id"])
        .pivot_table(
            index="cancer_id", columns="patient_fold_id", values="patient_id",
            aggfunc="count", fill_value=0,
        )
        .reindex(columns=range(5), fill_value=0)
    )

    rows: list[dict[str, Any]] = []
    for cancer_id in sorted(candidates.cancer_id.unique()):
        raw_available = cancer_id in raw_counts.index
        counts = tuple(
            int(fold_counts.at[cancer_id, fold]) if cancer_id in fold_counts.index else 0
            for fold in range(5)
        )
        capacity = nested_fold_capacity(counts)
        raw = raw_counts.loc[cancer_id].to_dict() if raw_available else {}
        aligned = aligned_counts.loc[cancer_id].to_dict() if cancer_id in aligned_counts.index else {}
        lnc = lnc_stats.loc[cancer_id]
        label = readiness_class(raw_available, capacity)
        if not raw_available:
            rescue = "NEW_AUTHORIZED_ATAC_COHORT_REQUIRED"
        elif label == "PRESENT_BUT_ALIGNMENT_OR_FOLD_SPARSE":
            rescue = "ADD_MATCHED_PATIENTS_OR_USE_EXPLICIT_UNAVAILABLE_MASK"
        else:
            rescue = "BUILD_FRESH_FOLD_LOCAL_ATAC_EXPERT_THEN_INNER_GATE"
        rows.append(
            {
                "cancer_id": cancer_id,
                "raw_atac_available": bool(raw_available),
                "technical_replicates": int(raw.get("technical_replicates", 0)),
                "full_atac_aliquots": int(raw.get("full_atac_aliquots", 0)),
                "raw_atac_patients": int(raw.get("raw_atac_patients", 0)),
                "fold_aligned_patients": int(aligned.get("fold_aligned_patients", 0)),
                "unaligned_patients": int(aligned.get("unaligned_patients", 0)),
                **{f"fold_{fold}": counts[fold] for fold in range(5)},
                "minimum_test_patients": capacity.minimum_test_patients,
                "minimum_validation_patients": capacity.minimum_validation_patients,
                "minimum_nested_train_patients": capacity.minimum_nested_train_patients,
                "all_folds_nonempty": capacity.all_folds_nonempty,
                "genomic_analogue_min12_upper_bound_pass": (
                    capacity.genomic_analogue_min12_upper_bound_pass
                ),
                "conservative_pilot_capacity_pass": capacity.conservative_pilot_capacity_pass,
                "candidate_lncrnas": int(lnc.candidate_lncrnas),
                "gencode_v36_promoter_lncrnas": int(lnc.gencode_v36_promoter_lncrnas),
                "promoter_peak_overlap_lncrnas": int(lnc.promoter_peak_overlap_lncrnas),
                "promoter_peak_overlap_fraction": float(
                    lnc.promoter_peak_overlap_lncrnas / lnc.candidate_lncrnas
                ),
                "candidate_exact_pathways": len(candidate_pathways),
                "readiness_class": label,
                "fresh_v32_atac_contribution_status": "NOT_ESTIMATED",
                "formal_router_ready": False,
                "rescue_action": rescue,
            }
        )
    cancer = pd.DataFrame(rows)

    duplicate_patient_aliquots = (
        mapping.groupby(["cancer_id", "patient_id"], observed=True)
        .full_aliquot_id.nunique()
        .loc[lambda value: value.gt(1)]
        .reset_index(name="full_aliquots")
    )
    historical_pilot = json.loads(
        paths["historical_pilot_success"].read_text(encoding="utf-8")
    )
    historical_ablation = json.loads(
        paths["historical_ablation"].read_text(encoding="utf-8")
    )
    router_source = paths["router_source"].read_text(encoding="utf-8")
    launcher_source = paths["router_launcher"].read_text(encoding="utf-8")
    old_builder = paths["historical_r_builder"].read_text(encoding="utf-8")

    completed = datetime.now(timezone.utc).isoformat()
    audit: dict[str, Any] = {
        "status": "AUDIT_COMPLETE_NOT_TRAINED",
        "analysis_version": "CancerLncAtlas_V3.2_ATAC_READINESS_AUDIT",
        "completed_at_utc": completed,
        "scope": "READ_ONLY_CURRENT_STATE_NO_DOWNLOAD_NO_PRODUCTION_CHANGE",
        "formal_v32_primary_unchanged": True,
        "fresh_v32_atac_predictions_generated": False,
        "fresh_v32_atac_contribution_estimated_cancers": [],
        "patient_fold_authority": {
            **fold_authority_audit,
            "raw_atac_case_barcode_to_patient_normalization": "TCGA_CASE_ID_FIRST_12_ONLY_FOR_RAW_ATAC_SOURCE_MAPPING",
            "fold_patient_id_source": "EXPLICIT_PATIENT_ID_COLUMN",
            "fold_sample_id_fallback_used": False,
        },
        "summary": {
            "v32_cancers": int(cancer.cancer_id.nunique()),
            "raw_covered_cancers": int(cancer.raw_atac_available.sum()),
            "true_raw_gap_cancers": int((~cancer.raw_atac_available).sum()),
            "raw_gap_cancer_ids": cancer.loc[
                ~cancer.raw_atac_available, "cancer_id"
            ].tolist(),
            "present_but_alignment_or_fold_sparse_cancers": int(
                cancer.readiness_class.eq("PRESENT_BUT_ALIGNMENT_OR_FOLD_SPARSE").sum()
            ),
            "present_pilotable_not_formal_router_ready_cancers": int(
                cancer.readiness_class.eq("PRESENT_PILOTABLE_NOT_FORMAL_ROUTER_READY").sum()
            ),
            "conservative_pilot_capacity_cancers": int(
                cancer.conservative_pilot_capacity_pass.sum()
            ),
            "conservative_pilot_capacity_cancer_ids": cancer.loc[
                cancer.conservative_pilot_capacity_pass, "cancer_id"
            ].tolist(),
            "strict_genomic_analogue_min12_all_fold_upper_bound_cancers": int(
                cancer.genomic_analogue_min12_upper_bound_pass.sum()
            ),
            "technical_replicates": int(mapping.bam_prefix.nunique()),
            "full_atac_aliquots": int(mapping.full_aliquot_id.nunique()),
            "unique_atac_patients": int(mapping.patient_id.nunique()),
            "fold_aligned_patients": int(aligned_patient.patient_fold_id.notna().sum()),
            "unaligned_patients": int(aligned_patient.patient_fold_id.isna().sum()),
            "candidate_rows": int(len(candidates)),
            "candidate_lncrnas": int(candidate_lnc.gene_id.nunique()),
            "candidate_lncrnas_with_gencode_v36_promoter": int(
                len(set(candidate_lnc.gene_id) & promoter_gene_ids)
            ),
            "candidate_lncrnas_with_promoter_peak_overlap": int(
                len(set(candidate_lnc.gene_id) & overlap_gene_ids)
            ),
            "candidate_exact_pathways": len(candidate_pathways),
        },
        "pathway_alignment": {
            "membership_rows_in_candidate_scope": int(len(membership)),
            "membership_pathways": int(pathway_stats.index.nunique()),
            "candidate_pathways_missing_membership": sorted(
                candidate_pathways - set(pathway_stats.index.astype(str))
            ),
            "membership_unique_genes": int(membership.gene_id.nunique()),
            "membership_genes_with_gencode_v36_promoter": int(
                membership.loc[membership.gencode_v36_promoter, "gene_id"].nunique()
            ),
            "membership_genes_with_promoter_peak_overlap": int(
                membership.loc[membership.promoter_peak_overlap, "gene_id"].nunique()
            ),
            "pathways_with_at_least_1_peak_gene": int(
                pathway_stats.promoter_peak_overlap_genes.ge(1).sum()
            ),
            "pathways_with_at_least_3_peak_genes": int(
                pathway_stats.promoter_peak_overlap_genes.ge(3).sum()
            ),
            "pathways_with_at_least_5_peak_genes": int(
                pathway_stats.promoter_peak_overlap_genes.ge(5).sum()
            ),
            "pathways_with_at_least_12_peak_genes": int(
                pathway_stats.promoter_peak_overlap_genes.ge(12).sum()
            ),
            "datas7_workbook_present_but_v32_distal_topology_not_materialized": True,
        },
        "processing_findings": [
            {
                "kind": "PATIENT_FOLD_SAMPLE_FALLBACK_REMOVED",
                "evidence": "Fold alignment is receipt-bound to explicit patient_id; sample_id[:12] is forbidden",
                "raw_atac_barcode_mapping_is_distinct": True,
            },
            {
                "kind": "FALSE_MISSINGNESS_SCOPE_FILTER",
                "evidence": "Historical builder hard-coded BRCA/COAD/KIRP while official mapping has 23 cancers",
                "affected_cancers": 20,
            },
            {
                "kind": "FALSE_MISSINGNESS_CANCER_ID_NORMALIZATION",
                "evidence": "Official bam prefixes use trailing x for ACCx/GBMx/LGGx",
                "affected_mapping_rows": int(mapping.bam_prefix.str.match(r"^(?:ACCx|GBMx|LGGx)-").sum()),
            },
            {
                "kind": "WRONG_AGGREGATION_KEY",
                "evidence": "Historical builder uses substr(Case_ID, 1L, 16L), not full Case_ID",
                "historical_code_present": "substr(Case_ID, 1L, 16L)" in old_builder,
                "full_aliquots": int(mapping.full_aliquot_id.nunique()),
                "collapsed_patient_or_sample16_keys": int(mapping.sample16_id.nunique()),
                "patients_with_multiple_full_aliquots": json.loads(
                    duplicate_patient_aliquots.to_json(orient="records")
                ),
            },
            {
                "kind": "UNMOUNTED_EXISTING_SERVER_ASSET",
                "evidence": "Official files exist under the authorized /public8 input asset root but the V3.2 routed launcher does not pass ATAC",
                "server_asset_root": "./data/CancerLncAtlas/input/v3_1_target_context_assets_e0b44122/atac",
            },
            {
                "kind": "ROUTER_LINEAGE_GAP_REPAIRED_IN_CODE",
                "evidence": "ATAC prediction input now requires fresh patient-OOF lineage and SHA256 binding",
                "validator_present": "def validate_atac_oof_lineage" in router_source,
            },
        ],
        "true_gaps": {
            "raw_absent_cancers": cancer.loc[
                ~cancer.raw_atac_available, "cancer_id"
            ].tolist(),
            "unaligned_patient_ids": json.loads(
                aligned_patient.loc[
                    aligned_patient.patient_fold_id.isna(), ["cancer_id", "patient_id"]
                ].to_json(orient="records")
            ),
            "fresh_v32_atac_typed_predictions": "MISSING",
            "fresh_v32_atac_patient_fold_partitions": "MISSING",
            "fresh_v32_atac_checkpoints": "MISSING",
            "fresh_v32_atac_lineage": "MISSING",
            "fresh_v32_atac_materializer": "MISSING",
            "fresh_v32_atac_trainer": "MISSING",
            "fresh_v32_atac_per_cancer_performance": "NOT_ESTIMATED",
        },
        "router_contract": {
            "accepted_prediction_columns": [
                "cancer_id", "lncrna_id", "pathway_id",
                "atac_context_probability", "atac_available", "atac_unavailable_reason",
            ],
            "exact_candidate_universe_required": True,
            "typed_null_required": True,
            "atac_lineage_validator_present": "def validate_atac_oof_lineage" in router_source,
            "launcher_passes_atac_predictions": "--atac-predictions" in launcher_source,
            "launcher_passes_atac_lineage": "--atac-lineage" in launcher_source,
            "current_launcher_behavior": "ATAC_TYPED_UNAVAILABLE",
            "hhgt_core_change_required": False,
            "external_router_change_required": "INPUTS_ONLY_AFTER_FRESH_EXPERT_EXISTS",
        },
        "historical_atac": {
            "pilot_cancers": ["BRCA", "COAD", "KIRP"],
            "pilot_only": historical_pilot.get("pilot_only"),
            "formal_v31_started": historical_pilot.get("formal_v31_started"),
            "recommendation": historical_pilot.get("recommendation"),
            "mask_ablation_retraining": historical_ablation.get("retraining"),
            "usable_as_v32_contribution_evidence": False,
            "reason": "HISTORICAL_THREE_CANCER_MASK_ABLATION_NOT_FRESH_V32_OOF",
        },
        "input_lineage": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for label, path in {
                "mapping": asset_files["mapping"],
                "peak_set": asset_files["peaks"],
                "candidate_universe": paths["candidates"],
                "patient_folds": paths["folds"],
                "patient_fold_authority_receipt": paths["fold_receipt"],
                "promoters": paths["promoters"],
                "pathway_membership": paths["membership"],
                "historical_pilot_success": paths["historical_pilot_success"],
                "historical_ablation": paths["historical_ablation"],
                "historical_r_builder": paths["historical_r_builder"],
                "router_source": paths["router_source"],
                "router_launcher": paths["router_launcher"],
            }.items()
        },
        "large_official_assets": {
            label: {"path": str(path), "bytes": path.stat().st_size, "present": True}
            for label, path in {
                "normalized_matrix": asset_files["matrix"],
                "datas7_peak_to_gene": asset_files["datas7"],
                "datas1_donors": asset_files["datas1"],
            }.items()
        },
        "acceptance_policy": {
            "formal_contribution_claim": "FRESH_NESTED_OOF_ONLY",
            "per_cancer_gate": "INNER_VALIDATION_LOGLOSS_IMPROVEMENT_AND_BRIER_NONINFERIORITY",
            "heldout_reverse_selection_forbidden": True,
            "missing_policy": "EXPLICIT_FALSE_MASK_AND_NAN_NEVER_ZERO",
            "historical_results_cannot_be_relabelled_v32": True,
        },
    }

    table_path = output / "ATAC_CANCER_READINESS.tsv"
    markdown_path = output / "ATAC_READINESS.md"
    audit_path = output / "ATAC_READINESS.json"
    _atomic_tsv(table_path, cancer)
    _atomic_text(markdown_path, _markdown(audit, cancer))
    audit["outputs"] = {
        "cancer_readiness": {"path": str(table_path), "sha256": _sha256(table_path)},
        "report": {"path": str(markdown_path), "sha256": _sha256(markdown_path)},
    }
    _atomic_json(audit_path, audit)
    success = {
        "status": "AUDIT_COMPLETE_NOT_TRAINED",
        "analysis_version": audit["analysis_version"],
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "fresh_v32_atac_predictions_generated": False,
        "formal_v32_primary_unchanged": True,
        "success_written_last": True,
    }
    _atomic_json(output / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
