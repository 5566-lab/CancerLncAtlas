#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.common import load_config, write_json, write_table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, target: Path, required: bool = True) -> None:
    if not source.exists():
        if required:
            raise FileNotFoundError(source)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize V3.0 clinical release")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = cfg["_results"]
    audit_path = result / "reports" / "V3_0_FINAL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("V3.0 clinical audit is not PASS")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release = result / "releases" / stamp / "final_release"
    release.mkdir(parents=True, exist_ok=False)
    v29 = cfg["_root"] / cfg["clinical"]["v29_result_root"]
    files = {
        v29 / "v2_9_downstream" / "v2_9_release" / "final_expert_fusion_table.parquet": release / "functional" / "v2_9_final_expert_fusion_table.parquet",
        v29 / "v2_9_downstream" / "lncrna_state_release" / "lncrna_state_final.parquet": release / "functional" / "v2_9_lncrna_state_final.parquet",
        v29 / "v2_9_downstream" / "lncrna_state_release" / "rnass_state_significance_report.parquet": release / "functional" / "v2_9_rnass_state_significance_report.parquet",
        result / "clinical_standardized" / "clinical_endpoint_availability.tsv": release / "clinical" / "clinical_endpoint_availability.tsv",
        result / "clinical_survival_release" / "patient_survival_oof_prediction.parquet": release / "clinical" / "patient_survival_oof_prediction.parquet",
        result / "clinical_survival_release" / "survival_oof_metrics.tsv": release / "clinical" / "survival_oof_metrics.tsv",
        result / "clinical_survival_release" / "survival_feature_importance.parquet": release / "clinical" / "survival_feature_importance.parquet",
        result / "clinical_association_release" / "clinical_association_fold_records.parquet": release / "clinical" / "clinical_association_fold_records.parquet",
        result / "clinical_association_release" / "clinical_association_summary.parquet": release / "clinical" / "clinical_association_summary.parquet",
        result / "clinical_expert_release" / "clinical_replication_oof_prediction.parquet": release / "clinical" / "clinical_replication_oof_prediction.parquet",
        result / "clinical_expert_release" / "clinical_endpoint_relevance.parquet": release / "clinical" / "clinical_endpoint_relevance.parquet",
        result / "clinical_expert_release" / "clinical_relevance_summary.parquet": release / "clinical" / "clinical_relevance_summary.parquet",
        result / "clinical_expert_release" / "v3_0_functional_clinical_table.parquet": release / "predictions" / "v3_0_functional_clinical_table.parquet",
        audit_path: release / "reports" / "V3_0_FINAL_AUDIT.json",
        result / "reports" / "V3_0_CLINICAL_AUDIT.tsv": release / "reports" / "V3_0_CLINICAL_AUDIT.tsv",
    }
    for source, target in files.items():
        copy(source, target)
    for path in sorted((result / "web_tables").glob("web_*.parquet")):
        copy(path, release / "web_tables" / path.name)
    copy(result / "web_tables" / "V3_0_CLINICAL_WEB_SUCCESS.json", release / "web_tables" / "V3_0_CLINICAL_WEB_SUCCESS.json")

    model_card = f"""# CancerLncAtlas CC-HHGT V3.0 Clinical Model Card

Generated: {datetime.now(timezone.utc).isoformat()}

## Architecture

V3.0 freezes the V2.9 functional discovery model and adds an independent clinical layer:

1. Multi-endpoint patient survival expert for OS, DSS, PFI, PFS, DFI and DFS using right-censoring-aware discrete hazards.
2. Cross-fitted adjusted Cox analyses for lncRNA, lncRNA-pathway interaction and lncRNA-state interaction candidates.
3. A clinical replication expert trained only on train-fold statistics to predict held-out clinical replication.
4. A separate translational priority score combining functional confidence and clinical relevance.

## Leakage boundary

- Survival endpoints never enter `discovery_ranking_probability`.
- Clinical preprocessing and time-bin selection are fitted on train patients only.
- Candidate clinical replication probabilities are OOF.
- Stage, age, sex, subtype, purity and treatment variables are covariates, not functional labels.

## Interpretation

`translational_priority_score` is a prioritization score, not a probability of survival benefit and not proof of causal pathway regulation.
"""
    (release / "V3_0_MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    rows = []
    for path in sorted(release.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(release)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(rows)
    write_table(manifest, release / "V3_0_RELEASE_MANIFEST.tsv")
    success = {
        "status": "COMPLETED",
        "version": "CC-HHGT_v3.0-clinical",
        "analysis_version": cfg["analysis_version"],
        "release_root": str(release),
        "manifest_files": int(len(manifest)),
        "v29_strict_models_retrained": False,
        "survival_endpoints_used_in_discovery": False,
        "required_endpoints": cfg["clinical"].get("required_endpoints", ["OS"]),
    }
    write_json(success, release / "SUCCESS.json")
    write_json(success, result / "V3_0_RELEASE_SUCCESS.json")
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
