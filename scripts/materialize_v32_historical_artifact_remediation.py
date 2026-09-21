from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.historical_artifact_remediation import (  # noqa: E402
    RemediationInputs,
    materialize_historical_artifact_remediation,
)


def default_inputs(repo_root: Path) -> RemediationInputs:
    artifacts = repo_root / "artifacts"
    return RemediationInputs(
        clinical_risk=artifacts
        / "v32_full_multitask"
        / "clinical"
        / "clinical_patient_risk.parquet",
        clinical_lineage=artifacts / "v32_full_multitask" / "clinical" / "MODULE_LINEAGE.json",
        clinical_entity=artifacts
        / "v32_full_multitask"
        / "clinical_entity"
        / "entity_clinical_associations.parquet",
        clinical_entity_lineage=artifacts
        / "v32_full_multitask"
        / "clinical_entity"
        / "ENTITY_CLINICAL_LINEAGE.json",
        genomic_predictions=artifacts
        / "v32_full_multitask"
        / "genomic_fresh_rerun1"
        / "mutation_cnv_typed_predictions.parquet",
        genomic_lineage=artifacts
        / "v32_full_multitask"
        / "genomic_fresh_rerun1"
        / "LINEAGE.json",
        mixed_asset_root=artifacts
        / "v32_staging"
        / "final_capacity_audit_20260825"
        / "mixed_query_assets_exact2135",
        mixed_query_code=repo_root / "cc_hhgt" / "v32" / "mixed_query.py",
        single_cell_root=artifacts / "v32_single_cell_fresh_20260826_r1",
        single_cell_fusion_binding=artifacts
        / "v32_single_cell_fusion_adapter_20260826_r1"
        / "SINGLE_CELL_FUSION_BINDING.json",
        hnsc_ucell_binding=artifacts
        / "single_cell_cell_level_hnsc_ucell_query_binding_20260826_r1"
        / "HNSC_UCELL_QUERY_BINDING.json",
        evidence_prediction=artifacts
        / "v32_evidence_fusion_adapter_20260826_r1"
        / "evidence_exact_fusion_expert.parquet",
        evidence_fusion_binding=artifacts
        / "v32_evidence_fusion_adapter_20260826_r1"
        / "EVIDENCE_FUSION_BINDING.json",
        evidence_model_binding=artifacts
        / "v32_evidence_output_binding_20260826_r2_local"
        / "EVIDENCE_OUTPUT_BINDING.json",
        evidence_direction_binding=artifacts
        / "v32_evidence_direction_probabilities_20260826_r1_fixed_seed_reinference"
        / "EVIDENCE_DIRECTION_PROBABILITY_BINDING.json",
        evidence_direction_audit_binding=artifacts
        / "v32_evidence_direction_probabilities_20260826_r3_independent_audit"
        / "INDEPENDENT_AUDIT_BINDING.json",
        multimodal_scores=artifacts
        / "v32_multimodal_fusion_formal_20260826_r2_transparent"
        / "multimodal_secondary_scores.parquet",
        multimodal_binding=artifacts
        / "v32_multimodal_fusion_formal_20260826_r2_transparent"
        / "MULTIMODAL_FUSION_BINDING.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--non-formal", action="store_true")
    args = parser.parse_args()
    result = materialize_historical_artifact_remediation(
        inputs=default_inputs(args.repo_root.resolve()),
        output_root=args.output_root,
        formal=not args.non_formal,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
