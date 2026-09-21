"""Fail-closed audit for cancer-native candidates and masked MoE outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(os.getenv("CC_HHGT_RELEASE_DIR", str(ROOT / "results" / "v2_8_cancer_native_moe_release")))
ADAPTER = ROOT / "results" / "cancer_adapter"
REPORT = RELEASE / "CANCER_NATIVE_MOE_AUDIT.json"


def main() -> int:
    final_path = RELEASE / "final_three_probability_table.parquet"
    rescued_path = RELEASE / "cancer_specific_rescued_candidates.parquet"
    oof_path = RELEASE / "cancer_specific_oof_prediction.parquet"
    gate_path = RELEASE / "moe_gate" / "confidence.pt"
    required = [final_path, rescued_path, oof_path, gate_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"required outputs are missing: {missing}")

    final = pd.read_parquet(final_path)
    oof = pd.read_parquet(oof_path)
    rescued = pd.read_parquet(rescued_path)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    check(
        "patient_native_output_present",
        "cancer_native_probability" in final,
        "cancer_native_probability" in final,
        True,
    )
    check(
        "rescued_candidates_exist",
        len(rescued) > 0 or int(final["strict_available"].fillna(0).eq(0).sum()) == 0,
        int(len(rescued)),
        "> 0 (or zero when strict covers every candidate)",
    )
    unavailable = final["strict_available"].fillna(0).eq(0)
    check(
        "strict_unavailable_weight_zero",
        bool(final.loc[unavailable, "expert_weight_strict"].fillna(0).abs().le(1e-6).all()),
        int((final.loc[unavailable, "expert_weight_strict"].fillna(0).abs() > 1e-6).sum()),
        0,
    )
    patient_unavailable = oof["strict_available"].fillna(0).eq(0)
    check(
        "patient_internal_joint_weight_zero",
        bool(oof.loc[patient_unavailable, "patient_gate_joint_weight"].fillna(0).abs().le(1e-6).all()),
        int((oof.loc[patient_unavailable, "patient_gate_joint_weight"].fillna(0).abs() > 1e-6).sum()),
        0,
    )
    full_weight = final[
        ["expert_weight_strict", "expert_weight_cancer", "expert_weight_evidence"]
    ].sum(axis=1)
    check(
        "confidence_gate_weight_sum",
        bool(np.allclose(full_weight, 1.0, atol=1e-4)),
        float((full_weight - 1).abs().max()),
        "<= 1e-4",
    )
    discovery_weight = final[
        ["expert_weight_strict_discovery", "expert_weight_cancer_discovery"]
    ].sum(axis=1)
    discovery_available = final[
        ["cross_cancer_probability", "cancer_native_probability"]
    ].notna().any(axis=1)
    check(
        "discovery_gate_weight_sum_when_available",
        bool(
            np.allclose(
                discovery_weight[discovery_available], 1.0, atol=1e-4
            )
        ),
        float((discovery_weight[discovery_available] - 1).abs().max()),
        "<= 1e-4",
    )
    check(
        "no_simple_mean_scope",
        final["prediction_scope"].astype(str).str.contains("masked_moe").all(),
        sorted(final["prediction_scope"].astype(str).unique().tolist()),
        "masked_moe",
    )
    check(
        "candidate_union_sources_present",
        {"candidate_from_strict", "candidate_from_patient", "candidate_from_evidence"}.issubset(final.columns),
        sorted(
            set(final.columns)
            & {"candidate_from_strict", "candidate_from_patient", "candidate_from_evidence"}
        ),
        ["candidate_from_evidence", "candidate_from_patient", "candidate_from_strict"],
    )

    failed = [item for item in checks if item["status"] == "FAIL"]
    payload = {
        "completed_at": datetime.now().isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "n_final_pairs": int(len(final)),
        "n_rescued_pairs": int(len(rescued)),
        "checks": checks,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failed:
        raise RuntimeError(f"cancer-native MoE audit failed: {[x['check'] for x in failed]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
