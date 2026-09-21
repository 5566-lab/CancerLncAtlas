#!/usr/bin/env python3
"""Fail-closed audit for direct lncRNA--tumor-state models."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULT = Path(os.getenv("CC_HHGT_RESULT_ROOT", str(ROOT / "results")))
DATA = Path(os.getenv("CC_HHGT_STATE_DATA_ROOT", str(RESULT / "lncrna_state_data")))
RUN = Path(os.getenv("CC_HHGT_STATE_RUN_ROOT", str(RESULT / "lncrna_state_run")))
RELEASE = Path(os.getenv("CC_HHGT_STATE_RELEASE_ROOT", str(RESULT / "lncrna_state_release")))
REQUIRE_TELOMERASE = os.getenv("CC_HHGT_REQUIRE_TELOMERASE_STATE", "1") == "1"
REQUIRE_STEMNESS = os.getenv("CC_HHGT_REQUIRE_STEMNESS_STATE", "1") == "1"
EXPERTS = ["state_patient_probability", "state_rgcn_probability", "state_hgt_probability", "state_cc_hhgt_strict_probability"]


def main() -> int:
    checks = []
    def add(name, ok, detail, required=True):
        checks.append({"requirement": name, "status": "PASS" if ok else "FAIL", "required": required, "detail": detail})

    for name, path in [
        ("state_data", DATA / "SUCCESS.json"),
        ("state_training", RUN / "SUCCESS.json"),
        ("state_moe", RELEASE / "LNCRNA_STATE_MODEL_SUCCESS.json"),
    ]:
        add(name, path.exists(), str(path))
    final_path = RELEASE / "lncrna_state_final.parquet"
    add("state_final_table", final_path.exists(), str(final_path))
    if final_path.exists():
        final = pd.read_parquet(final_path)
        states = sorted(final["state_id"].astype(str).unique())
        lower = "|".join(states).lower()
        add("telomerase_state_present", (not REQUIRE_TELOMERASE) or ("extend" in lower or "telomerase" in lower), states, REQUIRE_TELOMERASE)
        add("stemness_state_present", (not REQUIRE_STEMNESS) or any(token in lower for token in ["rnass", "dnass", "stemness", "ereg.expss", "ereg_expss"]), states, REQUIRE_STEMNESS)
        add("replication_classes", set(final["replication_class"]).issubset({"robust_core", "replicated", "exploratory"}), final["replication_class"].value_counts().to_dict())
        add("state_probability_range", final["lncrna_state_probability"].between(0, 1).all(), [float(final["lncrna_state_probability"].min()), float(final["lncrna_state_probability"].max())])

    fold_path = RELEASE / "lncrna_state_fold_prediction.parquet"
    add("state_fold_table", fold_path.exists(), str(fold_path))
    if fold_path.exists():
        fold = pd.read_parquet(fold_path)
        weight_columns = [f"state_weight_{expert}" for expert in EXPERTS]
        present = [column for column in weight_columns if column in fold]
        valid = fold["lncrna_state_probability"].notna()
        add("state_moe_weight_normalization", len(present) == len(weight_columns) and np.allclose(fold.loc[valid, present].sum(axis=1), 1, atol=1e-4), present)
        expert_coverage = {expert: int(fold[expert].notna().sum()) if expert in fold else 0 for expert in EXPERTS}
        add("state_patient_and_three_graph_experts_have_coverage", all(value > 0 for value in expert_coverage.values()), expert_coverage)
        violations = {}
        for expert in EXPERTS[1:]:
            probability = expert
            weight = f"state_weight_{expert}"
            if probability in fold and weight in fold:
                bad = fold.loc[fold[probability].isna(), weight].fillna(0).abs().gt(1e-6).sum()
                violations[expert] = int(bad)
        add("unavailable_state_graph_experts_masked", all(value == 0 for value in violations.values()), violations)

    provenance = DATA / "fold_provenance.parquet"
    add("train_only_state_feature_contract", provenance.exists(), str(provenance))
    if provenance.exists():
        prov = pd.read_parquet(provenance)
        ok = prov["candidate_selection_scope"].eq("train_patients_only_per_fold").all() and prov["residualizer_fit_scope"].eq("train_patients_only").all()
        add("train_only_state_feature_contract_values", ok, prov[["candidate_selection_scope", "residualizer_fit_scope"]].drop_duplicates().to_dict("records"))
        direction_scope_ok = (
            "direction_supervision_scope" in prov
            and prov["direction_supervision_scope"].eq("positive_membership_pairs_only").all()
        )
        add(
            "patient_state_direction_positive_only",
            direction_scope_ok,
            prov.get("direction_supervision_scope", pd.Series(dtype=str)).drop_duplicates().tolist(),
        )

    direction_violations = 0
    for path in sorted(DATA.glob("cancer_id=*/part-0.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "train_membership_label", "train_direction_label",
                "validation_membership_label", "validation_direction_label",
                "test_membership_label", "test_direction_label",
            ],
        )
        for split in ("train", "validation", "test"):
            direction_violations += int(
                (
                    frame[f"{split}_membership_label"].ne(1.0)
                    & frame[f"{split}_direction_label"].notna()
                ).sum()
            )
    add("patient_state_unlabeled_direction_count_zero", direction_violations == 0, direction_violations)

    model_root = Path(os.getenv("CC_HHGT_STATE_MODEL_ROOT", str(ROOT / "models" / "lncrna_state_experts")))
    checkpoints = sorted(model_root.glob("*/*/seed_*/best.pt"))
    add("state_model_checkpoints", len(checkpoints) > 0, len(checkpoints))
    graph_retrained = False
    direction_scope_failures = 0
    for path in checkpoints[:20]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        graph_retrained |= bool(payload.get("graph_encoder_retrained", True))
        direction_scope_failures += int(
            payload.get("direction_supervision_scope") != "positive_membership_pairs_only"
            or int(payload.get("state_unlabeled_direction_used_in_loss", -1)) != 0
        )
    add("strict_graph_encoder_not_retrained", not graph_retrained, graph_retrained)
    add("patient_checkpoint_direction_scope_valid", direction_scope_failures == 0, direction_scope_failures)

    required_failures = [row for row in checks if row["required"] and row["status"] != "PASS"]
    status = "PASS" if not required_failures else "FAIL"
    payload = {"status": status, "generated_at": datetime.now().isoformat(), "required_failures": required_failures, "checks": checks}
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / "LNCRNA_STATE_AUDIT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(RELEASE / "LNCRNA_STATE_AUDIT.tsv", sep="\t", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("lncRNA-state audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
