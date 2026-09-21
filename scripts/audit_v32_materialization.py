"""Audit V3.2 exact-pathway Gene Set and subtype materialization outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-root", required=True)
    args = parser.parse_args()
    root = Path(args.materialization_root)
    success_path = root / "MATERIALIZATION_SUCCESS.json"
    if not success_path.is_file():
        raise RuntimeError("materialization success marker is missing")

    master_paths = sorted((root / "geneset_master_parts").glob("cancer_id=*.parquet"))
    member_paths = sorted((root / "geneset_member_parts").glob("cancer_id=*.parquet"))
    if len(master_paths) != 33 or len(member_paths) != 33:
        raise RuntimeError("expected exactly 33 cancer master/member parts")
    member_by_cancer = {path.stem.split("=", 1)[1]: path for path in member_paths}
    coverage_rows: list[dict[str, object]] = []
    all_master: list[pd.DataFrame] = []
    violations: list[str] = []
    for master_path in master_paths:
        cancer = master_path.stem.split("=", 1)[1]
        master = pd.read_parquet(master_path)
        members = pd.read_parquet(member_by_cancer[cancer])
        all_master.append(master)
        if not master.empty:
            if set(master.pathway_target_level.astype(str)) != {"exact_pathway"}:
                violations.append(f"{cancer}: non-exact pathway target")
            if master.ranking_uses_regulatory_evidence.astype(bool).any():
                violations.append(f"{cancer}: regulatory evidence changed ranking")
            if master.member_count.lt(10).any() or master.member_count.gt(200).any():
                violations.append(f"{cancer}: member count outside 10..200")
            if set(master.direction.astype(str)) - {"positive", "negative"}:
                violations.append(f"{cancer}: invalid association direction")
        if int(master.member_count.sum()) != len(members):
            violations.append(f"{cancer}: master/member row count disagreement")
        coverage_rows.append(
            {
                "cancer_id": cancer,
                "publishable_genesets": int(len(master)),
                "exact_pathways": int(master.pathway_id.nunique()) if not master.empty else 0,
                "member_rows": int(len(members)),
                "positive_genesets": int(master.direction.astype(str).eq("positive").sum()) if not master.empty else 0,
                "negative_genesets": int(master.direction.astype(str).eq("negative").sum()) if not master.empty else 0,
                "coverage_status": "AVAILABLE" if len(master) else "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS",
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values("cancer_id", kind="stable")
    coverage.to_csv(root / "CANCER_GENESET_COVERAGE.tsv", sep="\t", index=False)
    master = pd.concat(all_master, ignore_index=True)

    conservation = pd.read_parquet(root / "PATHWAY_CONSERVATION.parquet")
    context = pd.read_parquet(root / "PATHWAY_CONTEXT_SUBTYPE.parquet")
    program = pd.read_parquet(root / "CANCER_PROGRAM_SUBTYPE.parquet")
    status = {
        "status": "PASS" if not violations else "FAIL",
        "cancers": 33,
        "cancers_with_genesets": int(coverage.publishable_genesets.gt(0).sum()),
        "cancers_without_genesets": coverage.loc[
            coverage.publishable_genesets.eq(0), "cancer_id"
        ].astype(str).tolist(),
        "genesets": int(len(master)),
        "member_rows": int(coverage.member_rows.sum()),
        "exact_pathways_with_genesets": int(master.pathway_id.nunique()),
        "pathway_context_status_counts": {
            str(key): int(value)
            for key, value in context.classification_status.value_counts(dropna=False).items()
        },
        "pathway_conservation_status_counts": {
            str(key): int(value)
            for key, value in conservation.classification_status.value_counts(dropna=False).items()
        },
        "cancer_program_status_counts": {
            str(key): int(value)
            for key, value in program.classification_status.value_counts(dropna=False).items()
        },
        "family_used_as_target": False,
        "regulatory_evidence_used_for_ranking": False,
        "violations": violations,
    }
    (root / "MATERIALIZATION_COVERAGE_AUDIT.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, sort_keys=True))
    if violations:
        raise RuntimeError("materialization coverage audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
