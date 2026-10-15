#!/usr/bin/env python3
"""Build observed rule evidence from native sources on 149; no training.

CURRENT_RULE_EVIDENCE_INPUT_V1: hash-bound candidates and exact_membership;
sources of kind native_exact_events (events/lineage), encode_functional
(audit/table) or rbns_sequence (audit/table). All paths may be relative.
"""
import argparse
import json
from pathlib import Path
import socket

import pandas as pd

from cc_hhgt.v32.current_core_export import sha256, verified_path
from cc_hhgt.v32.evidence_rule_sources import load_rule_sources, read_table
from cc_hhgt.v32.evidence_rule_summary import (KEYS, EVENT_COLUMNS, assign_study_components, attach_candidates,
    candidate_aliases, summarize_events, validate_events, stable_id, update_source_matches, complete_source_coverage)
from cc_hhgt.v32.evidence_rule_audit import sql_summary, assert_summary_equal


def prepare_sources(manifest_path):
    manifest_path = Path(manifest_path).resolve(); root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "CURRENT_RULE_EVIDENCE_INPUT_V1": raise ValueError("Unknown rule evidence input manifest")
    candidates = read_table(verified_path(manifest["candidates"], root))[KEYS]
    candidate_aliases(candidates)
    members = read_table(verified_path(manifest["exact_membership"], root))
    events, coverage = load_rule_sources(manifest["sources"], root, members)
    # Pandas/Arrow must retain numeric types even when all reviews are unknown.
    events["review_value"] = pd.to_numeric(events.review_value, errors="raise").astype(float)
    events["raw_occurrences"] = pd.to_numeric(events.raw_occurrences, errors="raise").astype("int64")
    validate_events(events)
    events = assign_study_components(events)
    return manifest, candidates, events, coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149": raise RuntimeError("Rule evidence materialization must execute on host 149")
    input_sha = sha256(args.manifest)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False,
        "source_manifest_sha256": input_sha}, indent=2))
    manifest, candidates, source_events, coverage = prepare_sources(args.manifest)
    def save(frame, name):
        path = output/name; frame.to_parquet(path, index=False, compression="zstd")
        return {"path": str(path), "sha256": sha256(path), "rows": len(frame)}
    source_record = save(source_events, "SOURCE_EVENTS.parquet")
    summaries = []; event_records = []; context_records = []; matches = {}
    for index, (cancer, universe) in enumerate(candidates.groupby("cancer_id", sort=True)):
        ledger = attach_candidates(source_events, universe)
        update_source_matches(matches, ledger)
        summary, contexts = summarize_events(ledger, universe)
        assert_summary_equal(summary, sql_summary(ledger, universe))
        event_records.append(save(ledger, f"EVENTS_{index:03d}.parquet"))
        if not contexts.empty: context_records.append(save(contexts, f"CONTEXTS_{index:03d}.parquet"))
        summaries.append(summary)
        print(json.dumps({"cancer_id": cancer, "candidates": len(universe), "evidence_records": len(ledger), "training_started": False}), flush=True)
    summary_record = save(pd.concat(summaries, ignore_index=True), "RULE_SUMMARY.parquet")
    coverage_record = save(complete_source_coverage(coverage, matches), "SOURCE_COVERAGE.parquet")
    if sha256(args.manifest) != input_sha: raise ValueError("Rule input manifest changed during materialization")
    from cc_hhgt.v32 import evidence_rule_sources, evidence_rule_summary, evidence_rule_audit
    implementation = {str(Path(p).resolve()): sha256(Path(p)) for p in
        (evidence_rule_sources.__file__, evidence_rule_summary.__file__, evidence_rule_audit.__file__, __file__)}
    run_id = stable_id("RULE-EVIDENCE:", [input_sha, implementation])
    binding = {"format": "CURRENT_RULE_EVIDENCE_RELEASE_V1", "status": "PASS", "run_id": run_id,
        "analysis_version": "CancerLncAtlas_V3.2_CURRENT_RULE_EVIDENCE", "target_host": "149",
        "input_manifest": {"path": str(args.manifest.resolve()), "sha256": input_sha},
        "implementation_sha256": implementation, "candidate_rows": len(candidates),
        "outputs": {"summary": summary_record, "source_events": source_record, "source_coverage": coverage_record},
        "event_parts": event_records, "context_parts": context_records,
        "training_started": False, "neural_score_available": False, "changes_primary_ranking": False,
        "independent_external_validation": False, "review_fraction_is_causal_probability": False,
        "study_count_semantics": "CONNECTED_PROVENANCE_GROUPS_NOT_PROOF_OF_STATISTICALLY_INDEPENDENT_REPLICATION",
        "audit_scope": "SOURCE_NORMALIZATION_BOUND; SUMMARY_INDEPENDENTLY_RECOMPUTED_BY_SQL"}
    binding_path = output/"RULE_EVIDENCE_BINDING.json"
    binding_path.write_text(json.dumps(binding, indent=2)+"\n")
    from cc_hhgt.v32.evidence_rule_release import audit_rule_release
    audit = audit_rule_release(binding_path, expected_sha=sha256(binding_path))
    audit_path = output/"INDEPENDENT_AUDIT.json"; audit_path.write_text(json.dumps(audit, indent=2)+"\n")
    (output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "training_started": False,
        "binding_sha256": sha256(binding_path), "independent_audit_sha256": sha256(audit_path)}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
