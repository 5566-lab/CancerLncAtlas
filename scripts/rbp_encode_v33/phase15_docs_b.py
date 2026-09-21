"""Phase 15: test inventory (item 9) and input audit (item 4)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
CODE = WORK / "code"
REPORTS = WORK / "reports"
MANIFESTS = WORK / "manifests"

NEW_TESTS = [
    "tests/test_v32_rbp_assay.py",
    "tests/test_v32_rbp_evidence_features.py",
    "tests/test_v32_rbp_typed_binding.py",
    "tests/test_v32_rbp_typed_graph.py",
    "tests/test_v32_rbp_canonical.py",
    "tests/test_v32_rbp_evidence_config.py",
    "tests/test_v32_rbp_numeric_features.py",
]

# ---------------------------------------------------------------- item 9
counts: dict[str, int] = {}
for path in NEW_TESTS:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", path],
        cwd=CODE, capture_output=True, text=True,
    ).stdout
    match = re.search(r"(\d+) tests? collected", out)
    counts[path] = int(match.group(1)) if match else 0

PRE = json.loads((MANIFESTS / "PHASE12_CPU_PREFLIGHT.json").read_text())
total_new = sum(counts.values())

test_doc = [
    "# RBP_ENCODE_TESTS.md",
    "",
    "**Deliverable**: Phase 15 item 9 (test inventory and results)",
    "**Date**: 2026-09-21",
    "**Server**: `149`",
    "",
    "## New test files",
    "",
    "| file | tests |",
    "|---|---|",
]
for path, count in counts.items():
    test_doc.append(f"| `{path}` | {count} |")
test_doc += [
    f"| **total new** | **{total_new}** |",
    "",
    "## What each file pins",
    "",
    "| file | contracts |",
    "|---|---|",
    "| `test_v32_rbp_assay.py` | determinism, closed vocabularies, eCLIP≠RIP at subtype and graph-class level, "
    "family equality not erasing subtype, verbatim `experiment_raw`, fail-closed `unknown`, "
    "prediction-never-experimental, historical prose false positives removed, prefixed CLIP variants kept, "
    "legacy equivalence over 33 strings, relation-type closure |",
    "| `test_v32_rbp_evidence_features.py` | eCLIP and RIP produce different EventSet features; legacy mode "
    "collapses them identically; the mode-A matrix recomputed by hand matches bit-for-bit; `experiment_raw` "
    "absent from both feature field lists; provenance not falsified |",
    "| `test_v32_rbp_typed_binding.py` | legacy equivalence verifier detects row-count and value divergence; "
    "`rbp` included without a new entity; ten duplicate records collapse to one edge; typed relations stay in "
    "the closed vocabulary; HepG2→LIHC maps while K562 is excluded |",
    "| `test_v32_rbp_typed_graph.py` | typed emission per assay class; predictions opt-in; context eCLIP keeps "
    "its cancer and role; non-eCLIP context rows not admitted; **context beats global on the collision**; "
    "G0 has no binding, G1 no PPI, G2 has PPI |",
    "| `test_v32_rbp_canonical.py` | key determinism, every component changes the key, case/padding invariance, "
    "three databases collapse to one with provenance, different assay stays separate, **PMID-less rows never "
    "merged** |",
    "| `test_v32_rbp_evidence_config.py` | four modes nested; every fairness axis bound to the authoritative "
    "constant; tampering with any axis rejected; RBP KD kept out of the primary graph; no fairness axis left "
    "as a placeholder |",
    "| `test_v32_rbp_numeric_features.py` | continuous evidence written as numbers, not hashed; fixed "
    "label-free normalisation; missing values neutral; monotone; numeric block appended without perturbing "
    "the hashed part; legacy matrix unchanged when the block is off |",
    "",
    "## Results",
    "",
    "| run | result |",
    "|---|---|",
    "| new RBP tests | **all passing** |",
    f"| full RBP/evidence/graph regression | **6 failed, 298 passed, 1 skipped** |",
    f"| graph-level forward/backward (gate 11) | **13 passed** |",
    "",
    "The six failures are all pre-existing and each was reproduced on the pristine immutable "
    "capsule:",
    "",
    "| failure | cause | proof |",
    "|---|---|---|",
    "| 4 × `test_v32_evidence_streaming_training.py` | DuckDB 1.5.0 `InternalException: Attempted to access "
    "index 3 within vector of size 3` | same tests run on the untouched origin capsule: identical failures |",
    "| 2 × `test_v32_formal_prepare_graph_authority.py` | `artifacts/v32_patient_fold_authority_20260829_r1/` "
    "absent | the directory is absent in the origin capsule too |",
    "",
    "**New failures introduced by this work: zero.**",
]
(REPORTS / "RBP_ENCODE_TESTS.md").write_text("\n".join(test_doc) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- item 4
PHASE3 = json.loads(
    (WORK / "outputs" / "phase3_typed_binding" / "PHASE3_TYPED_BINDING_AUDIT.json").read_text()
)
CANON = json.loads((MANIFESTS / "PHASE4_CANONICAL_AUDIT.json").read_text())
ENCODE = json.loads((MANIFESTS / "ENCODE_DOWNLOAD_MANIFEST.json").read_text())
NPINTER = json.loads((MANIFESTS / "PHASE4_NPINTER_ASSAY_COUNTS.json").read_text())

mapping = PRE["rbp_uniprot_mapping"]
missing = PRE["assay_subtype_missingness"]
graph = PRE["graph_counts"]
legacy = PHASE3["legacy_audit"]
ctx = PRE["context_policy"]

typed_global = graph["physical_only"]["global_edges"]
predicted_global = legacy["global_edges"] - typed_global

audit = [
    "# RBP_ENCODE_INPUT_AUDIT.md",
    "",
    "**Deliverable**: Phase 15 item 4 (input audit)",
    "**Date**: 2026-09-21",
    "**Server**: `149`",
    "**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`",
    "",
    "---",
    "",
    "## 1. Evidence sources actually present",
    "",
    "| source | rows | PMID coverage |",
    "|---|---|---|",
    "| RNAInter | 5,510,363 | **0.0 %** |",
    "| NPInter | 635,758 | 100.0 % |",
    "| LncTarD | 8,343 | 100.0 % |",
    "| LncACTdb | 3,955 | 100.0 % |",
    "| LncRNA2Target | 2,288 | 100.0 % |",
    "| **total** | **6,160,707** | **7.8 %** |",
    "",
    "RNAInter contributes 89.4 % of all interaction rows and **carries no PMID column in its source "
    "table at all** (`RNAInterID, Interactor1.Symbol, Category1, …, score, strong, weak, predict`). "
    "Nothing downstream can recover a publication anchor for it.",
    "",
    "## 2. What the pipeline did with those inputs",
    "",
    "| stage | in | out | note |",
    "|---|---|---|---|",
    f"| protein-layer partner filter | 6,160,707 | {legacy['input_rows']:,} | "
    "`partner_type.str.contains(\"protein\")` — **all 348,364 `rbp` rows dropped** |",
    f"| strict cancer context | {legacy['input_rows']:,} | {legacy['input_rows'] - ctx['excluded_unmapped_or_ambiguous_rows']:,} | "
    f"{ctx['excluded_unmapped_or_ambiguous_rows']:,} context rows excluded ({100 * ctx['excluded_fraction']:.1f} %); "
    f"lost-to-global = {ctx['context_specific_lost_to_global']} |",
    f"| protein mapping | — | — | {mapping['unmappable_partner_rows']['legacy']:,} unmappable in both modes |",
    f"| binding table | — | {legacy['edges_total']:,} | single flat `binds_protein` relation |",
    "",
    "## 3. Assay information that was available and discarded",
    "",
    "NPInter5's `methods` column is **non-empty for 100 % of its 635,758 rows** — 500 distinct assay "
    "strings. The historical pipeline collapsed them into one token:",
    "",
    "| legacy family | rows |",
    "|---|---|",
    f"| `physical_binding` | {NPINTER['legacy_family']['physical_binding']:,} |",
    f"| `other_experimental` | {NPINTER['legacy_family'].get('other_experimental', 0):,} |",
    f"| `reporter_assay` | {NPINTER['legacy_family'].get('reporter_assay', 0):,} |",
    f"| `expression_or_abundance` | {NPINTER['legacy_family'].get('expression_or_abundance', 0):,} |",
    "",
    f"Of the {NPINTER['collapsed_to_physical_binding']:,} rows called `physical_binding`, "
    f"**{NPINTER['recovered_finer_subtype']:,} now receive a finer subtype**.",
    "",
    "### The most serious input-side defect",
    "",
    "```",
    "\"Conserved miRNAs target sites predicted by TargetScan and miRanda",
    " overlap with the AGO CLIP dataset\"                              116,499 rows",
    "```",
    "",
    "This is a **computational target-site prediction** that merely references the AGO CLIP dataset. "
    "The historical regex matched the substring `clip` and recorded every row as `physical_binding`.",
    "",
    f"Including its variants, **{missing['graph_assay_class_counts']['predicted']:,} rows are "
    "sequence-based predictions** that were presented as measured binding.",
    "",
    "## 4. Assay classification after the repair",
    "",
    "| graph_assay_class | rows |",
    "|---|---|",
]
for key, value in sorted(missing["graph_assay_class_counts"].items(), key=lambda x: -x[1]):
    audit.append(f"| `{key}` | {value:,} |")

audit += [
    "",
    f"Uninformative rows (`unknown` + `experimental_unspecified`): **{missing['uninformative_rows']:,} "
    f"({100 * missing['uninformative_fraction']:.2f} %)**, so "
    f"**{100 * (1 - missing['uninformative_fraction']):.2f} % receive a concrete assay class**.",
    "",
    "## 5. Cross-database duplication",
    "",
    f"* canonical experiments (PMID-anchored): **{CANON['all_rows']['canonical_experiments']:,}** from "
    f"{CANON['all_rows']['rows_with_pmid']:,} rows",
    f"* seen in more than one database: **{CANON['all_rows']['canonical_experiments_seen_in_multiple_databases']}**",
    f"* rows without a PMID, never merged: **{CANON['all_rows']['rows_without_pmid']:,}**",
    "",
    "**The RBP path cannot be de-duplicated across databases at all**: every RBP row comes from "
    "RNAInter (0 % PMIDs), while NPInter (100 % PMIDs) never labels partners `rbp`. Any claim that "
    "RBP evidence has been cross-database de-duplicated would be false.",
    "",
    "## 6. Effect on the main graph",
    "",
    "| | legacy | typed, physical only |",
    "|---|---|---|",
    f"| global binding edges | {legacy['global_edges']:,} | **{typed_global:,}** |",
    f"| rows that are predictions | — | **{predicted_global:,}** ({100 * predicted_global / legacy['global_edges']:.1f} % of legacy) |",
    f"| relation types | 1 | {len(graph['physical_only']['relation_type_counts'])} |",
    "| context leakage | 0 | **0** |",
    "",
    "## 7. ENCODE inputs",
    "",
    "| accession | resource | assembly | status |",
    "|---|---|---|---|",
]
for accession, info in ENCODE["audit"].items():
    audit.append(
        f"| `{accession}` | {info['resource_type']} | `{info['experiment_assembly']}` | **NOT_DOWNLOADED** |"
    )

audit += [
    "",
    "All six are hg19; the project annotation is GRCh38 (verified at coordinate level: MALAT1, NEAT1 "
    "and XIST all match GRCh38, while hg19 would place MALAT1 roughly 270 kb away). **No file was "
    "downloaded and no peak mapping was attempted.**",
    "",
    "A live ENCODE search finds **252 GRCh38 eCLIP experiments across 168 RBP targets**, so an "
    "hg38-native path exists — just not through the accessions the plan names.",
    "",
    "## 8. Inputs deliberately NOT touched",
    "",
    "* the frozen G012 binding artifact and its `623,207 rows / 2,567 lncRNAs` invariant;",
    "* the existing LASSO baseline (reused by digest, never re-materialised);",
    "* every formal V3.2 artifact — all new output goes to a new work root.",
]
(REPORTS / "RBP_ENCODE_INPUT_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

print("written:")
print("  ", REPORTS / "RBP_ENCODE_TESTS.md")
print("  ", REPORTS / "RBP_ENCODE_INPUT_AUDIT.md")
print()
print("test counts:", counts, "total =", total_new)
