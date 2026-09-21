"""Phase 15: emit the remaining document deliverables from real state.

  4  RBP_ENCODE_INPUT_AUDIT.md
  7  modified-file list + SHA256
  8  old/new graph schema diff
  9  test inventory and results
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
CODE = WORK / "code"
sys.path.insert(0, str(CODE))

from cc_hhgt.v32.formal_graph import (  # noqa: E402
    CONTEXT_ECLIP_ROLE,
    FORMAL_RELATION_SCHEMA,
    G1_ROLES,
    GLOBAL_BINDING_ROLE,
)
from cc_hhgt.v32.rbp_assay import ASSAY_SUBTYPES, GRAPH_ASSAY_CLASSES  # noqa: E402

REPORTS = WORK / "reports"
MANIFESTS = WORK / "manifests"

NEW_MODULES = [
    "cc_hhgt/v32/rbp_assay.py",
    "cc_hhgt/v32/rbp_typed_binding.py",
    "cc_hhgt/v32/rbp_canonical.py",
    "cc_hhgt/v32/rbp_evidence_config.py",
]
MODIFIED_FILES = [
    "cc_hhgt/v32/evidence_training.py",
    "cc_hhgt/v32/evidence_streaming_training.py",
    "cc_hhgt/v32/formal_graph.py",
]
NEW_TESTS = [
    "tests/test_v32_rbp_assay.py",
    "tests/test_v32_rbp_evidence_features.py",
    "tests/test_v32_rbp_typed_binding.py",
    "tests/test_v32_rbp_typed_graph.py",
    "tests/test_v32_rbp_canonical.py",
    "tests/test_v32_rbp_evidence_config.py",
]

ORIGIN = Path(
    "${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/code"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_sha(path: str) -> tuple[str, str, str]:
    work = CODE / path
    origin = ORIGIN / path
    return (
        sha256(work) if work.exists() else "",
        sha256(origin) if origin.exists() else "(new file)",
        "NEW" if not origin.exists() else ("MODIFIED" if sha256(work) != sha256(origin) else "same"),
    )


# --------------------------------------------------------------- item 7
rows = []
for path in NEW_MODULES + NEW_TESTS:
    work_hash, origin_hash, status = rel_sha(path)
    rows.append((status, path, work_hash, origin_hash))
for path in MODIFIED_FILES:
    work_hash, origin_hash, status = rel_sha(path)
    rows.append((status, path, work_hash, origin_hash))

lines = [
    "# RBP_ENCODE_CHANGED_FILES.md",
    "",
    "**Deliverable**: Phase 15 item 7 (modified-file list + SHA256)",
    "**Date**: 2026-09-21",
    "",
    "Origin baseline capsule (never modified):",
    "`${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/code`",
    "",
    "| status | file | work-root SHA256 | origin SHA256 |",
    "|---|---|---|---|",
]
for status, path, work_hash, origin_hash in rows:
    lines.append(f"| {status} | `{path}` | `{work_hash[:32]}…` | `{origin_hash[:32] if origin_hash != '(new file)' else '(new file)'}…` |")

lines += [
    "",
    "## Files created outside the origin capsule",
    "",
    "| file | purpose |",
    "|---|---|",
    "| `cc_hhgt/v32/rbp_assay.py` | deterministic assay taxonomy (two levels + graph class) |",
    "| `cc_hhgt/v32/rbp_typed_binding.py` | typed lncRNA–protein/RBP binding materialiser + legacy-equivalence verifier |",
    "| `cc_hhgt/v32/rbp_canonical.py` | canonical experiment identity and cross-database collapse |",
    "| `cc_hhgt/v32/rbp_evidence_config.py` | `rbp_evidence` switch block, four modes, fairness enforcement |",
    "| `config/rbp_ablation/*.yaml` | four generated ablation configs |",
    "",
    "## Source files modified",
    "",
    "| file | change |",
    "|---|---|",
    "| `cc_hhgt/v32/evidence_training.py` | taxonomy columns reach the event frame; assay-aware feature fields; "
    "`preserve_assay_type` threaded through `_event_feature_matrix` / `build_bag_examples` / `run_evidence_training`. "
    "`experiment_raw` provenance fixed (never the fallback). |",
    "| `cc_hhgt/v32/evidence_streaming_training.py` | DuckDB streaming path kept column-identical: schema, "
    "`physical_all`, `all_exact_events`, `event_public_columns`. |",
    "| `cc_hhgt/v32/formal_graph.py` | `CONTEXT_ECLIP_ROLE` added to `G1_ROLES`; schema 9 → 18 rows; "
    "`materialize_typed_lnc_protein_binding` added; explicit global/context collision rule. "
    "`materialize_global_lnc_protein_binding` left byte-identical. |",
    "",
    "The frozen G012 invariant (`623,207 rows / 2,567 lncRNAs`) was never renegotiated: the legacy "
    "builder is untouched.",
]
(REPORTS / "RBP_ENCODE_CHANGED_FILES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# --------------------------------------------------------------- item 8
OLD_SCHEMA = {
    ("lncRNA", "binds_protein", "protein", GLOBAL_BINDING_ROLE, False),
}
typed_rows = [r for r in FORMAL_RELATION_SCHEMA if str(r[1]).startswith("binds_protein")]
diff = [
    "# RBP_ENCODE_GRAPH_SCHEMA_DIFF.md",
    "",
    "**Deliverable**: Phase 15 item 8 (old/new graph schema diff)",
    "**Date**: 2026-09-21",
    "",
    "## Relation schema",
    "",
    f"* before: **9** registered tuples",
    f"* after: **{len(FORMAL_RELATION_SCHEMA)}** registered tuples",
    f"* binding relations: **1** → **{len(typed_rows)}**",
    "",
    "### Binding relations after the change",
    "",
    "| relation_type | edge_role |",
    "|---|---|",
]
for row in typed_rows:
    diff.append(f"| `{row[1]}` | `{row[3]}` |")

diff += [
    "",
    "### Edge roles",
    "",
    "| role | before | after |",
    "|---|---|---|",
    f"| `{GLOBAL_BINDING_ROLE}` | present | present (unchanged) |",
    f"| `{CONTEXT_ECLIP_ROLE}` | absent | **new** |",
    "| `static_protein_gene_encoding` | present | present |",
    "| `static_symmetric_ppi` | present | present |",
    "",
    "### G0 / G1 / G2 membership",
    "",
    "`G1_ROLES` is now:",
    "",
    "```",
    "\n".join(sorted(G1_ROLES)),
    "```",
    "",
    "Both binding roles are G1 roles, so the scientific meaning of the three arms is unchanged:",
    "",
    "| arm | content | changed? |",
    "|---|---|---|",
    "| G0 | no binding of any kind | **no** |",
    "| G1 | + typed lncRNA–protein binding (global and approved context eCLIP) + protein→gene | extended only by approved binding roles |",
    "| G2 | G1 + PPI | **no** |",
    "",
    "### Node vocabulary",
    "",
    "**Unchanged.** No new node type. `rbp` partners map into the existing `protein` node space via "
    "the same `protein_gene_map` authority, in the same `UNIPROT:` / `GENE:` identifier space, so no "
    "`NONO:protein` / `NONO:RBP` duplication is constructible. A test asserts no `RBP` token appears in "
    "any emitted `protein_id` or `lncrna_id`.",
    "",
    "### A constraint discovered while wiring this",
    "",
    "`EDGE_KEYS = (source_type, source_id, relation_type, target_type, target_id)` carries neither "
    "`edge_role` nor `cancer_id`. The graph can therefore hold only **one edge per (source, relation, "
    "target) triple**, so a pair supported both by a context-free record and by a context-specific "
    "eCLIP record collides. The collision is resolved explicitly: **the context-specific edge wins** "
    "and the global duplicate is dropped. Keeping the global edge would broadcast context-specific "
    "evidence to every cancer.",
    "",
    "### Assay vocabulary",
    "",
    f"* `assay_subtype`: {len(ASSAY_SUBTYPES)} values",
    f"* `graph_assay_class`: {len(GRAPH_ASSAY_CLASSES)} values — `{', '.join(GRAPH_ASSAY_CLASSES)}`",
    "",
    "Only `graph_assay_class` is permitted to type a graph relation, which keeps relation-type "
    "cardinality bounded.",
]
(REPORTS / "RBP_ENCODE_GRAPH_SCHEMA_DIFF.md").write_text("\n".join(diff) + "\n", encoding="utf-8")

print("written:")
print("  ", REPORTS / "RBP_ENCODE_CHANGED_FILES.md")
print("  ", REPORTS / "RBP_ENCODE_GRAPH_SCHEMA_DIFF.md")
for status, path, work_hash, _ in rows:
    print(f"    {status:9s} {work_hash[:16]}  {path}")
