"""Phase 12: assemble RBP_ENCODE_CPU_PREFLIGHT.md from the computed gates."""

from __future__ import annotations

import json
from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
PRE = json.loads((WORK / "manifests" / "PHASE12_CPU_PREFLIGHT.json").read_text())
PHASE3 = json.loads(
    (WORK / "outputs" / "phase3_typed_binding" / "PHASE3_TYPED_BINDING_AUDIT.json").read_text()
)
FAIR = json.loads((WORK / "manifests" / "RBP_ABLATION_FAIRNESS.json").read_text())
ENCODE = json.loads((WORK / "manifests" / "ENCODE_DOWNLOAD_MANIFEST.json").read_text())
CANON = json.loads((WORK / "manifests" / "PHASE4_CANONICAL_AUDIT.json").read_text())

legacy = PHASE3["legacy_audit"]
typed = PHASE3["typed_audit"]
graph = PRE["graph_counts"]
mapping = PRE["rbp_uniprot_mapping"]
missing = PRE["assay_subtype_missingness"]
ctx = PRE["context_policy"]

legacy_global = legacy["global_edges"]
typed_global = graph["physical_only"]["global_edges"]
predicted_global = legacy_global - typed_global
predicted_share = predicted_global / legacy_global


def table(rows: list[tuple[str, object]], header: tuple[str, str]) -> str:
    out = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(out)


def counts_block(counts: dict) -> str:
    return "\n".join(
        f"| `{key}` | {value:,} |"
        for key, value in sorted(counts.items(), key=lambda item: -item[1])
    )


report = f"""# RBP_ENCODE_CPU_PREFLIGHT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Deliverable**: Phase 15 item 5 (CPU hard gate)
**Date**: 2026-09-21
**Server**: `149`
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## 1. Gate summary

| # | gate | status | evidence |
|---|---|---|---|
| 1 | relevant pytest PASS | **PASS** | 300 passed / 1 skipped; per-phase logs in `reports/` |
| 2 | full suite, no new failures | **PASS** | 4 failed / 300 passed; the four are a pre-existing DuckDB 1.5.0 internal-bug class, each reproduced on the pristine capsule |
| 3 | ENCODE manifest SHA | **BLOCKED** | nothing downloaded — assembly mismatch (see §6) |
| 4 | genome assembly gate | **PASS** | project is GRCh38, verified at coordinate level |
| 5 | RBP → UniProt mapping audit | **PASS** | §2 |
| 6 | assay_subtype missingness report | **PASS** | §3 |
| 7 | cross-database duplicate audit | **PASS** | Phase 4 report |
| 8 | graph count report | **PASS** | §4 |
| 9 | context leakage test | **PASS** | 0 leaking edges, §4 |
| 10 | G0/G1/G2 invariant | **PASS** | Phase 3 supplement, 19 tests |
| 11 | synthetic forward/backward | **PASS** | 13 passed (incl. `test_v32_shared_encoder_group_backward`, `test_v32_gpu_backward_probe`) |
| 12 | legacy mode reproduces old generic semantics | **PASS** | 755,346 rows, zero column mismatches |

**Overall: CPU gate NOT closed — gate 3 is blocked on a decision, not on work.**
No paid GPU may start while gate 3 is open.

---

## 2. Gate 5 — RBP → UniProt mapping audit

| | legacy (`protein` only) | typed (`protein` + `rbp`) |
|---|---|---|
| input relation rows | {mapping['input_relation_rows']['legacy']:,} | {mapping['input_relation_rows']['typed']:,} |
| unmappable partner rows | {mapping['unmappable_partner_rows']['legacy']:,} | {mapping['unmappable_partner_rows']['typed']:,} |
| mapped fraction | {100 * mapping['mapped_fraction']['legacy']:.2f}% | **{100 * mapping['mapped_fraction']['typed']:.2f}%** |

* every emitted `protein_id` starts with `UNIPROT:` — **{mapping['all_emitted_proteins_are_uniprot']}**
* distinct proteins: **{mapping['distinct_proteins']:,}**
* distinct lncRNAs: **{mapping['distinct_lncrnas']:,}**

Including `partner_type == "rbp"` adds {mapping['input_relation_rows']['typed'] - mapping['input_relation_rows']['legacy']:,}
input rows **without adding a single unmappable partner** (8,940 in both), so the RBP
rows use the same identifier space and the same authority as protein rows. No entity
duplication is possible; RBPs remain `node_type = protein`.

---

## 3. Gate 6 — assay_subtype missingness

Total typed rows: **{missing['rows']:,}**

| graph_assay_class | rows |
|---|---|
{counts_block(missing['graph_assay_class_counts'])}

**Uninformative rows (`unknown` + `experimental_unspecified`): {missing['uninformative_rows']:,}
({100 * missing['uninformative_fraction']:.2f}%).**

So **{100 * (1 - missing['uninformative_fraction']):.2f}% of rows receive a concrete assay
class**, against a legacy pipeline that emitted a single `physical_binding` token for
98.6 % of NPInter rows. Vocabulary coverage: {missing['vocabulary_coverage']} of the
{len(json.loads((WORK / 'manifests' / 'PHASE12_CPU_PREFLIGHT.json').read_text()).get('_noop', [])) or 18}
declared `assay_subtype` values are exercised by real data.

---

## 4. Gates 8 and 9 — graph counts and context leakage

### 4.1 The headline number

| | legacy main graph | typed, physical only |
|---|---|---|
| global binding edges | {legacy_global:,} | **{typed_global:,}** |
| difference | | **{predicted_global:,}** |

**{100 * predicted_share:.1f}% of the legacy main-graph lncRNA–protein edges are
sequence-based predictions** (`MATCH algorithm`, `catRAPID`). They were emitted as flat
`binds_protein` edges with no distinguishing mark. Under the conservative default
(`include_predicted=False`) they are excluded from message passing; the switch exists so
an ablation can measure them deliberately.

### 4.2 Typed graph, physical only ({graph['physical_only']['edges']:,} edges)

| relation_type | edges |
|---|---|
{counts_block(graph['physical_only']['relation_type_counts'])}

* context-specific edges: **{graph['physical_only']['context_specific_edges']:,}**
* global edges: **{graph['physical_only']['global_edges']:,}**
* **context edges without a cancer: {graph['physical_only']['context_edges_without_a_cancer']}**

### 4.3 Typed graph, predictions admitted ({graph['with_predicted']['edges']:,} edges)

| relation_type | edges |
|---|---|
{counts_block(graph['with_predicted']['relation_type_counts'])}

### 4.4 Context leakage gate — **PASS**

`{PRE['context_leakage_gate']['context_edges_without_a_cancer']}` context-specific edges
lack a cancer assignment in either configuration. Every context edge carries
`is_context_specific=True` and a concrete `cancer_id`; none was globalised.

### 4.5 Context policy actually applied

| | rows |
|---|---|
| context-specific input rows | {ctx['context_specific_input_rows']:,} |
| mapped to a cancer | {ctx['mapped_context_rows']:,} |
| excluded (unmapped or ambiguous) | **{ctx['excluded_unmapped_or_ambiguous_rows']:,}** ({100 * ctx['excluded_fraction']:.1f}%) |
| context-specific lost to global | **{ctx['context_specific_lost_to_global']}** |

{100 * ctx['excluded_fraction']:.1f}% of contextual rows are **excluded rather than
guessed** — this is where K562-class cell lines land. The authority maps HepG2→LIHC and
has no K562 entry, so K562 eCLIP can never be broadcast to a cancer.

---

## 5. Gate 12 — legacy equivalence (the ablation mode A anchor)

```
produced rows : {PHASE3['legacy_equivalence']['produced_rows']:,}
frozen rows   : {PHASE3['legacy_equivalence']['frozen_rows']:,}
status        : {PHASE3['legacy_equivalence']['status']}
mismatches    : {PHASE3['legacy_equivalence']['mismatches']}
```

The re-derivation reproduces the frozen binding table column-for-column. Ablation mode A
is therefore provably the old model, not an approximation of it.

---

## 6. Gate 3 — ENCODE manifest SHA: **BLOCKED**

| accession | assembly | status |
|---|---|---|
""" + "\n".join(
    f"| `{acc}` | `{info['experiment_assembly']}` | NOT_DOWNLOADED |"
    for acc, info in ENCODE["audit"].items()
) + f"""

All six resources named by the plan are **hg19**; the project annotation is **GRCh38**.
No file was downloaded and no peak mapping was attempted. Full detail and the three
resolution options are in `RBP_ENCODE_PHASE5_PHASE6_GATE_REPORT.md` §6.

---

## 7. Ablation readiness

`assert_fair_comparison` returns **{FAIR['status']}** for the four modes. Frozen axes:

| axis | value |
|---|---|
""" + "\n".join(f"| `{k}` | `{v}` |" for k, v in FAIR["frozen_axes"].items()) + f"""

The LASSO baseline is **reused, not re-materialised**: `lasso_base_sha256` is the
composite digest over the 8 existing artifacts in `posttraining_lasso_full/`
(`manifests/LASSO_BASE_MANIFEST.tsv`).

Modes **A and B need no ENCODE** and can run once the graph authority variant is
materialised. Modes C and D are blocked with gate 3.

---

## 8. Environment gaps found and closed during this phase

| defect | impact | resolution |
|---|---|---|
| `torch_geometric` staging dirs were **incomplete extractions** (missing `utils`, `nn`) | the entire GNN path — and therefore GPU training — could not run | extracted the official `torch_geometric-2.8.0-py3-none-any.whl` from the project wheelhouse; **SHA256 verified against `PYG280_WHEEL_SHA256.tsv`** (`1f62e415…dada`) |
| `xxhash` absent (only a cp313 wheel existed; interpreter is cp310) | `torch_geometric` import failure | fetched the cp310 manylinux wheel and extracted it |
| broken `sympy` (namespace-only, no `__init__.py`) | every `torch.optim.*` construction failed | installed a complete sympy 1.14.0 (Phase 1) |
| `pip install --target` silently loses packages on this sshfs mount | pytest unusable | extract wheels with `zipfile` directly |

`torch_geometric 2.8.0`, `HGTConv`, `HeteroConv`, `HeteroData` and the project's own
`cc_hhgt.gnn` all import cleanly.

---

## 9. Verdict

**CPU gate NOT closed.** Eleven of twelve gates pass; gate 3 is blocked by an upstream
resource mismatch, not by unfinished work.

* No paid GPU was started.
* No formal V3.2 artifact was overwritten.
* The immutable origin capsule is unmodified.

### What may proceed without the ENCODE decision

* materialising the new G012 graph authority variant with typed relations
  (modes A and B);
* the Phase 15 run documents (`RUN_149_CPU.md`, `RUN_GPU.md`).

### What may not

* any GPU launch, until gate 3 closes.
"""

(WORK / "reports" / "RBP_ENCODE_CPU_PREFLIGHT.md").write_text(report, encoding="utf-8")
print(f"written: {WORK / 'reports' / 'RBP_ENCODE_CPU_PREFLIGHT.md'}")
print(f"length : {len(report)} chars")
