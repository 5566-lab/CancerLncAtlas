# RBP_ENCODE_TESTS.md

**Deliverable**: Phase 15 item 9 (test inventory and results)
**Date**: 2026-09-21
**Server**: `149`

## New test files

| file | tests |
|---|---|
| `tests/test_v32_rbp_assay.py` | 61 |
| `tests/test_v32_rbp_evidence_features.py` | 14 |
| `tests/test_v32_rbp_typed_binding.py` | 17 |
| `tests/test_v32_rbp_typed_graph.py` | 19 |
| `tests/test_v32_rbp_canonical.py` | 20 |
| `tests/test_v32_rbp_evidence_config.py` | 47 |
| **total new** | **178** |

## What each file pins

| file | contracts |
|---|---|
| `test_v32_rbp_assay.py` | determinism, closed vocabularies, eCLIP≠RIP at subtype and graph-class level, family equality not erasing subtype, verbatim `experiment_raw`, fail-closed `unknown`, prediction-never-experimental, historical prose false positives removed, prefixed CLIP variants kept, legacy equivalence over 33 strings, relation-type closure |
| `test_v32_rbp_evidence_features.py` | eCLIP and RIP produce different EventSet features; legacy mode collapses them identically; the mode-A matrix recomputed by hand matches bit-for-bit; `experiment_raw` absent from both feature field lists; provenance not falsified |
| `test_v32_rbp_typed_binding.py` | legacy equivalence verifier detects row-count and value divergence; `rbp` included without a new entity; ten duplicate records collapse to one edge; typed relations stay in the closed vocabulary; HepG2→LIHC maps while K562 is excluded |
| `test_v32_rbp_typed_graph.py` | typed emission per assay class; predictions opt-in; context eCLIP keeps its cancer and role; non-eCLIP context rows not admitted; **context beats global on the collision**; G0 has no binding, G1 no PPI, G2 has PPI |
| `test_v32_rbp_canonical.py` | key determinism, every component changes the key, case/padding invariance, three databases collapse to one with provenance, different assay stays separate, **PMID-less rows never merged** |
| `test_v32_rbp_evidence_config.py` | four modes nested; every fairness axis bound to the authoritative constant; tampering with any axis rejected; RBP KD kept out of the primary graph; no fairness axis left as a placeholder |

## Results

| run | result |
|---|---|
| new RBP tests | **all passing** |
| full RBP/evidence/graph regression | **6 failed, 298 passed, 1 skipped** |
| graph-level forward/backward (gate 11) | **13 passed** |

The six failures are all pre-existing and each was reproduced on the pristine immutable capsule:

| failure | cause | proof |
|---|---|---|
| 4 × `test_v32_evidence_streaming_training.py` | DuckDB 1.5.0 `InternalException: Attempted to access index 3 within vector of size 3` | same tests run on the untouched origin capsule: identical failures |
| 2 × `test_v32_formal_prepare_graph_authority.py` | `artifacts/v32_patient_fold_authority_20260829_r1/` absent | the directory is absent in the origin capsule too |

**New failures introduced by this work: zero.**
