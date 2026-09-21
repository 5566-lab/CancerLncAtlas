# V3.2 G0/G1/G2 shared-base input layout (design, 2026-09-04)

The current corrected preparation stores five complete `PATIENT_FOLD_*.pt`
payloads independently under each of `G0`, `G1`, and `G2` (15 files, about
170 GB).  Most patient/candidate tensors are invariant across graph variants;
the repeated materialization is therefore a storage-layout issue, not a
scientific change.

## Proposed layout

```text
v32_g012_shared_base_<run>/
  BASE/
    PATIENT_FOLD_0.pt ... PATIENT_FOLD_4.pt
    BASE_MANIFEST.json
  DELTA_G0/
    PATIENT_FOLD_0.pt ... PATIENT_FOLD_4.pt
  DELTA_G1/
    PATIENT_FOLD_0.pt ... PATIENT_FOLD_4.pt
  DELTA_G2/
    PATIENT_FOLD_0.pt ... PATIENT_FOLD_4.pt
  SHARED_BASE_DELTA_BINDING.json
```

`BASE/PATIENT_FOLD_<f>.pt` contains the fold-independent payload fields:

- patient-fold and candidate-batch tensors;
- base logits, conservation context, labels and availability masks;
- label/input contracts and common artifact hashes;
- common model configuration.

Each `DELTA_<variant>/PATIENT_FOLD_<f>.pt` contains only:

- the variant-specific runtime graph bundle/edge tensors;
- the variant identifier and graph-authority binding;
- a reference to the matching base fold and its SHA-256.

The loader must reconstruct the exact legacy payload in memory before passing
it to training.  It must reject a missing base fold, a mismatched fold,
variant, graph-authority hash, or base SHA; it must never silently substitute a
different variant or fill a missing component with zero.

## Invariants

1. The existing 15-file corrected layout remains the authority until the
   shared-base package has passed byte/semantic equivalence tests.
2. No source `.pt` file is deleted or modified during conversion.
3. Reconstructed payloads must preserve candidate keys, row order, labels,
   masks, graph variant, and all declared lineage hashes.
4. G0/G1/G2 must still be independently addressable for fair comparison.
5. A failed conversion is recorded as `RUN_FAILED.json`; it must not emit a
   success binding.
6. The compact package is not a new model version and does not authorize
   training by itself; the existing corrected-input and GPU preflight gates
   still apply.

## Expected benefit

The five base-fold files are written once instead of three times.  The saving
depends on the graph-bundle fraction of each payload; it must be measured from
the materialized package rather than assumed.  The conversion should be run
on host `149` only, with an isolated output root and a finite storage budget.

## Current status

`DESIGN_ONLY_NOT_MATERIALIZED`.  The authoritative corrected input remains
`./data/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1`.

## Serialization reality check (2026-09-05)

The prepared `.pt` files are PyTorch zip archives whose dominant member is a
single stored `data.pkl` object (for example, G1/FOLD_0 is approximately
11.99 GB in that member).  The tensor storage members are comparatively tiny.
Consequently, a ZIP-level deduplication or hard-link pass cannot implement the
proposed BASE/DELTA semantics.  A real conversion must load/reconstruct the
payload, identify invariant and graph-specific fields, reserialize both
parts, and prove exact semantic equivalence with the original payload.  Until
that loader/converter and its equivalence tests exist, the layout remains
design-only and must not be advertised as a storage reduction already
achieved.
