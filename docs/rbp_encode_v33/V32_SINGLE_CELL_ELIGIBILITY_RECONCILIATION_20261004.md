# Single-cell eligibility reconciliation — 2026-10-04

Five of the 23 formally eligible single-cell datasets were being excluded from private-head
training while still declaring `formal_eligible=True`. This records what the raw data actually
says, what was changed, and how the result was verified.

## 1. What was wrong

`single_cell_training.normalise_dataset_manifest` raised on `formal_eligible=True` combined with
a failing `qualified` predicate, yet the manifest on disk contained five such rows — rows the
current producer would refuse to emit (`single_cell_input_builder` raises on the same three
combinations, at lines 360, 393 and 399).

Two independent causes, both contradicted by the raw data:

**(a) A hard-coded cancer-name blacklist.**

```python
KNOWN_FEATURE_UNIVERSE_LIMITATIONS = frozenset({"CESC", "UCS", "UVM"})
```

This forced `feature_universe_status = LIMITED` **by name**, regardless of any measurement, and
can only *reduce* eligibility.

**(b) Stale status columns.**

Measured directly from the raw 10x matrices, using the symbol↔ENSG annotation authority carried
in UVM's own `raw_feature_bc_matrix.h5` (33,660 pairs), intersected with
`FORMAL_LNCRNA_SCOPE.parquet` (16,889 lncRNAs):

| dataset | measured lncRNAs in universe | recorded | ratio | recorded semantics |
|---|---|---|---|---|
| CESC | **2,717** | 47 | 57.8× | `FRESH_DIRECT_STABLE_ID_...` |
| UVM | **12,624** | 446 | 28.3× | `FRESH_DIRECT_STABLE_ID_...` |
| BRCA | **9,536** | 2,909 | 3.3× | `READ_ONLY_AUDIT_COUNT_METADATA_BLOCKED_H5_NOT_SCANNED` |
| COAD | **9,536** | 2,909 | 3.3× | same |
| OV   | **9,536** | 2,909 | 3.3× | same |

The `LOW_LNCRNA_FEATURE_UNIVERSE` threshold is 1,000; all five measure well above it. Note that
BRCA/COAD/OV's recorded semantics states outright that the h5 was **never scanned** — 2,909 is an
inherited audit number, not a measurement.

For BRCA/COAD/OV the donor flag was also wrong. From the raw `cell_metadata.parquet`:

| dataset | cells | `patient_id_verified` | authority | distinct donors |
|---|---|---|---|---|
| BRCA | 40,694 | all `True` (100%) | `E-MTAB-8107.sdrf.txt` | 17 |
| COAD | 35,369 | all `True` (100%) | `E-MTAB-8107.sdrf.txt` | 7 |
| OV   | 41,524 | all `True` (100%) | `E-MTAB-8107.sdrf.txt` | 5 |

So `donor_metadata_available=False` was a stale declaration, not a data absence.

## 2. What changed

* **`single_cell_input_builder.py` / `single_cell_partition_builder.py`** — the name-based
  blacklist is removed from every eligibility decision; `LIMITED` is now decided solely by
  `lncrna_feature_universe_count < 1000`. `KNOWN_FEATURE_UNIVERSE_LIMITATIONS` survives only as a
  historical annotation constant (definition + `__all__`), never as a gate.
* **`single_cell_manifest_reconciliation.py`** (new) — `sync_verified_manifest` promotes a
  dataset only from hash-verified evidence, and still refuses when the evidence does not meet the
  bar:

  ```python
  if count < LOW_FEATURE_UNIVERSE_THRESHOLD or record["donor_metadata_available"] is not True:
      raise ValueError(f"Evidence does not meet feature/metadata requirements: {cancer}")
  ```

  `reconcile_bound_manifest` re-verifies every SHA-256 (cohort binding, independent audit,
  per-partition `SUCCESS.json`, `FILE_MANIFEST.parquet`, metadata), cross-checks cancer/dataset
  identity and all binding counts, and requires the metadata to actually carry non-empty
  `patient_id` and `cell_type_major`. `adapt_verified_head_tables` aligns `ENSG…` with
  `LNC:ENSG…` (rejecting an ambiguous namespace), supplies the `source_tier` and
  `n_observations` the consumer requires, and pools major cell types to the donor×compartment
  level with cell-count weights, preserving the declared expression scale.
* **`build_v32_single_cell_r11_rescued_contract.py`** — adds a `target_host == "149"` guard and a
  `LAUNCH_PREFLIGHT.json` receipt, tightens metadata validation to every required column, and
  calls the reconciliation instead of downgrading.
* **`scripts/deploy_single_cell_manifest_fix_149.py`**, **`scripts/reconcile_single_cell_head_inputs_149.py`**
  — deployment and reconciliation entry points.

This replaces a stale declaration with a verified measurement. It does not lower a bar: the
measured counts are 3–58× above the threshold, and the promotion path re-checks the bar itself.

## 3. Verified result

Recomputed on host 149 with the real consumers:

```
normalise_dataset_manifest        -> qualified = 23 / 33
excluded                          -> BLCA KICH KIRP LIHC LUAD PAAD PRAD STAD THCA UCS
```

The excluded set is exactly the frozen typed-unavailable ten, so coverage is 23 + 10 = 33 as
frozen — the change restores five datasets to the scope that was already declared, and does not
widen it.

| quantity | value |
|---|---|
| qualified datasets | **23** (was 18) |
| cancers with training rows | **22** |
| training rows after `exact_candidate_join` | **166,248** |
| fold block counts | `{0:5, 1:5, 2:4, 3:4, 4:4}` |
| restored cancers | BRCA, CESC, COAD, OV, UVM |

DLBC is qualified but contributes 0 rows — `NO_ASSOCIATION_ROWS_AFTER_DONOR_REPLICATION_FILTER`
(4 donors). The remaining 10 are unavailable for reasons unrelated to the feature universe:
seven lack an auditable accession→individual mapping, KICH/KIRP have only 3 independent donors,
and UCS exposes 38 directly-matched lncRNAs (43 with unique-symbol mapping) against a 1,000
threshold — its raw matrix holds 18,082 features, so the limitation is real.

Tests: 30 passed. The five-fold assignment check passed. No formal training was started; no
historical result was modified.

## 4. Boundary

This is a validation-tier input correction. It changes which datasets the single-cell private
head can consume; it does not touch the primary HHGT model, the frozen 23+10 release scope, or
any sealed-test authority.
