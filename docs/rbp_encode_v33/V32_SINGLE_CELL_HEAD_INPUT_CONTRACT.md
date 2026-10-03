# Consuming the audited formal-23 single-cell assets

Recorded 2026-10-03, from actually running `cc_hhgt.v32.single_cell_training`'s normalisers
against the audited partitions. Every claim below was observed, not inferred from schemas.

## Why this note exists

`COHORT_BINDING.json` (`BOUND_23_OF_23_ANNOTATION_V2_AUDITED`) gives one `output_root` per
formal cancer. Those roots hold `association_evidence.parquet`,
`lncrna_donor_celltype_summary.parquet`, `association_context_availability.parquet` and
`pathway_availability.parquet`. Passing them straight to
`scripts/run_v32_single_cell_training.py` **does not work**, for four independent reasons.

## 1. Source-row counts reconcile exactly

Concatenating the 23 binding-named roots gives **15,743,466** association rows, which is
exactly `SUCCESS.json`'s `total_association_evidence_rows`. That equality is the check that the
selection is neither short nor duplicated — notably, each partition also has a
`.pre_annotation_v2_20260907` backup beside it (BRCA: 218,054 current vs 116,191 backup), and
reading a partition directory wholesale would silently include both.

## 2. Column-name aliases the normalisers require

`single_cell_training._first_column` matches by an exact normalised token
(`re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")`), so these source names do not satisfy it:

| consumer wants one of | source column | consequence |
|---|---|---|
| `rho` \| `effect` \| `association_effect` \| `correlation` \| `beta` | `spearman_rho` | **hard failure** |
| `fdr` \| `q_value` \| `padj` \| `p_value` | `bh_q_global_tests` | **hard failure** |
| `detection_rate` | `detect_rate` | silently NaN |
| `n_cells` | `cell_count` | silently NaN |

The assembly adds these as **additive alias columns** and keeps every source column. The last
two matter as much as the first two: they fail quietly, so the head would train with
`lnc_detection_rate` / `lnc_n_cells` entirely missing and nothing would say so.

## 3. NULL `dataset_id` collapses into one fake dataset

Seven rows of `dataset_manifest_33c.parquet` have a NULL `dataset_id`.
`normalise_dataset_manifest` does `astype(str)` **before** grouping, so all seven become the
literal string `"None"` and trip:

```
SingleCellTrainingError: A single-cell dataset maps to multiple cancers
```

Those seven are `BLCA, LIHC, LUAD, PAAD, PRAD, STAD, THCA` — every one of them
`formal_eligible=False` with
`blocking_reason=SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED`.
Each is given an explicit unique `UNAVAILABLE_<CANCER>` placeholder, so the declaration of
unavailability is machine-readable instead of accidental.

## 4. Two rows are self-inconsistent under the module's own contract

`normalise_dataset_manifest` both **raises** on

```python
formal_eligible & ~feature_universe_status.eq("pass")
```

and, three lines later, defines the real gate as

```python
qualified = formal_eligible & cancer_id in FORMAL_SINGLE_CELL_CANCERS
          & quality_status in _FORMAL_QUALITY
          & source_tier in _FORMAL_SOURCE_TIERS
          & feature_universe_status.eq("pass")
          & donor_metadata_available
```

with the training flow filtering **every** downstream table on `qualified`.

Two of the 23 formal rows fail those conditions while still declaring `formal_eligible=True`:

| dataset_id | cancer | quality_status | feature_universe_status | lncRNA features |
|---|---|---|---|---|
| `SC_GSE297041_CESC` | CESC | LIMITED | LIMITED | 47 (audited 53) |
| `SC_GSE139829_UVM` | UVM | LIMITED | LIMITED | 446 |

So the module's contract already treats a LIMITED-feature dataset as **not formal**; the source
manifest simply declares otherwise. The resolution taken is the conservative one: emit a
`dataset_manifest.training.parquet` in which those two rows carry `formal_eligible=False`, with
the original values, the failing fields and the rationale recorded verbatim under
`formal_eligible_overrides` in the input manifest. This can only **shrink** coverage — never
expand it — the source manifest and its SHA-256 are untouched, and the change is reversible.

**Consequence:** the private head trains on the **21** cancers that satisfy `qualified`, not 23.
If 23-cancer coverage is required, an upstream authority must say so explicitly; that is a scope
decision, not something to encode by editing `formal_eligible` to make a gate pass.

## 5. How all of this is verified

The assembly script does not stop at hashes and row counts. It runs the real consumers —
`normalise_dataset_manifest`, `normalise_single_cell_associations`, `_normalise_lnc_celltype`,
`normalise_candidates` — against its own output and only reports success when all four pass.
Hashes prove the file is what was assembled; only the normalisers prove it is *consumable*.
