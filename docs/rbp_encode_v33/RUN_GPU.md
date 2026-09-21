# RUN_GPU.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Deliverable**: Phase 15 item 10 (execution document, GPU half)
**Date**: 2026-09-21

> ## STATUS: DO NOT START
>
> The CPU gate is **NOT closed**. Of twelve gates, eleven PASS and **gate 3
> (ENCODE manifest SHA) is BLOCKED**: all six ENCODE accessions named by the plan are
> hg19 while the project annotation is GRCh38, and silent liftOver is forbidden.
>
> Starting a paid instance now would burn money on an ablation set that cannot be
> completed. The plan is explicit: *"CPU gate 未闭合前不启动付费 GPU."*

---

## 1. Preconditions — all must hold

| # | precondition | current |
|---|---|---|
| 1 | `RBP_ENCODE_CPU_PREFLIGHT.md` reads PASS | **11/12, gate 3 BLOCKED** |
| 2 | input manifest frozen with SHA256 | yes |
| 3 | new graph authority variant frozen | **not yet materialised** |
| 4 | all hashes written to a manifest | yes |
| 5 | all four ablation configs generated | **yes** |
| 6 | same folds verified across modes | **yes** (`assert_fair_comparison` = FAIR) |
| 7 | no current production output overwritten | yes |
| 8 | VRAM / disk / runtime estimate recorded | **not yet** |

**Preconditions 3 and 8 are outstanding; gate 3 is blocked on a decision.**

---

## 2. Resolve gate 3 first

Pick one path (see `RBP_ENCODE_PHASE5_PHASE6_GATE_REPORT.md` §6):

* **A — substitute hg38-native ENCODE eCLIP accessions** (252 GRCh38 experiments, 168 RBP
  targets available). Keeps the methodology and every context rule; changes only which
  accessions are cited. Recommended.
* **B — explicit, non-silent liftOver.** Requires a pinned chain file with its own SHA256,
  per-peak mapping status, quarantine of unmapped peaks, a new authority variant, and the
  lifting declared in every downstream lineage record.
* **C — drop the ENCODE half** and deliver Phases 0–4.

Then regenerate `manifests/ENCODE_DOWNLOAD_MANIFEST.tsv` with real SHA256 values for every
downloaded file. **A download manifest without hashes does not close gate 3.**

---

## 3. Smoke run — one fold, bounded, mandatory

Only after every precondition holds. The plan: *"先跑 1 fold / bounded smoke."*

```bash
bash 03_train_all_linux.sh --fold 0 --max-steps 50 --arm typed_binding_only
```

The smoke passes only if **all** of the following are observed:

| check | how to confirm |
|---|---|
| forward PASS | run completes without exception |
| backward PASS | gradients non-zero on at least one core parameter |
| loss finite | every logged loss is finite (no `nan`, no `inf`) |
| checkpoint / resume PASS | write a checkpoint, resume from it, identical step count |
| **typed relations present in the graph metadata** | inspect `HeteroData` edge types: `('lncRNA', 'binds_protein_eclip', 'protein')` and the other typed relations must appear — **not** a single `binds_protein` |

The last row is the one that matters most. The plan: *"禁止 GPU 启动后再发现 ENCODE 没进模型."*
If the graph metadata still shows only `binds_protein`, the typed path is not wired and the
run must be stopped before the five-fold job.

---

## 4. Formal five-fold run

Only after the smoke passes on every row above.

```bash
for arm in legacy_generic_binding typed_binding_only \
           typed_binding_plus_encode_eclip full_rbp_evidence ; do
  bash run_model_workflow.sh config/rbp_ablation/$arm.yaml train
done
```

Requirements carried from the plan and enforced by `assert_fair_comparison`:

* identical patient folds, candidate universe, LASSO base, label definition, seeds,
  training budget and evaluation metric across all four arms;
* `changes_primary_ranking = false` in every arm;
* `auxiliary_gradients_into_core = false` in every arm;
* RBP knockdown stays **evidence-only** and never enters the primary graph;
* the shuffled-eCLIP control uses the same node count and approximately the same degree
  distribution, with real lncRNA–RBP pairings destroyed.

---

## 5. Cost discipline

* create the instance only when every precondition holds — never "to look around";
* the platform is a preemptible CompShare GPU; a 4090 is roughly ¥1.44–2.09/h plus
  ~¥0.07/h disk;
* run the one-fold smoke first and read its output before launching five folds;
* on any launcher failure, validation failure, or prolonged absence of optimizer progress,
  **stop the instance immediately**;
* after results are verified on both local and 149 with matching SHA256, release the
  instance **and its cloud disk**.

---

## 6. Reporting (pre-registered in `RBP_ENCODE_ABLATION_PLAN.md`)

Report AUPRC, AUROC, Precision@K, calibration and ΔAUPRC vs the reused LASSO base for every
arm, with per-fold values. Include the real-vs-shuffled eCLIP comparison and per-assay-subtype
Evidence performance.

**Pre-registered decision rule — no exceptions:**

* if C ≤ B + noise → ENCODE eCLIP does not help; **report FAIL**;
* if C ≈ shuffled eCLIP → the gain is degree, not biology; **report FAIL**;
* if D ≤ C → RBP knockdown adds nothing; report it;
* **no weight may be tuned, no label changed, and no mode redefined to convert a FAIL into
  a PASS.**

ENCODE eCLIP is *training input* in modes C and D and therefore cannot also be presented as
independent external validation. RBP knockdown, if reported, must be labelled **RBP-level
functional support — not lncRNA perturbation validation**.

---

## 7. Teardown

1. verify result SHA256 on local and 149;
2. copy results into the work root and refresh the manifest;
3. `compshare instance delete <id> --release-disk --yes --wait`, or release the disk from the
   console;
4. record the terminal receipt in `reports/`.
