# V3.2 fresh G0/G1/G2 preparation contract (2026-08-29, revision 3)

Status: **DESIGN ONLY - NOT APPROVED FOR PREPARATION OR TRAINING**

Revision 3 closes two design-level false-readiness risks. First, the complete
static G2 backbone is resident in every runtime chunk and only fold-local
coexpression is chunked. Second, the fixed 3.3-million training/evaluation
sample is no longer described as the website's complete eligible universe.
Neither correction authorizes implementation, preparation, inference, or
training.

The historical `PATIENT_FOLD_*.pt` files are diagnostic-only because their
declared code tree (`3ad7e4d9...`) cannot be reconstructed from an immutable
snapshot. No fresh run may load their tensors, graph bundle, labels, baseline
logits, checkpoint, prediction, or ranking. Their candidate/fold hashes may be
used only as equality assertions after the fresh pipeline regenerates them.

> **Status amendment (2026-09-04):** This document is a 2026-08-29 design
> snapshot.  Its readiness-matrix row that labels the full-33 CNV five-fold
> OOF `NOT_READY` is superseded by the independently audited
> `v32_cnv_head_oof_20260902_r6` (five fresh checkpoints, 165 prediction
> files, typed OOF).  The corrected local-CNV audit still requires primary
> G0/G1/G2 retraining; completion of the independent CNV head does not imply
> that primary training or release is complete.  See
> `docs/V32_LOCAL_CNV_AUDIT_RECONCILIATION_20260904.md`.

## 1. Frozen data-level inputs

Before preparation, an independent source freeze must inventory every file by
absolute path, byte count, and SHA-256. The allowlist is limited to:

- 33 edgeR/TMM lncRNA-expression parquets;
- 33 bulk-gene-expression parquets (165,106,010 rows, 11,233 samples,
  17,691 genes) for fold-local coexpression;
- the association-covariate table used only to residualize coexpression;
- the 16,889 x 33 detection-rate table;
- 33 exact-pathway activity parquets;
- static annotation rows selected under the typed relation contract below;
- frozen V3.2 code, config, task manifest, and runtime receipt.

The activity table is a data-level covariate, not an old model output. A
read-only audit proved that its 10,432 `(cancer_id, sample_id)` values are
exactly the intersection of the 11,233 upstream activity samples and the
10,493 fresh lncRNA-expression samples. All 21,303,825 formal rows occur, with
exact column values, in the 22,939,836-row upstream activity table; the
expected intersection also contains exactly 21,303,825 rows. File-level
inventories still have to be written and independently rehashed.

No literature, drug, perturbation, interaction, validation, or test data may
define `discovery_effect`, FDR, proxy labels, weak labels, direction labels, or
the L1 baseline. This implements
`annotation_sources_may_define_label=false`. Interaction annotations may enter
only the explicit G1/G2 graph ablation.

## 2. Candidate-scope contract and one common fold authority

There are three different counts and they must never be conflated:

- 16,889: annotation lncRNAs in the complete detection grid;
- 8,541: union of lncRNAs eligible in at least one cancer and therefore present
  in the 3.3-million-row candidate universe;
- 4,712: the pan-cancer shared subset eligible in at least three cancers.

Among the 8,541 candidate lncRNAs, 2,456 occur in one cancer, 1,373 in two,
and 4,712 in at least three. The V3.2 config explicitly says
`local_only_retained: true` and
`candidate_sampling_covers_all_cancer_eligible_lncRNAs: true`.
`prepare_v32_formal.py` selects `within_cancer_eligible`, not only the shared
subset. Training and evaluation consume every prepared candidate batch, and
the old graph contains exactly 8,541 lncRNA nodes. Thus 4,712 is a reporting
and context stratum, not a training/evaluation filter. Fresh audits must report
metrics separately for `shared`, `two-cancer local`, and `one-cancer local`.

The current frozen V3.2 code must regenerate, rather than copy:

1. the 100,000-pairs-per-cancer candidate universe (3,300,000 rows);
2. five patient folds from seed `20260726`;
3. train-patient discovery associations and labels;
4. validation/test replication labels only;
5. one train-only L1 baseline and its three label-view logits per fold.

For outer fold `f`, train excludes `f` and `(f+1)%5`, validation is
`(f+1)%5`, and test is `f`. Candidate keys, base logits, conservation context,
graph availability, and direction features must be bitwise identical across
the three label views. Only replication label-control tensors may differ. Test
labels are forbidden from fitting and early stopping.

The current formal activity cohort has 10,432 distinct
`(cancer,sample,patient)` rows and exactly 10,432 patient keys: no multi-sample
patient, cross-fold patient, or missing-fold patient is observed. Therefore no
actual patient leakage is asserted for this cohort. The builder is still
contract-fragile because it accepts/hashes `sample_id` while being named a
patient-fold builder. Fresh code must accept explicit `patient_id`, assign at
patient level, bind both sample and patient hashes, and assert sample-to-patient
many-to-one validity plus no patient crossing cancer or fold. The observed
one-to-one result is a required PASS gate, not permission to retain the weak
contract.

The common fold payload is stored once per fold. G0/G1/G2 manifests reference
the same five common-fold SHA-256 values; graph variants do not duplicate the
large label payloads.

### 2.1 Label, direction, and evaluation governance blockers

Fresh preparation cannot reuse old label tensors. Three implementation defects
must first be resolved and independently checked:

1. Partial-correlation degrees of freedom are inconsistent. The current
   association code uses `n - design.shape[1] - 2`, sampled associations use
   an intercept-only `n-3`, while another feature builder uses
   `n - residual_design_rank - 1`. The candidate formula is
   `n - rank(X) - 1`, where `X` includes the intercept, but it must be proved
   against an independent analytic toy and R implementation before adoption.
   Column count may not replace numerical matrix rank. Train discovery,
   validation replication, sealed test replication, FDR, labels, and the L1
   baseline are all regenerated together after this proof; no old label or
   baseline hash is an authority.
2. The current replication attachment replaces membership proxy fields but
   retains the train-patient `discovery_effect`; batch construction then derives
   validation/test direction labels from that train effect. Fresh payloads use
   distinct fields: `train_discovery_effect` is an allowed train-derived input,
   while each held-out split gets a label-only `replication_effect`,
   `replication_direction_label`, and availability mask. Held-out replication
   direction is never a model input. Validation direction loss/metrics use the
   validation replication mask; final test direction metrics use only the
   sealed test replication mask.
3. Historical preparation exposed test labels, test logits, and test AUPRC/
   AUROC before architecture lock. Consequently every historical test split is
   development data and cannot support a fresh untouched-test claim. New
   training authorities contain train and validation labels only. Test keys and
   a cryptographic commitment may be public, but test labels/probabilities live
   in a separate sealed artifact that the training process cannot read. An
   independent evaluator may unseal it only after winner, configuration,
   checkpoint, code, and input hashes are locked. A newly pre-registered split
   can reduce direct reuse but is not fully independent because the cohort has
   already been examined; definitive promotion therefore requires untouched
   external validation. Without that, claims are limited to strict nested-CV
   development evidence.

Preparation is forbidden from computing or publishing any test metric. The
audit must prove the training process cannot open the sealed path, and that
changing sealed test values leaves every prepared/trained hash unchanged up to
the locked evaluator hand-off.

## 3. Fold-specific common G0 base

Every accepted graph row must have finite typed endpoints and finite weight,
`observed=true`, `raw_effect=NULL`, `requires_fold_localization` false/null,
and both endpoints in the frozen node inventory. Outcome-derived and direct
lncRNA-to-pathway edges are forbidden.

All G0/G1/G2 arms in an outer fold share these G0 base relations:

| Typed relation | Source and transform |
| --- | --- |
| `lncRNA -> expressed_in -> cancer` | Fresh all-sample, outcome-free detection rate; exact candidate `(cancer,lncRNA)` pairs; fixed before folds |
| `lncRNA -> coexpressed_positive/negative -> gene` | Recomputed separately for each outer fold from train patients only; sign becomes relation type; magnitude is `abs(rho)` with an explicit relation polarity |
| `gene -> member_of_positive/negative -> pathway` | Static annotation; sign becomes relation type; magnitude is `abs(weight)` with explicit relation polarity; target is one of 2,135 candidate pathways |
| `pathway -> member_of_family -> pathway_family` | Static annotation; exact zero-weight rows are excluded |

The weighted message branch must multiply magnitude by registered relation
polarity (`+1` or `-1`). The HGT topology branch receives distinct positive and
negative relation types. Reverse directed relations retain the same polarity.
This prevents the topology branch from silently treating inhibitory edges as
ordinary positive adjacency.

The fresh `safe_graph` contract must authorize exact role/relation pairs; the
materializer may not bypass it:

| Source role | Allowed relation(s) | Split/outcome constraints |
| --- | --- | --- |
| `transductive_expression_eligibility` | `expressed_in` | pre-registered all-sample, outcome-free; never masquerades as `fold_local_expression` |
| `fold_train_coexpression` | `coexpressed_positive`, `coexpressed_negative` | exact outer-train patients only; sign derived directly from fresh signed rho |
| `static_signed_pathway_membership` | `member_of_positive`, `member_of_negative` | static annotation, outcome-free |
| `static_pathway_hierarchy` | `member_of_family` | static annotation, nonzero weight |
| `static_global_lnc_protein_binding` | `binds_protein` | global/non-context-specific only |
| `static_protein_gene_encoding` | `encoded_by` | static annotation |
| `static_symmetric_ppi` | `physical_interaction` | canonical non-self pair and shared symmetric relation |

Unknown roles/relations, direct lncRNA-to-pathway edges, label-derived rows,
non-train coexpression, context-specific binding in the static graph, signed
relations whose source sign disagrees with their typed polarity, nonpositive
magnitudes, and generic PPI reverse types are hard failures. The current
allowlist lacks several of these roles and would reject or mislabel them; that
is an implementation blocker, not evidence that the source data are absent.

### 3.1 `expressed_in` exact audit

The old combined graph has 58,964 `expressed_in` rows. The current full rates
table has 557,337 rows; thresholding at detection rate >=0.10 and restricting
to exact candidate cancer-lncRNA pairs yields 76,734 rows. This fresh relation
is the authority because the old graph used a narrower legacy lncRNA scope.

The old prepared G0 has exactly 8,541/8,541 lncRNA nodes and 33/33 cancer nodes
with degree feature zero; each type has only one unique `x=[0,1]` row. The
fresh 33 cancer degrees are all nonzero and all distinct:

```text
ACC 2029  BLCA 2295  BRCA 2173  CESC 2100  CHOL 2169  COAD 1838
DLBC 2112 ESCA 2585  GBM 4031   HNSC 1646  KICH 1972  KIRC 2251
KIRP 2346 LAML 4352  LGG 2615   LIHC 1713  LUAD 2403  LUSC 2337
MESO 2014 OV 2672    PAAD 2114  PCPG 2143  PRAD 2179  READ 1844
SARC 2329 SKCM 2072  STAD 2304  TGCT 2640  THCA 1961  THYM 2858
UCEC 2325 UCS 2697   UVM 1615
```

Candidate-lncRNA cancer degree ranges from 1 to 33, with quartiles
`[1,3,13]`; exactly 4,712 have degree >=3. The relation is outcome-free and
does not define labels. An independent audit must prove label hashes are
unchanged when this source is absent.

This is an explicitly pre-registered **transductive eligibility relation**.
It is calculated once from standardized tumour-expression measurements across
all available samples, before looking at any association outcome, label,
validation result, or test result. It is not described as a train-only
association signal. Its permitted uses are eligibility/detectability and
cancer-node identity only. The source is common to all folds and arms, and
removing it must leave candidate, patient-fold, train/validation/test label,
direction-label, and L1-baseline hashes unchanged. A paper must disclose this
transductive choice. A train-only sensitivity graph may be reported, but it
cannot be selected or tuned using test performance.

### 3.2 Fold-local coexpression contract

Use the frozen V3.1 statistical settings, but only outer-fold train patients:

- minimum 40 aligned samples;
- detection rate >=0.10 and variance >=0.01;
- at most 12,000 lncRNAs and 25,000 genes per cancer;
- covariate-residualized Spearman correlation;
- `abs(rho)>=0.20`, BH FDR <=0.05;
- at most 75 edges per lncRNA per sign;
- block size 128;
- candidate lncRNAs for that cancer and the fixed G2-union gene universe only.

Train sample counts are 6,243-6,275 per fold. CHOL, DLBC, and UCS are below
40 in every fold; KICH is also below 40 in folds 0 and 4. Their coexpression
edge count must be zero by contract, not silently filled from all-patient or
old edges. `expressed_in` still gives all 33 cancers graph identity.

The old all-patient source has 7,128,241 edges (3,620,470 positive and
3,507,771 negative). Filtering to the current candidate cancer-lncRNA pairs
and the 21,981-gene G2 union leaves 7,075,800 edges (3,579,946 positive,
3,495,854 negative) across 32 cancers. This is an estimate only, never an
input. The strict per-fold upper bound from 75 edges per sign is about
10.2-10.5 million edges after excluding cancers with <40 train samples; the
expected range is 4-10.5 million. Preparation capacity is budgeted to the
upper bound. The audit reports positive/negative counts, lncRNA source
coverage, and cancer coverage per fold; no missing sign is imputed.

The old canonical graph builder wrote one `coexpressed_with` relation with
`weight=abs(rho)` and did not preserve `direction`. Thus the 3,507,771 old
negative rows have already lost their sign before preparation. Old canonical
coexpression edges are forbidden even as a shortcut: fresh outer-train rho is
the only source from which positive/negative typed relations may be emitted.

`edge_chunk_size=250000` is the cap on **variable fold-local coexpression
source rows only**. It is not a cap on all graph rows or directed messages.
Coexpression rows use source/sign-stratified weight ordering and a complete
cycle over all retained rows. No destructive offline relation cap is
permitted. Every runtime chunk is the union of the complete resident static
backbone in section 7 and one coexpression chunk. A fold with no coexpression
still has one backbone-only chunk.

The current V3.2 trainer never calls `runtime_bundle_for_step`; therefore it
would effectively train on only the first scheduled chunk. It is not
compatible with this complete G0 and is forbidden until a chunk-aware training
loop, complete-coverage validation, and the reachability audit in section 6
are independently tested.

## 4. G1/G2 increments and signed/symmetric handling

| Variant | Added typed relation | Selection and transform |
| --- | --- | --- |
| G1 | `lncRNA -> binds_protein -> protein` | candidate lncRNA; only global rows (`cancer_id IS NULL`, `is_context_specific=false`); exact-key dedup by maximum weight |
| G1 | `protein -> encoded_by -> gene` | static UniProt mapping; exact-key dedup |
| G2 | `protein <-> physical_interaction <-> protein` | STRING; canonical `(min,max)` pair, reject self-loop, dedup by maximum weight, then exactly two reciprocal messages sharing one relation type |

Static counts are: 349,608 signed gene-pathway rows; 2,114 nonzero hierarchy
rows (21 zeros excluded); 623,207 global binding rows; 20,008 protein-gene
rows; and 77,426 canonical STRING pairs. The STRING source has zero reverse
duplicates and zero self-loops. Global binding covers 2,567 of the 8,541
candidate lncRNAs.

PPI bypasses the generic forward-plus-`rev_` materializer. For canonical pair
`(u,v,w)`, where `u<v`, it emits exactly `(u,v,w)` and `(v,u,w)` under the
single typed relation
`protein -> physical_interaction -> protein`. It must not emit a
`rev_physical_interaction` type, must not auto-reverse those two messages a
second time, and must satisfy `directed_message_count = 2 * 77,426 = 154,852`.
Both directions therefore share one HGT parameter set and cannot depend on
arbitrary lexical endpoint orientation.

The earlier preparation lower-clipped 125,194 negative `member_of` weights and
21 candidate hierarchy zero weights. It also omitted all coexpression and
`expressed_in` edges. Fresh preparation must not repeat these processing
errors.

## 5. Two-layer receptive-field proof

Each HGT layer and its weighted residual aggregate one reciprocal directed
hop. The pair decoder receives both `h_lncRNA^2` and `h_pathway^2`, so a
meta-path is usable when the two radius-2 endpoint neighborhoods intersect;
the entire path does not have to terminate inside only one endpoint embedding.

| Evidence path | Length | What two layers expose to the pair decoder |
| --- | ---: | --- |
| `lncRNA - coexpression - gene - pathway` | 2 | both endpoints reach/share the gene |
| `lncRNA - protein - gene - pathway` | 3 | lncRNA reaches gene; pathway reaches protein; neighborhoods overlap |
| `lncRNA - proteinA - PPI - proteinB - gene - pathway` | 4 | lncRNA and pathway both receive `proteinB` information at radius 2 |

Thus two layers are sufficient for the pre-registered estimand of one direct
PPI bridge. Paths containing two consecutive PPI edges (length >=5) are not
the G2 estimand and must not be claimed. If that estimand is later expanded,
one depth must be selected for all G0/G1/G2 arms using validation only, with a
shallower-model tie break; test results cannot choose depth.

## 6. Counterexamples and canonicalization tests

With the old single `member_of` relation, changing `+w` to `-w` leaves the
HGTConv topology result exactly unchanged because HGTConv consumes only edge
index/type. If the weighted projection is zero, the complete layer is also
unchanged. With two equal initial gene embeddings, swapping `+w/-w` between
the genes also leaves the old signed weighted sum at zero. Therefore preserving
a negative scalar alone is not a sufficient signed-topology contract.

The fresh audit must include two executable toy counterexamples:

1. flipping an edge sign moves it between positive/negative relation types,
   changes the signed-relation hash, and changes a deterministic toy encoder
   output by more than a registered epsilon;
2. randomly permuting input edge rows is canonicalized back to the same typed
   order, graph hash, and encoder output. Sign assignment permutation across
   endpoints must not canonicalize to the same signed graph.

It must additionally include these independent design oracles:

3. one canonical self-type PPI pair materializes as exactly two reciprocal
   messages under one relation type; the test rejects four messages, a
   `rev_physical_interaction` type, a self-loop, or different parameters by
   lexical orientation;
4. a two-chunk toy has a complete resident static backbone in both chunks.
   Each variable coexpression edge has its two-hop
   `lncRNA-gene-pathway` completion in the chunk that owns it; the same
   three-hop `lncRNA-protein-gene-pathway` and four-hop
   `lncRNA-proteinA-proteinB-gene-pathway` paths exist in every chunk;
5. shuffling coexpression row order or chunk execution order leaves the static
   path hash unchanged and, after canonical float64 logit aggregation, changes
   the final prediction by no more than a registered numerical epsilon;
6. G0/G1 typed masks yield empty edge-index/weight tensors for masked G1/G2
   relations, zero messages and zero weighted aggregate from those relations,
   while metadata, parameter names/counts, initialization values, node tensors,
   optimizer-step budget, and RNG draw budget match G2;
7. splitting the same validation rows into different batch sizes or row orders
   yields the same global nnPU risk, direction BCE numerator/denominator,
   shrinkage numerator/denominator, aggregate logits, and winner decision.

These are test specifications and independent oracles only. Passing a
self-contained toy does not mark the production materializer or trainer ready.

## 7. Fair capacity, resident backbone, and runtime estimand

Within each outer fold, all variants use the same G2-union node universe,
canonical node ordering, and full signed relation schema. Empty relation slots
remain present in G0/G1 so parameter count and initialization draw count do not
change by variant.

`node_x` is computed once from the complete common G0 base for that fold
(`expressed_in`, train-only signed coexpression, signed membership, and
hierarchy). The exact same tensor is supplied to G0/G1/G2 and every runtime
chunk. G1/PPI degree cannot enter G0 through node features; arm-specific degree
cannot confound the edge ablation. Audits require identical `node_x` SHA-256
across arms and prove that removing G1/G2 edges does not alter it.

### 7.1 Complete static backbone in every chunk

The G2 master backbone is resident in every chunk:

| Static source rows | Rows | Directed runtime messages |
| --- | ---: | ---: |
| `expressed_in` | 76,734 | 153,468 |
| signed `member_of` | 349,608 | 699,216 |
| nonzero hierarchy | 2,114 | 4,228 |
| global binding | 623,207 | 1,246,414 |
| protein-gene encoding | 20,008 | 40,016 |
| canonical PPI pairs | 77,426 | 154,852 |
| **total resident backbone** | **1,149,097** | **2,298,194** |

A full variable coexpression chunk contributes at most 250,000 source rows and
500,000 reciprocal messages. Peak G2 residency is therefore 1,399,097 source
rows and 2,798,194 directed messages. The static rows are not divided among
chunks. This is the correction to the old round-robin schedule that could put
binding, PPI, encoding, and membership segments of one path in different
runtime graphs.

Coexpression rows are canonically ordered by cancer, source, sign, magnitude,
and stable ID, then balanced across `K=ceil(N/250000)` chunks so chunk sizes
differ by at most one. This retains every row and avoids a tiny final chunk
receiving the same ensemble weight as a full chunk. With the expected
4-10.5-million fold edge range, `K` is 16-42, not 47. If `N=0`, `K=1` and the
single chunk is backbone-only.

The G2 backbone tensors and full metadata schema are instantiated for every
arm. G0 masks binding, encoding, and PPI; G1 masks PPI; G2 masks none. Masking
means empty edge-index and edge-weight tensors, not zero weights on live edge
indices: HGT topology must receive no masked message. Relation modules and
initial parameter values still exist in all arms. Dropout masks must be drawn
from an arm-independent node/layer/step schedule, and audits compare RNG state
before and after matched steps so an edge mask cannot change the random-number
budget.

### 7.2 Candidate-batch x chunk training schedule

Let `B` be the number of immutable, stratified candidate batches and `K` the
master chunk count for a fold. The same candidate-batch manifest, row order,
chunk order, and seeds are used by G0/G1/G2. Training retains minibatch nnPU
risk because its positive/unlabelled clamp is nonlinear; consequently batch
composition itself is an authority and cannot be regenerated per arm.

The patient-split authority is separate from model initialization. The split
seed/manifest is frozen once; model comparison uses the same pre-registered
three initialization seeds `{20260726, 20260827, 20260928}` for G0/G1/G2,
hierarchical gating, and the external router. No arm may keep its best seed.
Validation reports paired fold-by-seed differences and a pre-registered
interval/aggregate; the locked test evaluates the already selected ensemble.
This triples the one-seed training budget and must be included in the target
CUDA wall-time pilot. Historical one-seed reports are not stability evidence.

One candidate pass `s` visits every batch exactly once. Canonical batch `i` is
paired with chunk `pi[(i+s) mod K]`, where `pi` is a frozen fold/epoch chunk
permutation. A full-coverage epoch comprises `K` candidate passes,
`s=0..K-1`. Thus every candidate key is evaluated once against every chunk,
every chunk gets the same candidate exposure, and no candidate/chunk pair is
omitted or duplicated. This exact Cartesian supercycle is the formal estimand;
the current code's single traversal of `train_batches` is not an edge coverage
cycle. If its measured cost is unacceptable, a different unbiased estimator
must be separately pre-registered and compared against this oracle before use.

**2026-09-01 schedule amendment.** The exact Cartesian implementation remains
the `exact_cartesian_v1` oracle and the default when no alternative is named.
Paid runs may explicitly pre-register
`runtime_profile.candidate_chunk_schedule_mode=balanced_cyclic_single_pass_v1`.
In that mode, one training cycle selects exactly one of the existing Cartesian
passes: every candidate batch is visited once, and every runtime chunk receives
either `floor(B/K)` or `ceil(B/K)` candidate-batch exposures (the current
authority has `B=403 >= K=29`). Cartesian pass offsets are placed in a frozen
hash permutation derived only from seed and outer fold; cycle `c` selects
offset `rho[c mod K]`. Thus the first `K` completed cycles contain every
candidate-batch/chunk pair exactly once, while any individual cycle is a
balanced stochastic/quasi-Monte-Carlo estimator of the uniform chunk-marginal
training objective, **not** exact per-cycle Cartesian coverage. G0/G1/G2 use
the same batch manifest, chunk IDs, offset permutation, cycle schedule and
optimizer boundaries. The complete schedules for all configured cycles,
including mode and cycle count, are serialized into one SHA-256 authority;
resume is allowed only at a complete mode-specific cycle boundary. Inner
validation remains the exact all-chunk weighted aggregation in section 7.3.
This amendment changes neither patient-fold membership nor access to sealed
test labels.

**2026-09-01 group-shared execution amendment.** A second explicit candidate
mode is registered as
`balanced_group_latin_shared_encoder_v1`, with gradient algorithm
`SINGLE_SHARED_ENCODER_CONVENTIONAL_GROUP_BACKWARD_V1`. Canonical candidate
batches are frozen into consecutive optimizer groups of width `A` (the final
group may be shorter). For cycle `c` and group `g`, every batch in the group is
assigned chunk `pi[(g + rho[c mod K]) mod K]`. The current `B=403`, `A=4`
authority therefore has 101 optimizer groups: 100 groups of 32,768 rows and a
final `(400,401,402)` group of 23,200 rows. One optimizer group materializes
one graph, calls `encoder.encode` once, applies the decoders in original batch
order, evaluates the unchanged global nnPU/direction/shrinkage loss once over
the complete group, and calls backward once. A shared-mode OOM is a typed,
fail-closed error; falling back to fragment replay would change the registered
algorithm and is forbidden.

The first `K` completed group-Latin cycles still cover every candidate-batch /
chunk pair exactly once. That coverage fact does **not** make the per-step
objective exact or unbiased. Group-Latin deliberately changes optimizer-group
composition from the legacy mixed-chunk Cartesian oracle and is therefore a
biased estimator relative to that oracle. It must not be described as the old
oracle, an unbiased replacement, or a production winner. Before production
selection, the real prepared authority must compare it with the legacy
mixed-chunk oracle under a pre-registered acceptance contract covering loss
components and nnPU branch, gradients, optimizer updates, validation, runtime,
and memory. Merely passing synthetic equivalence and schedule tests does not
satisfy that real-data oracle gate.

Optimizer-step and gradient-accumulation boundaries are identical across arms.
For `exact_cartesian_v1`, early stopping and best-checkpoint decisions occur
only after a complete full-coverage cycle; for either explicitly registered
estimator, they occur only after its complete mode-specific estimator cycle.
Every mode then performs the same full validation aggregation, and no decision
may observe a partial cycle. A resumable checkpoint is atomic at an optimizer
boundary and stores model, optimizer, scheduler, AMP
scaler, epoch/pass/batch/chunk cursors, candidate/chunk manifest hashes,
accumulation state, and Python, NumPy, Torch CPU, and every CUDA-device RNG
state. Alternatively, only complete-epoch checkpoints are allowed. Reseeding
after interruption is forbidden. A continuous-versus-resumed equivalence test
must match parameters, logits, and schedule consumption within the registered
determinism tolerance.

**2026-09-01 formal-runtime amendment.** Both balanced schedule modes require
the literal runtime declaration
`allow_partial_candidate_chunk_rotation: true`; missing or non-Boolean values
fail closed, while the default `exact_cartesian_v1` behavior is unchanged. The
formal trainer checkpoint schema is
`CC_HHGT_V3_2_FULL_TRAINING_STATE_V4_GROUP_SCHEDULE`; the prior V3 streaming
schema is rejected with a typed migration error rather than being resumed under
new schedule semantics. G0/G1/G2 identity is bound across run ID, configuration,
prepared-artifact path and payload, then carried through the runtime contract,
checkpoint/resume comparison, heartbeats and terminal receipt. Resume validates
every history row and recomputes cumulative optimizer/encoder/decoder/global-
loss/backward counters before restoring model state. Training call telemetry is
explicitly scoped to optimizer groups and excludes exact validation calls.

### 7.3 Validation, test, and chunk aggregation

For each validation key, evaluation obtains one membership logit and one
direction logit from every chunk with dropout disabled. Chunk weights are
`w_k=n_coex,k/N_coex` (or 1 for the sole backbone-only chunk). In canonical
chunk-ID order, float64/Kahan summation computes
`z=sum_k(w_k*z_k)`; probability is `sigmoid(z)`. Probabilities are not averaged,
and mean embeddings are not decoded as a substitute. Optional weighted-mean
embeddings are visualization/export artifacts only and never define the score.

Every key must have exactly `K` contributions and total weight one. After
logit aggregation, the evaluator collects the complete fixed validation subset
and computes membership nnPU risk once globally; it may not average per-batch
clamped nnPU scalars. Direction BCE is a global eligible-row sum/count, and
residual shrinkage has an explicit global numerator, denominator, and weight.
This also replaces the current unweighted average of batch scalar losses, which
overweights a short final batch. Metrics and winner decisions must be invariant
to validation batch size and row order.

The same aggregation is used for sealed final test keys only after winner lock.
Chunk execution order is not allowed to change final logits beyond the
registered epsilon. Validation/test duplicate and denominator receipts are
stored per fold. No arm may choose a different chunk subset or evaluation key
subset.

## 8. Measured resource budget

Execution, if later approved, is restricted to
`./data/CancerLncAtlas`. `/public8` and `/dsk2` are near full. The server
has 502 GiB RAM (326 GiB available), 48 CPUs, and 3.3 TiB free on `${PRIVATE_WORK_ROOT}`.

A read-only BRCA/PF00 probe used the frozen numerical kernels on 657 outer-train
patients, 2,173 lncRNAs, and 14,754 genes. It performed 32,060,442 tests and
selected 295,813 edges (154,595 positive; 141,218 negative). With every
numerical thread fixed to one, wall time was 248.72 seconds and peak RSS was
4,344,048 KiB (4.14 GiB). Gene parquet load/pivot dominated at 156.99 seconds;
the BH and selection correlation passes used 18.34 and 23.69 seconds. This
demonstrates an I/O/materialization cost, not a missing-data problem.

Probe authority:
`./data/CancerLncAtlas/runtime/smoke/v32_g012_coexpression_resource_probe_20260829_r1/BRCA_PF00_RESOURCE.json`,
SHA-256 `8e3b521241c1ef96f1bd8d79e3d20d778061eecc8915e26484796cb34a68f27f`.
It explicitly records no formal preparation, persisted graph edges, training,
or predictions.

Hard preparation limits are revised to:

- initially at most four cancer workers, not eight; every worker has one BLAS/
  OpenMP/Torch thread;
- process-group RSS <=64 GiB soft budget and a 96-GiB typed-stop ceiling, plus
  a live server-available-memory gate before each worker launch;
- one outer fold at a time and bounded 128-lncRNA correlation blocks;
- 60 GiB disk reservation and >=100 GiB free-space preflight;
- atomic staging and no success marker after interruption or audit failure.

At the strict 10.5-million coexpression upper bound, there are at most 42
variable chunks. Five fold-local coexpression/schedule authorities are expected
to require roughly 5-20 GiB compressed; shared fold labels another 3.2-4.5 GiB.
The 60-GiB cap includes audit copies and serialization overhead. Any Arrow or
matrix-cache optimization that avoids repeated long-table pivoting must prove
edge/hash equivalence to the measured oracle before adoption.

A separate synthetic forward-only probe used the then-estimated 1,149,099-row resident
backbone plus 250,000 coexpression rows (2,798,198 directed messages), the
registered two layers, 96 hidden channels, two heads, and BF16. On the original
RTX 4070 Ti SUPER training environment it used 3,031,568,896 bytes peak
allocated, 3,605,004,288 bytes peak reserved, and 1.581 seconds for one forward.
Authority is
`artifacts/v32_g012_gpu_forward_resource_probe_20260829_r1/GPU_FORWARD_RESOURCE.json`,
SHA-256 `4e1631866f46896491d8b6d68740a41135e0080051d8e8781cae925f803ff4e4`.
That receipt is a superseded design-time estimate; the source-audited
1,149,097/2,798,194 counts require a new probe before training.

That GPU result is only a lower-bound capacity probe: topology is synthetic and
it excludes gradients, optimizer state, decoder batches, and retained training
activations. The COMPUTE_HOST login node has no visible `nvidia-smi`, and its
available Python lacks Torch. Therefore server GPU training remains NOT_READY.
Before authorization, the actual CUDA worker must repeat the exact-node probe
and a bounded backward/optimizer memory probe without producing a checkpoint or
prediction. No CPU fallback may be called the requested GPU comparison.

## 9. Post-selection full eligible-universe inference

The 3,300,000-row candidate table is a fixed training/fair-comparison sample,
not the website's complete universe. The 76,734 eligible
`(cancer,lncRNA)` pairs crossed with 2,135 exact pathways contain 163,827,090
combinations. The sample covers only 2.014%; each eligible cancer-lncRNA pair
has 22-62 sampled pathways (median 43). Training, validation, architecture
selection, and the locked final test remain on the unchanged 3.3-million rows.

Only after winner/configuration/checkpoint/test hashes are locked may an
independent inference process score the full Cartesian universe. It uses 656
tiles at a 250,000-pair cap, or an equivalent on-demand immutable cache. For
every unseen pair and every outer-fold model it must regenerate the L1/base
logit, conservation/context features, graph indices/availability, and any
fold-fitted transform strictly from that fold's train patients. The five
fold-derived logits are aggregated by one pre-registered rule; no full-universe
value may feed model selection or test evaluation.

The L1 contract is concrete. Its hashed identity terms can represent an unseen
pair, but its numeric inputs include train-discovery effect/FDR, detection,
sample count, and cross-cancer support/consistency. Before fresh baseline
fitting, each fold must therefore define a tiled full-eligible train-discovery
feature authority using the corrected df formula and one pre-registered
multiple-testing family over the full eligible grid. The fixed 3.3M rows used
to fit/select L1 are a subset of that authority. After winner lock, inference
applies the already locked fold L1 coefficients and hasher to unseen feature
rows; it does not refit L1. Computing FDR only on the 3.3M sample and later
changing its denominator for unseen rows is forbidden because it would break
sampled-overlap parity. Cross-cancer context is likewise recomputed from the
same fold-train full-grid effects, never copied from an old release.

The tiled path must also score all sampled 3.3-million rows and reproduce the
locked sampled inference under the same fold/ensemble rule within tolerance,
with exact key, fold, feature, and availability-mask parity. A missing tile or
uncalculated key is `NOT_EVALUATED_TYPED`, never numeric zero, `NO_DATA`, or a
biological/modality-unavailable state. Cache keys bind model, code, input,
fold, graph, endpoint, cancer, lncRNA, and pathway hashes.

If the winner enables Mutation, CNV, or ATAC, absence from the 3.3-million
sample is not modality unavailability. Each enabled head must prove it can
derive the same fold-specific callable/event/context features for unseen tiles
from raw patient assays and static pathway mapping. True assay gaps retain
their typed reason. If a head cannot meet this contract, the only permitted
fallback is a separately named, pre-registered full-universe endpoint that
excludes that modality; it cannot silently reuse the multimodal score name or
substitute zero. The sampled multimodal and full core-only endpoints remain
visibly distinct.

The full aggregate contains 163,827,090 rows and entails 819,135,450 fold-pair
evaluations if five logits are streamed. Fold logits should be reduced online
rather than stored five times. This path also needs five fold-specific
full-grid train-discovery feature authorities, so the earlier 80/120-GiB output
estimate is not a safe total-workspace bound. Provisionally reserve 250 GiB and
stop before materialization if the measured projection for feature authority,
output, index, audit, and cache exceeds 500 GiB. A representative end-to-end
tile pilot must replace both planning bounds before any full run. The proposed
authority root is
`./data/CancerLncAtlas/results/v32_full_eligible_inference_20260829_r1`;
it does not yet exist and is NOT_READY.

## 10. Mainline readiness matrix

`READY` applies only to the named deliverable and scope; it does not imply the
winner, website, or full model is ready.

| Deliverable | State | Authority path / SHA-256 | Exact scope or blocker |
| --- | --- | --- | --- |
| Fresh Mutation five-fold OOF expert | READY | `artifacts/v32_full_multitask/genomic_fresh_rerun1/mutation_cnv_typed_predictions.parquet` / `a0b4d4bfe1399dc7d34968dced86ed62d76ea76b5638236ad11694277a3a24a5`; `LINEAGE.json` / `d6004f65de2cdb0136577e53a1c8aec416edac4dea01c432904e24ec5b01c0bc` | Ready only as the fixed 3.3M expert table; future common-primary and full-unseen binding still require parity proofs. |
| Full-33 CNV five-fold OOF expert | NOT_READY | No OOF authority. Prerequisite measurement materialization: `./data/CancerLncAtlas/results/v32_routed_candidate/full33_segment_streaming_cnv_20260829_r2/SUCCESS.json` / `e03da1d6064dcb92f2092b7819b81626d8d66540af83b0acf8633bf500f29d08`; independent `AUDIT.json` / `edb1741d808d4365cf9cf9a441d7fb1c4fb3790734519226c65a879c1fa6d821` | Segment-to-patient/lncRNA/pathway mapping is complete and audited; this is not a trained fold expert. Five-fold CNV training/inference is absent. |
| ATAC r3 five-fold OOF expert | READY | `./data/CancerLncAtlas/results/v32_atac_fresh_oof_20260829_r3/fresh_patient_fold_oof_expert/atac_typed_predictions.parquet` / `ed2832f02995b713e876629c60e63afbd4f1ebc1fa502c04c0c042d263fe5751`; audit JSON / `2b3694fd09be09f99970b2dd6af85e15a25ccbc38b7c643e0fe91927bcdab663` | Fresh OOF with typed raw-coverage gaps; full-unseen tile context remains unproved. |
| Fresh G0/G1/G2 common folds and graphs | NOT_READY | No authority / no SHA | Revision 3 is design-only. Historical PTs/extractions are diagnostic-only; their declared code snapshot is unreconstructable. Label df, direction-label, signed-coexpression, safe-graph, backbone/chunk, mask, and sealed-test gates remain open. |
| Hierarchical/end-to-end routed OOF | NOT_READY | No `hierarchical_oof/SUCCESS.json`; source `cc_hhgt/v32/hierarchical_candidate_inference.py` / `02a67fe01368d220e49324abb95def43718ee6c8c98013567f00b36b77e80729` | Needs fresh common folds, full33 CNV OOF, CUDA worker, and audited chunk-aware trainer. |
| External router OOF | NOT_READY | No `external_router/SUCCESS.json`; source `cc_hhgt/v32/cancer_modality_router.py` / `5c4867be8114a9cabb94fcb5954ea0ef32a9d70e7db3c537299aa0ec08d35a98` | Code exists, but canonical common-primary binding and required modality OOF inputs are incomplete. |
| Fair hierarchical-vs-external winner | NOT_READY | No `fair_comparison/SUCCESS.json` / no SHA | Cannot compare until both OOF outputs share candidate, folds, labels, modality availability, budget, and sealed evaluation contract. |
| Full/on-demand 163.8M inference | NOT_READY | No output authority. Eligibility input `artifacts/input/formal_gdc_threshold/full_gdc_lncrna_detection_rates.tsv.gz` / `69241435b1326b99c177ac70f9497b7e25310b1eb898555f7e418c9ecce7be52`; sampled candidate / `cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f` | Needs winner lock, unseen L1/context generation, modality-head coverage or a separately named core-only endpoint, overlap parity, and tile resource pilot. |

The per-cancer expansion of these scoped states is
`docs/v32_g012_33_cancer_status_matrix_20260829.tsv`. It deliberately keeps
CNV measurement materialization separate from CNV OOF training and keeps the
fixed 3.3M Mutation expert separate from future common-primary binding.

The patient-fold input-specific audit is PASS:
`artifacts/v32_patient_fold_unit_audit_20260829_r1/AUDIT.json`, SHA-256
`9ba6f7c4d09f3dd5d46fe01651817d89d6b619b868b55488740181683fe8dd70`.
It proves the present one-sample-per-patient cohort has no observed cross-fold
patient leakage; it does not waive the patient-ID builder correction.

## 11. Gates before any preparation or training

Preparation is not authority until an independent process writes a semantic
PASS receipt binding:

- all source hashes and the exact activity-subset proof;
- current code/config/runtime/task hashes and an exact patient-ID fold contract;
- the independently verified correlation-df formula and freshly regenerated
  discovery/replication/FDR/label/L1 hashes;
- train-derived input direction separated from validation/sealed-test
  replication direction, with held-out feature-exclusion tests;
- a training authority containing no test labels/logits/metrics and a sealed
  evaluator inaccessible before winner/checkpoint lock;
- fresh candidate, patient-fold, common-feature, and split-label hashes;
- 8,541/4,712 scope counts and per-scope candidate/evaluation coverage;
- exact `safe_graph` role/relation allowlists and negative gates;
- per-fold coexpression train-sample, threshold, signed-rho provenance, source
  coverage, edge, and balanced runtime-chunk audits;
- G0/G1/G2 node/schema/active-edge/polarity/direction/duplicate/self-loop and
  availability audits;
- the source-audited 1,149,097-row resident backbone and exact 2,298,194/2,798,194 directed
  message counts, including the two-message one-relation PPI rule;
- 33-cancer transductive `expressed_in` degree/distinguishability checks and
  proof that removing it leaves every candidate/label/L1 hash unchanged;
- identical G0-derived `node_x`, node universe, relation schema, parameter
  initialization, optimizer budget, batch authority, chunk schedule, and RNG
  budget across arms; masked relations must produce zero messages;
- the split seed is bound separately from the shared three-initialization-seed
  authority, with all arms reporting the same paired fold-by-seed population
  rather than selecting a best seed;
- exact candidate-batch x chunk full-cycle coverage, resume equivalence, and
  global validation nnPU/direction/shrinkage aggregation invariance;
- all executable counterexample/reachability/order tests in section 6;
- measured preparation and actual-target CUDA training probes within gates;
- `old_prepared_loaded=false`, `old_checkpoint_loaded=false`,
  `old_predictions_used=false`, and `training_run=false`.

Only that receipt may authorize a later fair G0/G1/G2 training stage. Full
eligible-universe inference has its separate post-selection gates in section 9.
