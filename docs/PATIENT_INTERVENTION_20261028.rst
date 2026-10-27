Patient-level intervention and interaction attribution
=====================================================

Status: code implemented and fixture-verified on host 149. No optimizer, real
cohort, paid compute or deployment ran. The fixture model is an untrained graph
model built only to exercise the code path; nothing here is a patient result.

Two questions, two estimands
----------------------------

**C19 - how does the lncRNA-pathway relation change after P is inactivated?**

.. math::

   \Delta_{(L,pw)}(P) = f_{(L,pw)}(\theta, M; do(P = \text{inactivated}))
                      - f_{(L,pw)}(\theta, M; P = \text{observed})

``f`` is the frozen membership logit (``model.forward`` in ``eval`` mode, no
gradient, no optimizer). ``M`` is the graph and the observed pair features.
Exactly one omics channel of P's encoding gene changes; everything else - graph
topology, edge weights, other node values, pair features, model weights - stays
byte-identical. ``cc_hhgt/v32/patient_conditioning.py`` reports the per-pair
``delta``, ``sign_flip``, within-cancer rank movement and a per-cancer
top-k overlap summary.

This is a *model-conditional interventional contrast*. It is not a causal effect
in patients, not a prediction of a tumour, and not a biological result. The
record carries ``causal_claim: false`` and ``requires_orthogonal_validation:
true``.

**C20 - is P a mediating interaction node for that relation?**

It cannot be proven by this code, and the artifact says so: every record carries
``biological_mediation_proven: false``.

The reason is structural. A message-passing model can only move information along
edges, so "P's state affects the pair score" always provably travels through P's
outgoing edges. Removing those edges and observing that the effect disappears is
therefore a tautology, not evidence. What *is* informative is attribution and
falsification:

* ``attribution_share``: the fraction of the pair effect that disappears when
  P's edges into the specific lncRNA are removed, i.e. how much of the effect is
  carried by that declared path set rather than by other routes.
* ``rewire`` control: P's outgoing edges are redirected to random nodes of the
  same target type, keeping edge type, count and weight identical. If the pair
  effect survives rewiring, it was never about P's actual partners.
* ``placebo`` control: the same intervention is applied to a matched node with no
  path to the pair. If the pair effect appears there too, the estimator is
  measuring something global.

The verdict is one of ``NO_EFFECT``, ``VALUE_ONLY_EFFECT`` (too little of the
effect travels through the declared path set), ``NOT_INTERACTION_SPECIFIC`` (a
control failed) or ``INTERACTION_ATTRIBUTED_UNDER_FROZEN_MODEL``. Only the last
one allows the sentence "within this frozen model, P's interaction partners carry
the effect", and it still does not allow "P mediates the relation in biology".

Refusal conditions
------------------

``PATIENT_CONDITIONING_SUBJECT_SCOPE_REQUIRED`` (a cohort table cannot stand in
for a patient), ``..._SUBJECT_MISMATCH``, ``..._CANCER_MISMATCH``,
``..._UNKNOWN_MODE``, ``..._NODE_TYPE_UNKNOWN``, ``..._NODE_NOT_IN_GRAPH``,
``..._CHANNEL_MISSING``, ``..._PARTICIPANT_UNMEASURED`` (no measurement for that
patient, channel and node - never imputed), ``..._INTERVENTION_VALUE_MISSING``,
``..._NO_INTERVENTION_CONTRAST`` (counterfactual equals the observed value),
``..._VALUE_NOT_FINITE``, ``..._BUNDLE_NOT_AUGMENTED`` (multi-omics node input
must exist first), ``..._PAIRS_EMPTY``, ``..._PAIR_COLUMNS_MISSING``,
``..._PAIR_TARGET_MISSING``, ``..._PAIR_NODE_MISSING``,
``..._PAIR_FEATURE_INVALID``, ``..._BLOCK_WIDTH``, ``..._BLOCK_NOT_FINITE``,
``..._PATH_TARGET_NOT_IN_PAIRS``, ``..._GRAPH_KIND_UNSUPPORTED`` (edge surgery is
implemented for the heterogeneous graph only; the homogeneous model refuses
instead of silently skipping the controls), ``..._REWIRE_UNAVAILABLE``.

What the fixture checks show
----------------------------

On a seven-node synthetic graph (P = gene ``G1`` connected to ``L1`` and
``PW1``; a matched placebo ``G2`` connected only to ``PW2``; isolated ``L2``):

1. the intervention writes exactly one block cell and leaves the observed flag
   set - a declared counterfactual, not a missing assay;
2. the conditioned bundle is restored bit-identically after the contrast;
3. the pair ``(L1, PW1)`` changes while ``(L2, PW2)`` does not, so the reported
   delta follows graph connectivity rather than a global shift;
4. the null contrast (``mode="observed"``) produces exactly zero effect;
5. the attribution run on the connected graph passes both controls and reports
   ``INTERACTION_ATTRIBUTED_UNDER_FROZEN_MODEL`` with
   ``biological_mediation_proven: false``;
6. the same run with a null contrast reports ``NO_EFFECT``.

Non-claims
----------

* The fixture model is untrained and synthetic; the sign and size of any delta
  here are meaningless outside the test graph.
* No patient result exists: no real patient table, no trained multi-omics model.
* No mediation proof exists and none can be obtained from a deterministic
  message-passing model with these data.
* The rewiring and placebo controls are necessary conditions, not sufficient
  ones: passing them rules out two artefacts, it does not establish biology.
