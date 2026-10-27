Paired "with multi-omics" versus "without" comparison
=====================================================

Status: code implemented and fixture-verified on host 149. **No arm has been
trained**, so no effect is reported. The refusal is part of the contract, not a
missing step in the bookkeeping.

What the module does
--------------------

``cc_hhgt/v32/omics_ablation_comparison.py`` has three layers.

**Arm identity.** ``validate_arms`` takes exactly two ``Arm`` records, each
declaring ``arm_id``, ``omics_node_input``, the SHA256 of the candidate table,
label vector, fold assignment and training budget, the initialisation seed and
an optional trained checkpoint hash. It refuses anything that is not a
single-variable comparison:

* two arms only, distinct identifiers, exactly one with omics input;
* identical candidate, label, fold and budget digests;
* identical initialisation seed;
* well-formed 64-character lowercase hexadecimal digests.

A refusal here means the two arms cannot be compared, no matter what the scores
say.

**Estimator.** ``paired_difference`` computes per-cancer average precision and
ROC AUC, the overall delta average precision, a paired *stratified* bootstrap
interval (resampling within cancer, the same resampled indices for both arms),
an exact sign test over pairs, and a paired sign-flip permutation test. It is
verified on synthetic scores, including a planted improvement and a null case.

**The reporting gate.** ``compare_arms`` is the only entry point that would
report an effect. If either arm declares no trained checkpoint it returns
``status = REFUSED_UNTRAINED_ARMS`` with ``comparison_performed = false`` and
``effect_reported = false``. Even when both arms declare a checkpoint, the
record carries ``external_validation = false`` and states that an internal
paired comparison is not external validation.

Why the gate matters
--------------------

The tempting shortcut is to score the existing G2 model twice - once with the
omics block attached and once without - and call the difference the effect of
multi-omics. That would be wrong: the current checkpoint never saw those input
columns, so the second arm is reading random weights on new channels. The gate
refuses exactly that with ``ABLATION_REQUIRES_TRAINED_ARMS``.

Fixture evidence
----------------

1. average precision matches the step-wise definition on a hand-computed example;
2. ROC AUC handles perfect, reversed and fully tied orderings;
3. eleven unpaired or ambiguous arm declarations are refused with distinct codes;
4. identical scores produce a zero delta, a ``[0, 0]`` interval and ``p = 1``;
5. a planted improvement produces a positive delta, a bootstrap interval
   excluding zero, an exact sign-test ``p < 1e-6`` and a permutation
   ``p < 0.01``;
6. the estimator is deterministic for a fixed seed, and refuses length
   mismatch, non-finite scores and too-small resampling;
7. ``compare_arms`` refuses untrained arms, refuses missing scores once trained,
   and only then computes a paired estimate.

Non-claims
----------

* No trained multi-omics arm exists; the reported state is a refusal.
* A paired internal comparison, even when it exists, is not external validation
  and does not support a biological claim on its own.
* The estimator says nothing about whether the difference would replicate in
  another cohort.
