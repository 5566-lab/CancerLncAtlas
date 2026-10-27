Unvalidated binding candidates with held-out evaluation
======================================================

Status: code implemented and fixture-verified on host 149. No real binding
evidence was scored, no acquisition ran, and no candidate is claimed to bind.

What "discovery" can honestly mean here
---------------------------------------

An unfitted score over declared features cannot establish that a lncRNA binds a
protein. What it can do is produce a ranked candidate list whose *evaluation* is
honest: fixed candidate budget, held-out known edges, no leakage, and an
explicit comparator. ``cc_hhgt/v32/novel_binding_discovery.py`` implements that,
and refuses the cases where the evaluation would mean nothing.

The pipeline
------------

1. **Split.** Known edges of the declared evidence arm are split by
   ``SHA256(seed, lncRNA, protein)`` into a scored partition and a held-out
   partition. The split is a pure function of the seed and the edge identity: no
   label, degree or score participates, so the held-out set cannot be chosen to
   look good. A holdout smaller than ``minimum_holdout``, or a realized fraction
   further than ``0.25`` from the declared one, is refused.
2. **Universe.** The candidate universe is the product of the node identifiers
   present in the known graph minus every *scored* edge. Held-out known edges
   stay in the universe unmarked - they are the signal the evaluation must find.
3. **Score.** A declared additive score
   ``sum_f w_f * (x_f - center_f) / scale_f`` with frozen weights and frozen
   standardisation. Nothing is fitted. A missing feature contributes exactly
   zero and is listed per candidate, so no candidate can be promoted by an
   imputation.
4. **Evaluation.** At the fixed budget: precision@budget, recall@budget, average
   precision (the same step definition used by the ablation comparison module),
   and the same three numbers for a degree-prior baseline whose degrees come
   from the scored partition only.
5. **Emission.** ``budget`` rows, each tagged
   ``candidate_status = "unvalidated_candidate"`` and
   ``evidence_required = "orthogonal_experimental_validation"``.

Refusals
--------

``DISCOVERY_CONFIG_TYPE``, ``DISCOVERY_WEIGHTS_MISSING``,
``DISCOVERY_WEIGHTS_NOT_FINITE``, ``DISCOVERY_PRIMARY_FEATURE_UNWEIGHTED``,
``DISCOVERY_HOLDOUT_FRACTION_INVALID``, ``DISCOVERY_BUDGET_INVALID``,
``DISCOVERY_MINIMUM_HOLDOUT_INVALID``, ``DISCOVERY_STANDARDISATION_MISSING``,
``DISCOVERY_STANDARDISATION_INVALID``, ``DISCOVERY_KNOWN_COLUMNS_MISSING``,
``DISCOVERY_KNOWN_EDGE_DUPLICATE``, ``DISCOVERY_KNOWN_EDGES_EMPTY``,
``DISCOVERY_HOLDOUT_TOO_SMALL``, ``DISCOVERY_SCORED_PARTITION_EMPTY``,
``DISCOVERY_HOLDOUT_FRACTION_DRIFT``, ``DISCOVERY_UNIVERSE_EMPTY``,
``DISCOVERY_FEATURE_COLUMNS_MISSING``, ``DISCOVERY_LABEL_LEAKAGE`` (any feature
column named ``label``, ``is_known``, ``is_positive``, ``outcome`` or
``known_edge``), ``DISCOVERY_FEATURE_ROW_DUPLICATE``,
``DISCOVERY_PRIMARY_FEATURE_ABSENT`` (no candidate carries the primary feature),
``DISCOVERY_BUDGET_EXCEEDS_UNIVERSE``, ``DISCOVERY_HOLDOUT_EMPTY``,
``DISCOVERY_PARTITION_OVERLAP``, ``DISCOVERY_SCORES_NOT_FINITE``.

Fixture evidence
----------------

The fixture plants a latent rule (high motif affinity **and** an RNA-binding
domain) on a 20 x 15 identifier grid, samples the known edges from that rule,
and checks that:

1. the split is deterministic, seed-sensitive and stable in size; duplicates,
   missing columns and a too-small holdout are refused;
2. the universe excludes scored edges and keeps every held-out known edge;
3. the score equals the declared weighted sum, a missing feature contributes
   zero and is recorded, and label columns, duplicate rows or a wholly missing
   primary feature are refused;
4. fixed-budget precision, recall, average precision and the degree-prior
   baseline match hand-computed values on a four-candidate example;
5. end to end, average precision exceeds the holdout prevalence by more than
   0.4, precision@budget is at least three times the prevalence, and both lifts
   over the degree prior are positive;
6. every emitted row is an unvalidated candidate, ranks are 1..budget, and no
   emitted candidate is a scored known edge;
7. the pipeline is deterministic for a fixed seed and changes with the seed.

Non-claims
----------

* The fixture is synthetic; the planted rule is invented, so the measured power
  says nothing about real lncRNA-protein binding.
* Passing these checks means the *evaluation* is sound, not that a candidate is
  real. Every row still requires orthogonal experimental validation.
* No existing binding/PPI graph was modified, no acquisition ran, and the
  untrained October expansion edges remain untrained.
