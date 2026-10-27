Recorded SC-HHGT LASSO evidence to model-input derivation
==========================================================

Status: code implemented and fixture-verified on host 149. No LASSO, PCA or
neural parameter fitting ran; no real cohort, acquisition, paid compute or
deployment ran.

What the model consumes
-----------------------

A completed episode payload carries:

* ``lasso_response_summary`` - the original ``BatchedLassoResult.response_frame()``
  columns (``response_index``, ``response_id``, ``available``,
  ``unavailable_reason``, ``selected_alpha``, ``inner_cv_mse``,
  ``inner_null_mse``, ``bootstrap_successes``);
* ``sparse_lasso_edges`` - the localized ``lnc_index`` / ``gene_index`` /
  ``coefficient`` / ``stability`` tensors that the HGT model actually reads;
* ``node_global_indices``, ``pair_features_by_arm`` and ``row_metadata``.

Until now nothing re-derived those tensors: a payload could keep self-consistent
hashes while its pair features, node set or coefficients had been rewritten.

What is now replayed
--------------------

``cc_hhgt/v32/single_cell_lasso_replay.py`` rebuilds, from the recorded values
plus the pinned static graph assets only:

1. row identities (``row_metadata.lncrna_id`` / ``pathway_id`` mapped onto the
   static node universe) and hence the response universe;
2. response identity - the response order must equal the unique row lncRNA order
   and every recorded ``response_id`` must be that node's canonical identifier;
3. the coefficient/stability edge set: sorted by ``(response, gene)``, unique,
   finite, ``|coefficient| > 0``, ``0 <= stability <= 1``, and every edge must
   belong to an available response;
4. ``local_node_indices`` - the lncRNA, gene, pathway and pathway-family
   candidate nodes, exactly;
5. ``lasso_pair_features`` - the three LASSO-derived pair-feature columns
   (pathway stable support, inner relative MSE gain, availability);
6. ``localized_lasso_edges`` - the sparse edge tensors the model consumes.

`lasso_pair_features` and `localized_lasso_edges` are the original episode
functions, so the comparison is against the same implementation the materializer
called. ``without_lasso`` must zero exactly the masked columns and preserve the
three base columns.

Tolerance and the stability boundary
-----------------------------------

The payload stores coefficients and stability as float32 while the materializer
computed the features from float64, so float comparisons use ``atol = 1e-6``;
integer indices, node identities and edge ordering are compared exactly.

A recorded stability value within ``1e-6`` of ``lasso_stability_threshold`` can
legitimately change the support mask by one float32 rounding step. Those edges
are counted as ``boundary_stability_edges``; if the comparison fails while such
an edge exists, the replay raises a distinct
``SC_LASSO_REPLAY_BOUNDARY_STABILITY`` error instead of reporting a content
mismatch. Resolving it requires re-materializing the episode with float64
evidence, not editing the payload.

Where it is enforced
--------------------

``native_single_cell_model_sources.inspect`` calls ``verify_episodes`` before the
model-use inventory is finalised and registers the result as a
``NATIVE_SC_LASSO_MODEL_INPUT_REPLAY_V1`` transform replay covering every
recorded episode and every outer fold. A mismatch aborts the inspection, so a
signed single-cell expert cannot be bound to a payload whose LASSO-derived model
inputs do not reproduce.

What is explicitly *not* claimed
--------------------------------

* The recorded coefficients are never re-solved. KKT/objective consistency and
  the provenance of the fit itself (who fitted what, on which rows) remain open
  and are declared as ``SC_LASSO_COEFFICIENT_OBJECTIVE_NOT_REFIT`` and
  ``SC_FITTED_LASSO_COEFFICIENTS_NOT_RECONSTRUCTED``.
* The bootstrap stability ratio is taken as recorded evidence; it is checked for
  domain and for downstream use, not recomputed from resamples.
* Static graph assets are pinned by hash; their own biological provenance is a
  separate, unclosed gap.
* This is same-implementation reconstruction, not independent biological
  validation and not a statement about predictive performance.
