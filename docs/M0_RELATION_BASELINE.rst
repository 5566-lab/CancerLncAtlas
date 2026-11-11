M0: within-cancer linear prediction of exact pathway relationships
=================================================================

Endpoint and inputs
-------------------

Each row is cancer x lncRNA x exact pathway. M0 consumes the same verified
CCHHGT prepared ``label``, ``available``, ``direction_label``,
``direction_available`` and weights. It does not recalculate thresholds.
This relation endpoint differs from the historical patient-score regression.

For each cancer the default model is

.. math::

   \operatorname{logit}\widehat P(Y_{c,l,k}=1)
       = b_c + \beta_c^\top u_{c,l,k}^{\mathrm{source}}.

The numeric vector is the existing prepared source-patient block. Its first
five columns are source rho, absolute rho, negative log10 source q, log(1+n),
and source availability. Subsequent columns contain the declared expression,
CNV, mutation and other context. Source patients and label patients are
separate; label-patient rho, p and q are not predictors. This evaluates whether
an association observed in discovery patients reproduces in independent
patients. It is not an expression-only model without association information.

Clinical information and purity retain the patient-stage adjustment recipe
used to produce the prepared labels. They are not fabricated as relation-level
covariates. Missing labels stay unavailable.

The default ``--identity-mode numeric_only`` does not add lncRNA/pathway IDs
to the design matrix. ``--identity-mode exact`` is a separate ablation adding
the existing sparse identity terms c, l, t, c_l, c_t and l_t. Its vocabulary
is fitted only on available training rows. Both arms use a training-only
numeric scaler. L1 logistic regression is the default; ``--penalty l2`` uses
L2 logistic regression. Direction uses its own availability mask and head.

Scope and validation
--------------------

* Default cancers are BRCA, LUAD and COAD, fitted separately.
* Only active ``analysis_role=primary_pathway`` targets enter the main task.
  A TF or kinase displayed as Pathway does not automatically become a main
  prediction target or receive its loss weight.
* Filtering precedes scaling, vocabulary fitting, training and model selection.
* The prepared ``RELATION_TARGET_POLICY.json`` pins the policy and source-ID
  map under ``relation_target_policy/``. M0 verifies these checksums and uses
  the sealed map automatically. A supplied alternative must match. Legacy
  root-level ``TARGET_POLICY.tsv`` bindings are also checked.
* Membership C is selected by validation AP; direction C by validation BCE.
  Per-target metrics retain exact pathway IDs.
* The runner never opens sealed test labels. Patient folds, source/label
  disjointness, arrays and source lineage are verified by the existing
  prepared validator.
* One-to-one source-to-canonical ID renames require an explicit mapping table.
  Many-to-one identity changes require rematerialized labels, not renaming.

The fused FunctionalRelationContext is a different label contract: unsupported
relations are unlabelled and its continuous effects are metadata. That directory
cannot be supplied to this binary runner by setting unlabelled rows to zero or
substituting an effect column. M0 and CCHHGT comparisons use the same prepared
endpoint; a future continuous objective must be adapted in both models.

Comparison with CCHHGT
----------------------

M0 matches the numeric block, candidates, labels, masks and patient splits.
CCHHGT additionally uses graph-node inputs and context that are not fully
flattened into this numeric block. A difference therefore measures the full
model increment, not graph topology alone. Isolating topology requires an arm
with identical node inputs and no message passing.

Preparation keeps exact identity terms for old configurations without an
``identity_mode`` field. New configurations may explicitly set
``baseline.identity_mode=numeric_only``; the shared baseline design and batched
prediction interfaces then omit every identity column. Preparation may pool
cancers; this standalone M0 fits cancers separately. Its predictions must not
be silently substituted for the prepared graph model's baseline logits. To
compare residual models strictly, prepare and train both arms with the same
baseline mode and cancer-sharing design, or use one cancer per capsule.

Running the demo
----------------

Use a CCHHGT capsule prepared with the fixed backbone and explicit target
policy, on server 149. Each capsule contains one outer fold; do not concatenate
training arrays from different outer folds.

.. code-block:: bash

   /dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python scripts/run_m0_relation.py \
     --prepared /path/to/verified_scoped_prepared \
     --policy /path/to/TARGET_POLICY.tsv \
     --out /path/to/new_m0_relation_directory \
     --cancers BRCA LUAD COAD --penalty l1 --identity-mode numeric_only

For an explicit identity ablation, rerun into a new output directory using
``--identity-mode exact``. Optional ``--identity-map`` TSV columns are
``source_target_id`` and ``canonical_id``.

Outputs include coefficients, fitted scaler, vocabulary, validation predictions,
exact-target metrics and ``M0_RELATION_RECEIPT.json``. The receipt records
policy/prepared/input hashes, folds, label recipe, numeric source semantics
and identity mode. It does not claim that graph node inputs were matched.

Verification
------------

``tests/test_m0_relation.py`` checks exclusion before fitting; display/analysis
role separation; rejection of many-to-one label renaming; masked-label isolation;
per-cancer fitting; sealed-test isolation; whitelisted feature loading; rejection
of implicit fused-to-binary conversion; identity-invariant numeric-only fitting;
explicit exact-identity features; bound-policy consistency; and mode-specific
comparison receipts.
