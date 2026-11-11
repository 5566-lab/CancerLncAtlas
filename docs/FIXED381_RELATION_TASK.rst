Fixed381 relation task
======================

The primary task is cancer x lncRNA x exact pathway. The backbone freezes the
381 historically scoreable KEGG_MEDICUS source definitions, including their
NETWORK N identifiers and original members. Official classification supports
122 of these entries as Cancer / Cancer signaling. That annotation does not
filter the user-fixed 381. Reference and Variant identities remain distinct.

Catalog and supervision
-----------------------

TARGET_MASTER roles describe display categories. TARGET_POLICY separately
specifies analysis_role, loss_group, loss_weight and active_in_analysis.

* All 381 backbone entries are primary_pathway, with Pathway weight 1.0.
* TF, kinase and PROGENy entries display as Pathway and supply mechanism
  context by default. Their category does not create an independent loss.
* Genuine State definitions remain state_auxiliary, with State weight 0.5.
* Protein overlap supports context links; it does not assert identity equality
  or create additional primary targets. Zero-overlap context is excluded.
* Mixed identities retain source-specific roles in SOURCE_TASK_ROLES. For
  example, PROGENy Hypoxia is Pathway and Buffa Hypoxia is State.

Preparation and label views
--------------------------

build_kegg381_backbone.py verifies the historical membership snapshot.
build_relation_catalog.py builds the explicit analysis policy.
build_functional_target_view.py projects existing fused labels without changing
values, missingness, unlabelled states or the original p/q correction family.
configure_kegg381_demo.py writes separate BRCA, LUAD and COAD configurations.

Prepared data seal the policy, identity map, backbone and hashes. Lineage
validation checks the canonical axes. Demo configurations retain all 381
pathways and enforce within-cancer preparation.

Each fused-label context contains 381 primary columns and 194 existing State
columns. Three other State definitions lack a separate source label column.
LABEL_IDENTITY_MAP explicitly selects original KEGG columns. For GPI, Mismatch
repair and NHEJ, identity redirects change output identifiers without replacing
the original labels with Reactome labels.

M0 comparison
-------------

The default per-cancer relation model is::

    s(c, l, k) = b(c) + beta(c)^T u_source(c, l, k)

Source features come from the source-patient partition. Label-side rho/p/q
values are not features. numeric_only omits lncRNA/pathway identity terms;
exact identity terms are a separate ablation. Existing binary prepared labels
use sparse logistic regression with L1 regularization; L2 is optional. This is
not a Ridge patient-score reconstruction task.

M0 and HHGT share target policy, folds, masks and Y definition. The full HHGT
comparison also adds node information. A separate no-message-passing model
with identical node inputs is needed to isolate message-passing benefit.

Current fused labels use membership=0 for unlabelled relations, not negatives.
Their use requires a shared PU or other explicitly defined objective in M0 and
HHGT; the legacy binary objective must not silently reinterpret these values.
Multiple State readouts mapping to one identity also require an explicit
aggregation rule. Existing checks reject ambiguous duplication and old-label
fallbacks. Scoped configuration files alone do not constitute a real training
run.
