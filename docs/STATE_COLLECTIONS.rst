State collections and component observations
===========================================

The primary State list contains 48 entries: one cellular-composition collection,
one SBS mutational-signature spectrum and 46 existing State entries. Together
with 476 primary pathways, the main catalog contains 524 entries.

The cellular-composition collection contains 91 source identities. Its detail
page keeps cell types and estimation methods separate. These identities link to
96 original readout definitions; enrichment scores and fractions retain their
source units. The SBS collection contains 60 separate mutational signatures.
Grouping changes the navigation and observation ownership, without averaging,
rescaling or deleting measured values. Source-specific missingness is retained.

``STATE_TOP_LEVEL_TARGETS.tsv`` defines the 48-entry State list.
``STATE_GROUP_MEMBERSHIP.tsv`` records 151 ``component_of`` relationships.
``DISPLAY_TARGETS.tsv`` defines the 524-entry combined list.
``State目录.json`` supplies the State navigation tree and source methods.
Original component identities remain resolvable in ``STATE_DETAIL_TARGETS.tsv``;
they are biological components, not equivalent-name aliases of the parent.

The measured registry routes component observations to their parent and records
``state_component_ids`` on each link. The original component links and targets
remain in ``STATE_CHILD_OBSERVATIONS.tsv.gz`` and ``STATE_CHILD_TARGETS.tsv``.
The existing ``load_target`` accessor returns each collection as an original,
multidimensional patient-observation matrix, with observed masks and source
metadata. Shared observations are linked once per parent and retain all origins.

Classification does not add independent losses. Pathway and State task weights
remain 1.0 and 0.5. M0 and graph-based models must use the same unique observation
dimensions, masks and patient partitions; a parent entry is not a new scalar
label or an additional observation alongside its components.
