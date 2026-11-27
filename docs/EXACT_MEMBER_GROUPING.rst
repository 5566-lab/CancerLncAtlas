Exact-member pathway observation groups
======================================

Primary pathways with nonempty, identical original-source UniProt member sets
share one observation-group identity. Equality is exact (Jaccard = 1); missing
annotations and partial overlaps do not establish a merge. Existing reference
mechanisms are preferred as representatives, followed by canonical-ID order.
Database source names and event definitions remain separately traceable.

The current backbone has 477 source mechanisms and 476 observation groups. Its
single exact-member merge joins N00017 (SMO activation) and N00010 (PTCH1 loss),
each with the same 29 proteins. The representative ID is
CLA-P-08A1A41F05A3E9ED and its display name is "Hedgehog signaling, shared
SMO/PTCH1 downstream module". N00062 has a different member set and remains a
separate reference definition.

Patient observations are unioned by observation ID. Their numeric matrices,
missingness and patient partitions are unchanged. Source-specific readout
definitions remain available. A shared member module does not require one
particular upstream event; the original SMO and PTCH1 event matrices remain
separate and unknown where evidence is absent. Module applicability does not
label either event positive.

The accepted measured-Y registry has 476 primary groups plus 197 State
identities (673 targets) for BRCA, LUAD and COAD. M0 and graph-model comparisons
must use this same accepted registry. Source definitions and aliases must not
automatically create extra losses or repeat a shared observation's weight.
Pathway = 1.0 and State = 0.5 remain unchanged.

Implementation: cla_engine/exact_member_groups.py. The versioned materializer,
dictionary publisher, validation receipts and source-group mapping are in
provenance/exact_member_merge_20261114/. This change does not calculate lncRNA
associations or fit models.
