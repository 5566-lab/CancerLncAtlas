# Graph-edge leakage inspection (fold 0) — 2026-10-04

The label of the primary task is an lncRNA x pathway association. If the graph contained any
edge connecting an lncRNA to a pathway, a model could read the answer off its own input. This
inspection enumerates the **actual** edge table of the C global-binding bundle rather than
trusting the guard that is supposed to prevent it.

Artifact inspected: `G2_GLOBAL_FOLD_0.pt` (read from the staged local copy, not the sshfs mount).
Receipt: `LEAK_EDGE_INSPECTION.json`.

## Result

```
status                   : PASS_NO_LNCRNA_PATHWAY_EDGE
edges                    : 7,190,167
lncRNA AND pathway edges : 0
outcome_derived          : {"False": 7190167}
observed                 : {"True": 7190167}
```

**Zero edges touch both an lncRNA and a pathway.** The complete relation inventory:

| relation_type | source -> target | count |
|---|---|---|
| coexpressed_negative | lncRNA -> gene | 3,403,498 |
| coexpressed_positive | lncRNA -> gene | 3,031,107 |
| member_of_positive | gene -> pathway | 224,414 |
| physical_interaction | protein -> protein | 154,852 |
| member_of_negative | gene -> pathway | 125,194 |
| expressed_in | lncRNA -> cancer | 76,734 |
| binds_protein_experimental_unspecified | lncRNA -> protein | 66,056 |
| binds_protein_other_clip | lncRNA -> protein | 41,663 |
| binds_protein_eclip | lncRNA -> protein | 41,150 |
| encoded_by | protein -> gene | 20,008 |
| binds_protein_rna_capture | lncRNA -> protein | 3,150 |
| member_of_family | pathway -> pathway_family | 2,114 |
| binds_protein_rip | lncRNA -> protein | 191 |
| binds_protein_other_physical | lncRNA -> protein | 21 |
| binds_protein_unknown | lncRNA -> protein | 15 |

This matches the guard: `safe_graph.build_safe_graph` rejects `direct_target` pairs
(`lncRNA` <-> `pathway`/`exact_pathway`) and `formal_graph.py:655` calls it when building the
authority, so the exclusion is structural rather than incidental. Every edge is also
`outcome_derived=False` and `observed=True`, and the authority gates record
`outer_train_coexpression_only` and `outer_train_expression_only` as true, so a validation
row's own expression does not enter its own graph.

## What this does and does not establish

It rules out the **direct** shortcut: the graph cannot hand the model an lncRNA-pathway edge
that restates the label.

It does **not** rule out indirect inference, and should not be reported as if it did. The graph
still connects lncRNA to pathway in two hops, via gene:

    lncRNA --coexpressed_{positive,negative}--> gene --member_of_{positive,negative}--> pathway

That is the intended mechanism of a graph model, and it is computed from the outer-training
split only — but it means "no direct edge" is a narrower statement than "the graph cannot
inform the label". A shuffled-label or shuffled-edge control would be needed to bound how much
of the observed gain is attributable to real pairing, and no such control was run.

The other boundaries on the evaluation stand unchanged: validation tier only, no sealed-test
access, only the G2 arm trained so the gain cannot be attributed to ENCODE eCLIP or assay
typing specifically, and the figures sit almost exactly on the historical `[H5]` diagnostic
whose provenance was later downgraded.