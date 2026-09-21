# Applicability to protein-coding genes

## Short conclusion

The architecture is transferable, but the current trained model cannot be used directly with a protein-coding gene substituted into `lncrna_id`.

## Why expression data similarity is insufficient

lncRNAs and coding genes are both measured in RNA-seq, so the patient-native numerical branch can use analogous expression and pathway-association features. However, the graph target semantics are different:

- lncRNA–pathway membership is usually unknown and is the prediction target;
- coding genes often have an explicit gene→pathway membership edge;
- using that edge while predicting the same gene–pathway relation leaks the answer;
- coding genes additionally connect directly to proteins, domains, variants, targets and pathways.

## What can be shared

- patient-fold expression/pathway association framework;
- Tumor State, single-cell, mutation and drug features;
- EventSetEncoder and direct/indirect evidence policy;
- graph encoder node representations;
- availability-masked stacking and final MoE implementation.

## What must be separate

A coding-gene model should predict:

```text
cancer × coding_gene × pathway_family regulatory association
```

rather than ordinary pathway membership.

It needs:

1. generic `subject_type` / `subject_id` candidate keys;
2. a coding-gene-specific decoder/head;
3. coding-gene-specific patient labels;
4. masking of the target gene’s direct pathway membership edge and near-duplicate annotations;
5. separate OOF evaluation and calibration;
6. separate output names and website semantics.

## Recommended implementation

Use the frozen graph encoder where possible, but train a separate coding-gene regulatory head and patient-native model. Do not mix coding-gene and lncRNA labels in one head until a multi-task benchmark proves that this improves both tasks.
