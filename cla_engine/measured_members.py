"""Exact member-to-assay bindings and patient values, independent of lncRNA.

An antibody group or a phosphosite remains one measurement. Binding it to
multiple pathways does not duplicate its value or create a supervised loss.
"""
from __future__ import annotations
import collections
import json
import numpy as np
import pandas as pd


def annotation_features(annotation, feature_ids, *, gene_column='gene_id'):
    """Preserve every annotation record and explicit multiple-gene mappings."""
    if len(feature_ids)!=len(set(feature_ids)):
        raise ValueError('MEMBER_DUPLICATE_FEATURE_AXIS')
    if not {'feature_id',gene_column}<=set(annotation):
        raise ValueError('MEMBER_ANNOTATION_SCHEMA')
    annotation=annotation[annotation.feature_id.isin(feature_ids)]
    grouped={key:group for key,group in annotation.groupby('feature_id',sort=False)}
    records=[]
    for fid in feature_ids:
        group=grouped.get(fid,pd.DataFrame(columns=annotation.columns))
        genes=set()
        for value in group[gene_column].fillna('').astype(str):
            genes.update(x for x in value.split('|') if x)
        clean=group.fillna('').drop_duplicates()
        records.append({'feature_id':fid,'gene_ids':'|'.join(sorted(genes)),
            'gene_count':len(genes),'source_annotation_json':json.dumps(clean.to_dict('records'),ensure_ascii=False,sort_keys=True)})
    return pd.DataFrame(records,columns=['feature_id','gene_ids','gene_count','source_annotation_json'])


def bind_members(members, features, conditions):
    """Only original KEGG members define direct member binding.

    Partial antibody-group overlap is retained as ambiguous context, with no
    assertion that the measurement specifically describes a pathway member.
    """
    if not {'canonical_id','ensembl_gene_id'}<=set(members):
        raise ValueError('MEMBER_BACKBONE_SCHEMA')
    if not features.feature_id.is_unique or not conditions.canonical_id.is_unique:
        raise ValueError('MEMBER_DUPLICATE_BINDING_AXIS')
    if not set(members.canonical_id)<=set(conditions.canonical_id):
        raise ValueError('MEMBER_CONDITION_AXIS')
    pathway_genes=members.groupby('canonical_id').ensembl_gene_id.agg(set).to_dict()
    by_gene=collections.defaultdict(set)
    for cid,genes in pathway_genes.items():
        for gene in genes:by_gene[gene].add(cid)
    policy=conditions.set_index('canonical_id')
    rows=[]
    for feature in features.itertuples():
        genes=set(feature.gene_ids.split('|'))-{''}
        targets=set().union(*(by_gene[g] for g in genes)) if genes else set()
        for cid in sorted(targets):
            hit=genes&pathway_genes[cid]
            full=genes<=pathway_genes[cid]
            required=policy.loc[cid,'condition_required']
            if not isinstance(required,(bool,np.bool_)):
                raise ValueError('MEMBER_CONDITION_REQUIRES_BOOLEAN')
            rows.append({'canonical_id':cid,'feature_id':feature.feature_id,
                'assay_gene_ids':'|'.join(sorted(genes)),'matched_member_gene_ids':'|'.join(sorted(hit)),
                'measurement_role':'member_measurement' if full else 'ambiguous_group_context',
                'gene_specific':len(genes)==1,'condition_required':bool(required),
                'condition_kind':policy.loc[cid,'condition_kind'],
                'binding_establishes_mechanism_condition':False})
    fields=['canonical_id','feature_id','assay_gene_ids','matched_member_gene_ids','measurement_role','gene_specific','condition_required','condition_kind','binding_establishes_mechanism_condition']
    result=pd.DataFrame(rows,columns=fields)
    if result.duplicated(['canonical_id','feature_id']).any():
        raise ValueError('MEMBER_DUPLICATE_MEASUREMENT_BINDING')
    return result


def align_patients(frame, patients):
    """Reindex explicitly; unmeasured patients and values remain missing."""
    if not frame.index.is_unique or not frame.columns.is_unique or len(patients)!=len(set(patients)):
        raise ValueError('MEMBER_DUPLICATE_PATIENT_OR_FEATURE')
    result=frame.reindex(patients)
    if np.isinf(result.to_numpy(float)).any():raise ValueError('MEMBER_INFINITE_MEASUREMENT')
    result.index.name='patient_id'
    observed=result.notna()
    return result,observed


def coverage(values, observed, bindings, canonical_ids):
    """Measurement availability, without variance or association requirements."""
    pd.testing.assert_frame_equal(observed,values.notna())
    if not set(bindings.feature_id)<=set(values):raise ValueError('MEMBER_BOUND_FEATURE_ABSENT')
    groups={cid:group for cid,group in bindings.groupby('canonical_id')}
    rows=[]
    for cid in canonical_ids:
        group=groups.get(cid,bindings.iloc[:0])
        direct=group.loc[group.measurement_role.eq('member_measurement'),'feature_id'].tolist()
        ambiguous=group.loc[group.measurement_role.eq('ambiguous_group_context'),'feature_id'].tolist()
        rows.append({'canonical_id':cid,'patients_in_scope':len(values),
            'bound_member_features':len(direct),'ambiguous_group_features':len(ambiguous),
            'measured_member_features':int(observed[direct].any(axis=0).sum()),
            'patients_with_member_measurement':int(observed[direct].any(axis=1).sum()),
            'patients_with_ambiguous_group_measurement':int(observed[ambiguous].any(axis=1).sum())})
    return pd.DataFrame(rows)


def condition_qualified_mask(observed, applicability):
    """Condition eligibility is separate from measured values and missingness.

    Pass one pathway's condition series. The caller still chooses which
    observation tasks to enable; this function does not create a loss.
    """
    if not observed.index.equals(applicability.index):
        raise ValueError('MEMBER_CONDITION_PATIENT_ORDER')
    if not all(dtype==bool for dtype in observed.dtypes):
        raise ValueError('MEMBER_OBSERVED_MASK_TYPE')
    if not applicability.dropna().isin([0.,1.]).all():
        raise ValueError('MEMBER_CONDITION_STATE')
    return observed.mul(applicability.eq(1),axis=0).astype(bool)
