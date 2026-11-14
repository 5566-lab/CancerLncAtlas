"""Patient observations for frozen WikiPathways source definitions.

Source identities remain distinct from backbone identities. Molecular assay
columns use the existing observation IDs; RNA and protein summaries have
source-specific IDs and never stand for absolute pathway activation.
"""
from __future__ import annotations
import collections
import json
import numpy as np
import pandas as pd
from cc_hhgt.m0_joint_scores import raw_ssgsea


def bind_source_members(members, features):
    """Bind an assay once per source; keep partial multigene hits as context."""
    required={'source_target_id','ensembl_gene_id'}
    if not required <= set(members) or not {'feature_id','gene_ids'} <= set(features):
        raise ValueError('WP_MEMBER_SCHEMA')
    if not features.feature_id.is_unique:
        raise ValueError('WP_DUPLICATE_ASSAY')
    sets=members.loc[members.ensembl_gene_id.ne('')].groupby('source_target_id').ensembl_gene_id.agg(set).to_dict()
    by_gene=collections.defaultdict(set)
    for sid, genes in sets.items():
        for gene in genes:by_gene[gene].add(sid)
    rows=[]
    for feature in features.itertuples():
        genes=set(str(feature.gene_ids).split('|'))-{''}
        targets=set().union(*(by_gene[g] for g in genes)) if genes else set()
        for sid in sorted(targets):
            direct=genes <= sets[sid]
            rows.append(dict(source_target_id=sid,feature_id=feature.feature_id,
                evidence_role='member_measurement' if direct else 'ambiguous_group_context',
                is_source_observation=direct,assay_gene_ids='|'.join(sorted(genes)),
                matched_member_gene_ids='|'.join(sorted(genes & sets[sid])),
                member_scope='original_WikiPathways_source_definition',
                independent_loss_enabled=False))
    return pd.DataFrame(rows,columns=['source_target_id','feature_id','evidence_role',
        'is_source_observation','assay_gene_ids','matched_member_gene_ids','member_scope','independent_loss_enabled'])


def score_rna(expression, annotation, source_sets):
    """Existing common-coding-background ssGSEA, raw ES, no fitted range."""
    if not annotation.gene_id.is_unique or not annotation.gene_biotype.eq('protein_coding').all():
        raise ValueError('WP_RNA_REQUIRES_CODING_ANNOTATION')
    background=set(annotation.reference_symbol)-{''}
    if set(expression.columns)!=background or not expression.columns.is_unique or not expression.index.is_unique:
        raise ValueError('WP_RNA_BACKGROUND_AXIS')
    scores,coverage=raw_ssgsea(expression,source_sets,integer_ranks=True,minimum=10,maximum=500)
    coverage['score_method']='GSVA_SSGSEA_ALPHA025_COMMON_PC_BACKGROUND_RAW_ES'
    coverage['minimum_mapped_genes']=10
    coverage['maximum_definition_genes']=500
    coverage['reason']=np.where(coverage.eligible,'',np.where(coverage.mapped_genes.lt(10),
        'FEWER_THAN_10_MAPPED_CODING_GENES',np.where(coverage.total_genes.gt(500),
        'MORE_THAN_500_DEFINITION_GENES','NO_COMPLEMENT_BACKGROUND')))
    return scores,coverage


def score_rppa(values, features, source_gene_sets, reference_patients):
    """Reference-z antibodies -> one gene-group mean -> source group mean.

    This reproduces materialize_rppa_functional_evidence.fit_z/score_matrix:
    >=3 finite reference values, positive SD, >=3 observed groups per sample.
    Multiple antibodies for the same gene group contribute one group value.
    """
    if not values.columns.is_unique or not values.index.is_unique or not features.feature_id.is_unique:
        raise ValueError('WP_RPPA_DUPLICATE_AXIS')
    refs=list(reference_patients)
    if len(refs)!=len(set(refs)) or not set(refs)<=set(values.index):
        raise ValueError('WP_RPPA_REFERENCE_PATIENTS')
    info=features.set_index('feature_id').reindex(values.columns)
    if info.gene_ids.isna().any():raise ValueError('WP_RPPA_ANNOTATION_AXIS')
    reference=values.loc[refs]
    mean=reference.mean();sd=reference.std(ddof=1);n=reference.notna().sum()
    good=n.ge(3)&sd.gt(1e-12)&np.isfinite(sd)
    z=(values-mean)/sd;z.loc[:,~good]=np.nan
    group_map=collections.defaultdict(list)
    for feature,genes in info.gene_ids.items():
        key='|'.join(sorted(set(str(genes).split('|'))-{''}))
        if key:group_map[key].append(feature)
    groups=pd.DataFrame({key:z[ids].mean(axis=1) for key,ids in sorted(group_map.items())},index=values.index)
    out={};counts={};coverage=[]
    for sid,members in source_gene_sets.items():
        selected=[key for key in groups if set(key.split('|'))<=set(members)]
        count=groups[selected].notna().sum(axis=1)
        score=groups[selected].mean(axis=1).where(count.ge(3))
        out[sid]=score;counts[sid]=count
        coverage.append(dict(target_id=sid,total_genes=len(set(members)),mapped_groups=len(selected),
            observed_score_patients=int(score.notna().sum()),minimum_observed_groups=3,
            reason='' if score.notna().any() else ('FEWER_THAN_3_ANNOTATED_GENE_GROUPS' if len(selected)<3 else 'FEWER_THAN_3_VALID_OBSERVED_GROUPS_PER_PATIENT')))
    fit=pd.DataFrame({'feature_id':values.columns,'reference_mean':mean,'reference_sd':sd,
        'reference_n':n,'reference_eligible':good}).reset_index(drop=True)
    return pd.DataFrame(out,index=values.index),pd.DataFrame(counts,index=values.index),pd.DataFrame(coverage),fit


def protein_symbol_matrix(values, feature_alignment):
    """Mirror CPTAC protein input aggregation: coding symbols, native mean."""
    ann=feature_alignment.loc[feature_alignment.layer.eq('protein')].drop_duplicates('feature_id').set_index('feature_id')
    ann=ann.reindex(values.columns)
    valid=ann.reference_presence.eq('GENCODE_V36') & ann.reference_symbol.fillna('').ne('') & ann.gene_biotype.eq('protein_coding')
    frame=values.loc[:,valid].T.copy();frame.index=ann.loc[valid,'reference_symbol']
    result=frame.groupby(level=0,sort=True).mean().T
    result.index.name='patient_id';result.columns.name=None
    return result


def verify_reused_observations(values, metadata, old_values, old_metadata):
    """Shared molecule IDs must preserve the old package's exact raw values."""
    old=set(old_metadata.observation_id);shared=[x for x in metadata.observation_id if x in old]
    if shared:
        pd.testing.assert_frame_equal(values[shared],old_values[shared],check_names=False)
    return len(shared)


def existing_measurement_ids(metadata, old_metadata):
    """Reuse pinned observation IDs by their assay identity, never by a new hash.

    Older bundles may use an earlier ID serialization; a molecular measurement
    must keep that identity when additional pathway memberships are added.
    """
    keys=['cohort','kind','layer','feature_id']
    for frame in (metadata,old_metadata):
        if frame.duplicated(keys).any() or not frame.observation_id.is_unique:
            raise ValueError('WP_DUPLICATE_ASSAY_IDENTITY')
    old=old_metadata.set_index(keys).observation_id.to_dict()
    remap={row.observation_id:old.get(tuple(getattr(row,k) for k in keys),row.observation_id)
           for row in metadata.itertuples(index=False)}
    if len(set(remap.values()))!=len(remap):
        raise ValueError('WP_OBSERVATION_ID_COLLISION')
    return remap
