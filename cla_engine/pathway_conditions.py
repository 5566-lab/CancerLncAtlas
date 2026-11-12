"""Biological context requirements, independent of lncRNA or pathway scores."""
from __future__ import annotations
import json,re
import numpy as np
import pandas as pd

EVIDENCE={
 'none':[],
 'copy_number_amplification':['validated_amplification_call'],
 'activating_mutation':['curated_activating_variant'],
 'inactivating_mutation':['curated_loss_of_function_variant'],
 'fusion':['verified_fusion'],
 'rearrangement':['verified_rearrangement'],
 'increased_expression':['reference_qualified_expression_increase'],
 'decreased_expression':['reference_qualified_expression_decrease'],
 'protein_positive':['clinical_receptor_status','validated_protein_positive_assay'],
 'functional_loss':['validated_functional_loss'],
 'duplication_or_activating_mutation':['verified_internal_tandem_duplication','curated_activating_variant'],
 'pathogen_or_effector':['matched_pathogen_assay','matched_effector_assay'],
 'exposure':['documented_exposure','measured_exposure'],
}


def condition_definition(row):
    name=row['network_name'];source_type=row['network_type'];network=row['network_id']
    correction=''
    if network=='N01392':
        if name!='Arsenic to electron transfer in complex II' or 'C20296' not in row['expanded_definition']:
            raise ValueError('N01392_EXPOSURE_CURATION_SOURCE_CHANGED')
        kind='exposure';correction='Arsenic/C20296 is the upstream chemical condition; source Pathogen classification retained as provenance.'
    elif source_type=='Reference':kind='none'
    elif source_type=='Env factor':kind='exposure'
    elif source_type=='Pathogen':kind='pathogen_or_effector'
    elif source_type=='Variant':
        rules=[('Amplified ','copy_number_amplification'),('Mutation-activated ','activating_mutation'),
               ('Mutation-inactivated ','inactivating_mutation'),('Duplication or mutation-activated ','duplication_or_activating_mutation'),
               ('Loss of ','functional_loss')]
        kind=next((v for prefix,v in rules if name.startswith(prefix)),None)
        if kind is None:
            for token,v in [(' fusion','fusion'),('-rearrangement','rearrangement'),('-overexpression','increased_expression'),
                            ('-reduced expression','decreased_expression'),('-positive','protein_positive')]:
                if token in name:kind=v;break
        if kind is None:raise ValueError('UNCLASSIFIED_KEGG_VARIANT_CONDITION:'+network)
    else:raise ValueError('UNKNOWN_KEGG_NETWORK_TYPE:'+source_type)
    return {'canonical_id':row['canonical_id'],'network_id':network,'network_name':name,
       'source_network_type':source_type,'condition_kind':kind,'condition_required':kind!='none',
       'mechanism_definition':row['mechanism_definition'],'expanded_definition':row['expanded_definition'],
       'variant_reference_ids':';'.join(sorted(set(re.findall(r'\b\d+v\d+\b',row['expanded_definition'])))),
       'accepted_evidence_types':json.dumps(EVIDENCE[kind]),'curation_note':correction,
       'member_readouts_establish_condition':False}


def applicability_matrix(rules,patient_ids,observations):
    """Absent evidence stays unknown. A qualified negative is distinct from missing."""
    if len(patient_ids)!=len(set(patient_ids)) or not rules.network_id.is_unique or not rules.canonical_id.is_unique:
        raise ValueError('CONDITION_DUPLICATE_AXIS')
    lookup=rules.set_index('network_id')
    columns=list(rules.canonical_id);index=pd.Index(patient_ids,name='patient_id')
    value=pd.DataFrame(np.nan,index=index,columns=columns)
    for row in rules.to_dict('records'):
        if not row['condition_required']:value[row['canonical_id']]=1.
    if not observations.empty:
        needed={'patient_id','network_id','status','evidence_type','source_id'}
        if not needed<=set(observations):raise ValueError('CONDITION_EVIDENCE_SCHEMA')
        if not set(observations.patient_id)<=set(index):raise ValueError('CONDITION_PATIENT_OUTSIDE_SCOPE')
        if not set(observations.network_id)<=set(lookup.index):raise ValueError('CONDITION_UNKNOWN_NETWORK')
        for (patient,network),group in observations.groupby(['patient_id','network_id']):
            rule=lookup.loc[network]
            accepted=set(json.loads(rule.accepted_evidence_types))
            if not set(group.evidence_type)<=accepted:raise ValueError('CONDITION_EVIDENCE_NOT_SPECIFIC_ENOUGH:'+network)
            if not group.status.isin(['present','absent']).all() or group.source_id.fillna('').eq('').any():
                raise ValueError('CONDITION_EVIDENCE_STATUS_OR_SOURCE')
            status=set(group.status)
            value.loc[patient,rule.canonical_id]=float(next(iter(status))=='present') if len(status)==1 else np.nan
    return value,value.notna()
