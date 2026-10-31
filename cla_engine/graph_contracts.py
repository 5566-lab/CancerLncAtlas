"""Exact membership and source-preserving relation validation.

Signed membership is an effect/annotation field. Binding/PPI weight is a
non-negative confidence, never an inhibitory effect. Unknown is not positive.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from .common import fail, digest
from .identity import frame_ids, gene, token, cancer, unique_values

UNMAPPED={'unmapped','mapping_failed','not_mapped','rejected','invalid','ambiguous','unresolved'}
UNSIGNED={'unsigned','explicit_unsigned','mapped_unsigned','unspecified'}
MAPPED={'mapped','mapped_unique','unique','mapped_ensembl','mapped_alias','exact','direct'}

def membership_join(exact, weighted):
    exact=frame_ids(exact);weighted=frame_ids(weighted)
    keys=['pathway_id','gene_id']
    if set(keys)-set(exact) or set(keys+['weight'])-set(weighted):fail('SCHEMA_MISMATCH','membership')
    current=unique_values(exact,keys,[],'exact membership')
    w=weighted.copy();raw=w.weight;v=pd.to_numeric(raw,errors='coerce')
    state=w.get('mapping_status',pd.Series('mapped',index=w.index)).astype('string').str.lower().str.strip()
    if state.isna().any() or not state.isin(UNMAPPED|UNSIGNED|MAPPED).all():
        fail('MEMBERSHIP_MAPPING_STATUS_UNKNOWN','Mapping status must be explicit',values=state.dropna().unique().tolist())
    unmapped=state.isin(UNMAPPED)
    if ((raw.notna()&v.isna())|np.isinf(v.to_numpy(float))).any():fail('SIGNED_MEMBERSHIP_INVALID','Unparseable/infinite weight is not unsigned')
    mapped=~unmapped
    if w.loc[mapped,keys].isna().any().any():fail('SIGNED_MEMBERSHIP_MISSING_ENDPOINT','Mapped membership must have both endpoints')
    # mapping_status here annotates gene-id mapping, not signedness; the weight
    # column alone carries sign (NaN = explicit unsigned). Rows with a mapped
    # status and no weight fall into the contract UNSIGNED vocabulary
    # ('mapped_unsigned'), matching the frozen v2_9/g012 source semantics: the
    # graph then emits membership_unspecified edges instead of signed member edges.
    unsigned=state.isin(UNSIGNED)|(state.isin(MAPPED)&v.isna())
    if (unsigned&v.notna()).any():fail('SIGNED_MEMBERSHIP_CONFLICT','Unsigned status/signature with a numeric signed weight')
    if (mapped & v.eq(0)).any():fail('SIGNED_MEMBERSHIP_ZERO','Zero is neither a signed relation nor explicit unsigned')
    w['weight']=v;valid=w.loc[mapped,keys+['weight']].drop_duplicates()
    if valid.duplicated(keys).any():fail('SIGNED_MEMBERSHIP_CONFLICT','Conflicting signed/unsigned or signed weights for one exact key')
    out=current.merge(valid,on=keys,how='left',validate='one_to_one')
    validkeys=set(valid[keys].itertuples(index=False,name=None));exactkeys=set(current[keys].itertuples(index=False,name=None))
    report={'current_rows':len(current),'signed_matches':int(out.weight.notna().sum()),
            'explicit_unsigned':int(out.weight.isna().sum()),'unmapped_source_rows_excluded':int(unmapped.sum()),
            'signed_only_keys':len(validkeys-exactkeys),'exact_only_keys':len(exactkeys-validkeys),
            'policy':'CURRENT_EXACT_KEYS; MAPPED_FINITE_SIGNED; EXPLICIT_UNMAPPED_EXCLUDED; MAPPED_NAN_UNSIGNED; UNMATCHED_UNSIGNED',
            'namespace_policy':'GENE:ENSG equals ENSG; no symbol or fuzzy mapping',
            'excluded_rows_sha256':digest(w.loc[unmapped].fillna('').astype(str).to_dict('records')),
            'mapped_nan_reclassified_unsigned':int((state.isin(MAPPED)&v.isna()).sum())}
    return out,report

def validate_membership(frame):
    f=frame_ids(frame);keys=['pathway_id','gene_id']
    if 'weight' in f:
        raw=f.weight;v=pd.to_numeric(raw,errors='coerce')
        if (raw.notna()&v.isna()).any() or np.isinf(v).any():fail('EDGE_WEIGHT_INVALID','membership')
        f['weight']=v
        f=unique_values(f,keys,['weight'],'membership signed/unsigned conflict')
    else:f=unique_values(f,keys,[],'membership')
    return f

def confidence_frame(frame,keys,role, *, weight='weight', selfloops=False):
    f=frame_ids(frame)
    if set(keys+[weight])-set(f):fail('SCHEMA_MISMATCH',role)
    if f[keys].isna().any().any():fail('EDGE_ENDPOINT_MISSING',role)
    for k in keys:f[k]=f[k].map(lambda x:token(x,k))
    f[weight]=pd.to_numeric(f[weight],errors='coerce')
    if not np.isfinite(f[weight]).all() or f[weight].le(0).any():
        fail('PHYSICAL_CONFIDENCE_INVALID',role+' confidence must be finite and >0, not signed effect')
    for col in ['is_prediction','is_predicted','is_computational','model_derived','outcome_derived','is_computed']:
        if col in f:
            from .tables import strict_bool
            if strict_bool(f[col]).any():fail('INELIGIBLE_STATIC_RELATION',role,field=col)
    if not selfloops and len(keys)==2 and keys[0].startswith('protein_id') and keys[1].startswith('protein_id'):
        if f[keys[0]].eq(f[keys[1]]).any():fail('PPI_SELF_LOOP_FORBIDDEN','Original physical-PPI contract excludes source self-loops')
    return f

def validate_provenance(f,role):
    """Schema matches pinned source tables; unsupported metadata cannot be inferred.

    Missing species is reported as unrecorded, not relabelled human experimental
    evidence. Explicit nonhuman, predicted, ambiguous or causal/regulatory claims
    cannot enter the static binding/PPI namespace.
    """
    required={'source_database','relation_type','edge_id'}
    if role=='binding':required.add('mapping_status')
    if role=='ppi':required.add('physical_score')
    if required-set(f):fail('STATIC_PROVENANCE_MISSING',role,fields=sorted(required-set(f)))
    for key in required-{'physical_score'}:
        if f[key].isna().any() or f[key].astype(str).str.strip().eq('').any():fail('STATIC_PROVENANCE_MISSING',role,field=key)
    for key in ('species','organism','taxon_id','taxonomy_id'):
        if key in f:
            known=f[key].notna() & ~f[key].astype(str).str.lower().isin(['','unknown','unspecified','na'])
            accepted=f[key].astype(str).str.strip().str.lower().isin(['human','homo sapiens','homo_sapiens','9606','9606.0'])
            if (known&~accepted).any():fail('NONHUMAN_STATIC_RELATION',role,field=key)
    from cc_hhgt.v32.evidence_semantics_r5 import predicted_method
    for key in ('source_dataset','experiment_type','method','experiment_method'):
        if key in f:
            values=f[key].fillna('').astype(str)
            if values.map(predicted_method).any() or values.str.contains(r'(?i)(?:^|[-_ /])(?:predicted|computational|in[ _-]?silico)(?:$|[-_ /])',regex=True).any():
                fail('PREDICTED_STATIC_RELATION',role,field=key)
    if 'mapping_status' in f:
        accepted={'mapped','mapped_unique','unique','exact','direct','mapped_ensembl','mapped_alias','mapped_uniprot','resolved','mapped_expanded','expanded','mapped_multi'}
        if not f.mapping_status.astype(str).str.lower().str.strip().isin(accepted).all():fail('AMBIGUOUS_STATIC_MAPPING',role)
    if 'mapping_multiplicity' in f:
        x=pd.to_numeric(f.mapping_multiplicity,errors='coerce')
        if not np.isfinite(x).all() or x.lt(1).any():fail('AMBIGUOUS_STATIC_MAPPING',role)
        # multiplicity>1 annotates crosswalk attribution: the protein resolves to
        # its gene through protein_gene_map with a fractional mapping_weight, and
        # the binding/PPI endpoint ids themselves are explicit, so these rows
        # remain valid static relations (frozen v3_state/g012 source convention;
        # count surfaced in the binding_context audit).
    allowed={
        'binding':{'binding','physical_binding','rna_protein_binding','lncrna_protein_binding','lncrna_protein','binds','binds_to','experimental_binding','physical_interaction','binds_protein'},
        'ppi':{'physical_interaction','physical','physical_ppi','ppi','protein_protein_interaction','string_physical','reported_physical_interaction'}
    }
    values=f.relation_type.astype(str).str.strip().str.lower().str.replace('-', '_', regex=False).str.replace(' ', '_', regex=False)
    if not values.isin(allowed[role]).all():fail('STATIC_RELATION_ONTOLOGY_MISMATCH',role,values=sorted(set(values)-allowed[role]))


def normalize_binding(frame):
    f=confidence_frame(frame,['lncrna_id','protein_id'],'binding')
    validate_provenance(f,'binding')
    if 'cancer_id' not in f:fail('CANCER_CONTEXT_MISSING','Binding context cannot be invented')
    from .tables import strict_bool
    specific=strict_bool(f.is_context_specific) if 'is_context_specific' in f else np.zeros(len(f),bool)
    null=f.cancer_id.isna()|f.cancer_id.fillna('').astype(str).str.strip().eq('')
    if np.any(null & specific):fail('CANCER_CONTEXT_CONFLICT','Context-specific relation lacks a cancer')
    if null.any() and 'is_context_specific' not in f:fail('CANCER_CONTEXT_UNKNOWN','Null cancer requires explicit is_context_specific=False')
    f['cancer_id']=[cancer(v,True,explicit_global=bool(n and not sp)) for v,n,sp in zip(f.cancer_id,null,specific)]
    if np.any(specific & f.cancer_id.eq('GLOBAL').to_numpy()):fail('CANCER_CONTEXT_CONFLICT','Specific binding cannot be global')
    if 'relation_type' in f:
        bad=f.relation_type.astype(str).str.contains(r'(?i)prediction|coexpression|outcome|ceRNA|sponge|function',regex=True)
        if bad.any():fail('BINDING_RELATION_TYPE_INVALID','Nonbinding claims cannot be relabelled binding')
    return f

def binding_context(frame,cancer_id,lncs=None):
    """One source of truth for GLOBAL plus exact-cancer binding selection.

    Both the graph and genomic partner input must consume normalize_binding's
    ontology/context validation. Explicit null + non-context-specific is GLOBAL;
    unknown null context is never guessed. Gene-level aggregation deduplicates
    endpoints later so GLOBAL and local copies cannot double-count a partner.
    """
    f=normalize_binding(frame);key=cancer(cancer_id)
    allowed=f.cancer_id.isin(['GLOBAL',key])
    if lncs is not None:allowed &= f.lncrna_id.isin(lncs)
    out=f.loc[allowed].copy()
    report={'normalizer':'normalize_binding','cancer_id':key,'source_rows':len(f),
        'selected_rows':len(out),'global_rows':int(out.cancer_id.eq('GLOBAL').sum()),
        'context_rows':int(out.cancer_id.eq(key).sum()),'distinct_lnc_protein_pairs':len(out[['lncrna_id','protein_id']].drop_duplicates()),
        'null_context_not_treated_as_negative':True}
    if 'mapping_multiplicity' in f:
        x=pd.to_numeric(f.mapping_multiplicity,errors='coerce')
        report['mapped_multiplicity_gt1_rows']=int(x.gt(1).sum())
        report['mapped_multiplicity_gt1_proteins']=int(f.loc[x.gt(1),'protein_id'].nunique()) if len(f) else 0
    return out,report


def normalize_ppi(frame):
    f=confidence_frame(frame,['protein_id_a','protein_id_b'],'PPI')
    validate_provenance(f,'ppi')
    if 'physical_score' in f:
        x=pd.to_numeric(f.physical_score,errors='coerce')
        if not np.isfinite(x).all() or x.le(0).any():fail('PPI_PHYSICAL_SCORE_INVALID','Physical score must be finite positive')
    if 'relation_type' in f:
        bad=f.relation_type.astype(str).str.contains(r'(?i)functional|predicted|coexpression|regulat',regex=True)
        if bad.any():fail('PPI_RELATION_TYPE_INVALID','Functional/derived associations are not physical-PPI evidence')
    left=f.protein_id_a.to_numpy(str);right=f.protein_id_b.to_numpy(str)
    f['protein_id_a']=np.where(left<right,left,right);f['protein_id_b']=np.where(left<right,right,left)
    # Reciprocal source copies count once, retaining source identifiers in the sidecar audit.
    import json
    source_cols=[c for c in f.columns if c not in {'protein_id_a','protein_id_b'}]
    records={}
    for key,group in f.groupby(['protein_id_a','protein_id_b'],sort=True):
        # Sorted JSON is order-invariant; repetitions are records, never independent studies.
        records[key]=json.dumps(sorted({json.dumps(r,sort_keys=True,default=str) for r in group[source_cols].fillna('').to_dict('records')}),ensure_ascii=False)
    f=f.sort_values(['weight','edge_id'],ascending=[False,True],kind='stable').drop_duplicates(['protein_id_a','protein_id_b'])
    f['source_records_json']=[records[(a,b)] for a,b in f[['protein_id_a','protein_id_b']].itertuples(index=False,name=None)]
    return f

def source_summary(frame,role):
    cols=[c for c in ['edge_id','source_database','relation_type','mapping_status','n_pmids','n_source_records','physical_score','source_records_json','species','organism','is_computational'] if c in frame]
    return {'role':role,'source_rows':len(frame),'source_metadata_columns':cols,
            'metadata_sha256':hashlib.sha256(pd.util.hash_pandas_object(frame[cols].fillna('').astype(str),index=False).to_numpy().tobytes()).hexdigest() if cols else None,
            'experimental_validation_inferred':False,'aggregation':'MAX_CONFIDENCE_PER_ENDPOINT_CONTEXT; SOURCE_TABLE_HASH_RETAINED'}
