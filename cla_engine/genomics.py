"""R6 mutation context, preserving observed calls and valid denominators.

An observed mutant is evidence of that event; it is NOT proof that every other
patient was measured and WT. Sparse sentinels never fill a dense WT matrix.
Gene summaries are joined to protein identity and pathway/partner membership,
not turned into new primary labels or inferred loss-of-function scores.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from .common import fail,sha,digest
from .tables import table,require,strict_bool
from .identity import gene,token,frame_ids

SENTINEL='__MC3_WES_ASSAY_CALLABLE__'
FEATURES=('log1p_observed_mutant_patients','log1p_observed_variant_count',
          'explicit_callable_patient_fraction','callable_mutation_rate','callable_rate_available',
          'log1p_observed_truncating_patients','log1p_observed_missense_patients',
          'log1p_observed_synonymous_patients','annotated_positive_fraction','observed_or_callable_available')
TRUNC={'frame_shift_del','frame_shift_ins','nonsense_mutation','splice_site','stop_gained','frameshift_variant','splice_acceptor_variant','splice_donor_variant'}
MISSENSE={'missense_mutation','missense_variant'}
SYN={'silent','synonymous_variant'}

@lru_cache(maxsize=8)
def _parquet_cancer_ids(path,size,mtime_ns):
    """Bounded identity-only scan cache; input bytes are separately SHA-verified."""
    import pyarrow.dataset as ds
    dataset=ds.dataset(path,format='parquet')
    if 'cancer_id' not in dataset.schema.names:fail('MUTATION_SCHEMA','cancer_id missing')
    raw=set()
    for batch in dataset.scanner(columns=['cancer_id'],batch_size=65536).to_batches():
        raw.update(batch.column(0).unique().to_pylist())
    if None in raw:fail('MUTATION_CANCER_ID_MISSING','Mutation cancer identities cannot be null',path=path)
    return tuple(sorted(raw,key=str))


def read_cancer_source(path,cancer):
    """Select canonical cancer identities, rejecting empty/undeclared cohorts.

    A nonempty sentinel-only cohort is permitted and means assay metadata, not
    per-locus WT. Zero rows after cancer selection is a source/coverage failure,
    never an automatic all-unavailable genomic feature block.
    """
    from .identity import cancer as cancer_key
    p=Path(path);expected=cancer_key(cancer)
    if p.suffix=='.parquet':
        try:
            import pyarrow.dataset as ds
            dataset=ds.dataset(str(p),format='parquet')
            if 'cancer_id' not in dataset.schema.names:fail('MUTATION_SCHEMA','cancer_id missing')
            stat=p.stat();raw=_parquet_cancer_ids(str(p.resolve()),stat.st_size,stat.st_mtime_ns)
            selected=[x for x in raw if cancer_key(x)==expected]
            if not selected:fail('MUTATION_CANCER_SCOPE_EMPTY','Expected cancer absent from source; do not interpret as no mutations',
                                 path=str(p),cancer=expected,observed_cancer_ids=sorted(map(str,raw)))
            f=frame_ids(dataset.to_table(filter=ds.field('cancer_id').isin(selected)).to_pandas())
        except ImportError:
            f=table(p)
    else:f=table(p)
    require(f,['cancer_id'],'mutation')
    f=frame_ids(f)
    if f.cancer_id.isna().any():fail('MUTATION_CANCER_ID_MISSING','Mutation cancer identities cannot be null',path=str(p))
    observed=sorted(set(f.cancer_id))
    result=f.loc[f.cancer_id.eq(expected)].copy()
    if result.empty:fail('MUTATION_CANCER_SCOPE_EMPTY','Expected cancer absent from source; no silent unavailable block',
                        path=str(p),cancer=expected,observed_cancer_ids=observed)
    return result

@dataclass
class MutationCalls:
    patients: list[str]
    entities: list[str]
    events: np.ndarray  # nan = unknown; positive calls are allowed without a WT denominator
    counts: np.ndarray
    callable: np.ndarray
    annotations: dict[str,np.ndarray]
    report: dict

    def summaries(self,indices,minimum=12):
        ix=np.asarray(indices,int)
        if len(ix)!=len(set(ix.tolist())) or np.any(ix<0) or np.any(ix>=len(self.patients)):
            fail('MUTATION_COHORT_INDEX','Invalid or repeated cohort indices')
        e=self.events[ix];count=self.counts[ix];call=self.callable[ix]
        # A positive observation outside the denominator is retained as a count only.
        pos=(e==1);ncall=call.sum(0);nmut=pos.sum(0)
        rateok=ncall>=minimum
        rate=np.divide((pos&call).sum(0),ncall,out=np.zeros(len(self.entities)),where=rateok)
        annotated=np.zeros_like(pos)
        terms=[]
        for name in ('truncating','missense','synonymous'):
            observed=self.annotations[name][ix]&pos;annotated|=observed
            terms.append(np.log1p(observed.sum(0)))
        available=(pos.any(0)|call.any(0))
        values=np.column_stack([np.log1p(nmut),np.log1p(np.nansum(np.where(pos,count,np.nan),axis=0)),
            ncall/max(len(ix),1),rate,rateok.astype(float),*terms,
            np.divide(annotated.sum(0),nmut,out=np.zeros(len(self.entities)),where=nmut>0),available.astype(float)]).astype(np.float32)
        return values,{'cohort_patient_sha256':digest([self.patients[i] for i in ix]),'cohort_n':len(ix),
            'rate_available_entities':int(rateok.sum()),'observed_mutant_entities':int(pos.any(0).sum()),
            'all_rates_unavailable':not bool(rateok.any()),'columns':list(FEATURES),
            'no_sequence_consequence_means':'UNKNOWN_NOT_ASSUMED_DAMAGING',**self.report}


def normalize_mutation(frame,patients,entities,kind,cancer):
    """Accept the user's actual sparse gene/lncRNA schema, preserving leading zeros.

    Repeated contradictory facts fail. Exact duplicates may collapse. Unknown
    negatives without per-entity callable evidence remain unknown, not WT.
    """
    if len(frame)==0:fail('MUTATION_CANCER_SCOPE_EMPTY','Empty per-cancer table requires an explicit unavailable-source workflow, not silent features',cancer=cancer,kind=kind)
    from .identity import cancer as canonical_cancer
    cancer=canonical_cancer(cancer)
    f=frame_ids(frame);entity='gene_id' if kind=='gene' else 'lncrna_id'
    ev='is_mutated' if kind=='gene' else 'lncrna_any_mutation'
    cv='mutation_callable' if kind=='gene' else 'lncrna_mutation_available'
    bc='mutation_count' if kind=='gene' else 'exonic_variant_count'
    require(f,['cancer_id','patient_id',entity,ev,cv,bc],'mutation '+kind)
    patients=list(map(str,patients));entities=[gene(x) if kind=='gene' else token(x,entity) for x in entities]
    if len(set(patients))!=len(patients) or len(set(entities))!=len(entities):fail('MUTATION_DUPLICATE_ID','Matrix identity lists are not unique')
    if not f.cancer_id.eq(cancer).all():fail('MUTATION_CANCER_DRIFT','Table not filtered to expected cancer')
    # Rows outside the authority are audited then excluded, not coerced into another patient.
    unmatched=f.loc[~f.patient_id.isin(patients)]
    f=f.loc[f.patient_id.isin(patients)].copy()
    if len(frame)>0 and f.empty:fail('MUTATION_PATIENT_ID_COVERAGE','No input patient matches the frozen cancer cohort; do not silently produce an all-unavailable modality')
    raw=f[bc];burden=pd.to_numeric(raw,errors='coerce')
    bad=raw.notna()&(~np.isfinite(burden)|burden.lt(0)|burden.ne(np.floor(burden)))
    if bad.any():fail('MUTATION_BURDEN_INVALID','Counts must be finite nonnegative integers')
    event=strict_bool(f[ev]);event_known=f[ev].notna().to_numpy()
    call_declared=strict_bool(f[cv]);sentinel=f[entity].eq(SENTINEL).to_numpy()
    nonsentinel=~sentinel
    # Gene summaries count the event itself (is_mutated <=> mutation_count>0).
    # lncRNA MC3 summaries report the locus event set separately from the
    # EXONIC subset: an event with zero exonic variants (non-exonic-only
    # record) is legitimate under the frozen build_v32_mc3_assay_callability
    # semantics and must not fail the whole modality.
    full_count = (kind == 'gene')
    if np.any(nonsentinel&event_known&event&full_count&(~np.isfinite(burden.to_numpy())|(burden.to_numpy()<=0))):
        fail('MUTATION_EVENT_COUNT_CONFLICT','Positive event requires an observed positive count')
    if np.any(nonsentinel&event_known&~event&(burden.fillna(0).to_numpy()>0)):
        fail('MUTATION_EVENT_COUNT_CONFLICT','Negative event with positive variant count')
    # Combining contradictory callability aliases with OR is forbidden.
    explicit=np.zeros(len(f),bool)
    for name in ('locus_callable','per_locus_callable'):
        if name in f:
            value=strict_bool(f[name]);known=f[name].notna().to_numpy()
            if np.any(known&~value&call_declared&explicit):fail('MUTATION_CALLABILITY_CONFLICT',name)
            explicit|=value
    if 'callability_level' in f:
        explicit|=f.callability_level.astype('string').str.lower().isin(['locus','per_locus','gene','per_gene']).to_numpy()
    if 'absence_is_wildtype' in f:explicit|=strict_bool(f.absence_is_wildtype)
    for name in ('locus_callable','per_locus_callable'):
        if name in f:
            known=f[name].notna().to_numpy();value=strict_bool(f[name])
            if np.any(known&~value&explicit):fail('MUTATION_CALLABILITY_CONFLICT','Explicit uncallable cannot be overridden by a WT assertion')
    measured=call_declared&explicit&event_known&np.isfinite(burden.to_numpy())&nonsentinel
    f['__event']=np.where(event_known,event.astype(float),np.nan);f['__count']=burden
    f['__call']=measured;f=f.loc[nonsentinel].copy()
    f[entity]=f[entity].map(gene if kind=='gene' else lambda x:token(x,entity))
    consequence_cols=[x for x in ('Variant_Classification','variant_classification','consequence','functional_class') if x in f]
    f['__consequence']=f[consequence_cols].fillna('').astype(str).agg('|'.join,axis=1) if consequence_cols else ''
    semantic=[entity,'patient_id','__event','__count','__call','__consequence']
    f=f[semantic].drop_duplicates()
    if f.duplicated(['patient_id',entity]).any():
        fail('MUTATION_DUPLICATE_CONFLICT','Input is a per-patient/entity summary; variant rows require a separate explicit aggregation')
    n,m=len(patients),len(entities);e=np.full((n,m),np.nan,np.float32);cnt=e.copy();call=np.zeros((n,m),bool)
    ann={k:np.zeros((n,m),bool) for k in ('truncating','missense','synonymous')}
    pi={x:i for i,x in enumerate(patients)};gi={x:i for i,x in enumerate(entities)}
    ignored=0
    for ent,pat,evv,co,ca,cons in f.itertuples(index=False,name=None):
        if ent not in gi:ignored+=1;continue
        i,j=pi[pat],gi[ent]
        # Definite positive observations do not disappear with an unavailable denominator.
        if ca or evv==1:e[i,j]=evv;cnt[i,j]=co
        call[i,j]=bool(ca)
        import re
        terms=set(re.split(r'[|,;&]+',str(cons).lower()))
        for name,vocabulary in [('truncating',TRUNC),('missense',MISSENSE),('synonymous',SYN)]:ann[name][i,j]=bool(terms&vocabulary)
    return MutationCalls(patients,entities,e,cnt,call,ann,{
        'raw_rows':len(frame),'assay_sentinel_rows':int(sentinel.sum()),'out_of_authority_rows':len(unmatched),
        'out_of_entity_scope_rows':ignored,'distinct_records':len(f),'callability_not_broadcast':True,
        'consequence_columns':consequence_cols,'mutant_does_not_imply_loss_of_function':True})


def load_gene_calls(cancer,patients,genes,cfg):
    custom=cfg.get('genomics',{}).get('gene_mutation_path')
    p=Path(custom) if custom else Path(cfg['imports_root'])/'genomic/sample_gene_mutation_mc3_assay_callable.parquet'
    if cfg.get('synthetic_fixture') and not p.exists():p=p.with_suffix('.tsv')
    if not p.exists():fail('GENE_MUTATION_INPUT_MISSING','Known raw coding-gene mutation summary is required; no model predictions substitute',path=str(p))
    calls=normalize_mutation(read_cancer_source(p,cancer),patients,genes,'gene',cancer)
    calls.report.update({'source_path':str(p),'source_sha256':sha(p)})
    return calls


@dataclass
class GenomicProjection:
    identity: tuple
    matrices: dict
    totals: dict
    report: dict

    def apply(self,values):
        values=np.asarray(values,np.float32)
        genes=self.identity[0]
        if values.shape!=(len(genes),len(FEATURES)) or not np.isfinite(values).all():
            fail('GENOMIC_FEATURE_SHAPE','Finite coding-gene features must match the declared genes and columns')
        observed=values[:,FEATURES.index('observed_or_callable_available')]>0
        available_values=values*observed[:,None]
        result={'gene':values}
        for key,mat in self.matrices.items():
            n=np.asarray(mat@observed.astype(np.float32)).ravel();sums=np.asarray(mat@available_values)
            average=np.divide(sums,n[:,None],out=np.zeros_like(sums),where=n[:,None]>0)
            fraction=np.divide(n,self.totals[key],out=np.zeros_like(n),where=self.totals[key]>0)
            result[key]=np.column_stack([average,fraction,(n>0).astype(np.float32)]).astype(np.float32)
        result['audit']=dict(self.report)
        return result


def build_genomic_projection(genes,protein_gene,proteins,membership,pathways,binding,lncs,cancer,variant):
    """Normalize binding once per cancer, then reuse sparse maps across views.

    No genomic measurements or label cohorts are used to construct this map.
    G0 still has an empty partner projection; PPI is not added to binding partners.
    """
    from scipy import sparse
    from .graph_contracts import binding_context
    genes=tuple(gene(x) for x in genes);proteins=tuple(proteins);pathways=tuple(pathways);lncs=tuple(lncs)
    gi={x:i for i,x in enumerate(genes)}
    if len(gi)!=len(genes):fail('GENOMIC_GENE_ID_DUPLICATE','Projection genes must be unique canonical IDs')
    pg=frame_ids(protein_gene);require(pg,['protein_id','gene_id'],'protein gene mapping')
    pg=pg[['protein_id','gene_id']].drop_duplicates()
    if pg.groupby('protein_id').gene_id.nunique().gt(1).any():fail('PROTEIN_GENE_AMBIGUOUS','Resolve one protein mapping to multiple coding genes')
    mapping=pg.set_index('protein_id').gene_id.to_dict()
    protein_groups={p:[mapping[p]] for p in proteins if p in mapping}
    member=frame_ids(membership);require(member,['gene_id','pathway_id'],'membership')
    pathway_groups=member.groupby('pathway_id').gene_id.apply(list).to_dict()
    partners={};binding_report={'status':'BINDING_NOT_USED_IN_G0','selected_rows':0,'global_rows':0,'context_rows':0}
    if variant in {'G1','G2'}:
        b,binding_report=binding_context(binding,cancer,lncs)
        for l,p in b[['lncrna_id','protein_id']].drop_duplicates().itertuples(index=False,name=None):
            if p in mapping:partners.setdefault(l,[]).append(mapping[p])
    elif variant!='G0':fail('GRAPH_VARIANT_INVALID',variant)
    matrices={};totals={}
    for kind,groups,ids in [('protein',protein_groups,proteins),('pathway',pathway_groups,pathways),('partner',partners,lncs)]:
        rr=[];cc=[];denom=np.zeros(len(ids),np.float32)
        for i,key in enumerate(ids):
            members=set(groups.get(key,()));denom[i]=len(members)
            cols=sorted(gi[g] for g in members if g in gi)
            rr.extend([i]*len(cols));cc.extend(cols)
        matrices[kind]=sparse.csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(len(ids),len(genes)))
        totals[kind]=denom
    return GenomicProjection((genes,proteins,pathways,lncs,cancer,variant),matrices,totals,
        {'columns':list(FEATURES)+['known_member_fraction','aggregate_available'],
         'gene_feature_dim':len(FEATURES),'aggregate_feature_dim':len(FEATURES)+2,
         'partner_context_uses_binding':variant!='G0','binding_context':binding_report,
         'distinct_lnc_partner_gene_pairs':int(matrices['partner'].nnz),
         'genomic_values_ever_used_as_labels':False,'mapping_cache_scope':'ONE_CANCER_VARIANT_REUSED_ACROSS_SOURCE_VIEWS'})


def project_gene_context(values,genes,protein_gene,proteins,membership,pathways,binding,lncs,cancer,variant,*,projection=None):
    identity=(tuple(gene(x) for x in genes),tuple(proteins),tuple(pathways),tuple(lncs),cancer,variant)
    if projection is None:projection=build_genomic_projection(genes,protein_gene,proteins,membership,pathways,binding,lncs,cancer,variant)
    if not isinstance(projection,GenomicProjection) or projection.identity!=identity:
        fail('GENOMIC_PROJECTION_CONTEXT_DRIFT','Cannot reuse a mapping across different cancer/variant/entity contexts')
    return projection.apply(values)
