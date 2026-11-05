"""Common-background single-sample components and train-fitted State transforms.

No patient-cohort normalization is fitted in raw_ssgsea. Its integer average
ranks and stable descending order follow GSVA 2.4.9's internal ssgsea function.
The explicit measured universe is retained, including constant genes, so a new
query cannot change the scoring background. This is a new harmonized version,
not a claim that legacy cohort-normalized values are numerically unchanged.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from scipy.sparse import csr_matrix
from scipy.optimize import nnls

def raw_ssgsea(expression,sets,*,integer_ranks=True,minimum=1,maximum=None):
    """Patients x genes -> patients x sets; incomplete background stays missing."""
    if not expression.columns.is_unique:raise ValueError('Duplicate background genes')
    genes=list(expression.columns);lookup={g:i for i,g in enumerate(genes)};n=len(genes)
    rows=[];cols=[];coverage=[]
    for j,(name,gg) in enumerate(sets.items()):
        mapped=sorted({lookup[g] for g in gg if g in lookup});m=len(mapped)
        eligible=m>=minimum and m<n and (maximum is None or len(set(gg))<=maximum)
        coverage.append(dict(target_id=name,total_genes=len(set(gg)),mapped_genes=m,eligible=eligible))
        if eligible:rows.extend([j]*m);cols.extend(mapped)
    membership=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(sets),n))
    counts=np.asarray(membership.sum(axis=1)).ravel();eligible=counts>0
    data=expression.to_numpy(float);valid=np.isfinite(data).all(axis=1)
    scores=np.full((len(data),len(sets)),np.nan)
    for start in range(0,valid.sum(),64):
        ids=np.flatnonzero(valid)[start:start+64];v=data[ids]
        ranks=rankdata(v,axis=1,method='average')
        if integer_ranks:ranks=ranks.astype(np.int32)
        order=np.argsort(-ranks,axis=1,kind='stable');pos=np.empty_like(order)
        np.put_along_axis(pos,order,np.arange(n)[None,:],axis=1)
        remaining=n-pos;weights=ranks**.25
        den=(membership@weights.T).T
        numerator=(membership@(weights*remaining).T).T
        outsum=n*(n+1)/2-(membership@remaining.T).T
        with np.errstate(divide='ignore',invalid='ignore'):
            score=numerator/den-outsum/(n-counts)
        score[:,~eligible]=np.nan;scores[ids]=score
    return pd.DataFrame(scores,index=expression.index,columns=list(sets)),pd.DataFrame(coverage)

def fit_range(raw,rows):
    v=np.asarray(raw)[np.asarray(rows)]
    finite=v[np.isfinite(v)]
    return float(np.ptp(finite)) if len(finite) and np.ptp(finite)>1e-12 else np.nan

def weighted_programs(expression,network,train):
    """Freeze gene z-scores and program z-scores on training patients only."""
    expr=expression.to_numpy(float);train=np.asarray(train,int)
    mu=np.nanmean(expr[train],axis=0);sd=np.nanstd(expr[train],axis=0,ddof=1)
    pos={g:i for i,g in enumerate(expression.columns)};net=network.copy()
    net=net.loc[net.gene.isin(pos)];net['i']=net.gene.map(pos)
    net=net.loc[np.isfinite(sd[net.i]) & (sd[net.i]>1e-12)]
    all_ids=sorted(network.state_id.unique());ids={s:i for i,s in enumerate(all_ids)}
    weight=csr_matrix((net.weight,(net.i,net.state_id.map(ids))),shape=(expr.shape[1],len(ids)))
    good=sd>1e-12;z=np.zeros_like(expr);z[:,good]=(expr[:,good]-mu[good])/sd[good]
    # Sparse multiplication propagates only missing genes with nonzero weights.
    result=np.asarray(weight.T@z.T).T;norm=np.sqrt(np.asarray(weight.power(2).sum(axis=0))).ravel()
    with np.errstate(divide='ignore',invalid='ignore'):result=result/norm
    count=net.groupby('state_id').size().reindex(all_ids,fill_value=0).to_numpy();result[:,count<5]=np.nan
    pm=np.nanmean(result[train],axis=0);ps=np.nanstd(result[train],axis=0,ddof=1);ps[ps<=1e-12]=np.nan
    result=(result-pm)/ps
    state=dict(genes=list(expression.columns),gene_mean=mu,gene_sd=sd,program_ids=all_ids,program_mean=pm,program_sd=ps,detected=count)
    return pd.DataFrame(result,index=expression.index,columns=['STATE|native_program::'+s for s in all_ids]),state

def frozen_xcell(raw,train,reference):
    """Retain published spillover matrix; freeze both native minimum offsets."""
    x=raw.to_numpy(float);train=np.asarray(train,int)
    minima=np.nanmin(x[train],axis=0);x=x-minima
    groups={}
    for j,name in enumerate(raw.columns):groups.setdefault(name.split('%')[0],[]).append(j)
    cells=reference['cells'];agg=np.column_stack([np.mean(x[:,groups[c]],axis=1) for c in cells])
    cm=np.nanmin(agg[train],axis=0);v=np.maximum((agg-cm)/5000,0)
    fv=np.asarray(reference['fv'],float);v=(v**fv[:,1])/(2*fv[:,2])
    K=np.asarray(reference['K'],float)*.5;np.fill_diagonal(K,1)
    scored=np.full_like(v,np.nan)
    for i in range(len(v)):
        if np.isfinite(v[i]).all():scored[i]=nnls(K,v[i],maxiter=10000)[0]
    return pd.DataFrame(scored,index=raw.index,columns=['STATE|xCell::'+c for c in cells]),dict(signature_min=minima,cell_min=cm)

def frozen_caf(expression,genes,train):
    genes=[g for g in genes if g in expression];x=expression[genes].to_numpy(float);t=x[train]
    mu=np.nanmean(t,axis=0);sd=np.nanstd(t,axis=0,ddof=1);good=np.isfinite(sd)&(sd>1e-12)
    # Fitted means are permitted for PCA fitting only; observed label components
    # themselves are never imputed. Incomplete query labels remain unavailable.
    complete=np.isfinite(t[:,good]).all(axis=1)
    if good.sum()<2 or complete.sum()<3:return np.full(len(x),np.nan),dict(status='INSUFFICIENT_PCA_DATA')
    _,_,vt=np.linalg.svd((t[complete][:,good]-mu[good])/sd[good],full_matrices=False)
    w=np.abs(vt[0]);w/=w.sum();score=x[:,good]@w
    return score,dict(genes=np.asarray(genes)[good],mean=mu[good],sd=sd[good],weight=w)

def frozen_extend(expression,train):
    markers=['C18orf54','CCT6P1','CEP72','POLE2','MCM4','HELLS','TERC','CCT6AP1','C21orf81','LOC81691','AC004381.6','ANKRD20A11P','EHMT2','LIN9','PAXIP1','TERT']
    m=[g for g in markers if g in expression];x=expression.to_numpy(float)
    ranks=rankdata(x,axis=1,method='average',nan_policy='propagate')*10000/x.shape[1]
    pos={g:i for i,g in enumerate(expression.columns)};tel=[g for g in ['TERT','TERC'] if g in m]
    other=[g for g in m if g not in tel];marker_sum=ranks[:,[pos[g] for g in other]].sum(axis=1)
    if tel:
        component=ranks[:,[pos[g] for g in tel]].max(axis=1);total=marker_sum+component
        ok=np.isfinite(total[train])&np.isfinite(component[train]);rho=spearmanr(total[train][ok],component[train][ok]).statistic
        penalty=1-rho
        raw=marker_sum+component/penalty if np.isfinite(penalty) and penalty>1e-12 else np.full(len(x),np.nan)
    else:raw=marker_sum;penalty=np.nan
    lo=np.nanmin(raw[train]);hi=np.nanmax(raw[train]);score=(raw-lo)/(hi-lo) if hi>lo else np.full(len(x),np.nan)
    return score,dict(markers=m,penalty=penalty,minimum=lo,maximum=hi)


class JointScoreBuilder:
    """Fit a single scoring reference for both cohorts in each training split."""
    def __init__(self,aligned,cache,apm):
        from pathlib import Path
        import json
        from cla_engine.common import sha
        self.aligned=Path(aligned);self.cache=Path(cache);self.apm=Path(apm)
        receipt=json.loads((self.cache/'COMPLETE.json').read_text())
        if receipt['input_complete_sha256']!=sha(self.aligned/'COMPLETE.json'):raise ValueError('Alignment changed')
        for name,digest in receipt['files'].items():
            if sha(self.cache/name)!=digest:raise ValueError('Cache drift '+name)
        if not (self.apm/'COMPLETE.json').exists():raise ValueError('Frozen APM materialization incomplete')
        apm_receipt=json.loads((self.apm/'COMPLETE.json').read_text())
        for name,digest in apm_receipt.get('files',{}).items():
            if isinstance(digest,str) and sha(self.apm/name)!=digest:raise ValueError('APM artifact drift '+name)
        self.e=pd.read_parquet(self.aligned/'SYMBOL_LOGTPM.parquet');self.e.index.name='patient_id'
        self.frames={k:pd.read_parquet(self.cache/(k+'.parquet')).reindex(self.e.index) for k in ['PATHWAY_RAW','STATE_SSGSEA_RAW','HALLMARK_RAW','STATE_FIXED_RAW','XCELL_RAW']}
        self.defs=json.loads((self.cache/'DEFINITIONS.json').read_text());self.xref=json.loads((self.cache/'XCELL_REFERENCE.json').read_text())
        self.network=pd.read_csv(self.cache/'PROGRAM_NETWORK.tsv',sep='\t')
        self.catalog=json.loads((self.cache/'CONFIG_TARGETS.json').read_text())
        self.native={c:pd.read_parquet(self.aligned/(c+'_NATIVE_Y.parquet')).reindex(self.e.index) for c in ['TCGA','CPTAC']}
        self.target_metadata=None

    def fit_apply(self,train,split):
        import json
        train=np.asarray(train,int);f=self.frames;states={'split':split,'train_patient_ids':self.e.index[train].tolist(),'normalizations':{}}
        pieces=[];meta=[]
        def add(frame,layer,method,scope='SHARED_RECOMPUTED'):
            pieces.append(frame)
            meta.extend(dict(target_id=t,layer=layer,method=method,scope=scope) for t in frame.columns)
        def norm(frame,name):
            d=fit_range(frame.to_numpy(),train);states['normalizations'][name]=dict(kind='DIVIDE_TRAIN_SCORE_RANGE',scale=d)
            return frame/d
        add(norm(f['PATHWAY_RAW'],'RNA_PATHWAY'),'RNA_PATHWAY','GSVA_ssGSEA_alpha025_COMMON_PC_BACKGROUND_TRAIN_RANGE')
        s=f['STATE_SSGSEA_RAW']
        for group,key in [('EMT','EMT_Miow'),('ECM','ECM_Chakravarthy'),('SenMayo','SenMayo')]:
            cols=[c for c in s if key in c];add(norm(s[cols],group),'STATE','GSVA_ssGSEA_COMMON_ALLGENE_BACKGROUND_TRAIN_RANGE')
        h=norm(f['HALLMARK_RAW'],'HALLMARK_50_STATE')
        names=['HALLMARK_OXIDATIVE_PHOSPHORYLATION','HALLMARK_DNA_REPAIR'];hh=h[names].copy();hh.columns=['STATE|HALLMARK::'+c for c in names]
        add(hh,'STATE','GSVA_ssGSEA_50_HALLMARK_TRAIN_RANGE')
        # The cosine purity conversion is not calibrated on this RNA background.
        # Preserve source purity endpoints separately; never train on this conversion.
        fixed=f['STATE_FIXED_RAW'].drop(columns=['STATE|ESTIMATE::TumorPurity']).copy()
        col='STATE|signifinder::Hypoxia_Buffa';mu=fixed[col].iloc[train].mean();sd=fixed[col].iloc[train].std(ddof=1)
        fixed[col]=(fixed[col]-mu)/sd if sd>1e-12 else np.nan;states['normalizations'][col]=dict(kind='TRAIN_Z',mean=mu,sd=sd)
        for name in ['RNAss','EREG.EXPss']:
            col='STATE|stemness_rna::'+name;lo=fixed[col].iloc[train].min();hi=fixed[col].iloc[train].max()
            fixed[col]=(fixed[col]-lo)/(hi-lo) if hi>lo else np.nan;states['normalizations'][col]=dict(kind='TRAIN_MINMAX_NO_TEST_CLIPPING',minimum=lo,maximum=hi)
        add(fixed,'STATE','EXACT_SIGNATURE_FORMULAS_COMMON_BACKGROUND_TRAIN_REFERENCE_WHERE_REQUIRED')
        programs,st=weighted_programs(self.e,self.network,train);states['programs']=st;add(programs,'STATE','WEIGHTED_GENE_Z_THEN_PROGRAM_Z_TRAIN_ONLY')
        xc,st=frozen_xcell(f['XCELL_RAW'],train,self.xref);states['xcell']=st;add(xc,'STATE','XCELL_NATIVE_SIGNATURES_PUBLISHED_SPILLOVER_FROZEN_TRAIN_MINIMA')
        out=pd.DataFrame(index=self.e.index)
        gg=[r['SYMBOL'] for r in self.defs['LRRC15CAF_Dominguez']]
        out['STATE|signifinder::LRRC15CAF_Dominguez'],states['caf']=frozen_caf(self.e,gg,train)
        dr=pd.DataFrame(self.defs['DNArep_Kang']);dr=dr.loc[dr.SYMBOL.isin(self.e.columns)]
        values=self.e[dr.SYMBOL].to_numpy(float);threshold=np.nanmedian(values[train],axis=0)
        high=dr['class'].eq('high').to_numpy();score=np.where(high,values>threshold,values<threshold).sum(axis=1).astype(float)
        score[~np.isfinite(values).all(axis=1)]=np.nan;out['STATE|signifinder::DNArep_Kang']=score
        states['dna_repair']=dict(genes=dr.SYMBOL.tolist(),high=high,training_median=threshold,implementation='GENEWISE_THRESHOLD_COMPARISON_NO_R_VECTOR_RECYCLING')
        out['STATE|EXTEND::published_score'],states['extend']=frozen_extend(self.e,train)
        ap=pd.read_csv(self.apm/(split+'.tsv'),sep='\t').set_index('patient_id').score.reindex(self.e.index)
        aprec=json.loads((self.apm/(split+'.json')).read_text())
        if list(aprec['train_patients'])!=self.e.index[train].tolist():raise ValueError('APM reference patient mismatch')
        out['STATE|signifinder::APM_Wang']=ap;states['apm']=dict(receipt=str(self.apm/(split+'.json')),method='FROZEN_GAUSSIAN_GSVA')
        add(out,'STATE','TRAIN_FITTED_NATIVE_STATE_REFERENCE')
        harmonized=set(c for part in pieces for c in part.columns)
        # Native assay targets are separate measured endpoints. Non-recomputable
        # State values are explicitly cohort-qualified, NEVER silently pooled.
        for cohort,yy in self.native.items():
            extra=yy[[c for c in yy if c.startswith('STATE|') and c not in harmonized]].copy()
            extra.columns=['STATE_NATIVE_'+cohort+'|'+c.split('|',1)[1] for c in extra]
            add(extra,'STATE_NATIVE','PRESERVED_SOURCE_ENDPOINT_NOT_HARMONIZED',scope=cohort+'_ONLY_NATIVE_REFERENCE')
            for layer in ['PROTEIN_PATHWAY','PHOSPHO_PTMSEA','METHYLATION_PATHWAY']:
                cols=[c for c in yy if c.startswith(layer+'|')]
                if cols:add(yy[cols],layer,'MEASURED_NATIVE_ASSAY_TARGET',scope=cohort+'_ONLY_NATIVE_REFERENCE')
        result=pd.concat(pieces,axis=1)
        if not result.columns.is_unique:raise ValueError('Duplicate target axes')
        md=pd.DataFrame(meta)
        if self.target_metadata is not None and not md.equals(self.target_metadata):raise ValueError('Target axis changed between splits')
        self.target_metadata=md
        return result,states
