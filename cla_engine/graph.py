"""Typed hierarchical molecular graph. Dynamic edges belong to one cancer and one feature cohort."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .common import fail
from .tables import require
from .statistics import basis
from .residual_variation import rank_residual_estimable
from .identity import frame_ids, coverage
from .graph_contracts import validate_membership, normalize_binding, normalize_ppi, confidence_frame, source_summary

def nodes_and_static(membership,hierarchy,binding,protein_gene,ppi,lncs,pathways):
    require(membership,['pathway_id','gene_id'],'exact membership')
    membership=validate_membership(membership);hierarchy=frame_ids(hierarchy);protein_gene=frame_ids(protein_gene)
    binding=normalize_binding(binding);ppi=normalize_ppi(ppi)
    require(protein_gene,['protein_id','gene_id','source_database','mapping_multiplicity'],'protein-gene authority')
    multiplicity=pd.to_numeric(protein_gene.mapping_multiplicity,errors='coerce')
    if not np.isfinite(multiplicity).all() or multiplicity.lt(1).any():fail('PROTEIN_GENE_AMBIGUOUS','Resolve multi-gene mappings before graph construction')
    if protein_gene.groupby('protein_id').gene_id.nunique().gt(1).any():fail('PROTEIN_GENE_AMBIGUOUS','Conflicting canonical gene identities')
    # multiplicity>1 on this frozen source records fractional crosswalk attribution
    # (one row per protein; mapping_weight = 1/multiplicity). Endpoint pairing stays
    # single-gene; verify the fraction is coherent instead of rejecting the row.
    w=pd.to_numeric(protein_gene.mapping_weight,errors='coerce') if 'mapping_weight' in protein_gene else None
    if w is not None:
        if not np.isfinite(w).all() or w.le(0).any():fail('PROTEIN_GENE_WEIGHT_INVALID','mapping_weight must be finite positive')
        if not np.isclose(w.to_numpy(float),1/multiplicity.to_numpy(float)).all():fail('PROTEIN_GENE_ATTRIBUTION_INCONSISTENT','mapping_weight must equal 1/multiplicity')
    for col in ('species','organism','taxon_id'):
        if col in protein_gene:
            valid=protein_gene[col].astype(str).str.lower().isin(['human','homo sapiens','9606','9606.0'])
            if not valid.all():fail('PROTEIN_GENE_SPECIES_INVALID','Human-mapped core requires consistent taxon metadata')
    for name,table_,cols in [('hierarchy',hierarchy,['pathway_id','pathway_family_id']),('protein-gene',protein_gene,['protein_id','gene_id'])]:
        if table_[cols].isna().any().any():fail('EDGE_ENDPOINT_MISSING',name)
    gene=sorted(set(membership.gene_id.astype(str))|set(protein_gene.gene_id.astype(str)));protein=sorted(set(protein_gene.protein_id.astype(str)))
    famcol='pathway_family_id';require(hierarchy,['pathway_id',famcol],'hierarchy')
    fam=sorted(set(hierarchy[famcol].astype(str)))
    ids={'lncRNA':list(lncs),'gene':gene,'protein':protein,'pathway':list(pathways),'pathway_family':fam}
    maps={t:{v:i for i,v in enumerate(vs)} for t,vs in ids.items()};relations=[];edges={};audit=[source_summary(binding,'binding'),source_summary(ppi,'ppi'),source_summary(protein_gene,'protein_gene')]
    def add(src,rel,dst,frame,a,b,w=None,reverse=True,context=False):
        require(frame,[a,b]+([w] if w else []),rel)
        f=frame.copy();s=f[a].astype(str).map(maps[src]);d=f[b].astype(str).map(maps[dst]);valid=s.notna()&d.notna()
        if len(f) and not valid.any():fail('GRAPH_ENDPOINT_COVERAGE_ZERO',rel,source_rows=len(f))
        weight=pd.to_numeric(f[w],errors='coerce') if w else pd.Series(np.ones(len(f)),index=f.index)
        if weight.isna().any() or not np.isfinite(weight).all():fail('EDGE_WEIGHT_INVALID',rel)
        contextval=f['cancer_id'].fillna('').astype(str).to_numpy() if context and 'cancer_id' in f else np.repeat('',len(f))
        for sign,name in [(1,rel+'_positive'),(-1,rel+'_negative')]:
            take=valid & (weight*sign>0)
            if not take.any():continue
            key=src+'|'+name+'|'+dst
            arr=np.stack([s[take].to_numpy(int),d[take].to_numpy(int)])
            vals=np.abs(weight[take].to_numpy(np.float32));ctx=contextval[take]
            # Evidence confidence weights are priors, not calibrated causal strengths.
            rows=pd.DataFrame({'s':arr[0],'d':arr[1],'w':vals,'ctx':ctx}).groupby(['s','d','ctx'],as_index=False,observed=True).w.max()
            entry={'edge_index':rows[['s','d']].to_numpy(np.int64).T,'weight':rows.w.to_numpy(np.float32),'context':rows.ctx.to_numpy(str),'sign':sign}
            edges[key]=entry;relations.append((src,name,dst))
            if reverse:
                k=dst+'|rev_'+name+'|'+src;edges[k]={**entry,'edge_index':entry['edge_index'][::-1].copy()};relations.append((dst,'rev_'+name,src))
        audit.append({'relation':rel,'source_rows':len(f),'mapped_rows':int(valid.sum()),'excluded_unmapped_rows':int((~valid).sum())})
    # Current exact membership defines topology. Historical signed membership is not substituted.
    if 'weight' in membership:
        known=membership.loc[membership.weight.notna()]
        unknown=membership.loc[membership.weight.isna()].drop(columns='weight')
        if len(known):add('gene','member','pathway',known,'gene_id','pathway_id','weight')
        if len(unknown):add('gene','membership_unspecified','pathway',unknown,'gene_id','pathway_id')
    else:add('gene','membership_unspecified','pathway',membership,'gene_id','pathway_id')
    add('pathway','member_of_family','pathway_family',hierarchy,'pathway_id',famcol,'membership_weight' if 'membership_weight' in hierarchy else None)
    protein_gene=confidence_frame(protein_gene,['protein_id','gene_id'],'protein_gene',weight='mapping_weight' if 'mapping_weight' in protein_gene else 'weight',selfloops=True)
    add('protein','encoded_by','gene',protein_gene,'protein_id','gene_id','mapping_weight' if 'mapping_weight' in protein_gene else None)
    if len(ppi):coverage(set(ppi.protein_id_a)|set(ppi.protein_id_b),protein,role='PPI protein namespace',minimum=1.)
    require(binding,['lncrna_id','protein_id','weight','cancer_id'],'binding')
    binding=binding.loc[binding.lncrna_id.isin(lncs)].copy()
    if len(binding):coverage(binding.protein_id,protein,role='binding protein namespace',minimum=1.)
    add('lncRNA','reported_binding','protein',binding,'lncrna_id','protein_id','weight',context=True)
    # Preserve the source relation semantics; not automatically called physical interaction.
    require(ppi,['protein_id_a','protein_id_b','weight'],'PPI')
    reciprocal=ppi.rename(columns={'protein_id_a':'protein_id_b','protein_id_b':'protein_id_a'})
    symmetric=pd.concat([ppi,reciprocal],ignore_index=True)
    add('protein','reported_PPI','protein',symmetric,'protein_id_a','protein_id_b','weight',reverse=False)
    for sign in ['positive','negative']:
        relations.extend([('lncRNA','coexpression_'+sign,'gene'),('gene','rev_coexpression_'+sign,'lncRNA')])
    return ids,maps,relations,edges,audit

def coexpression(X,G,C,call,Z,cohort,topk=8,min_n=12,cnv_transform='raw_signed',*,return_audit=False,lnc_batch=64,gene_batch=256):
    """Pairwise-complete source-cohort rank association; no molecular imputation.

    Rank transforms and nuisance coefficients are recomputed on EACH lncRNA-gene
    common mask. Missingness groups share computations only when the masks are
    identical. Peak temporary correlation storage is lnc_batch x gene_batch;
    exact global top-k per lncRNA is merged across gene blocks deterministically.
    This preserves the existing lncRNA-only A1 action and raw/rank option.
    """
    X=np.asarray(X,dtype=float);G=np.asarray(G,dtype=float);C=np.asarray(C,dtype=float)
    Z=np.asarray(Z,dtype=float);rawcall=np.asarray(call)
    if any(a.ndim!=2 for a in (X,G,C,Z,rawcall)) or X.shape!=C.shape or X.shape!=rawcall.shape or any(a.shape[0]!=X.shape[0] for a in (G,Z)):
        fail('COEXPRESSION_SHAPE_MISMATCH','Aligned two-dimensional X/G/C/call/Z required')
    if not np.isin(rawcall,[0,1]).all():fail('COEXPRESSION_CALLABILITY_INVALID','CNV callability must be boolean, never NaN')
    if cnv_transform not in {'raw_signed','rank'}:fail('CNV_TRANSFORM_INVALID',cnv_transform)
    if any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) or v<1 for v in (topk,min_n,lnc_batch,gene_batch)):
        fail('COEXPRESSION_CONFIGURATION_INVALID','Positive integer minimum_pairs, topk and block sizes required')
    raw_ix=np.asarray(cohort)
    if raw_ix.ndim!=1 or (len(raw_ix) and (not np.issubdtype(raw_ix.dtype,np.integer) or len(set(raw_ix.tolist()))!=len(raw_ix) or np.any(raw_ix<0) or np.any(raw_ix>=X.shape[0]))):
        fail('COEXPRESSION_COHORT_INVALID','Source indices must be unique valid integers')
    ix=raw_ix.astype(np.int64);source_n=len(ix);ix=ix[np.isfinite(Z[ix]).all(1)]
    x=X[ix];g=G[ix];c=C[ix];z=Z[ix];calls=rawcall[ix].astype(bool)
    lm=np.isfinite(x)&np.isfinite(c)&calls;gm=np.isfinite(g)
    nlnc=X.shape[1];ngene=G.shape[1];k=min(int(topk),ngene)
    best=np.full((nlnc,k),np.nan);bestgene=np.full((nlnc,k),ngene,np.int64)
    bestn=np.zeros((nlnc,k),np.int32);bestrank=np.zeros((nlnc,k),np.int16)
    def groups(mask):
        if mask.shape[1]==0 or mask.shape[0]==0:return []
        _,labels_=np.unique(np.packbits(mask.T,axis=1),axis=0,return_inverse=True)
        return [np.flatnonzero(labels_==i) for i in range(int(labels_.max())+1)]
    lg=groups(lm);gg=groups(gm)
    below=0;estimable=0;degenerate=0;mask_evaluations=0
    for lcols in lg:
        for gcols in gg:
            common=lm[:,lcols[0]]&gm[:,gcols[0]];n=int(common.sum());count=len(lcols)*len(gcols);mask_evaluations+=1
            if n<min_n:below+=count;continue
            Q,q=basis(z[common])
            for start in range(0,len(lcols),lnc_batch):
                li=lcols[start:start+lnc_batch]
                raw_x=rankdata(x[common][:,li],axis=0);xx=raw_x.copy()
                original_cn=c[common][:,li]
                cc=rankdata(original_cn,axis=0) if cnv_transform=='rank' else original_cn.copy()
                xx-=Q@(Q.T@xx);cc-=Q@(Q.T@cc)
                den=np.sum(cc*cc,axis=0);tol=1e-18*np.maximum(1,np.sum(original_cn*original_cn,axis=0))
                active=den>tol;beta=np.divide(np.sum(xx*cc,axis=0),den,out=np.zeros(len(li)),where=active)
                xr=xx-cc*beta;xn=np.sqrt(np.sum(xr*xr,axis=0));ranks=q+active.astype(np.int16)
                xok=rank_residual_estimable(raw_x,xr,z.shape[1])
                for gstart in range(0,len(gcols),gene_batch):
                    gj=gcols[gstart:gstart+gene_batch]
                    raw_y=rankdata(g[common][:,gj],axis=0);yy=raw_y-Q@(Q.T@raw_y)
                    yn=np.sqrt(np.sum(yy*yy,axis=0));norm=xn[:,None]*yn[None,:]
                    good=((norm>1e-10)&((n-ranks-1)>0)[:,None]
                          &xok[:,None]&rank_residual_estimable(raw_y,yy,z.shape[1])[None,:])
                    corr=np.divide(xr.T@yy,norm,out=np.full(norm.shape,np.nan),where=good)
                    corr=np.clip(corr,-1,1);estimable+=int(np.isfinite(corr).sum());degenerate+=int((~np.isfinite(corr)).sum())
                    candidates=np.concatenate([best[li],corr],axis=1)
                    indices=np.concatenate([bestgene[li],np.broadcast_to(gj,(len(li),len(gj)))],axis=1)
                    counts=np.concatenate([bestn[li],np.full(corr.shape,n,np.int32)],axis=1)
                    ranks_=np.concatenate([bestrank[li],np.broadcast_to(ranks[:,None],corr.shape)],axis=1)
                    strength=np.where(np.isfinite(candidates),np.abs(candidates),-np.inf)
                    order=np.lexsort((indices,-strength),axis=1)[:,:k]
                    best[li]=np.take_along_axis(candidates,order,axis=1)
                    bestgene[li]=np.take_along_axis(indices,order,axis=1)
                    bestn[li]=np.take_along_axis(counts,order,axis=1)
                    bestrank[li]=np.take_along_axis(ranks_,order,axis=1)
    edges={};support={};included=np.zeros(nlnc,bool)
    for sign,name in ((1,'positive'),(-1,'negative')):
        keep=np.isfinite(best)&(best*sign>=.15)&(bestgene<ngene)
        ls,ks=np.where(keep);included[ls]=True
        edges[name]=(np.vstack([ls,bestgene[ls,ks]]).astype(np.int64),np.abs(best[ls,ks]).astype(np.float32))
        support[name]={'n':bestn[ls,ks],'rank':bestrank[ls,ks]}
    audit={'contract':'PAIRWISE_COMMON_MASK_COEXPRESSION_R62_V1','cnv_transform':cnv_transform,
        'local_cnv_action':'LNC_ONLY_A1','molecular_missingness':'PAIRWISE_COMPLETE_NO_IMPUTATION',
        'ranks_and_nuisance_refit_on_common_mask':True,'source_patients':source_n,'covariate_complete_patients':len(ix),
        'lncrnas':nlnc,'genes':ngene,'minimum_pairs':int(min_n),'topk_per_lncrna':int(topk),
        'all_patients_callable_lncrnas':int((calls & np.isfinite(c)).all(0).sum()) if len(ix) else 0,
        'lncrnas_with_minimum_measured_patients':int((lm.sum(0)>=min_n).sum()),
        'lncrna_mask_groups':len(lg),'gene_mask_groups':len(gg),'mask_intersections_evaluated':mask_evaluations,
        'pairs_below_minimum_n':below,'estimable_pairs':estimable,'rank_or_variance_degenerate_pairs':degenerate,
        'lncrnas_with_selected_edges':int(included.sum()),'selected_edges':{s:int(e[0].shape[1]) for s,e in edges.items()},
        'block_sizes':{'lncrna':int(lnc_batch),'gene':int(gene_batch)},'edge_support':support}
    if not estimable:audit['empty_reason']='NO_ESTIMABLE_COMMON_MASK_PAIRS'
    elif not included.any():audit['empty_reason']='NO_CORRELATION_EXCEEDS_FROZEN_THRESHOLD'
    else:audit['empty_reason']=None
    return (edges,audit) if return_audit else edges

def graph_npz(ids,edges,cancer,dynamic,features):
    """No cross-cancer coexpression averaging; static binding context is filtered explicitly."""
    result={f'x::{t}':v.astype(np.float32) for t,v in features.items()}
    for key,v in edges.items():
        ctx=v['context'];keep=np.isin(ctx,['','GLOBAL',cancer])
        selected=v['edge_index'][:,keep];weights=v['weight'][keep]
        # A global and cancer-specific copy of the same graph fact is one message.
        # Confidence is max-aggregated; original rows remain hashed in the source audit.
        if selected.shape[1]:
            df=pd.DataFrame({'s':selected[0],'d':selected[1],'w':weights}).groupby(['s','d'],as_index=False,sort=True).w.max()
            selected=df[['s','d']].to_numpy(np.int64).T;weights=df.w.to_numpy(np.float32)
        result['edge::'+key]=selected;result['weight::'+key]=weights
    for s,(e,w) in dynamic.items():
        key='lncRNA|coexpression_'+s+'|gene';rev='gene|rev_coexpression_'+s+'|lncRNA'
        result['edge::'+key]=e;result['weight::'+key]=w;result['edge::'+rev]=e[::-1].copy();result['weight::'+rev]=w
    if 'cancer' in ids:
        ci=ids['cancer'].index(cancer)
        for kind in ('lncRNA','gene'):
            # Connect only actually measured nodes, not all identities.
            observed=np.flatnonzero(features[kind][:,2]>0)
            edge=np.vstack([observed,np.full(len(observed),ci,dtype=np.int64)])
            result[f'edge::{kind}|measured_in|cancer']=edge
            result[f'weight::{kind}|measured_in|cancer']=np.ones(len(observed),np.float32)
            result[f'edge::cancer|rev_measured_in|{kind}']=edge[::-1].copy()
            result[f'weight::cancer|rev_measured_in|{kind}']=np.ones(len(observed),np.float32)
    return result
