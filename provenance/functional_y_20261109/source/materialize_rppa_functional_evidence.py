"""RPPA protein abundance and matched directional phosphosite evidence."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import socket,sys,json,gzip,re,os,time,traceback
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'rppa_functional_evidence_demo_r1';P=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo');sys.path.insert(0,str(P))
from cla_engine.common import sha
from cla_engine.statistics import design
from block_associations import associate
RAW=Path('/dell_2/DSC/CancerLncAtlas/inputs/rppa500_20261104_r4')
AUTH=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv')
COV=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_full_pyg_r6_2/covariates_merged/covariates_r62_20260908.parquet');CFG=Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json')
def js(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)))
def contexts():return [(f'train_label_fold{j}',[q for q in [2,3,4] if q!=j],[j]) for j in [2,3,4]]+[('validation',[2,3,4],[1]),('sealed_test',[2,3,4],[0])]
def fit_z(x,train):
 ref=x[train];mu=np.nanmean(ref,axis=0);sd=np.nanstd(ref,axis=0,ddof=1);n=np.isfinite(ref).sum(0);good=(n>=3)&np.isfinite(sd)&(sd>1e-12);z=np.full_like(x,np.nan);z[:,good]=(x[:,good]-mu[good])/sd[good];return z,{'mean':mu,'sd':sd,'n':n,'valid':good}
def score_matrix(z,W,minimum=3):
 finite=np.isfinite(z);count=finite.astype(float)@(W!=0).astype(float);den=finite.astype(float)@np.abs(W);out=np.full(count.shape,np.nan);np.divide(np.nan_to_num(z,nan=0)@W,den,out=out,where=(count>=minimum)&(den>0));return out,count

def prepare():
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','cancers':['BRCA','LUAD','COAD'],'operation':'RPPA_MEASURED_FUNCTIONAL_EVIDENCE','reference_fit':'SOURCE_PATIENTS_ONLY','protein_score':'Mean training-standardized antibody gene groups; each gene group counted once','phospho_score':'Signed mean training-standardized phosphosite residuals after matched-total-protein regression','minimum_observed_groups_for_aggregate':3,'missing_score':None,'training_started':False})
 complete=json.loads((RAW/'COMPLETE.json').read_text());files={}
 for name in ['PROTEIN.parquet','PHOSPHORYLATION.parquet','FEATURE_CATALOG.tsv','PATIENT_AUTHORITY.parquet']:
  h=sha(RAW/name);v=complete['files'][name];expected=v['sha256'] if isinstance(v,dict) else v;assert h==expected;files[name]=h
 js(O/'INPUT_SHA256.json',files)
 cat=pd.read_csv(R/'measurement_mechanism_r1/RPPA_FEATURE_SITE_REVIEW.tsv',sep='\t').fillna('');prot=cat.loc[cat.molecular_layer.eq('PROTEIN')&cat.gene_ids.ne('')].copy();prot['gene_group']=prot.gene_ids.map(lambda x:'|'.join(sorted(set(x.split('|')))))
 groups=prot.groupby('gene_group',sort=True).feature_id.agg(list).to_dict();js(O/'PROTEIN_GROUPS.json',groups)
 master=pd.read_csv(R/'canonical_targets_r2/TARGET_MASTER.tsv',sep='\t');pw=master.loc[master.roles.str.contains('Pathway')].copy();comp=pd.read_csv(R/'canonical_targets_r2/PATHWAY_GENE_COMPOSITION.tsv.gz',sep='\t');members=comp.groupby('canonical_id').member_id.agg(set).to_dict();gkeys=list(groups);W=np.zeros((len(gkeys),len(pw)));coverage=[]
 for j,d in enumerate(pw.itertuples()):
  genes=members.get(d.canonical_id,set());ix=[i for i,key in enumerate(gkeys) if set(key.split('|'))<=genes];W[ix,j]=1;coverage.append({'canonical_id':d.canonical_id,'display_name':d.display_name,'assay_groups':len(ix),'pathway_genes':len(genes),'gene_groups':' ; '.join(gkeys[i] for i in ix),'score_method':'RPPA_TRAIN_Z_GENE_GROUP_MEAN','measurement_layer':'PROTEIN_ABUNDANCE'})
 pd.DataFrame(coverage).to_csv(O/'PROTEIN_TARGETS.tsv',sep='\t',index=False);np.save(O/'PROTEIN_MEMBERSHIP.npy',W)
 # Canonical reviewed human sequence anchors. Antibody groups remain single observations.
 seqs={};current=None
 with gzip.open(R/'sources/uniprot_reviewed_human_20261109.fasta.gz','rt') as f:
  for line in f:
   if line.startswith('>'):
    gn=re.search(r'\bGN=([^ ]+)',line);current=(gn.group(1),line.split('|')[1]) if gn else None
    if current:seqs[current]=''
   elif current:seqs[current]+=line.strip()
 ann=pd.read_csv(R/'common_rna_scores_r1/COMMON_GENE_ANNOTATION.tsv',sep='\t').set_index('gene_id');sym=ann.reference_symbol.to_dict();ph=cat.loc[cat.molecular_layer.eq('PHOSPHORYLATION')].copy();site_rows=[];matches=[]
 site_def=pd.read_csv(R/'canonical_targets_r2/PHOSPHO_READOUT_COMPOSITION.tsv.gz',sep='\t');sig_by_site={k:g for k,g in site_def.groupby('site_identifier')}
 for row in ph.itertuples():
  genes=row.gene_ids.split('|') if row.gene_ids else [];labels=row.corrected_site_labels.split('|') if row.corrected_site_labels else [];rec={'feature_id':row.feature_id,'gene_ids':row.gene_ids,'site_labels':row.corrected_site_labels,'validation_status':row.validation_status,'sequence_status':'MULTI_GENE_OR_MULTI_SITE','matched_total_group':'','flanking_sequence':'','uniprot_accessions':''}
  if len(genes)==1 and len(labels)==1 and re.fullmatch(r'[STY][0-9]+',labels[0]):
   gene=genes[0];g=sym.get(gene);site=labels[0];pos=int(site[1:]);hits=[]
   for (gn,acc),seq in seqs.items():
    if gn==g and pos<=len(seq) and seq[pos-1]==site[0]:
     padded='_'*7+seq+'_'*7;flank=padded[pos-1:pos+14]+'-p';hits.append((acc,flank))
   flanks={h[1] for h in hits};rec['uniprot_accessions']='|'.join(h[0] for h in hits)
   if len(flanks)==1:
    flank=next(iter(flanks));rec['sequence_status']='CANONICAL_RESIDUE_AND_POSITION_MATCH';rec['flanking_sequence']=flank
    if gene in groups:
     rec['matched_total_group']=gene
     if flank in sig_by_site:
      for h in sig_by_site[flank].itertuples():matches.append({'source_readout_id':h.source_readout_id,'feature_id':row.feature_id,'direction':h.direction,'weight':1 if h.direction=='u' else -1,'flanking_sequence':flank,'matched_total_group':gene})
   else:rec['sequence_status']='NO_UNIQUE_CANONICAL_SEQUENCE_MATCH'
  site_rows.append(rec)
 sr=pd.DataFrame(site_rows);sr.to_csv(O/'PHOSPHOSITE_ALIGNMENT.tsv',sep='\t',index=False);mm=pd.DataFrame(matches);assert set(mm.direction)<={'u','d'}
 # A feature with contradictory signs within a signature is not assigned a sign.
 contradiction=mm.groupby(['source_readout_id','feature_id']).weight.nunique();bad=set(contradiction[contradiction>1].index);mm=mm.loc[[(r.source_readout_id,r.feature_id) not in bad for r in mm.itertuples()]].drop_duplicates(['source_readout_id','feature_id']);mm.to_csv(O/'PHOSPHO_SIGNATURE_MEMBERS.tsv',sep='\t',index=False)
 defs=pd.read_csv(R/'canonical_targets_r2/PHOSPHO_READOUT_DEFINITIONS.tsv',sep='\t');extra=pd.read_csv(R/'score_registry_r2/ADDITIONAL_SOURCE_IDENTITIES.tsv',sep='\t').set_index('source_target_id').canonical_id.to_dict();defs['canonical_id']=defs.source_readout_id.map(extra);assert defs.canonical_id.notna().all();defs['matched_sites']=defs.source_readout_id.map(mm.groupby('source_readout_id').feature_id.nunique()).fillna(0).astype(int);defs.to_csv(O/'PHOSPHO_TARGETS.tsv',sep='\t',index=False)
 rec={'target_host':'149','protein_groups':len(groups),'pathway_targets':len(pw),'protein_targets_with_3_groups':sum(x['assay_groups']>=3 for x in coverage),'phospho_features':len(sr),'unique_sequence_matches':int(sr.sequence_status.eq('CANONICAL_RESIDUE_AND_POSITION_MATCH').sum()),'sequence_and_total_matched':int(sr.matched_total_group.ne('').sum()),'phospho_signature_edges':len(mm),'phospho_targets_with_3_sites':int((defs.matched_sites>=3).sum()),'contradictory_feature_signature_pairs_excluded':len(bad)};js(O/'ALIGNMENT_COMPLETE.json',rec);print('RPPA_ALIGNMENT',rec,flush=True)

def work(cancer):
 out=O/cancer;out.mkdir();a=pd.read_csv(AUTH,sep='\t');a=a.loc[a.cancer_id.eq(cancer)];patients=a.patient_id.tolist();folds=a.patient_fold_id.to_numpy(int);ad=R/'common_rna_scores_r1/aligned'/('TCGA_'+cancer)
 x=pd.read_parquet(ad/'LNC_LOGTPM_FULL.parquet').reindex(patients);lncs=x.columns.tolist();X=x.to_numpy(float);cn=R/'full_lnc_local_cnv_r1'/cancer;cp=json.loads((cn/'patient_ids.json').read_text());cl=json.loads((cn/'lncrna_ids.json').read_text());pi=np.array([cp.index(p) for p in patients]);li=np.array([cl.index(l) for l in lncs]);C=np.load(cn/'lncrna_signed_mean.npy',mmap_mode='r')[np.ix_(pi,li)].astype(float);call=np.load(cn/'lncrna_callable.npy',mmap_mode='r')[np.ix_(pi,li)].astype(bool)
 protein=pd.read_parquet(RAW/'PROTEIN.parquet').set_index('patient_id').reindex(patients);phospho=pd.read_parquet(RAW/'PHOSPHORYLATION.parquet').set_index('patient_id').reindex(patients);groups=json.loads((O/'PROTEIN_GROUPS.json').read_text());gkeys=list(groups);W=np.load(O/'PROTEIN_MEMBERSHIP.npy');pt=pd.read_csv(O/'PROTEIN_TARGETS.tsv',sep='\t');ht=pd.read_csv(O/'PHOSPHO_TARGETS.tsv',sep='\t');hm=pd.read_csv(O/'PHOSPHO_SIGNATURE_MEMBERS.tsv',sep='\t');hs=pd.read_csv(O/'PHOSPHOSITE_ALIGNMENT.tsv',sep='\t').fillna('');pcols=list(protein);pc={k:i for i,k in enumerate(pcols)};hf=sorted(hm.feature_id.unique());hfi={k:i for i,k in enumerate(hf)};hids=ht.source_readout_id.tolist();hi={k:i for i,k in enumerate(hids)};HW=np.zeros((len(hf),len(hids)))
 for row in hm.itertuples():HW[hfi[row.feature_id],hi[row.source_readout_id]]=row.weight
 htot=hs.set_index('feature_id').matched_total_group.to_dict();cfg=json.loads(CFG.read_text());cov=pd.read_parquet(COV);summary=[];frozen={}
 js(out/'AXES.json',{'patient_ids':patients,'lncrna_ids':lncs});pr=protein.to_numpy(float)
 for name,sf,lf in contexts():
  start=time.monotonic();dest=out/name;dest.mkdir();si=np.flatnonzero(np.isin(folds,sf));yi=np.flatnonzero(np.isin(folds,lf));sp=[patients[i] for i in si];lp=[patients[i] for i in yi];assert not set(sp)&set(lp)
  z,ref=fit_z(pr,si);gz=np.column_stack([pd.DataFrame(z[:,[pc[f] for f in groups[k]]]).mean(axis=1).to_numpy() for k in gkeys]);ps,pn=score_matrix(gz,W)
  residual=np.full((len(patients),len(hf)),np.nan);rr=[]
  for j,feature in enumerate(hf):
   total=gz[:,gkeys.index(htot[feature])];y=phospho[feature].to_numpy(float);valid=np.isfinite(total[si])&np.isfinite(y[si]);ix=si[valid];r={'feature_id':feature,'total_gene_group':htot[feature],'n_train':len(ix)}
   if len(ix)>=3 and np.ptp(total[ix])>1e-12:
    mat=np.column_stack([np.ones(len(ix)),total[ix]]);coef=np.linalg.lstsq(mat,y[ix],rcond=None)[0];good=np.isfinite(total)&np.isfinite(y);residual[good,j]=y[good]-coef[0]-coef[1]*total[good];r.update(intercept=float(coef[0]),slope=float(coef[1]))
   rr.append(r)
  hz,href=fit_z(residual,si);qs,qn=score_matrix(hz,HW)
  js(dest/'SCORING_REFERENCE.json',{'target_host':'149','train_patients':sp,'protein_feature_ids':pcols,'protein_standardization':ref,'phospho_regression':rr,'phospho_feature_ids':hf,'phospho_residual_standardization':href,'minimum_observed_groups':3})
  Z,report=design(cov,patients,sp,family='pathway',cfg=cfg,admission_cohorts={'source':sp,'label':lp});js(dest/'ADJUSTMENT.json',report);js(dest/'SPLIT.json',{'target_host':'149','outer_fold':0,'source_patients':sp,'label_patients':lp,'source_folds':sf,'label_folds':lf})
  for layer,Y,N,md in [('PROTEIN',ps,pn,pt),('PHOSPHO',qs,qn,ht)]:
   ld=dest/layer;ld.mkdir();ids=md.canonical_id.tolist();pd.DataFrame(Y,index=pd.Index(patients,name='patient_id'),columns=ids).to_parquet(ld/'PATIENT_S.parquet',compression='zstd');pd.DataFrame(N,index=pd.Index(patients,name='patient_id'),columns=ids).to_parquet(ld/'OBSERVED_COMPONENT_COUNT.parquet',compression='zstd')
   if name=='validation':frozen[layer]=Y.copy()
   if name=='sealed_test':assert np.array_equal(Y,frozen[layer],equal_nan=True)
   stats=associate(X,Y,C,call,Z,yi,min_n=cfg['minimum_pairs']);strict=stats['available']&(np.abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(np.abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict
   np.savez_compressed(ld/'ASSOCIATIONS.npz',**{k:stats[k] for k in ['r','p','q','n','rank','available']},strict_statistical_support=strict,weak_statistical_support=weak)
   ix,jx=np.where(strict|weak);pd.DataFrame({'lncrna_id':np.asarray(lncs)[ix],'canonical_id':np.asarray(ids)[jx],'rho':stats['r'][ix,jx],'p':stats['p'][ix,jx],'q':stats['q'][ix,jx],'n':stats['n'][ix,jx],'statistical_support':np.where(strict[ix,jx],'strict','weak'),'evidence_layer':layer,'platform':'TCPA_RPPA500'}).to_parquet(ld/'SUPPORTED_RELATIONS.parquet',compression='zstd')
   counts=md.copy();counts['patients_with_score']=np.isfinite(Y).sum(0);counts['eligible_lncrna_pairs']=stats['available'].sum(0);counts['strict_statistical_support']=strict.sum(0);counts['weak_statistical_support']=weak.sum(0);counts.to_csv(ld/'TARGET_COUNTS.tsv',sep='\t',index=False)
   row={'cancer_id':cancer,'context':name,'layer':layer,'patients_source':len(si),'patients_label':len(yi),'score_targets':Y.shape[1],'observed_score_targets':int(np.isfinite(Y).any(0).sum()),'eligible_relations':int(stats['available'].sum()),'strict_statistical_support':int(strict.sum()),'weak_statistical_support':int(weak.sum())};js(ld/'COMPLETE.json',dict(row,target_host='149',files={p.name:sha(p) for p in ld.iterdir() if p.is_file()}));summary.append(row);print('RPPA_CONTEXT_COMPLETE',row,flush=True)
  pd.DataFrame(summary).to_csv(out/'CONTEXT_SUMMARY.tsv',sep='\t',index=False);js(dest/'COMPLETE.json',{'target_host':'149','seconds':time.monotonic()-start,'context':name})
 js(out/'COMPLETE.json',{'target_host':'149','cancer':cancer,'contexts':summary});return cancer,summary
if __name__=='__main__':
 prepare();results={};failures={}
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(work,c):c for c in ['BRCA','LUAD','COAD']}
  for f in as_completed(fs):
   c=fs[f]
   try:_,rows=f.result();results[c]=rows
   except Exception as err:failures[c]=repr(err);print('RPPA_FAILED',c,traceback.format_exc(),flush=True)
   js(O/'PROGRESS.json',{'target_host':'149','complete':results,'failures':failures})
 js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','results':results,'failures':failures})
