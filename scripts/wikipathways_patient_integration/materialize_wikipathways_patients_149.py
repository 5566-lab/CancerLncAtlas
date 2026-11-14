from pathlib import Path
import os,socket,json,hashlib,sys,subprocess,time
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';REPO=R/'source_integration/repo'
O=W/'wiki_patient_integration_20261114_r1/patient_sources';C=R/'canonical_targets_r11_state_groups_20261114_r1';OLD=W/'fixed_y_remake_20261114_r1/Y_bundle';BASE=W/'measured_y_definition_r1'
sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from cla_engine.measured_y_registry import observation_id,verify_matrix
from cla_engine.measured_members import annotation_features,align_patients
from cla_engine.wikipathways_observations import bind_source_members,score_rna,score_rppa,protein_symbol_matrix,verify_reused_observations,existing_measurement_ids

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def wr(x,p):x.to_csv(p,sep='\t',index=False)
verified={}
def verify(p,digest):
    p=Path(p)
    if str(p) not in verified:
        print('VERIFY',p,flush=True)
        if sha(p)!=digest:raise ValueError('WP_INPUT_DRIFT:'+str(p))
        verified[str(p)]=digest
    elif verified[str(p)]!=digest:raise ValueError('WP_INPUT_HASH_CONFLICT')
    return p
def file_in(root,rec,name):
    h=rec['files'][name];return verify(root/name,h['sha256'] if isinstance(h,dict) else h)
def ref(record,base=None):
    p=Path(record['path']);p=base/p if base and not p.is_absolute() else p
    return verify(p,record['sha256'])
def receipt(folder):return json.loads((folder/'COMPLETE.json').read_text())
def hashes(folder):return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'COMPLETE.json'}

O.mkdir(parents=True,exist_ok=True)
assert not (O/'COMPLETE.json').exists(),'WP_COLLECTION_ALREADY_COMPLETE'
sj(O/'LAUNCH_PREFLIGHT.json',dict(target_host='149',actual_host=socket.gethostname(),operation='WikiPathways source patient observations and RNA/protein scores',training=False,lncrna_associations=False,sealed_outer_test_opened=False,code_sha256=sha(__file__)))
test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_wikipathways_observations.py','tests/test_measured_members.py'],cwd=REPO,capture_output=True,text=True)
sj(O/'TESTS.json',dict(returncode=test.returncode,output=test.stdout+test.stderr));print(test.stdout,flush=True);assert test.returncode==0,test.stdout+test.stderr

cr=receipt(C)
defs=rd(file_in(C,cr,'DEFINITION_TO_IDENTITY.tsv'));defs=defs[defs.database_family.eq('WIKIPATHWAYS')].copy();assert len(defs)==193
composition=rd(file_in(C,cr,'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz'));composition=composition[composition.node_id.isin(defs.node_id)].copy()
annotation=rd(file_in(C,cr,'PROTEIN_ANNOTATION.tsv'))
first=BASE/'protein_coding_inputs/TCGA_BRCA';first_rec=receipt(first)
coding=rd(file_in(first,first_rec,'SCORING_GENE_ANNOTATION.tsv'));coding_symbols=coding.set_index('gene_id').reference_symbol.to_dict()
members=composition.merge(defs[['node_id','authority_id','display_name']],on='node_id',validate='many_to_one').merge(annotation,on='uniprot_id',how='left',validate='many_to_many').fillna('')
members=members.rename(columns={'authority_id':'source_target_id','symbol':'annotation_symbol'})
members['reference_symbol']=[coding_symbols.get(g,s) for g,s in zip(members.ensembl_gene_id,members.annotation_symbol)]
members['coding_background_member']=members.ensembl_gene_id.isin(coding.gene_id)
members=members.drop_duplicates().sort_values(['source_target_id','uniprot_id','ensembl_gene_id'])
wr(members,O/'SOURCE_MEMBERS.tsv.gz')
source_ids=sorted(defs.authority_id.tolist());assert len(set(source_ids))==193
gene_sets={sid:set(members.loc[members.source_target_id.eq(sid),'ensembl_gene_id'])-{''} for sid in source_ids}
symbol_sets={sid:set(members.loc[members.source_target_id.eq(sid),'reference_symbol'])-{''} for sid in source_ids}
union_genes=set().union(*gene_sets.values())
targets=defs[['authority_id','display_name','node_id','integrated_canonical_id','protein_count']].rename(columns={'authority_id':'source_target_id','integrated_canonical_id':'mapped_identity_id'})
targets['definition_id']='WIKIPATHWAYS:'+targets.source_target_id;targets['source_database']='WIKIPATHWAYS'
targets['mapped_gene_count']=targets.source_target_id.map({s:len(g) for s,g in gene_sets.items()})
targets['mapped_symbol_count']=targets.source_target_id.map({s:len(g) for s,g in symbol_sets.items()})
targets['score_dimensions_registered']=2;targets['registered_score_layers']='RNA|PROTEIN';wr(targets,O/'SOURCE_TARGETS.tsv')
with (O/'WIKIPATHWAYS_SYMBOLS.gmt').open('w') as f:
    for sid in source_ids:f.write(sid+'\tWikiPathways frozen source definition\t'+'\t'.join(sorted(symbol_sets[sid]))+'\n')

binding=json.loads((W/'RELATION_MOLECULAR_BINDINGS.json').read_text())
rr=ref(binding['tcpa_rppa500']);tr=rr.parent;tcpa=json.loads(rr.read_text())
tcpa_ann=rd(file_in(tr,tcpa,'FEATURE_CATALOG.tsv'))
rr=ref(binding['cptac_aligned']);cpr=rr.parent;cptac=json.loads(rr.read_text())
cptac_ann=pd.read_parquet(file_in(cpr,cptac,'FEATURE_ALIGNMENT.parquet'))
rr=ref(binding['gene_node_omics']);gnr=rr.parent;gn=json.loads(rr.read_text())
gene_path=ref(gn['genes'],gnr);gene_axis=pd.read_parquet(gene_path).gene_id.tolist();gene_positions={g:i for i,g in enumerate(gene_axis)}
gene_selected=[g for g in gene_axis if g in union_genes];gene_indices=[gene_positions[g] for g in gene_selected]
rscript=REPO/'scripts/score_cptac_protein.R'
Rlibs=json.loads(Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo/config/cptac_149_sources_20261104.json').read_text())['R_libraries']
all_summary=[];all_coverage=[];completed={}

for cancer in ['BRCA','LUAD','COAD']:
  for cohort in ['TCGA','CPTAC']:
    case=cohort+'_'+cancer;dest=O/'cohorts'/case;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        previous=receipt(dest)
        for n,h in previous['files'].items():assert sha(dest/n)==h
        completed[case]=previous;all_summary.append(previous['summary']);all_coverage.extend(rd(dest/'SOURCE_LAYER_COVERAGE.tsv').to_dict('records'))
        print('REUSE',case,flush=True);continue
    print('CASE_START',case,flush=True)
    oldrec=receipt(OLD/'cohorts'/case);split_path=file_in(OLD/'cohorts'/case,oldrec,'PATIENT_PARTITION.json');split=json.loads(split_path.read_text())
    patients=split['reference_patients']+split['query_patients'];references=split['reference_patients']
    assert split['outer_fold']==0 and not (set(split['reference_folds']+split['query_folds'])&{0}) and len(patients)==len(set(patients))
    sj(dest/'PATIENT_PARTITION.json',split)
    sj(dest/'LAUNCH_PREFLIGHT.json',dict(target_host='149',case=case,patients=len(patients),reference=len(references),query=len(split['query_patients']),training=False,outer_test_opened=False))
    frames=[];obsrows=[];links=[];contexts=[];layer_frames={};layer_links={};input_records=[];scorecoverage=[]
    def add_members(layer,values,features,units,sourcefiles):
        bindings=bind_source_members(members,features)
        ids=set(bindings.feature_id);selected=[g for g in values if g in ids]
        values,mask=align_patients(values[selected],patients)
        meta=features.set_index('feature_id').loc[selected].reset_index()
        lookup={fid:observation_id(cohort,'member_measurement',layer,fid) for fid in selected}
        for row in meta.to_dict('records'):
            fid=row['feature_id'];obsrows.append(dict(observation_id=lookup[fid],cohort=cohort,kind='member_measurement',layer=layer,
                matrix_id=cohort+'::member::'+layer,feature_id=fid,definition_id=fid,source_target_id='',display_category='Pathway',canonical_display_roles='Pathway',
                measurement_layer=layer,method='original measured member value',units=units,definition_json=row['source_annotation_json'],method_variant='SOURCE_MEASUREMENT',
                supervision_eligible=True,eligibility_basis='verified original measured member',patients_measured=int(mask[fid].sum()),independent_loss_enabled=False,bound_target_count=0))
        bindings['observation_id']=bindings.feature_id.map(lookup);bindings['layer']=layer
        for row in bindings.to_dict('records'):
            if row['is_source_observation']:links.append(row)
            else:contexts.append(row)
        values.columns=[lookup[fid] for fid in selected];frames.append(values)
        layer_frames[layer]=values;layer_links[layer]=bindings
        input_records.extend(sourcefiles)
        print('MEMBERS',case,layer,len(values.columns),len(bindings),flush=True)
    def add_scores(layer,scores,method,coverage,details):
        scores=scores.reindex(index=patients,columns=source_ids);scores.index.name='patient_id'
        assert not np.isinf(scores.to_numpy()).any()
        columns=[]
        for sid in source_ids:
            feature=sid+'::'+layer+('::raw_ssgsea' if 'SSGSEA' in method.upper() else '::rppa_reference_z_group_mean')
            oid=observation_id(cohort,'source_readout',layer,feature);columns.append(oid)
            info=dict(source_target_id=sid,definition_id='WIKIPATHWAYS:'+sid,source_database='WIKIPATHWAYS',method=method,**details)
            obsrows.append(dict(observation_id=oid,cohort=cohort,kind='source_readout',layer=layer,matrix_id=cohort+'::WP_readout::'+layer,
                feature_id=feature,definition_id='WIKIPATHWAYS:'+sid,source_target_id=sid,display_category='Pathway',canonical_display_roles='Pathway',
                measurement_layer='RNA_EXPRESSION' if layer=='RNA' else 'PROTEIN_ABUNDANCE',method=method,units='source-specific raw ES' if 'SSGSEA' in method.upper() else 'reference-standardized antibody gene-group mean',
                definition_json=json.dumps(info,ensure_ascii=False,sort_keys=True),method_variant='WIKIPATHWAYS_SOURCE_V1',supervision_eligible=True,
                eligibility_basis='verified coding/member source dependency',patients_measured=int(scores[sid].notna().sum()),independent_loss_enabled=False,bound_target_count=0))
            links.append(dict(source_target_id=sid,feature_id=feature,observation_id=oid,evidence_role='source_pathway_readout',is_source_observation=True,
                assay_gene_ids='',matched_member_gene_ids='',member_scope='original_WikiPathways_source_definition',independent_loss_enabled=False,layer=layer))
        scores.columns=columns;frames.append(scores)
        coverage=coverage.copy();coverage['case']=case;coverage['layer']=layer;scorecoverage.extend(coverage.to_dict('records'))
        print('SCORES',case,layer,int(scores.notna().any().sum()),flush=True)

    coding_root=BASE/'protein_coding_inputs'/case;coding_rec=receipt(coding_root)
    annp=file_in(coding_root,coding_rec,'SCORING_GENE_ANNOTATION.tsv');ann=rd(annp)
    ep=file_in(coding_root,coding_rec,'SYMBOL_LOGTPM.parquet');expression=pd.read_parquet(ep).reindex(patients)
    assert set(expression.index)==set(patients) and list(expression.index)==patients
    rnas,rcov=score_rna(expression,ann,symbol_sets)
    rcov['observed_score_patients']=rcov.target_id.map(rnas.notna().sum().to_dict())
    add_scores('RNA',rnas,'GSVA_SSGSEA_ALPHA025_COMMON_PC_BACKGROUND_RAW_ES',rcov,dict(alpha=.25,minimum_mapped=10,maximum_definition_size=500,normalization='none; raw ES',background_genes=len(expression.columns)))
    input_records.extend([dict(path=str(ep),sha256=sha(ep)),dict(path=str(annp),sha256=sha(annp))])
    gp=file_in(coding_root,coding_rec,'GENE_LOGTPM.parquet');selected=[g for g in ann.gene_id if g in union_genes]
    gene_values=pd.read_parquet(gp,columns=selected).reindex(patients)
    af=ann[ann.gene_id.isin(selected)].copy();af['feature_id']=af.gene_id
    add_members('RNA',gene_values,annotation_features(af,selected),'log2(TPM+1), protein-coding gene IDs',[dict(path=str(gp),sha256=sha(gp))])
    if cohort=='TCGA':
        for layer,name in [('PROTEIN','PROTEIN.parquet'),('PHOSPHO','PHOSPHORYLATION.parquet')]:
            p=file_in(tr,tcpa,name);fields=[x for x in pq.ParquetFile(p).schema_arrow.names if x!='patient_id']
            feat=annotation_features(tcpa_ann,fields,gene_column='gene_ids')
            raw=pd.read_parquet(p,columns=['patient_id',*fields],filters=[('patient_id','in',patients)]).set_index('patient_id').reindex(patients)
            if layer=='PROTEIN':
                ps,counts,pcov,fit=score_rppa(raw,feat,gene_sets,references)
                counts.to_parquet(dest/'PROTEIN_SCORE_OBSERVED_GROUPS.parquet');wr(fit,dest/'RPPA_REFERENCE.tsv')
                sj(dest/'RPPA_REFERENCE.json',dict(reference_patients=references,minimum_reference=3,minimum_observed_groups=3,code_source=str(R/'materialize_rppa_functional_evidence.py'),code_source_sha256=sha(R/'materialize_rppa_functional_evidence.py')))
                add_scores('PROTEIN',ps,'RPPA_TRAIN_Z_GENE_GROUP_MEAN',pcov,dict(minimum_reference=3,minimum_observed_groups=3,normalization='antibody z fitted to reference patients only; mean per gene group; mean selected observed groups'))
            add_members(layer,raw,feat,'published TCPA RPPA500 Level 4 antibody signal',[dict(path=str(p),sha256=sha(p))])
        partition=next(x for x in gn['partitions'] if x['cancer_id']==cancer)
        pidp=ref(partition['patients'],gnr);ids=json.loads(pidp.read_text());position={p:i for i,p in enumerate(ids)}
        present=[p for p in patients if p in position];indices=[position[p] for p in present]
        for channel,layer in [('cnv_signed','CNV'),('methylation_promoter_beta','METHYLATION')]:
            vp=ref(partition['channels'][channel]['values'],gnr);mp=ref(partition['channels'][channel]['available'],gnr)
            matrix=np.load(vp,mmap_mode='r',allow_pickle=False);mask=np.load(mp,mmap_mode='r',allow_pickle=False)
            val=np.array(matrix[np.ix_(indices,gene_indices)]);available=np.array(mask[np.ix_(indices,gene_indices)],bool)
            assert np.array_equal(np.isfinite(val),available)
            value=pd.DataFrame(val,index=present,columns=gene_selected);f=annotation_features(pd.DataFrame({'feature_id':gene_selected,'gene_id':gene_selected}),gene_selected)
            add_members(layer,value,f,gn['units'][channel],[dict(path=str(p),sha256=sha(p)) for p in [vp,mp,pidp,gene_path]])
    else:
        for layer,native in [('PROTEIN','protein'),('PHOSPHO','phosphorylation'),('METHYLATION','methylation_gene')]:
            name=f'matrices/{cancer}/{native}_Tumor.parquet'
            if name not in cptac['files']:continue
            p=file_in(cpr,cptac,name);cols=pq.ParquetFile(p).schema_arrow.names;use=[p+'::T' for p in patients if p+'::T' in cols]
            raw=pd.read_parquet(p,columns=['feature_id',*use]);assert raw.feature_id.is_unique
            native_values=raw.set_index('feature_id').T;native_values.index=[x[:-3] for x in native_values.index]
            native_values=native_values.reindex(patients);native_values.index.name='patient_id'
            native_ann=cptac_ann[cptac_ann.layer.eq(native)]
            if layer=='PROTEIN':
                symbol_values=protein_symbol_matrix(native_values,cptac_ann)
                sinput=dest/'CPTAC_PROTEIN_SYMBOL.tsv.gz'
                if not sinput.exists():symbol_values.T.rename_axis('gene_symbol').to_csv(sinput,sep='\t',na_rep='NA',compression='gzip')
                else:
                    previous_input=pd.read_csv(sinput,sep='\t',index_col=0,float_precision='round_trip').T
                    pd.testing.assert_frame_equal(previous_input,symbol_values,check_names=False)
                sout=dest/'CPTAC_PROTEIN_SCORE'
                if not (sout/'COMPLETE.json').exists():
                    with (dest/'CPTAC_PROTEIN_SCORE.log').open('w') as log:
                        proc=subprocess.run(['/usr/bin/Rscript',str(rscript),'--r-library',':'.join(Rlibs),'--input',str(sinput),'--output',str(sout),'--gmt',str(O/'WIKIPATHWAYS_SYMBOLS.gmt')],stdout=log,stderr=subprocess.STDOUT,timeout=900)
                    assert proc.returncode==0,(dest/'CPTAC_PROTEIN_SCORE.log').read_text()[-5000:]
                prec=receipt(sout)
                assert prec['input_md5']==hashlib.md5(sinput.read_bytes()).hexdigest()
                assert prec['gmt_md5']==hashlib.md5((O/'WIKIPATHWAYS_SYMBOLS.gmt').read_bytes()).hexdigest()
                ps=pd.read_csv(sout/'PROTEIN_PATHWAY_ES.tsv.gz',sep='\t').set_index('pathway_id').T.apply(pd.to_numeric,errors='raise')
                # Empty fields are explicit missing scores, never zeros.
                ps=ps.replace('',np.nan).astype(float).reindex(patients)
                pcov=rd(sout/'PROTEIN_PATHWAY_COVERAGE.tsv').rename(columns={'pathway_id':'target_id','gene_count':'total_genes','observed_in_assay':'mapped_genes'})
                pcov['observed_score_patients']=pcov.target_id.map(ps.notna().sum().to_dict()).fillna(0).astype(int)
                pcov['reason']=np.where(pcov.observed_score_patients.gt(0),'',np.where(pcov.mapped_genes.lt(10),'FEWER_THAN_10_ASSAYED_GENES',np.where(pcov.total_genes.gt(500),'MORE_THAN_500_DEFINITION_GENES','FEWER_THAN_10_OBSERVED_GENES_PER_PATIENT')))
                add_scores('PROTEIN',ps,'GSVA_SSGSEA_ALPHA025_SAMPLEWISE_RAW_ES_NA_RM',pcov,dict(minimum_observed_genes=10,maximum_definition_size=500,alpha=.25,normalization='none; raw ES; use=na.rm',R_script_sha256=sha(rscript),background_genes=len(symbol_values.columns)))
            relevant=set(native_ann.loc[native_ann.gene_id.isin(union_genes),'feature_id']);use=[x for x in native_values if x in relevant]
            feat=annotation_features(native_ann,use)
            add_members(layer,native_values[use],feat,'published CPTAC native scale; exact source feature retained',[dict(path=str(p),sha256=sha(p))])

    values=pd.concat(frames,axis=1);values.index.name='patient_id';assert values.columns.is_unique
    meta=pd.DataFrame(obsrows);assert meta.observation_id.is_unique
    meta=meta.set_index('observation_id').loc[values.columns].rename_axis('observation_id').reset_index()
    observed=values.notna();verify_matrix(values,observed,patients)
    sourcelinks=pd.DataFrame(links).drop_duplicates(['source_target_id','observation_id','evidence_role'])
    meta['bound_source_count']=meta.observation_id.map(sourcelinks.groupby('observation_id').source_target_id.nunique()).fillna(0).astype(int)
    oldfolder=OLD/'cohorts'/case;oldmeta=rd(file_in(oldfolder,oldrec,'OBSERVATIONS.tsv.gz'))
    remap=existing_measurement_ids(meta,oldmeta)
    values.columns=[remap[x] for x in values.columns];observed.columns=values.columns
    meta['observation_id']=meta.observation_id.map(remap);sourcelinks['observation_id']=sourcelinks.observation_id.map(remap)
    for row in contexts:row['observation_id']=remap[row['observation_id']]
    shared=sorted(set(meta.observation_id)&set(oldmeta.observation_id))
    oldvalues=pd.read_parquet(file_in(oldfolder,oldrec,'Y_VALUES.parquet'),columns=shared).reindex(patients)
    reused=verify_reused_observations(values,meta,oldvalues,oldmeta)
    values.to_parquet(dest/'Y_VALUES.parquet',compression='zstd');observed.to_parquet(dest/'Y_OBSERVED.parquet',compression='zstd')
    pd.testing.assert_frame_equal(pd.read_parquet(dest/'Y_VALUES.parquet'),values);pd.testing.assert_frame_equal(pd.read_parquet(dest/'Y_OBSERVED.parquet'),observed)
    wr(meta,dest/'OBSERVATIONS.tsv.gz');wr(sourcelinks,dest/'SOURCE_OBSERVATIONS.tsv.gz');wr(pd.DataFrame(contexts),dest/'SOURCE_CONTEXT_OBSERVATIONS.tsv.gz')
    wr(pd.DataFrame(scorecoverage),dest/'SCORE_COVERAGE.tsv');sj(dest/'INPUT_REFERENCES.json',dict(files=input_records,source_catalog=str(C),source_members_sha256=sha(O/'SOURCE_MEMBERS.tsv.gz')))
    coverage=[]
    for sid in source_ids:
        for layer in ['RNA','PROTEIN','PHOSPHO','CNV','METHYLATION']:
            ls=sourcelinks[sourcelinks.source_target_id.eq(sid)&sourcelinks.layer.eq(layer)]
            mm=ls[ls.evidence_role.eq('member_measurement')].observation_id.tolist();ss=ls[ls.evidence_role.eq('source_pathway_readout')].observation_id.tolist()
            cov=dict(case=case,cohort=cohort,cancer=cancer,source_target_id=sid,layer=layer,patients_total=len(patients),defined_member_observations=len(mm),measured_member_observations=int(observed[mm].any().sum()),patients_with_member_values=int(observed[mm].any(axis=1).sum()),defined_score_observations=len(ss),patients_with_score=int(observed[ss].any(axis=1).sum()),original_gene_count=len(gene_sets[sid]))
            if layer not in layer_frames:cov['score_status']='SOURCE_LAYER_NOT_AVAILABLE'
            elif layer=='PHOSPHO':cov['score_status']='EXACT_SITE_MEMBER_VALUES;NO_DIRECTIONAL_SITE_SIGNATURE_IN_WP_PROTEIN_LIST'
            elif layer in ['CNV','METHYLATION']:cov['score_status']='MEMBER_VALUES;NO_PATHWAY_ACTIVITY_SCORE_ASSUMED'
            else:cov['score_status']='SOURCE_SCORE_WITH_MEASUREMENT_ELIGIBILITY'
            coverage.append(cov)
    wr(pd.DataFrame(coverage),dest/'SOURCE_LAYER_COVERAGE.tsv');all_coverage.extend(coverage)
    summary=dict(case=case,patients=len(patients),source_definitions=193,unique_observations=len(meta),reused_existing_observation_ids=reused,new_observation_ids=len(meta)-reused,member_observations=int(meta.kind.eq('member_measurement').sum()),RNA_score_sources=int(meta.kind.eq('source_readout').mul(meta.layer.eq('RNA')).mul(meta.patients_measured.gt(0)).sum()),PROTEIN_score_sources=int(meta.kind.eq('source_readout').mul(meta.layer.eq('PROTEIN')).mul(meta.patients_measured.gt(0)).sum()),sources_with_any_measurement=sum(int(observed[sourcelinks.loc[sourcelinks.source_target_id.eq(sid),'observation_id']].any().any()) for sid in source_ids))
    result=dict(format='CLA_WIKIPATHWAYS_PATIENT_OBSERVATIONS_V1',target_host='149',summary=summary,files=hashes(dest));sj(dest/'COMPLETE.json',result)
    completed[case]=result;all_summary.append(summary);sj(O/'PROGRESS.json',dict(target_host='149',completed={k:v['summary'] for k,v in completed.items()}))
    print('CASE_COMPLETE',json.dumps(summary),flush=True)

wr(pd.DataFrame(all_summary),O/'COHORT_SUMMARY.tsv');wr(pd.DataFrame(all_coverage),O/'SOURCE_LAYER_COVERAGE.tsv')
finite_layers={sid:sorted({x['layer'] for x in all_coverage if x['source_target_id']==sid and x['patients_with_score']>0}) for sid in source_ids}
targets['has_finite_patient_score']=targets.source_target_id.map(lambda sid:bool(finite_layers[sid]))
targets['measured_score_layers']=targets.source_target_id.map(lambda sid:'|'.join(finite_layers[sid]))
wr(targets,O/'SOURCE_TARGETS.tsv')
sj(O/'SCORING_POLICY.json',dict(target_host='149',RNA='Existing common protein-coding symbol background ssGSEA alpha=.25, minimum10, maximum500; raw ES without cohort range fit',TCGA_PROTEIN='Existing RPPA reference-only antibody z; one value per gene group; observed mean with >=3 groups',CPTAC_PROTEIN='Existing R GSVA ssGSEA alpha=.25 normalize=FALSE use=na.rm; >=10 observed genes, max500',PHOSPHO='Native site/antibody members only; a protein list does not define a directional phosphosite activity signature',CNV_METHYLATION='Original gene measurements; no activity score',source_scope='193 original WikiPathways definitions; direct equality/context routing is assigned in the catalog integration, not inferred from these patient values',missingness='Missing values remain NA; constant finite original measurements remain observed',lncrna_independent=True,training=False))
sj(O/'VERIFIED_INPUTS.json',verified)
sj(O/'COMPLETE.json',dict(format='CLA_WIKIPATHWAYS_PATIENT_SOURCE_COLLECTION_V1',target_host='149',sources=193,cohorts=all_summary,patients=sum(r['patients'] for r in all_summary),training=False,lncrna_independent=True,files=hashes(O)))
print('WIKIPATHWAYS_PATIENT_COLLECTION_COMPLETE',flush=True)
