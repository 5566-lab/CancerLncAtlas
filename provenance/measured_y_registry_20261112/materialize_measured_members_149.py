import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,socket,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
WORK=ROOT/'kegg_cancer_redefinition_20261111_r1';BASE=WORK/'measured_y_definition_r1'
REPO=ROOT/'source_integration/repo';OUT=BASE/'member_measurements_r1'
assert socket.gethostname().split('.')[0]=='149'
OUT.mkdir(exist_ok=True);sys.path.insert(0,str(REPO))
from cla_engine.common import sha
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
preflight={'target_host':'149','operation':'export exact patient member measurements and binding metadata','model_training':False,'lncrna_associations':False,'time_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__)}
save(OUT/'LAUNCH_PREFLIGHT.json',preflight)
tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_measured_members.py'],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
save(OUT/'TESTS.json',dict(preflight,exit_code=tests.returncode,output=tests.stdout))
print(tests.stdout,flush=True)
if tests.returncode:raise SystemExit(tests.returncode)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from cla_engine.measured_members import annotation_features,bind_members,align_patients,coverage
from cla_engine.pathway_conditions import applicability_matrix
from cla_engine.measured_readouts import read_partition

verified={}
def verify(path,digest):
    path=Path(path)
    if verified.get(str(path))!=digest:
        if sha(path)!=digest:raise ValueError('MEMBER_INPUT_DRIFT:'+str(path))
        verified[str(path)]=digest
    return path
def verify_record(record,base=None):
    path=Path(record['path']);path=base/path if base is not None and not path.is_absolute() else path
    return verify(path,record['sha256'])
def file_in(root,receipt,name):
    item=receipt['files'][name]
    return verify(root/name,item['sha256'] if isinstance(item,dict) else item)

struct=WORK/'pathway_graph_sources_r2'
sr=json.loads((struct/'COMPLETE.json').read_text())
member_path=file_in(struct,sr,'MEMBERS.tsv')
members=pd.read_csv(member_path,sep='\t',keep_default_na=False)
genes=set(members.ensembl_gene_id)
condition_dir=BASE/'condition_definitions_r1';cr=json.loads((condition_dir/'COMPLETE.json').read_text())
condition_path=file_in(condition_dir,cr,'PATHWAY_CONDITIONS.tsv')
conditions=pd.read_csv(condition_path,sep='\t',keep_default_na=False)
assert conditions.condition_required.dtype==bool and len(conditions)==477
canonical_ids=conditions.canonical_id.tolist()

bindings=json.loads((WORK/'RELATION_MOLECULAR_BINDINGS.json').read_text())
products={}
for key in ('tcpa_rppa500','cptac_aligned','gene_node_omics'):
    p=verify_record(bindings[key]);products[key]=(p.parent,json.loads(p.read_text()))
tcpa_root,tcpa=products['tcpa_rppa500']
tcpa_annotation_path=file_in(tcpa_root,tcpa,'FEATURE_CATALOG.tsv')
tcpa_ann=pd.read_csv(tcpa_annotation_path,sep='\t',keep_default_na=False)
cptac_root,cptac=products['cptac_aligned']
cptac_annotation_path=file_in(cptac_root,cptac,'FEATURE_ALIGNMENT.parquet')
cptac_ann=pd.read_parquet(cptac_annotation_path)
gn_root,gn=products['gene_node_omics']
assert gn['format']=='GENE_PATIENT_ASSAYS_VERIFIED_V1' and gn['matrix_orientation']=='patient_by_gene'
gene_path=verify_record(gn['genes'],gn_root)
gene_axis=pd.read_parquet(gene_path).gene_id.tolist()
assert len(gene_axis)==len(set(gene_axis))
gene_position={g:j for j,g in enumerate(gene_axis)}
gene_selected=[g for g in gene_axis if g in genes]
gene_columns=[gene_position[g] for g in gene_selected]

completed=[]
def write_block(case,cohort,layer,patient_ids,values,features,source_files,units):
    dest=OUT/case/layer
    if dest.exists():raise ValueError('MEMBER_OUTPUT_EXISTS:'+str(dest))
    links=bind_members(members,features,conditions)
    retained=set(links.feature_id)
    ordered=[fid for fid in values.columns if fid in retained]
    values,mask=align_patients(values[ordered],patient_ids)
    features=features[features.feature_id.isin(ordered)].set_index('feature_id').loc[ordered].reset_index()
    assert list(features.feature_id)==list(values)
    dest.mkdir(parents=True)
    save(dest/'LAUNCH_PREFLIGHT.json',dict(preflight,case=case,cohort=cohort,layer=layer))
    values.to_parquet(dest/'VALUES.parquet',compression='zstd')
    mask.to_parquet(dest/'OBSERVED.parquet',compression='zstd')
    features.to_csv(dest/'FEATURES.tsv.gz',sep='\t',index=False,compression='gzip')
    links.to_csv(dest/'PATHWAY_MEMBER_BINDINGS.tsv.gz',sep='\t',index=False,compression='gzip')
    cov=coverage(values,mask,links,canonical_ids)
    cov.to_csv(dest/'MEASUREMENT_COVERAGE.tsv',sep='\t',index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(dest/'VALUES.parquet'),values)
    pd.testing.assert_frame_equal(pd.read_parquet(dest/'OBSERVED.parquet'),mask)
    result={'format':'CLA_PATIENT_MEMBER_MEASUREMENTS_V1','target_host':'149','case':case,'cohort':cohort,'layer':layer,'patients':len(values),
        'features':len(values.columns),'observed_features':int(mask.any(axis=0).sum()),'observed_cells':int(mask.to_numpy().sum()),
        'pathways_with_member_measurement':int(cov.patients_with_member_measurement.gt(0).sum()),
        'pathways_with_ambiguous_group_measurement':int(cov.patients_with_ambiguous_group_measurement.gt(0).sum()),
        'units':units,'original_measurements_preserved':True,'measurement_duplicated_per_gene':False,
        'source_files':source_files,'backbone_members':{'path':str(member_path),'sha256':sha(member_path)},
        'condition_definitions':{'path':str(condition_path),'sha256':sha(condition_path)},
        'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
    save(dest/'COMPLETE.json',result)
    completed.append({k:v for k,v in result.items() if k not in ('files','source_files')})
    save(OUT/'PROGRESS.json',{'target_host':'149','completed':completed})
    print('MEMBER_COMPLETE '+case+' '+layer+' '+str(result['features'])+' features; '+str(result['pathways_with_member_measurement'])+' measured pathways',flush=True)

for cancer in ('BRCA','LUAD','COAD'):
    measured=BASE/'measurements_coding_r3'/cancer/'validation'
    mr=json.loads((measured/'COMPLETE.json').read_text())
    sp=file_in(measured,mr,'PATIENT_PARTITIONS.json')
    splits=json.loads(sp.read_text())
    for cohort in ('TCGA','CPTAC'):
        case=cohort+'_'+cancer;split=splits[cohort]
        assert split['outer_fold']==0 and 0 not in split['reference_folds']+split['query_folds']
        patients=split['reference_patients']+split['query_patients']
        assert len(patients)==len(set(patients))
        (OUT/case).mkdir(exist_ok=True)
        save(OUT/case/'PATIENT_PARTITIONS.json',split)

        coding=BASE/'protein_coding_inputs'/case
        rr=json.loads((coding/'COMPLETE.json').read_text())
        rp=file_in(coding,rr,'GENE_LOGTPM.parquet')
        aa=file_in(coding,rr,'SCORING_GENE_ANNOTATION.tsv')
        ann=pd.read_csv(aa,sep='\t',keep_default_na=False)
        assert ann.gene_biotype.eq('protein_coding').all()
        selected=[g for g in pq.ParquetFile(rp).schema_arrow.names if g in genes]
        v=read_partition(rp,patients,selected)
        af=ann.rename(columns={'gene_id':'feature_id'}).copy();af['gene_id']=af.feature_id
        f=annotation_features(af,selected)
        write_block(case,cohort,'RNA',patients,v,f,{'matrix':{'path':str(rp),'sha256':sha(rp)},'annotation':{'path':str(aa),'sha256':sha(aa)}},'log2(TPM+1), protein-coding gene IDs')

        if cohort=='TCGA':
            for layer,name in [('PROTEIN','PROTEIN.parquet'),('PHOSPHO','PHOSPHORYLATION.parquet')]:
                p=file_in(tcpa_root,tcpa,name)
                fields=[x for x in pq.ParquetFile(p).schema_arrow.names if x!='patient_id']
                f=annotation_features(tcpa_ann,fields,gene_column='gene_ids')
                linked=set(bind_members(members,f,conditions).feature_id)
                use=[x for x in fields if x in linked]
                v=pd.read_parquet(p,columns=['patient_id',*use],filters=[('patient_id','in',patients)]).set_index('patient_id')
                write_block(case,cohort,layer,patients,v,f[f.feature_id.isin(use)],{'matrix':{'path':str(p),'sha256':sha(p)},'annotation':{'path':str(tcpa_annotation_path),'sha256':sha(tcpa_annotation_path)}},'published TCPA RPPA500 Level 4 antibody signal')
            partition=next(x for x in gn['partitions'] if x['cancer_id']==cancer)
            pid_path=verify_record(partition['patients'],gn_root)
            ids=json.loads(pid_path.read_text());positions={x:j for j,x in enumerate(ids)}
            if len(ids)!=len(positions):raise ValueError('TCGA_GENE_DUPLICATE_PATIENTS')
            present=[p for p in patients if p in positions]
            indices=[positions[p] for p in present]
            for channel,layer in [('cnv_signed','CNV'),('methylation_promoter_beta','METHYLATION')]:
                value_path=verify_record(partition['channels'][channel]['values'],gn_root)
                mask_path=verify_record(partition['channels'][channel]['available'],gn_root)
                raw=np.load(value_path,mmap_mode='r',allow_pickle=False)
                raw_mask=np.load(mask_path,mmap_mode='r',allow_pickle=False)
                assert raw.shape==raw_mask.shape==(len(ids),len(gene_axis))
                values=np.array(raw[np.ix_(indices,gene_columns)])
                masks=np.array(raw_mask[np.ix_(indices,gene_columns)],dtype=bool)
                assert np.array_equal(np.isfinite(values),masks)
                if layer=='METHYLATION':assert ((values[masks]>=0)&(values[masks]<=1)).all()
                v=pd.DataFrame(values,index=present,columns=gene_selected)
                af=pd.DataFrame({'feature_id':gene_selected,'gene_id':gene_selected})
                f=annotation_features(af,gene_selected)
                write_block(case,cohort,layer,patients,v,f,{'matrix':{'path':str(value_path),'sha256':sha(value_path)},'mask':{'path':str(mask_path),'sha256':sha(mask_path)},'annotation':{'path':str(gene_path),'sha256':sha(gene_path)},'patients':{'path':str(pid_path),'sha256':sha(pid_path)}},gn['units'][channel])
        else:
            for layer,native in [('PROTEIN','protein'),('PHOSPHO','phosphorylation'),('METHYLATION','methylation_gene')]:
                name=f'matrices/{cancer}/{native}_Tumor.parquet'
                if name not in cptac['files']:
                    save(OUT/case/(layer+'_SOURCE_ABSENT.json'),{'target_host':'149','case':case,'layer':layer,'reason':'assay absent from pinned aligned product','source_receipt_sha256':bindings['cptac_aligned']['sha256']})
                    continue
                p=file_in(cptac_root,cptac,name)
                columns=pq.ParquetFile(p).schema_arrow.names
                if any(not x.endswith('::T') for x in columns if x!='feature_id'):raise ValueError('CPTAC_NON_TUMOR_AXIS')
                use=[p+'::T' for p in patients if p+'::T' in columns]
                raw=pd.read_parquet(p,columns=['feature_id',*use])
                if raw.feature_id.duplicated().any():raise ValueError('CPTAC_DUPLICATE_FEATURES')
                ann=cptac_ann[cptac_ann.layer.eq(native)]
                relevant=ann[ann.gene_id.isin(genes)].feature_id.unique()
                raw=raw[raw.feature_id.isin(relevant)]
                v=raw.set_index('feature_id').T
                v.index=[x[:-3] for x in v.index]
                f=annotation_features(ann,raw.feature_id.tolist())
                write_block(case,cohort,layer,patients,v,f,{'matrix':{'path':str(p),'sha256':sha(p)},'annotation':{'path':str(cptac_annotation_path),'sha256':sha(cptac_annotation_path)}},'published CPTAC native scale; exact source feature retained')

        condition,known=applicability_matrix(conditions,patients,pd.DataFrame())
        condition.to_parquet(OUT/case/'CONDITION_APPLICABILITY.parquet',compression='zstd')
        known.to_parquet(OUT/case/'CONDITION_KNOWN.parquet',compression='zstd')
        save(OUT/case/'CONDITION_SOURCE.json',{'target_host':'149','qualified_patient_observations_loaded':0,'reference_mechanisms':334,'conditional_mechanisms':143,'unknown_conditions_are_not_negative':True,'member_measurements_remain_independent_of_condition_status':True})
save(OUT/'COMPLETE.json',{'target_host':'149','format':'CLA_MEASURED_MEMBER_COLLECTION_V1','blocks':completed,'verified_inputs':verified,
    'no_lncRNA_input':True,'no_model_training':True,'automatic_auxiliary_losses':False,
    'case_metadata':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*/*') if p.is_file() and p.parent.name.startswith(('TCGA_','CPTAC_'))},
    'block_receipts':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*/*/COMPLETE.json')}})
print('MEMBER_COLLECTION_COMPLETE '+str(len(completed)),flush=True)
