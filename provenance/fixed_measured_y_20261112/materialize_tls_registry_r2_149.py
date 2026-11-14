import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json,socket,sys,subprocess,shutil
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1';REPO=ROOT/'source_integration/repo'
BASE=WORK/'measured_y_definition_r1';SRC=BASE/'observation_registry_r1';OUT=BASE/'observation_registry_r2';TLS=BASE/'tls12_coding_r1';EXT=BASE/'external_lineage_sources_r1'
sys.path.insert(0,str(REPO))
sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
from cla_engine.common import sha
from cla_engine.coding_fixed_signature import score_fixed_coding_set
from cla_engine.measured_y_registry import observation_id,load_target,verify_matrix
import pandas as pd
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
OUT.mkdir(exist_ok=True);TLS.mkdir(exist_ok=True)
save(OUT/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'append coding-background TLS12 readout with exact GDC signature','training':False,'lncrna_association':False,'outer_test_opened':False})
run=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_coding_fixed_signature.py','tests/test_measured_y_registry.py','tests/test_measured_members.py','tests/test_pathway_conditions.py'],cwd=REPO,capture_output=True,text=True)
save(OUT/'TESTS.json',{'target_host':'149','returncode':run.returncode,'output':run.stdout+run.stderr});print(run.stdout,flush=True)
if run.returncode:raise RuntimeError(run.stderr)
source_receipt=json.loads((SRC/'COMPLETE.json').read_text())
for cancer,h in source_receipt['receipt_hashes'].items():assert sha(SRC/cancer/'COMPLETE.json')==h
sources=json.loads((EXT/'SOURCES.json').read_text());gene_source=next(x for x in sources['files'] if x['file']=='PanImmune_GeneSet_Definitions.xlsx')
assert sha(EXT/gene_source['file'])==gene_source['sha256']
import openpyxl
book=openpyxl.load_workbook(EXT/gene_source['file'],read_only=True,data_only=True)
assert ('Wolf','Chemokine12_score') in list(book['setSources'].values)
genes=[r[1] for r in book['Genes'].values if r[0]=='Chemokine12_score'];book.close()
assert len(genes)==12 and len(set(genes))==12
CID='CLA-S-4F0486ABF1B8D70A';METHOD='TLS12_GDC_SSGSEA_PC_RAW_V1';DEFINITION='STATE|TLS12_Chemokine_GDC@@'+METHOD
definition={'biological_definition':'TLS-associated 12-chemokine transcriptional signature','source_definition':'PanImmune / Wolf / Chemokine12_score','source_url':gene_source['source_url'],'source_file_sha256':gene_source['sha256'],'members':genes,'method':'ssGSEA alpha=0.25, fixed coding gene background, unnormalized enrichment','expression_units':'log2(TPM+1)','patient_cohort_reference_fitted':False,'numerical_equivalence_to_published_PanImmune_scores':False}
save(TLS/'DEFINITION.json',definition)
cases=[];counts=[]
def verify_case(folder,cancer):
    check=[]
    for cohort in ['TCGA','CPTAC']:
        v,m,a,metadata=load_target(folder,CID,cohort)
        oid=observation_id(cohort,'readout','STATE',DEFINITION)
        expected=pd.read_parquet(TLS/(cohort+'_'+cancer)/'VALUES.parquet')[DEFINITION]
        assert oid in v and metadata.supervision_eligible.all()
        pd.testing.assert_series_equal(v[oid].rename(DEFINITION),expected)
        pd.testing.assert_series_equal(m[oid].rename(DEFINITION),expected.notna())
        check.append({'cohort':cohort,'TLS12_patients_measured':int(m[oid].sum()),'published_reference_unchanged':True,'missing_background_retained':True})
    save(folder/'ACCESSOR_VALIDATION.json',{'target_host':'149','checks':check})
for cancer in ['BRCA','LUAD','COAD']:
    old=SRC/cancer;new=OUT/cancer
    if (new/'COMPLETE.json').exists():
        prior=json.loads((new/'COMPLETE.json').read_text())
        for name,h in prior['files'].items():assert sha(new/name)==h
        verify_case(new,cancer);cases.append(prior)
        for cohort in ['TCGA','CPTAC']:counts.append(json.loads((TLS/(cohort+'_'+cancer)/'COMPLETE.json').read_text()))
        print('TLS_REGISTRY_REUSED',cancer,flush=True)
        continue
    new.mkdir(exist_ok=True)
    rec=json.loads((old/'COMPLETE.json').read_text())
    for name,h in rec['files'].items():assert sha(old/name)==h
    for name in ['TARGETS.tsv','PATIENT_PARTITIONS.json','INPUT_REFERENCES.json']:
        shutil.copyfile(old/name,new/name)
    obs=pd.read_csv(old/'OBSERVATIONS.tsv.gz',sep='\t',keep_default_na=False)
    links=pd.read_csv(old/'PATHWAY_OBSERVATIONS.tsv.gz',sep='\t',keep_default_na=False)
    methods=pd.read_csv(old/'SOURCE_READOUT_METHODS.tsv.gz',sep='\t',keep_default_na=False)
    coverage=pd.read_csv(old/'COVERAGE.tsv',sep='\t',keep_default_na=False)
    matrices=json.loads((old/'MATRICES.json').read_text())
    for cohort in ['TCGA','CPTAC']:
        case=cohort+'_'+cancer;input_dir=BASE/'protein_coding_inputs'/case;out=TLS/case;out.mkdir(exist_ok=True)
        inputs=json.loads((input_dir/'COMPLETE.json').read_text())
        assert inputs['scoring_background']=='PROTEIN_CODING_GENE_IDS_BEFORE_SYMBOL_AGGREGATION'
        for name,h in inputs['files'].items():assert sha(input_dir/name)==h
        expression=pd.read_parquet(input_dir/'SYMBOL_LOGTPM.parquet');annotation=pd.read_csv(input_dir/'SCORING_GENE_ANNOTATION.tsv',sep='\t')
        patients=matrices['patients'][cohort];assert set(expression.index)==set(patients)
        expression=expression.reindex(patients)
        scores,mapping=score_fixed_coding_set(expression,annotation,genes,DEFINITION)
        verify_matrix(scores,scores.notna(),patients)
        scores.to_parquet(out/'VALUES.parquet',compression='zstd');scores.notna().to_parquet(out/'OBSERVED.parquet',compression='zstd');mapping.to_csv(out/'GENE_COVERAGE.tsv',sep='\t',index=False)
        done={'target_host':'149','case':case,'patients':len(scores),'patients_measured':int(scores.notna().sum().iloc[0]),'members':12,'scoring_background_genes':len(expression.columns),'definition_sha256':sha(TLS/'DEFINITION.json'),'coding_input_receipt_sha256':sha(input_dir/'COMPLETE.json'),'model_training':False,'lncrna_association':False,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}}
        save(out/'COMPLETE.json',done);counts.append(done)
        mid=cohort+'::TLS12::STATE';oid=observation_id(cohort,'readout','STATE',DEFINITION)
        matrices['blocks'][mid]={'cohort':cohort,'values':{'path':str(out/'VALUES.parquet'),'sha256':sha(out/'VALUES.parquet')},'observed':{'path':str(out/'OBSERVED.parquet'),'sha256':sha(out/'OBSERVED.parquet')}}
        row=dict(observation_id=oid,cohort=cohort,kind='source_readout',layer='STATE',matrix_id=mid,feature_id=DEFINITION,definition_id=DEFINITION,source_target_id='TLS12_Chemokine_Published',display_category='State',canonical_display_roles='State',measurement_layer='RNA_EXPRESSION',method=definition['method'],units='unnormalized ssGSEA enrichment',definition_json=json.dumps(definition,ensure_ascii=False),method_variant=METHOD,supervision_eligible=True,eligibility_basis='protein-coding gene-ID input and exact GDC member definition',patients_measured=done['patients_measured'],independent_loss_enabled=False,bound_target_count=1)
        obs=pd.concat([obs,pd.DataFrame([row])],ignore_index=True)
        links=pd.concat([links,pd.DataFrame([dict(canonical_id=CID,observation_id=oid,evidence_role='same_state_readout',condition_required=False,scope_relation='same_identity',routing_basis='exact GDC TLS12 gene definition with versioned scoring method',molecular_components='|'.join(genes),is_target_observation=True,independent_loss_enabled=False)])],ignore_index=True)
        original=methods[(methods.cohort==cohort)&(methods.source_target_id=='TLS12_Chemokine_Published')].iloc[0].to_dict()
        original.update(target_id=DEFINITION,column_index=0,original_target_id=original['target_id'],coding_expression_basis_verified=True,measurement_definition='FIXED_CODING_GDC_TLS12',applied_method=definition['method'],method_variant=METHOD,patients_measured=done['patients_measured'],measured_fraction=done['patients_measured']/len(scores),source_metadata_json=json.dumps(definition),nonconstant=bool(scores.iloc[:,0].nunique()>1),category_basis='SOURCE_TASK_ROLES')
        methods=pd.concat([methods,pd.DataFrame([original])],ignore_index=True)
        where=(coverage.cohort==cohort)&(coverage.canonical_id==CID)&(coverage.layer=='STATE');assert where.sum()==1
        v,m,_,_=load_target(old,CID,cohort)
        union=m.any(axis=1)|scores.iloc[:,0].notna()
        coverage.loc[where,'defined_observations']+=1;coverage.loc[where,'observations_with_measurement']+=int(done['patients_measured']>0)
        coverage.loc[where,'patients_measured']=int(union.sum());coverage.loc[where,'measured_fraction']=float(union.mean())
    assert obs.observation_id.is_unique and not links.duplicated(['canonical_id','observation_id','evidence_role']).any()
    for frame,name in [(obs,'OBSERVATIONS.tsv.gz'),(links,'PATHWAY_OBSERVATIONS.tsv.gz'),(methods,'SOURCE_READOUT_METHODS.tsv.gz')]:frame.to_csv(new/name,sep='\t',index=False,compression='gzip')
    coverage.to_csv(new/'COVERAGE.tsv',sep='\t',index=False);save(new/'MATRICES.json',matrices)
    refs=json.loads((new/'INPUT_REFERENCES.json').read_text());refs['previous_registry']={'path':str(old/'COMPLETE.json'),'sha256':sha(old/'COMPLETE.json')};refs['TLS12_definition']={'path':str(TLS/'DEFINITION.json'),'sha256':sha(TLS/'DEFINITION.json')};save(new/'INPUT_REFERENCES.json',refs)
    rec.update(format='CLA_PATHWAY_PATIENT_OBSERVATION_REGISTRY_V2',observations=len(obs),target_bindings=len(links),matrix_blocks=len(matrices['blocks']),files={p.name:sha(p) for p in new.iterdir() if p.is_file() and p.name!='COMPLETE.json'})
    save(new/'COMPLETE.json',rec)
    verify_case(new,cancer);cases.append(rec)
    print('TLS_REGISTRY_COMPLETE',cancer,flush=True)
save(TLS/'COMPLETE.json',{'target_host':'149','cases':counts,'definition_sha256':sha(TLS/'DEFINITION.json'),'source_definition':gene_source,'training':False,'lncrna_association':False})
save(OUT/'COMPLETE.json',dict(source_receipt,format='CLA_PATHWAY_PATIENT_OBSERVATION_COLLECTION_V2',cases=cases,receipt_hashes={x['cancer']:sha(OUT/x['cancer']/'COMPLETE.json') for x in cases},TLS12_definition=str(TLS/'DEFINITION.json'),external_reference_policy='Original published RNA endpoints retain source values; coding TLS12 is an explicitly different scoring version',unresolved_external_RNA_lineage=['ESTIMATE::TumorPurity']))
print('TLS_COLLECTION_COMPLETE',flush=True)
