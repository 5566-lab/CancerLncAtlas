"""Describe actual CPTAC State calculations and covariate coverage per cohort."""
from pathlib import Path
import json,socket,hashlib
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;P=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1');O=R/'cptac_method_index_r1';O.mkdir(exist_ok=False)
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'INDEX_NATIVE_CPTAC_METHODS_AND_COVARIATES'}))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
policy=json.loads((P/'bundle_r1/METHOD_POLICY.json').read_text());bound=json.loads((P/'bundle_r1/BOUND_SOURCES.json').read_text());native=json.loads((P/'native_states_r1/COMPLETE.json').read_text());extra=json.loads((P/'extra_states_r1/COMPLETE.json').read_text());receipts={};source_hashes={}
for record in bound:
 if record['role']=='RNA_METHOD_RECEIPT':
  p=Path(record['path']);assert sha(p)==record['sha256'];receipts[p.stem]=json.loads(p.read_text());source_hashes[str(p)]=record['sha256']
coverage=pd.read_csv(R/'supervision_summary_r1/CURRENT_SCORE_COVERAGE.tsv.gz',sep='\t');selected=coverage.loc[coverage.cohort.str.startswith('CPTAC:')&coverage.method.eq('Source-specific State score')&coverage.patients_with_score.gt(0),['canonical_id','source_readout_id','cohort','cancer_id','measurement_layer']].drop_duplicates();assert not selected.duplicated(['cohort','source_readout_id']).any();rows=[]
for row in selected.itertuples():
 source=row.source_readout_id;cohort=row.cohort.split(':',1)[1];rec=receipts[cohort];methods=rec['method_records'];declared=policy['State_native_methods'][source];method='';reference='All tumor RNA samples in '+row.cohort;unit='Signature-specific score';scale='';software='';parameters=''
 if source.startswith('signifinder::'):
  matches=[v for v in methods.values() if source in ([v.get('columns')] if isinstance(v.get('columns'),str) else v.get('columns',[]))];assert len(matches)==1;v=matches[0];method='signifinder '+v['function_name'];software='signifinder '+rec['package_version'];scale=v['input_scale'];parameters=json.dumps(v.get('arguments',{}),sort_keys=True)
 elif source.startswith('native_program::'):
  v=methods['Routed_programs'];method='Weighted gene-expression z score followed by program z score';scale=v['input_scale'];unit='Program z score';parameters=json.dumps({'gene_sd_ddof':v['gene_sd_ddof'],'program_sd_ddof':v['program_sd_ddof'],'minimum_detected_genes':v['min_detected']},sort_keys=True)
 elif source.startswith('SenMayo::'):
  v=methods['SenMayo'];method='GSVA ssGSEA with score-range normalization';scale=v['input_scale'];software='GSVA '+rec['GSVA_version'];unit='Normalized enrichment score';parameters=json.dumps({k:v[k] for k in ['gene_set','gene_set_size','genes_measured','normalize']},sort_keys=True)
 elif source.startswith('HALLMARK::'):
  method='GSVA ssGSEA in the 50-Hallmark scoring context';scale='log2(TPM+1)';software='GSVA '+rec['GSVA_version'];unit='Normalized enrichment score'
 elif source.startswith('CYT::'):
  method='Mean log2(TPM+1) of GZMA and PRF1';reference='Within-patient expression of two genes';scale='log2(TPM+1)';unit='Mean log2(TPM+1)'
 elif source.startswith('MCPcounter::'):
  v=methods['MCPcounter'];method='MCPcounter.estimate';software='MCPcounter '+v['version'];scale=v['input_scale'];unit='MCP-counter abundance score'
 elif source.startswith(('IPS::','ESTIMATE::')):
  v=methods['IPS_ESTIMATE'];method='hacksig original '+('IPS' if source.startswith('IPS::') else 'ESTIMATE')+' score';software='hacksig '+v['version'];scale=v['input_scale'];unit='Estimated tumor cell fraction' if source=='ESTIMATE::TumorPurity' else ('Immunophenoscore' if source.startswith('IPS::') else 'ESTIMATE enrichment score')
 elif source.startswith('stemness_rna::'):
  key=source.split('::',1)[1];v=native['OCLR_reference'][key];method='Spearman correlation with frozen Malta OCLR weights, followed by min-max scaling';reference='All selected CPTAC tumors with RNA';scale='RNA expression ranks';unit='Reference-scaled stemness score';parameters=json.dumps({k:v[k] for k in ['formula','minimum','maximum']},sort_keys=True)
 elif source.startswith('cibersort_relative::'):
  method='Published 22 absolute CIBERSORT components divided by their within-sample sum';reference='Within-patient sum across 22 LM22 cell types';unit='Relative immune-cell composition';scale='Published absolute component scores'
 elif source.startswith('xCell::'):
  method='xCell native enrichment and spillover adjustment';software='xCell '+str(extra['xcell_version']);reference='Pooled CPTAC RNA reference';scale='RNA expression';unit='xCell enrichment score'
 elif source=='EXTEND::published_score':
  method='EXTEND native score with cancer-specific min-max scaling';software='EXTEND '+str(extra['extend_version']);unit='Cancer-reference-scaled EXTEND score';scale='RNA expression'
 elif source.startswith(('absolute_purity_ploidy::','mutation_load::')):
  method='Published CPTAC / LinkedOmics phenotype';reference='Original source-cohort measurement definition';scale='Published phenotype';unit={'absolute_purity_ploidy::purity':'Tumor cell fraction','absolute_purity_ploidy::ploidy':'Ploidy','mutation_load::Non-silent per Mb':'Non-silent mutations per Mb'}[source]
 else:raise ValueError('State method not resolved: '+source)
 assert method and reference and unit
 rows.append(dict(canonical_id=row.canonical_id,source_readout_id=source,cohort=row.cohort,cancer_id=row.cancer_id,measurement_layer=row.measurement_layer,method=method,reference=reference,score_unit=unit,input_scale=scale,software=software,reference_parameters=parameters))
table=pd.DataFrame(rows);table.to_csv(O/'CPTAC_STATE_METHODS.tsv',sep='\t',index=False)
clinical=pd.read_parquet(P/'bundle_r1/CLINICAL_ALIGNED.parquet');state=pd.read_parquet(P/'bundle_r1/STATE_PATIENT.parquet',columns=['patient_id','source_cohort','leukocyte_fraction::Leukocyte_fraction','absolute_purity_ploidy::purity','ESTIMATE::TumorPurity','cibersort_relative::Macrophages.M0','cibersort_relative::Macrophages.M1','cibersort_relative::Macrophages.M2']);joined=clinical.merge(state,on=['patient_id','source_cohort'],validate='one_to_one');cov=[]
for (cohort,cancer),f in joined.groupby(['source_cohort','cancer_id']):
 row={'cohort':'CPTAC:'+cohort,'cancer_id':cancer,'patients':len(f)}
 for output,source in [('age','age_years'),('sex','sex'),('stage','stage'),('published_genomic_purity','absolute_purity_ploidy::purity'),('RNA_ESTIMATE_purity','ESTIMATE::TumorPurity'),('methylation_leukocyte_fraction','leukocyte_fraction::Leukocyte_fraction'),('macrophages_M0','cibersort_relative::Macrophages.M0'),('macrophages_M1','cibersort_relative::Macrophages.M1'),('macrophages_M2','cibersort_relative::Macrophages.M2')]:row[output+'_observed']=int(f[source].notna().sum())
 cov.append(row)
cov=pd.DataFrame(cov);assert len(cov)==10 and cov.patients.sum()==1043 and cov.methylation_leukocyte_fraction_observed.sum()==0;cov.to_csv(O/'CPTAC_COVARIATE_COVERAGE.tsv',sep='\t',index=False)
record={'target_host':'149','status':'COMPLETE','state_readout_cohort_definitions':len(table),'distinct_state_readouts':table.source_readout_id.nunique(),'cohorts':table.cohort.nunique(),'clinical_patients':1043,'source_method_receipts':source_hashes,'source_policy_sha256':sha(P/'bundle_r1/METHOD_POLICY.json'),'native_source_sha256':sha(P/'native_states_r1/COMPLETE.json'),'extra_source_sha256':sha(P/'extra_states_r1/COMPLETE.json'),'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'COMPLETE.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='source_method_receipts'}))
