from pathlib import Path
import ast,copy,inspect
import numpy as np,pandas as pd,pytest
from cc_hhgt.v32.evidence_semantics_r5 import cancer_context,predicted_method,physical_method,relation_kind
from cc_hhgt.v32.evidence_training import build_exact_event_bags,normalize_id_map,_is_physical,_source_columns,EvidenceTrainingContractError
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('term',['catRAPID','catRAPID omics','RNAInter catRAPID prediction','in silico'])
def test_prediction_not_physical(term):
    assert predicted_method(term) and not physical_method(term)
    f=pd.DataFrame([{'lncrna_id':'L','partner_id':'G','cancer_id':'BRCA','experiment_raw':term,'experiment_family':'physical_binding','is_experimental':False,'is_prediction':True,'relation_type':'binding','source_record_id':'X'}])
    assert not _is_physical(f.iloc[0],_source_columns(f))
    b=build_exact_event_bags(pd.DataFrame(),f,pd.DataFrame({'gene_id':['G'],'pathway_id':['P']}))
    assert b.physical_facts.empty
    if not b.events.empty:
        assert b.events.is_computational.all();assert not b.events.is_physical.any()
        assert b.events.direction_target.eq(-1).all()

@pytest.mark.parametrize('term',['RIP','eCLIP','ChIRP','RAP-MS','RNA pull-down'])
def test_measured_physical_assay_recognized(term):
    assert physical_method(term)
    f=pd.DataFrame([{'lncrna_id':'L','partner_id':'G','cancer_id':'BRCA','experiment_type':term,'is_experimental':True,'is_predicted':False,'relation_type':'binding','source_record_id':'X'}])
    b=build_exact_event_bags(pd.DataFrame(),f,pd.DataFrame({'gene_id':['G'],'pathway_id':['P']}))
    assert len(b.physical_facts)==1 and b.physical_facts.is_prediction.eq(False).all()

def test_missing_context_not_pan_and_real_aliases():
    assert cancer_context('Breast cancer')=='BRCA'
    assert cancer_context('Liver cancer')=='LIHC'
    assert cancer_context('TCGA-BRCA')=='BRCA'
    for x in [None,'','Alzheimer’s disease','muscle differentiation','embryonic stem cells']:
        assert cancer_context(x).startswith('CONTEXT_')
    assert cancer_context('PAN-CANCER')=='PAN_CANCER'

def test_noncancer_event_not_broadcast_to_cancers():
    f=pd.DataFrame([{'lncrna_id':'L','partner_id':'G','cancer_id':'Alzheimer’s disease','source_record_id':'N','direction':'positive'}])
    b=build_exact_event_bags(pd.DataFrame(),f,pd.DataFrame({'gene_id':['G'],'pathway_id':['P']}))
    assert b.events.empty;assert b.rejected.rejection_reason.str.contains('CONTEXT').any()

def test_partner_direction_not_copied_to_targets():
    f=pd.DataFrame([{'lncrna_id':'L','partner_id':'G','cancer_id':'BRCA','source_record_id':'D','direction':'positive','manual_review_status':'approved'}])
    b=build_exact_event_bags(pd.DataFrame(),f,pd.DataFrame({'gene_id':['G','G'],'pathway_id':['P1','P2'],'weight':[1.,-1.]}))
    assert len(b.events)==2 and b.events.direction_target.eq(-1).all() and b.events.confidence_target.isna().all()

def test_cerna_not_promoted_to_direct_binding_or_pathway():
    assert relation_kind('ceRNA sponge')=='ceRNA_mediated'
    f=pd.DataFrame([{'lncrna_id':'L','partner_id':'G','cancer_id':'BRCA','source_record_id':'D','relation_type':'ceRNA_mediated','direction':'positive'}])
    b=build_exact_event_bags(pd.DataFrame(),f,pd.DataFrame({'gene_id':['G'],'pathway_id':['P']}))
    assert b.events.empty and b.physical_facts.empty

def test_ambiguous_alias_not_last_row_wins():
    f=pd.DataFrame({'source_id':['A','A'],'target_id':['G1','G2']})
    with pytest.raises(EvidenceTrainingContractError,match='AMBIGUOUS'):normalize_id_map(f)
    with pytest.raises(EvidenceTrainingContractError):normalize_id_map(f.iloc[::-1])

@pytest.mark.parametrize('value',[-3.,np.inf,-np.inf])
def test_genomic_normalizers_reject_invalid_burden(value):
    from cc_hhgt.v32.genomic_training import _normalise_mutation_calls,GenomicTrainingError
    f=pd.DataFrame({'cancer_id':['BRCA'],'patient_id':['P'],'lncrna_id':['L'],'lncrna_any_mutation':[True],'lncrna_mutation_available':[True],'exonic_variant_count':[value]})
    with pytest.raises(GenomicTrainingError):_normalise_mutation_calls(f,'lncrna')

@pytest.mark.parametrize('value',[np.inf,-np.inf])
def test_genomic_cnv_rejects_inf(value):
    from cc_hhgt.v32.genomic_training import normalise_cnv_calls,GenomicTrainingError
    f=pd.DataFrame({'cancer_id':['BRCA'],'patient_id':['P'],'gene_id':['G'],'cnv_value':[value],'cnv_callable':[True]})
    with pytest.raises(GenomicTrainingError):normalise_cnv_calls(f,entity_kind='gene',event_threshold=.2)

def test_sample_sentinel_not_locus_wildtype():
    from cc_hhgt.v32.genomic_training import _call_matrix,ASSAY_CALLABLE_SENTINEL
    f=pd.DataFrame({'patient_id':['P','P'],'entity_id':[ASSAY_CALLABLE_SENTINEL,'L'],'event':[False,False],'callable':[True,False],'burden':[0,np.nan]})
    e,b,_=_call_matrix(f,['P'],['L','M']);assert np.isnan(e).all() and np.isnan(b).all()

def test_duplicate_aggregate_not_double_mutation_count():
    from cc_hhgt.v32.genomic_training import _normalise_mutation_calls,GenomicTrainingError
    f=pd.DataFrame({'cancer_id':['BRCA']*2,'patient_id':['P']*2,'lncrna_id':['L']*2,'lncrna_any_mutation':[True]*2,'lncrna_mutation_available':[True]*2,'exonic_variant_count':[2.,2.]})
    out=_normalise_mutation_calls(f,'lncrna');assert len(out)==1 and out.burden.iloc[0]==2.
    f.loc[1,'exonic_variant_count']=3
    with pytest.raises(GenomicTrainingError):_normalise_mutation_calls(f,'lncrna')

def test_singlecell_context_activity_not_broadcast():
    from cc_hhgt.v32.single_cell_training import build_domain_features,_DOMAIN_FEATURES
    assoc=pd.DataFrame({'dataset_id':['D','D'],'cancer_id':['BRCA','BRCA'],'lncrna_id':['L','L'],'pathway_id':['W','W'],'cell_type_major':['Tcell','Bcell']})
    lnc=pd.DataFrame({'dataset_id':['D','D'],'cancer_id':['BRCA','BRCA'],'lncrna_id':['L','L'],'cell_key':['tcell','bcell'],'lnc_detection_rate':[.5,.5],'lnc_mean_log_expression':[2,2],'lnc_specificity_tau':[.2,.2],'lnc_n_cells':[10,90]})
    act=pd.DataFrame({'dataset_id':['D','D'],'pathway_id':['W','W'],'activity_kind':['activity','activity'],'activity_value':[1.,9.],'n_cells':[10,90],'mean_pseudotime':[.5,.5],'cell_type':['Tcell','Bcell']})
    out=build_domain_features(assoc,lnc,act)
    assert out[:,list(_DOMAIN_FEATURES).index('pathway_activity')].tolist()==[1.,9.]

def test_singlecell_state_preserved():
    from cc_hhgt.v32.single_cell_training import build_domain_features,_DOMAIN_FEATURES
    assoc=pd.DataFrame({'dataset_id':['D']*2,'cancer_id':['BRCA']*2,'lncrna_id':['L']*2,'pathway_id':['W']*2,'cell_type_major':['Tcell']*2,'cell_state':['A','B']})
    lnc=pd.DataFrame({'dataset_id':['D']*2,'cancer_id':['BRCA']*2,'lncrna_id':['L']*2,'cell_key':['tcell']*2,'state_key':['a','b'],'lnc_detection_rate':[.2,.6],'lnc_mean_log_expression':[2,8],'lnc_specificity_tau':[.2,.2],'lnc_n_cells':[10,10]})
    act=pd.DataFrame({'dataset_id':['D']*2,'cancer_id':['BRCA']*2,'pathway_id':['W']*2,'activity_kind':['activity']*2,'activity_value':[1.,9.],'n_cells':[10,90],'mean_pseudotime':[.5,.5],'cell_key':['tcell']*2,'state_key':['a','b']})
    out=build_domain_features(assoc,lnc,act)
    assert out[:,list(_DOMAIN_FEATURES).index('pathway_activity')].tolist()==[1.,9.]

def test_empty_singlecell_validation_stops_before_optimizer():
    from cc_hhgt.v32.single_cell_training import fit_private_head,SingleCellTrainingConfig,SingleCellTrainingError
    with pytest.raises(SingleCellTrainingError,match='validation'):
        fit_private_head(np.ones((2,3)),np.ones((2,4)),np.array([0.,1.]),np.ones((0,3)),np.ones((0,4)),np.array([]),fold=0,config=SingleCellTrainingConfig())

def test_singlecell_does_not_default_metadata_pass():
    from cc_hhgt.v32.single_cell_training import normalise_dataset_manifest,SingleCellTrainingError
    f=pd.DataFrame({'dataset_id':['D'],'cancer_id':['BRCA'],'formal_eligible':[True],'source_tier':['formal'],'quality_status':['pass']})
    with pytest.raises(SingleCellTrainingError):normalise_dataset_manifest(f)

def test_raw_p_cannot_stand_in_for_fdr():
    from cc_hhgt.v32.single_cell_training import normalise_single_cell_associations,SingleCellTrainingError
    f=pd.DataFrame({'cancer_id':['BRCA'],'dataset_id':['D'],'cell_type':['Tcell'],'lncrna_id':['L'],'pathway_id':['W'],'rho':[.3],'p_value':[.01]})
    with pytest.raises(SingleCellTrainingError,match='fdr|q_value|padj'):normalise_single_cell_associations(f,pd.DataFrame())

def test_atac_no_outer_patients_not_available():
    from cc_hhgt.v32.atac_training import _fold_frame
    f=pd.DataFrame({'cancer_id':['CESC'],'lncrna_id':['L'],'pathway_id':['P']})
    out=_fold_frame(f,0,np.array([.9]),np.array([''],object),2,0)
    assert not out.atac_available.any();assert out.atac_context_probability.isna().all()
    assert out.atac_unavailable_reason.str.contains('OUTER_TEST').all()

def test_atac_existing_test_patients_prediction_retained():
    from cc_hhgt.v32.atac_training import _fold_frame
    f=pd.DataFrame({'cancer_id':['CESC'],'lncrna_id':['L'],'pathway_id':['P']})
    out=_fold_frame(f,0,np.array([.9]),np.array([''],object),20,3)
    assert out.atac_available.all()

def test_distal_leakage_not_claimed_fixed_by_within_fold_training():
    text=(ROOT/'cc_hhgt/v32/epigenetic_expression_head.py').read_text()
    assert 'DISTAL_TOPOLOGY_SCOPE_UNVERIFIED' in text
    # Static assertion only: no DataS7 cohort or PyArrow path executed here.

def _cdr(monkeypatch,tmp_path,main_rows,extra_rows):
    from cc_hhgt.v32.clinical_training import standardize_tcga_cdr_workbook
    p=tmp_path/'source.xlsx';p.write_bytes(b'fixture-with-reader-injection-not-an-excel-file')
    main=pd.DataFrame(main_rows);extra=pd.DataFrame(extra_rows)
    monkeypatch.setattr(pd,'read_excel',lambda path,sheet_name:main.copy() if sheet_name=='TCGA-CDR' else extra.copy())
    return standardize_tcga_cdr_workbook(p)

def _cdrrow(p='P',**kw):
    r={'bcr_patient_barcode':p,'type':'BRCA','OS':1,'OS.time':12,'DSS':1,'DSS.time':12,'PFI':1,'PFI.time':12,'DFI':1,'DFI.time':12,'PFS':1,'PFS.time':12}
    r.update(kw);return r

def test_cdr_conflicts_do_not_keep_first(monkeypatch,tmp_path):
    from cc_hhgt.v32.clinical_training import ClinicalTrainingError
    with pytest.raises(ClinicalTrainingError,match='Conflicting'):_cdr(monkeypatch,tmp_path,[_cdrrow(),_cdrrow(**{'OS.time':35})],[_cdrrow()])

def test_cdr_extra_only_patient_kept_for_pfs(monkeypatch,tmp_path):
    e,c=_cdr(monkeypatch,tmp_path,[_cdrrow()],[_cdrrow(),_cdrrow('EXTRA')])
    p=e.loc[e.patient_id.eq('EXTRA')&e.clinical_endpoint.eq('PFS')]
    assert len(p)==1 and p.endpoint_available.all()

def test_cdr_conflicting_covariates_rejected(monkeypatch,tmp_path):
    from cc_hhgt.v32.clinical_training import ClinicalTrainingError
    with pytest.raises(ClinicalTrainingError,match='covariates'):_cdr(monkeypatch,tmp_path,[_cdrrow(gender='MALE'),_cdrrow(gender='FEMALE')],[_cdrrow()])

def test_clinical_no_measurement_cannot_be_significant():
    # Execute the original computational function bodies without importing unavailable PyArrow IO.
    from scipy.special import ndtr
    names={'ClinicalEntityTrainingError','_normal_two_sided_p','fold_local_residualize','cox_null_score_one_step','_cox_complete_score_one_step'}
    path=ROOT/'cc_hhgt/v32/clinical_entity_training.py';tree=ast.parse(path.read_text())
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    ns={'np':np,'pd':pd,'ndtr':ndtr};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
    rng=np.random.default_rng(3);z=np.linspace(-1,1,100);x=2*z+rng.normal(0,.25,100);train=np.arange(100)<80;x[~train]=np.nan
    residual,_,_=ns['fold_local_residualize'](x[:,None],z[:,None],train,min_train_measurements=12)
    assert np.isnan(residual[~train]).all()
    r=ns['cox_null_score_one_step'](residual[~train],np.arange(1,21),np.ones(20),min_patients=10,min_events=4)
    assert not r['estimable'].any()
