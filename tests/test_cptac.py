import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cla_engine.common import Stop, sha
from cla_engine.cptac_scope import select_cohorts
from cla_engine.cptac_materialize import case_ids, read_matrix, sample_identity, gene_id
from cla_engine.cptac_omics_prepare import methylation_means, site_input

def test_headerless_numeric_case_is_not_lost(tmp_path):
    p=tmp_path/'cases.txt';p.write_text('604\n01BR001\n')
    assert case_ids(p)==['604','01BR001']

@pytest.mark.parametrize('header',['A\tB\n','idx\tA\tB\n','\tA\tB\n'])
def test_real_r_write_table_header_variants_keep_first_sample(tmp_path,header):
    p=tmp_path/'assay.txt';p.write_text(header+'ENSG1\t1\t2\n')
    f=read_matrix(p)
    assert list(f.columns)==['A','B'] and f.loc['ENSG1','A']==1

@pytest.mark.parametrize('header',['A\tA\n','idx\tA\tA\n'])
def test_duplicate_headers_fail_before_pandas_can_mangle_them(tmp_path,header):
    p=tmp_path/'assay.txt';p.write_text(header+'ENSG1\t1\t2\n')
    with pytest.raises(Stop):read_matrix(p)

def test_source_inf_is_audited_and_never_imputed_zero(tmp_path):
    p=tmp_path/'assay.txt';p.write_text('idx\tA\tB\nsite\t-inf\t2\n')
    f=read_matrix(p)
    assert pd.isna(f.loc['site','A']) and f.loc['site','B']==2
    assert f.attrs['nonfinite_source'][0]['native_sample_id']=='A'

def test_tumor_and_adjacent_normal_get_distinct_sample_ids():
    t=sample_identity('LUAD','C3L-1_T','rna','rna.txt',{'C3L-1'},{'C3L-1'})
    n=sample_identity('LUAD','C3L-1_A','rna','rna.txt',{'C3L-1'},{'C3L-1'})
    assert t['patient_id']==n['patient_id'] and t['sample_id']!=n['sample_id']
    assert t['cancer_id']=='LUAD' and n['tissue_type']=='Normal'

def test_unpublished_rna_extra_cannot_enter_formal_cohort():
    assert sample_identity('BRCA','extra','rna','rna.txt',{'604'},set()) is None

def test_normal_control_cannot_be_mislabeled_as_tumor():
    assert sample_identity('UCEC','NX1','protein','UCEC_Normal.txt',{'C3L-1'},{'NX1'})['tissue_type']=='Normal'

def test_reference_version_removal_preserves_par_y():
    assert gene_id('ENSG000001.12_PAR_Y')=='ENSG000001_PAR_Y'
    assert gene_id('ENSG000001.12')=='ENSG000001'

def test_demo_accepts_both_cptac_and_tcga_cancer_aliases():
    assert select_cohorts('list',['KIRC,HNSC','LUSC','PAAD'],{'CCRCC','HNSCC','LSCC','PDAC'})=={'CCRCC','HNSCC','LSCC','PDAC'}
    with pytest.raises(Stop):select_cohorts('list',['LIHC'],{'BRCA'})
    with pytest.raises(Stop):select_cohorts('list',[],{'BRCA'})

def test_methylation_mean_uses_measured_values_and_masks_insufficient_sample():
    f=pd.DataFrame({'p1':[.1,.3,np.nan],'p2':[.9,np.nan,np.nan]},index=['A','B','C'])
    score,n=methylation_means(f,{'P':['A','B','C']},minimum=2)
    assert score.loc['P','p1']==pytest.approx(.2)
    assert pd.isna(score.loc['P','p2']) and n.loc['P','p2']==1

def test_phosphosites_of_same_gene_remain_distinct():
    f=pd.DataFrame({'p1':[1.,5.],'p2':[3.,9.]},index=['a','b'])
    meta=pd.DataFrame({'source_feature_id':['a','b'],'gene_id':['G','G'],
      'ptm_sequence_id':['AAAAAAASAAAAAAA-p','AAAAAAATAAAAAAA-p'],
      'protein_isoform':['P','P'],'site':['S8','T9'],'flanking_sequence':['AAAAAAASAAAAAAA','AAAAAAATAAAAAAA']})
    z,ref,bridge,amb=site_input(f,meta)
    assert z.shape==(2,2) and len(bridge)==2 and not amb
    np.testing.assert_allclose(ref.to_numpy(),[2.,7.])

def test_ambiguous_site_sequence_across_two_genes_is_excluded():
    f=pd.DataFrame({'p1':[1.,3.],'p2':[2.,4.]},index=['a','b'])
    meta=pd.DataFrame({'source_feature_id':['a','b'],'gene_id':['G1','G2'],
      'ptm_sequence_id':['AAAAAAASAAAAAAA-p']*2,'protein_isoform':['P1','P2'],
      'site':['S8','S8'],'flanking_sequence':['AAAAAAASAAAAAAA']*2})
    z,ref,bridge,amb=site_input(f,meta)
    assert z.empty and len(amb)==1

def test_download_refuses_changed_frozen_sha_before_network(tmp_path):
    from cla_engine.cptac_download import download_one
    root=tmp_path/'BRCA';root.mkdir();p=root/'data.txt';p.write_text('x')
    rec={'cohort':'BRCA','name':'data.txt','url':'https://example.invalid/data','sha256':'0'*64}
    p.with_name(p.name+'.receipt.json').write_text(json.dumps({'url':rec['url'],'sha256':sha(p),'bytes':1}))
    with pytest.raises(Stop):download_one(rec,tmp_path,attempts=1)

def test_relative_composition_never_treats_absolute_as_a_fraction():
    from cla_engine.cptac_native_states import CELL_MAP,relative_composition
    f=pd.DataFrame([[2.]*22,[0.]*22,[1.]*21+[np.nan]],columns=['CIBERSORT_'+k for k in CELL_MAP])
    r=relative_composition(f)
    np.testing.assert_allclose(r.iloc[0].to_numpy(),np.repeat(1/22,22))
    assert r.iloc[1:].isna().all().all()

def test_rna_stemness_uses_spearman_not_dot_product_or_imputation():
    from cla_engine.cptac_native_states import spearman_weight_score
    values=np.array([[1.,9.],[3.,4.],[10.,1.],[np.nan,3.]])
    score,n=spearman_weight_score(values,np.array([1.,2.,4.,5.]))
    assert n==3
    np.testing.assert_allclose(score,[1.,-1.])

def test_gct13_annotation_rows_do_not_become_patient_measurements(tmp_path):
    from cla_engine.cptac_bundle import read_gct
    p=tmp_path/'data.gct'
    p.write_text('#1.3\n2\t2\t1\t1\nid\tdesc\tA\tB\nbatch\tna\tB1\tB2\nP1\tq\t1\tNA\nP2\tx\t3\t4\n')
    f=read_gct(p)
    assert f.shape==(2,2) and pd.isna(f.loc['P1','B']) and f.loc['P2','A']==3

def test_normal_or_wrong_cohort_cannot_supply_a_training_target():
    from cla_engine.cptac_bundle import sample_to_patient
    allowed={'CPTAC:BRCA:604'}
    assert sample_to_patient('CPTAC:BRCA:604::T','BRCA',allowed)=='CPTAC:BRCA:604'
    for bad in ['CPTAC:BRCA:604::N','CPTAC:OV:604::T','TCGA-XX-0001']:
        with pytest.raises(Stop):sample_to_patient(bad,'BRCA',allowed)

def test_survival_without_time_is_not_a_complete_clinical_endpoint():
    from cla_engine.cptac_bundle import clinical_table
    f=pd.DataFrame({'patient_id':['a'],'source_cohort':['BRCA'],'cancer_id':['BRCA'],'native_case_id':['604'],
       'Age':['50'],'Sex':['Female'],'Stage':['Stage II'],'OS_days':[np.nan],'OS_event':['0'],'PFS_days':['12'],'PFS_event':['1']})
    c=clinical_table(f)
    assert not c.OS_available.iloc[0] and pd.isna(c.OS_event.iloc[0])
    assert c.PFS_available.iloc[0] and c.PFS_days.iloc[0]==12

def test_sparse_ptm_direction_does_not_create_a_false_zero_activity():
    from cla_engine.cptac_bundle import ptm_observed_mask
    f=pd.DataFrame({'a':[1.,2.,3.,4.,5.],'b':[1.,2.,3.,np.nan,5.]},index=['s1','s2','s3','s4','s5'])
    sets={'P':['s1;u','s2;u','s3;u','s4;u','s5;d']}
    count,ok=ptm_observed_mask(f,sets)
    assert count.loc['P','a']==5 and not ok.loc['P','a'] and not ok.loc['P','b']

def test_unvalidated_cptac_pooling_rejected_before_reading_data():
    from cla_engine.cptac_binding import validate_binding,ROLE
    from cla_engine.cptac_bundle import FORMAT
    with pytest.raises(Stop):
        validate_binding({'format':FORMAT,'role':ROLE,'training_pooling':True})
