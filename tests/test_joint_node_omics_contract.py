"""Prepared admission checks measured channels before any model is created."""
import copy
import numpy as np
import pytest
from cla_engine.common import Stop,atomic_json,atomic_npz,digest,read_json
from cla_engine.node_omics import VerifiedGeneNodeOmics,FEATURE_ORDER,verify_view_node_omics
from test_joint_node_omics import _product

def sealed(tmp_path):
    config=_product(tmp_path/'product');reader=VerifiedGeneNodeOmics(config,['ENSG000001','ENSG000002'],['BRCA']).verify_cancers(['BRCA'])
    block,audit=reader.build_view('BRCA',['p0','p1'])
    root=tmp_path/'prepared';folder=root/'views/BRCA__train2';folder.mkdir(parents=True)
    spec={'nodes':{'gene':['ENSG000001','ENSG000002'],'lncRNA':['L1']},'cancers':['BRCA'],
      'gene_node_omics_enabled':True,'node_input_dim':6,
      'node_input_feature_order':['log1p_degree','constant',*FEATURE_ORDER],
      'gene_node_omics_complete_sha256':config['complete']['sha256']}
    view={'id':'BRCA__train2','cancer_index':0}
    atomic_npz(folder/'graph.npz',**{'node_omics::gene':block,'node_omics::lncRNA':np.zeros((1,4),np.float32)})
    atomic_json(folder/'NODE_OMICS_AUDIT.json',audit)
    atomic_json(folder/'PROVENANCE.json',{'feature_patient_hash':digest(['p0','p1']),'gene_node_omics':audit})
    atomic_json(root/'INPUT_HASHES.json',reader.input_hashes())
    cfg={'gene_node_omics':config};verify_view_node_omics(root,spec,view,cfg)
    return root,folder,spec,view,cfg

@pytest.mark.parametrize('change,code',[
 ('missing_block','NODE_OMICS_NODE_TYPE_SET'),('mask','NODE_OMICS_MASK_INVALID'),
 ('missing_value','NODE_OMICS_MISSING_VALUE_NONZERO'),('non_gene','NODE_OMICS_NON_GENE_MEASUREMENT'),
 ('beta','NODE_OMICS_BETA_INVALID'),('value','NODE_OMICS_VIEW_AUDIT_DRIFT')])
def test_semantically_invalid_blocks_fail_before_encoder(tmp_path,change,code):
    root,folder,spec,view,cfg=sealed(tmp_path)
    with np.load(folder/'graph.npz') as source:arrays={k:source[k] for k in source.files}
    if change=='missing_block':arrays.pop('node_omics::lncRNA')
    elif change=='mask':arrays['node_omics::gene'][0,1]=.5
    elif change=='missing_value':arrays['node_omics::gene'][1,0]=1
    elif change=='non_gene':arrays['node_omics::lncRNA'][0,:2]=[1,1]
    elif change=='beta':arrays['node_omics::gene'][0,2]=1.1
    else:arrays['node_omics::gene'][0,0]+=1
    atomic_npz(folder/'graph.npz',**arrays)
    with pytest.raises(Stop) as error:verify_view_node_omics(root,spec,view,cfg)
    assert error.value.code==code

def test_node_view_cannot_claim_label_patient_hash_as_feature_source(tmp_path):
    root,folder,spec,view,cfg=sealed(tmp_path)
    provenance=read_json(folder/'PROVENANCE.json');provenance['feature_patient_hash']=digest(['p2'])
    atomic_json(folder/'PROVENANCE.json',provenance)
    with pytest.raises(Stop) as error:verify_view_node_omics(root,spec,view,cfg)
    assert error.value.code=='NODE_OMICS_VIEW_AUDIT_DRIFT'

def test_six_column_capsule_cannot_load_as_disabled_ablation(tmp_path):
    root,folder,spec,view,cfg=sealed(tmp_path)
    cfg['gene_node_omics']={'enabled':False};spec['gene_node_omics_enabled']=False
    with pytest.raises(Stop) as error:verify_view_node_omics(root,spec,view,cfg)
    assert error.value.code=='NODE_OMICS_DISABLED_INPUT'
