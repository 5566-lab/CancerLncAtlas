from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cla_engine.common import Stop, read_json, sha
from cla_engine.state_catalog import ROUTED_CATALOG_ID, AUGMENTED_CATALOG_ID, expanded_targets
from cla_engine.state_augment import append_columns
from cla_engine.state_protein_map import ProteinResolver, pinned
from cla_engine.execution_contract import validate_config_contract

ROOT = Path(__file__).resolve().parents[1]


def test_new_catalog_only_appends_reviewed_scores_and_preserves_joint_contract():
    old=expanded_targets(ROUTED_CATALOG_ID);new=expanded_targets(AUGMENTED_CATALOG_ID)
    assert len(old)==483 and new[:483]==old and len(new)==485
    assert [t['id'] for t in new[483:]]==['TLS12_Chemokine_Published','DNARepair_Hallmark']
    cfg=read_json(ROOT/'config/149.json');validate_config_contract(cfg)
    assert cfg['targets']==new and sum(t['role']=='primary' for t in new)==484
    assert cfg['model']['task_weights']=={'pathway':1.,'state':.5}
    assert cfg['pathway_fusion']['enabled'] is False
    assert cfg['pathway_target_policy']['status']=='REPLACEMENT_REQUIRED'


def test_append_joins_by_patient_and_preserves_missingness_and_native_values():
    base=pd.DataFrame({'patient_id':['p2','p1'],'cancer_id':['A','A'],'old':[np.nan,7.]})
    added=pd.DataFrame({'new':[np.nan,0.]},index=['p1','p2'])
    out=append_columns(base,added,[{'column':'old'},{'column':'new'}])
    assert out.patient_id.tolist()==['p2','p1']
    assert out['new'].iloc[0]==0. and np.isnan(out['new'].iloc[1])
    assert np.isnan(out.old.iloc[0]) and out.old.iloc[1]==7.


def test_append_rejects_unreviewed_columns_and_duplicate_patients():
    base=pd.DataFrame({'patient_id':['p'],'cancer_id':['A'],'old':[1.]})
    with pytest.raises(Stop):append_columns(base,pd.DataFrame({'invented':[1.]},index=['p']),[{'column':'old'},{'column':'new'}])
    with pytest.raises(Stop):append_columns(pd.concat([base,base]),pd.DataFrame({'new':[1.]},index=['p']),[{'column':'old'},{'column':'new'}])


def resolver_fixture():
    genes={'G1':{'symbol':'NEW1','biotype':'protein_coding'},'G2':{'symbol':'NEW2','biotype':'protein_coding'},
           'N1':{'symbol':'RNA1','biotype':'lncRNA'}}
    h=pd.DataFrame([
      {'ensembl_gene_id':'G1','status':'Approved','symbol':'NEW1','prev_symbol':'OLD1','alias_symbol':'SHARED','uniprot_ids':'P1'},
      {'ensembl_gene_id':'G2','status':'Approved','symbol':'NEW2','prev_symbol':'','alias_symbol':'SHARED','uniprot_ids':'P2'},
      {'ensembl_gene_id':'N1','status':'Approved','symbol':'RNA1','prev_symbol':'','alias_symbol':'','uniprot_ids':'P_FAKE'}])
    graph=pd.DataFrame({'gene_id':['GENE:G1','GENE:N1'],'protein_id':['UNIPROT:P_OLD_GRAPH','UNIPROT:P_FAKE']})
    return ProteinResolver(genes,{'NEW1':{'G1'},'NEW2':{'G2'},'RNA1':{'N1'}},h,graph)


def test_reference_alias_requires_unique_gene_and_preserves_graph_disagreement():
    r=resolver_fixture();x=r.resolve('OLD1')
    assert x['gene_id']=='G1' and x['uniprot_accessions']=='P1'
    assert x['current_graph_protein_ids']=='UNIPROT:P_OLD_GRAPH'
    assert r.resolve('SHARED')['mapping_status']=='AMBIGUOUS_SYMBOL'
    assert r.resolve('unknown')['mapping_status']=='UNRESOLVED_REFERENCE_GENE'
    assert r.resolve('OLD1','G_NOT_PRESENT')['gene_id']==''


def test_noncoding_signature_member_never_receives_a_protein_by_alias_or_graph():
    x=resolver_fixture().resolve('RNA1')
    assert x['mapping_status']=='NON_PROTEIN_CODING_MEMBER'
    assert x['uniprot_accessions']=='' and x['current_graph_protein_ids']==''


def test_source_change_rejected(tmp_path):
    p=tmp_path/'source.txt';p.write_text('reviewed');rec={'path':str(p),'sha256':sha(p)}
    assert pinned(rec)==p.resolve()
    p.write_text('changed')
    with pytest.raises(Stop) as e:pinned(rec)
    assert e.value.code=='STATE_MAP_SOURCE_HASH'


def test_new_state_ids_enter_prepared_embeddings_and_receive_shared_gradients(tmp_path):
    import torch
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    from cla_engine.lineage import validate_prepared
    from cla_engine.train import load_npz,tensor_batch
    from cla_engine.model import JointHGT,to_graph,loss_one
    torch.set_num_threads(1)
    cfg=make_fixture(tmp_path/'input',state_catalog_version=AUGMENTED_CATALOG_ID)
    cfg['cancer_type']='list';cfg['cancer_list']=['C1']
    p=Path(cfg['sources']['state']['path']);f=pd.read_csv(p,sep='\t')
    # A completely unmeasured target remains represented with unavailable labels.
    f['SBS::SBS3']=np.nan;f.to_csv(p,sep='\t',index=False)
    cfg['sources']['state'].update(sha256=sha(p),bytes=p.stat().st_size)
    out=tmp_path/'prepared';prepare(cfg,out)
    _,spec,stored,_=validate_prepared(out)
    assert spec['state_catalog']==AUGMENTED_CATALOG_ID and len(spec['states'])==484
    assert all(p.startswith('MSIGDB:') for p in spec['nodes']['pathway'])
    view=out/'views/C1__train2';a=load_npz(view/'state.npz')
    absent=spec['states'].index('SBS::SBS3')
    assert not a['available'][a['target']==absent].any()
    model=JointHGT(spec,cfg['model']);model.train()
    assert model.joint.state_embedding.num_embeddings==484
    graph=to_graph(load_npz(view/'graph.npz'),'cpu');encoded=model.encode(graph,0)
    new_ids=['TLS12_Chemokine_Published','DNARepair_Hallmark']
    positions=[]
    for sid in new_ids:
        hit=np.flatnonzero((a['target']==spec['states'].index(sid)) & a['available'])
        assert len(hit),sid
        positions.append(hit[0])
    batch=tensor_batch(a,np.asarray(positions),'cpu')
    loss=loss_one(model.forward_task('state',encoded,batch),batch,cfg['model']['direction_weight'])
    assert torch.isfinite(loss)
    before=model.joint.state_embedding.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3);optimizer.zero_grad();(.5*loss).backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.encoder.parameters() if p.requires_grad)
    optimizer.step()
    for sid in new_ids:
        i=spec['states'].index(sid);assert not torch.equal(before[i],model.joint.state_embedding.weight[i])


def test_hgnc_only_gene_maps_protein_without_inventing_a_gencode_node():
    h=pd.DataFrame([{'ensembl_gene_id':'G_NEW','status':'Approved','symbol':'NEW','prev_symbol':'','alias_symbol':'','uniprot_ids':'P_NEW','locus_group':'protein-coding gene','entrez_id':'123'}])
    graph=pd.DataFrame(columns=['gene_id','protein_id'])
    resolver=ProteinResolver({}, {}, h, graph)
    x=resolver.resolve('LOC123')
    assert x['mapping_status']=='MAPPED_PROTEIN' and x['uniprot_accessions']=='P_NEW'
    assert x['reference_presence']=='HGNC_ONLY_NOT_IN_GENCODE_V36'
    assert x['current_graph_protein_ids']==''
