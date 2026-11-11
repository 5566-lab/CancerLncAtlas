import pytest
from cla_engine.relation_catalog import compile_policy

def row(cid,roles='State',entity='state'):
    return dict(canonical_id=cid,display_name=cid,roles=roles,entity_type=entity)
def fixture():
    master=[row('main','Pathway','network_module'),row('tf'),row('kinase',entity='kinase_substrate_program'),row('progeny'),row('state'),row('other','Pathway','pathway'),row('zero','Pathway','pathway')]
    states=[dict(canonical_id='tf',state_family='transcription_factor'),dict(canonical_id='progeny',state_family='pathway_footprint'),dict(canonical_id='state',state_family='emt')]
    anchors=[dict(canonical_id='main',backbone_id='KEGG.NETWORK:N00056',official_cancer_supported=False)]
    proteins={cid:{'P1'} for cid in ('main','tf','kinase','progeny','state','other')}
    return master,states,anchors,proteins
def test_fixed_backbone_not_filtered_by_official_annotation():
    _,p=compile_policy(*fixture())
    assert [r['canonical_id'] for r in p if r['analysis_role']=='primary_pathway']==['main']
def test_regulator_display_never_creates_loss():
    m,p=compile_policy(*fixture());pol={r['canonical_id']:r for r in p}
    for cid in ('tf','kinase','progeny'):
        assert pol[cid]['roles']=='Pathway'
        assert pol[cid]['analysis_role']=='context_only'
        assert pol[cid]['loss_weight']==0
    assert pol['state']['loss_weight']==.5
def test_overlap_is_context_not_identity_or_primary():
    _,p=compile_policy(*fixture());pol={r['canonical_id']:r for r in p}
    assert pol['other']['analysis_role']=='context_only'
    assert pol['other']['backbone_anchor_ids']==''
    assert pol['zero']['analysis_role']=='excluded'
    assert not pol['zero']['active_in_analysis']
def test_kinase_word_in_pathway_name_is_not_regulator():
    m,s,a,p=fixture();m[0]['display_name']='Fusion kinase to PI3K pathway'
    _,rows=compile_policy(m,s,a,p)
    assert rows[0]['analysis_role']=='primary_pathway'
    assert rows[0]['mechanism_kind']==''
def test_shared_state_and_progeny_retains_explicit_state_loss():
    m,s,a,p=fixture();s.append(dict(canonical_id='progeny',state_family='hypoxia'))
    _,rows=compile_policy(m,s,a,p);r=next(r for r in rows if r['canonical_id']=='progeny')
    assert r['roles']=='Pathway|State'
    assert r['analysis_role']=='state_auxiliary'
    assert r['loss_weight']==.5
def test_missing_backbone_identity_fails():
    m,s,a,p=fixture();a[0]['canonical_id']='absent'
    with pytest.raises(ValueError,match='IDENTITY_MISMATCH'):
        compile_policy(m,s,a,p)

def test_source_category_is_separate_from_shared_identity():
    from cla_engine.relation_catalog import source_display_category
    defs={'Buffa':dict(state_family='hypoxia'),'PROGENY:Hypoxia':dict(state_family='pathway_footprint')}
    policy=dict(roles='Pathway|State',mechanism_kind='PROGENy')
    assert source_display_category(dict(source_target_id='Buffa'),defs,policy)=='State'
    assert source_display_category(dict(source_target_id='PROGENY:Hypoxia'),defs,policy)=='Pathway'
    assert source_display_category(dict(source_target_id='MSIGDB:HALLMARK:HALLMARK_HYPOXIA'),defs,policy)=='Pathway'

def test_demo_separates_cancers_without_replacing_y_contract(tmp_path):
    import json
    from cla_engine.relation_catalog import demo_configurations
    (tmp_path/'PREPARE_POLICY_BINDING.json').write_text(json.dumps({'relation_target_policy':{'path':'fixed'}}))
    base={'baseline':{},'profiles':{'diagnostic':{'max_pathways':64},'formal':{'max_pathways':None}},'pathway_target_policy':{'status':'REPLACEMENT_REQUIRED'}}
    output=demo_configurations(base,tmp_path)
    assert list(output)==['BRCA','LUAD','COAD']
    for cancer,config in output.items():
        assert config['cancer_list']==[cancer]
        assert config['baseline']['identity_mode']=='numeric_only'
        assert config['profiles']['diagnostic']['max_pathways'] is None
        assert config['pathway_target_policy']==base['pathway_target_policy']
    assert base['baseline']=={}
