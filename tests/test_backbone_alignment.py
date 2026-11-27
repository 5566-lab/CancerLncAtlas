from cla_engine.backbone_alignment import build_alignment

def test_overlap_enrichment_never_merges_identity_or_inherits_variant_context():
    master=[dict(canonical_id=x,display_name=x,source_database='R') for x in ['a','b','small','excluded']]
    backbone=[dict(canonical_id='a',network_id='N1',display_name='RAS mutation',network_type='Variant')]
    proteins={'a':{'p1','p2','p3','p4'},'b':{'p1','p2','p3','p4'},
        'small':{'p1'},'excluded':{'p1','p2','p3','p4'},'background':{'z'+str(i) for i in range(500)}}
    links,top,supported,summary=build_alignment(master,backbone,proteins,['excluded'])
    assert supported=={'b'}
    assert summary['tested_pairs']==2
    assert all(r['identity_merge']=='False' and r['mechanism_context_inherited']=='False' for r in links)
    assert not any(r['source_canonical_id']=='excluded' for r in links)
    assert next(r for r in links if r['source_canonical_id']=='small')['admitted_context_link']=='False'

def test_single_kinase_identity_is_not_subject_to_multigene_enrichment_gate():
    master=[dict(canonical_id=x,display_name=x,source_database='R') for x in ['a','kinase','unrelated']]
    anchors=[dict(canonical_id='a',network_id='N1',display_name='mechanism',network_type='Reference')]
    proteins={'a':{'K','A','B'},'kinase':{'K'},'unrelated':{'K'}}
    links,top,supported,_=build_alignment(master,anchors,proteins,regulator_proteins={'kinase':{'K'}})
    assert supported=={'kinase'}
    row=next(r for r in links if r['source_canonical_id']=='kinase')
    assert row['relation']=='regulator_component_in_mechanism'
    assert row['direct_regulator_proteins']=='K'
    assert row['identity_merge']=='False'

def test_kegg_anchor_cannot_expand_to_cross_database_union():
    from cla_engine.backbone_alignment import pin_anchor_components
    sets={'anchor':{'KEGG1','OTHER_LIBRARY'},'other':{'X'}}
    original={'source':{'KEGG1'}}
    out=pin_anchor_components(sets,[{'canonical_id':'anchor','source_target_id':'source'}],original)
    assert out['anchor']=={'KEGG1'}
    assert sets['anchor']=={'KEGG1','OTHER_LIBRARY'}
    assert out['other']==sets['other']
