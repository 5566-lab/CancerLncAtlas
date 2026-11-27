import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cla_engine.measured_readouts import measurement_mask, read_partition, summarize_target_coverage, materialize_measured_readouts
from cla_engine.relation_evidence_bindings import file_sha, write_table, FORMAT


def test_zero_and_constant_are_observed_missing_is_not_zero():
    f=pd.DataFrame({'zero':[0.,0.],'constant':[3.,3.],'missing':[np.nan,2.]},index=['a','b'])
    assert measurement_mask(f).to_numpy().tolist()==[[True,True,False],[True,True,True]]
    with pytest.raises(ValueError,match='INFINITE'):
        measurement_mask(pd.DataFrame({'x':[np.inf]}))


def test_read_partition_excludes_other_patients(tmp_path):
    f=pd.DataFrame({'v':[0.,2.,999.]},index=pd.Index(['source','label','sealed'],name='patient_id'))
    f.to_parquet(tmp_path/'scores.parquet')
    actual=read_partition(tmp_path/'scores.parquet',['label','source'],['v'])
    assert actual.index.tolist()==['label','source']
    assert actual.v.tolist()==[2.,0.]
    with pytest.raises(ValueError,match='PATIENT_AXIS'):
        read_partition(tmp_path/'scores.parquet',['unknown'],['v'])


def _fixture(tmp_path):
    cat=tmp_path/'cat';cat.mkdir()
    policies=[{'canonical_id':x,'analysis_role':role,'roles':'State' if role=='state_auxiliary' else 'Pathway','active_in_analysis':'True'}
              for x,role in [('P','primary_pathway'),('V','primary_pathway'),('S','state_auxiliary')]]
    write_table(cat/'TARGET_POLICY.tsv',policies,tuple(policies[0]))
    backbone=[{'canonical_id':'P','network_id':'N1','network_type':'Reference'},
              {'canonical_id':'V','network_id':'N2','network_type':'Variant'}]
    write_table(cat/'backbone.tsv',backbone,tuple(backbone[0]))
    src=tmp_path/'TCGA/BRCA/validation/PROTEIN';src.mkdir(parents=True)
    split={'outer_fold':0,'source_folds':[2,3,4],'label_folds':[1],'source_patients':['a'],'label_patients':['b']}
    (src.parent/'SPLIT.json').write_text(json.dumps(split))
    md=[{'target_id':'measured','source_target_id':'original','canonical_id':'P','measurement_layer':'PROTEIN'}]
    write_table(src/'TARGETS.tsv',md,tuple(md[0]))
    pd.DataFrame({'measured':[0.,0.,987.]},index=pd.Index(['a','b','sealed'],name='patient_id')).to_parquet(src/'PATIENT_S.parquet')
    (src/'COMPLETE.json').write_text(json.dumps({'target_host':'149','files':{n:file_sha(src/n) for n in ['TARGETS.tsv','PATIENT_S.parquet']}}))
    b=tmp_path/'bindings';b.mkdir()
    rows=[{'cohort':'TCGA','layer':'PROTEIN','column_index':0,'target_id':'measured','output_canonical_id':x,
           'evidence_role':'equivalent_pathway_readout' if x=='P' else 'molecular_component_readout'} for x in ['P','V']]
    write_table(b/'READOUT_BINDINGS.tsv',rows,tuple(rows[0]))
    d=[{'cohort':'TCGA','layer':'PROTEIN','column_index':0,'resolved_source_canonical_id':'P'}]
    write_table(b/'SOURCE_DISPOSITIONS.tsv',d,tuple(d[0]))
    refs={'structural':{key:{'path':str(cat/name),'sha256':file_sha(cat/name)} for key,name in [('TARGET_POLICY.tsv','TARGET_POLICY.tsv'),('backbone','backbone.tsv')]},
          'streams':[{'cohort':'TCGA','layer':'PROTEIN','directory':str(src),'source_receipt_sha256':file_sha(src/'COMPLETE.json')}]}
    (b/'INPUT_REFERENCES.json').write_text(json.dumps(refs))
    (b/'COMPLETE.json').write_text(json.dumps({'target_host':'149','format':FORMAT,'files':{n:file_sha(b/n) for n in ['INPUT_REFERENCES.json','READOUT_BINDINGS.tsv','SOURCE_DISPOSITIONS.tsv']}}))
    return b,src


def test_export_never_needs_lnc_or_association_files(tmp_path,monkeypatch):
    b,src=_fixture(tmp_path)
    original=Path.open
    def deny_associations(path,*args,**kwargs):
        if path.name in {'ASSOCIATIONS.npz','AXES.json','LNC_LOGTPM_FULL.parquet','TARGET_COUNTS.tsv'}:
            raise AssertionError('Forbidden lncRNA-derived input')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',deny_associations)
    out=tmp_path/'output'
    r=materialize_measured_readouts(b,out,cancer='BRCA',context='validation')
    assert r['biological_targets']==3 and r['state_identities']==1
    coverage=pd.read_csv(out/'TARGET_MEASUREMENT_COVERAGE.tsv',sep='\t').set_index('canonical_id')
    assert coverage.loc['P','patients_with_identity_measurement']==2
    assert coverage.loc['V','patients_with_member_measurement']==2
    assert bool(coverage.loc['V','condition_required'])
    assert coverage.loc['S','patients_with_any_measurement']==0
    assert pd.read_parquet(out/'TCGA/PROTEIN/OBSERVED.parquet').measured.all()
    assert pd.read_parquet(out/'TCGA/PROTEIN/VALUES.parquet').measured.tolist()==[0.,0.]


def test_receipt_drift_and_sealed_context_rejected(tmp_path):
    b,src=_fixture(tmp_path)
    with pytest.raises(ValueError,match='OPEN_DEMO'):
        materialize_measured_readouts(b,tmp_path/'sealed',cancer='BRCA',context='sealed_test')
    assert not (tmp_path/'sealed').exists()
    (src/'TARGETS.tsv').write_text('changed')
    with pytest.raises(ValueError,match='CONTENT_DRIFT'):
        materialize_measured_readouts(b,tmp_path/'changed',cancer='BRCA',context='validation')


def test_association_y_entry_is_disabled_before_io():
    from cla_engine.multimodal_relation_y import materialize_context
    with pytest.raises(RuntimeError,match='ASSOCIATION_STATISTICS_ARE_NOT_MEASURED_Y'):
        materialize_context(root=None,catalog=None,binding_dir=None,output_root=None,cancer='BRCA',context='validation',empirical_fisher=None,bh=None,basis=None,algorithm_path=None)
