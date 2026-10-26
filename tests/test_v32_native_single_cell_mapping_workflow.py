"""Actual H5 mapping and native cached-input consumers; no PCA/LASSO/model fit."""
from pathlib import Path
import socket

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from cc_hhgt.v32 import native_single_cell_mapping_workflow as workflow,native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from scripts import materialize_single_cell_metacell_149 as native
from scripts import run_v32_single_cell_verified_streaming as streaming


def forbidden(*a,**k):raise AssertionError('No PCA, LASSO or model fitting')


@pytest.fixture(scope='module',autouse=True)
def no_fit():
    from cc_hhgt.v32.single_cell_metacells import LocalSpace
    from cc_hhgt.v32 import single_cell_hhgt_episodes
    assert socket.gethostname()=='149'
    with pytest.MonkeyPatch.context() as patch:
        for name in ('Adam','AdamW','SGD'):patch.setattr(torch.optim,name,forbidden)
        patch.setattr(LocalSpace,'fit',forbidden)
        patch.setattr(single_cell_hhgt_episodes,'fit_context_lasso',forbidden)
        yield


def source(root,scale='raw_counts'):
    root.mkdir();r7=root/'single_cell_cell_level_r7_portable_supersession_20260829';r7.mkdir()
    assets=r7/'authority';assets.mkdir();raw=root/'processed/sc_tool_input/BRCA';raw.mkdir(parents=True)
    genes=[f'ENSG{n:06d}' for n in range(1,8)];count=123
    feature_ids=[genes[0]+'.1',genes[0]+'.2',*genes[1:],'UNMAPPED'] if scale=='raw_counts' else [*genes,'UNMAPPED']
    values=(1+np.arange(len(feature_ids)*count).reshape(len(feature_ids),count)%13).astype(float)
    values[-1]=100
    if scale=='log_normalized':values=np.log1p(values)
    matrix=sparse.csc_matrix(values)
    h5=raw/'input.h5'
    with h5py.File(h5,'w') as handle:
        group=handle.create_group('matrix')
        for key,v in (('data',matrix.data),('indices',matrix.indices),('indptr',matrix.indptr),('shape',matrix.shape)):
            group.create_dataset(key,data=v)
        group.create_dataset('barcodes',data=np.array([f'C{n:04d}' for n in range(count)],dtype='S'))
        features=group.create_group('features')
        features.create_dataset('id',data=np.array(feature_ids,dtype='S'));features.create_dataset('name',data=np.array(feature_ids,dtype='S'))
    annotation=assets/'annotation.parquet'
    pd.DataFrame({'gene_id':genes,'gene_symbol':[f'S{n}' for n in range(7)],'gene_class':['protein_coding']*5+['lncRNA']*2,
                  'unique_symbol':True}).to_parquet(annotation,index=False)
    membership=assets/'membership.parquet'
    pd.DataFrame([(f'P{p:04d}',gene) for p in range(2135) for gene in genes[:2]],columns=['pathway_id','gene_id']).to_parquet(membership,index=False)
    authority=assets/'cell_types.json';write(authority,{'synthetic':True,'cell_type':'T_cell'})
    policy=assets/'review.json';write(policy,{'synthetic':True,'excluded':['C0002']})
    metadata=pd.DataFrame({key:['Not_assessed']*count for key in streaming.REQUIRED_METADATA})
    metadata['cell_id']=[f'C{n:04d}' for n in range(count)];metadata['patient_id']=[f'D{max(n-3,0)//12:02d}' for n in range(count)]
    metadata['dataset_id']='SYNTHETIC';metadata['cancer_id']='BRCA';metadata['sample_id']=metadata.patient_id
    metadata['sample_class']='Tumor';metadata.loc[0,'sample_class']='Normal'
    metadata['cell_type_major']='T_cell';metadata['compartment']='immune';metadata['malignant_status']='NON_MALIGNANT'
    metadata['annotation_authority_sha256']=record(authority)['sha256'];metadata['is_doublet']=False;metadata.loc[1,'is_doublet']=True
    metadata['analysis_eligible_tumor_context']=np.arange(count)>=3
    metadata['source_review_include']=True;metadata.loc[2,'source_review_include']=False
    metadata['patient_id_verified']=True;metadata['source_dataset_id']='SYNTHETIC';metadata['source_review_evidence']='Synthetic source policy'
    metadata=metadata.sample(frac=1,random_state=19).reset_index(drop=True)
    metadata_path=raw/'cell_metadata.parquet';metadata.to_parquet(metadata_path,index=False)
    dataset=r7/'dataset_manifest_33c.parquet'
    pd.DataFrame({'cancer_id':['BRCA'],'dataset_id':['SYNTHETIC'],'measurement_scale':[scale],'formal_eligible':[True]}).to_parquet(dataset,index=False)
    write(r7/'TRAINING_HANDOFF.json',{'historical_assets_relabelled_fresh':False,'assets':{},'dataset_manifest_sha256':record(dataset)['sha256']})
    per_cancer={'formal_eligible':True,'blocking_reason':None,'dataset_id':'SYNTHETIC','fresh_direct_id_lncrna_feature_count':2,
        'annotation_authority_path':str(authority),'annotation_authority_file_sha256':record(authority)['sha256'],
        'annotation_authority_sha256':record(authority)['sha256'],'source_review_policy_path':str(policy),'source_review_policy_sha256':record(policy)['sha256']}
    for prefix,path in (('h5',h5),('metadata',metadata_path),('annotation',annotation),('exact_membership',membership)):
        per_cancer[prefix+'_path']=str(path);per_cancer[prefix+'_sha256']=record(path)['sha256']
    status=r7/'RUN_STATUS.json';write(status,{'formal_eligible_cancers':['BRCA'],'formal_eligible_cancer_count':1,
                                          'derived_assets_bound':False,'per_cancer':{'BRCA':per_cancer}})
    preflight=r7/'SERVER_PREFLIGHT.json';write(preflight,{'format':'CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_SERVER_PREFLIGHT_V1',
        'target_host':'149','r7_root':str(r7),'all_formal_raw_schemas_pass':True,'safe_to_start_fresh_cell_level_compute':True,'heavy_recompute_started':False})
    spec={'format':workflow.INPUT,'cancer_id':'BRCA','r7_root':str(r7),'run_status':record(status),'server_preflight':record(preflight)}
    manifest=root/'INPUT.json';write(manifest,spec)
    return manifest,spec,values,policy


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp('sc_mapping');manifest,spec,values,policy=source(root/'raw')
    chain_input=root/'CHAIN_INPUT.json';write(chain_input,{'format':chain.INPUT,'chain_id':'sc-mapping','stages':[
        {'id':'mapping','operation':'single_cell_mapping.run','inputs':spec}]})
    chain.initialize(chain_input,root/'chain');chain.advance(root/'chain');state=read(root/'chain/CHAIN.json')
    assert state['status']=='DECLARED_STAGES_COMPLETE',state
    pin=state['results']['mapping'];binding=read(pin['path']);mapped=Path(binding['mapped_success']['path']).parent
    return root,manifest,spec,values,policy,pin,binding,mapped


def test_raw_mapping_oracle_and_native_metacell_input_reuse(completed,monkeypatch):
    root,manifest,spec,values,policy,pin,binding,mapped=completed
    workflow.validate(pin,root);metadata=pd.read_parquet(mapped/'CELL_METADATA.parquet')
    assert metadata.cell_id.tolist()==[f'C{n:04d}' for n in range(3,123)]
    coding=np.vstack((values[0]+values[1],values[2:6]))[:,3:].T
    np.testing.assert_array_equal(sparse.load_npz(mapped/'CODING_MEASURED.npz').toarray(),coding)
    np.testing.assert_array_equal(sparse.load_npz(mapped/'LNC_MEASURED.npz').toarray(),values[6:8,3:].T)
    library=values[:,3:].sum(axis=0);np.testing.assert_array_equal(np.load(mapped/'ALL_GENE_LIBRARY.npy'),library)
    np.testing.assert_allclose(sparse.load_npz(mapped/'CODING_LOG_FOR_GROUPING.npz').toarray(),np.log1p(coding/library[:,None]*10000),rtol=1e-7)
    assert len(read(mapped/'IDS.json')['pathway_ids'])==2135
    assert str(policy) in {p['path'] for p in read(binding['preflight']['path'])['source_files']}
    monkeypatch.setattr(native,'_load_upstream_bundle',forbidden)
    receipt=native._materialize_mapped_matrices('BRCA',mapped,root/'unused',float('inf'))
    assert receipt['eligible_cells']==120 and receipt['donors']==10
    monkeypatch.setattr(chain,'dispatch',forbidden);chain.advance(root/'chain')


def test_mapped_counts_reach_actual_aggregation_and_ucell(completed):
    from cc_hhgt.v32.single_cell_metacells import aggregate_measurements
    _,_,_,values,_,_,binding,mapped=completed
    coding=sparse.load_npz(mapped/'CODING_MEASURED.npz');library=np.load(mapped/'ALL_GENE_LIBRARY.npy')
    groups=np.arange(120)//4
    aggregated,ids,policy=aggregate_measurements(coding,groups,scale='raw_counts',total_library=library)
    expected=np.log1p(coding.toarray().reshape(30,4,5).sum(axis=1)/library.reshape(30,4).sum(axis=1)[:,None]*10000)
    np.testing.assert_allclose(aggregated.toarray(),expected,rtol=1e-7)
    activity=native._ucell_from_metacells(aggregated,sparse.load_npz(mapped/'EXACT_SIGNATURE_MATRIX.npz'),
        np.load(mapped/'SIGNATURE_GENE_COUNTS.npy'),np.load(mapped/'MISSING_GENE_COUNTS.npy'),float('inf'))
    assert activity.shape==(30,2135) and np.isfinite(activity).all()
    np.testing.assert_array_equal(activity[:,0],activity[:,-1])


def test_provided_log_scale_is_not_inverted(tmp_path):
    manifest,spec,values,policy=source(tmp_path/'raw','log_normalized')
    binding=read(workflow.run(manifest,tmp_path/'out'));mapped=Path(binding['mapped_success']['path']).parent
    np.testing.assert_array_equal(sparse.load_npz(mapped/'CODING_MEASURED.npz').toarray(),values[:5,3:].T.astype(np.float32))
    np.testing.assert_array_equal(sparse.load_npz(mapped/'CODING_LOG_FOR_GROUPING.npz').toarray(),values[:5,3:].T.astype(np.float32))
    assert read(mapped/'MAPPED_SUCCESS.json')['aggregation_policy']=='PROVIDED_LOG_SCALE_ARITHMETIC_MEAN_NO_INVERSION'


@pytest.mark.parametrize('fault',['coding','library','cell_order'])
def test_rehashed_mapping_cannot_replace_raw_reconstruction(completed,fault):
    *_,binding,mapped=completed;audit_path=mapped/'MAPPED_SUCCESS.json';audit=read(audit_path)
    name={'coding':'CODING_MEASURED.npz','library':'ALL_GENE_LIBRARY.npy','cell_order':'CELL_METADATA.parquet'}[fault]
    path=mapped/name;original={p:p.read_bytes() for p in (path,audit_path)};workflow.validate_mapped(audit_path)
    try:
        if fault=='coding':
            frame=sparse.load_npz(path);frame.data[0]+=.5;sparse.save_npz(path,frame)
        elif fault=='library':
            frame=np.load(path);frame[0]+=1;np.save(path,frame)
        else:
            frame=pd.read_parquet(path).iloc[::-1].reset_index(drop=True);frame.to_parquet(path,index=False)
        audit['files'][name]={**record(path),'bytes':path.stat().st_size};write(audit_path,audit)
        with pytest.raises(ValueError,match='mapping replay differs|cell order/metadata differs'):
            native._materialize_mapped_matrices('BRCA',mapped,mapped/'unused',float('inf'))
    finally:
        for p,data in original.items():p.write_bytes(data)


def test_source_review_policy_change_invalidates_warm_cache(completed):
    _,_,_,_,policy,_,_,mapped=completed;workflow.validate_mapped(mapped/'MAPPED_SUCCESS.json');original=policy.read_bytes()
    try:
        policy.write_bytes(original+b' ')
        with pytest.raises(Exception,match='SHA256|sha256|mismatch|drift'):workflow.validate_mapped(mapped/'MAPPED_SUCCESS.json')
    finally:policy.write_bytes(original)


def test_duplicate_raw_barcodes_rejected_even_after_repinning(tmp_path):
    manifest,spec,_,_=source(tmp_path/'raw');status=read(spec['run_status']['path']);row=status['per_cancer']['BRCA'];path=Path(row['h5_path'])
    with h5py.File(path,'r+') as handle:handle['matrix/barcodes'][1]=handle['matrix/barcodes'][0]
    row['h5_sha256']=record(path)['sha256'];write(Path(spec['run_status']['path']),status)
    spec['run_status']=record(spec['run_status']['path']);write(manifest,spec)
    with pytest.raises(Exception,match='barcodes are duplicated'):workflow.plan(manifest,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_plan_writes_nothing_and_requires_149(completed,tmp_path,monkeypatch):
    from scripts.run_native_single_cell_mapping_workflow import main
    _,manifest,_,_,_,_,_,_=completed
    assert main(['--manifest',str(manifest),'--output',str(tmp_path/'out'),'--plan'])==0
    assert not (tmp_path/'out').exists()
    monkeypatch.setattr(socket,'gethostname',lambda:'local')
    with pytest.raises(RuntimeError,match='149'):workflow.plan(manifest,tmp_path/'out')
