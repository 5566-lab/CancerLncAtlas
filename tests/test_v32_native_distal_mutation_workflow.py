"""Synthetic MC3 rows with the unchanged official chain and actual consumers."""
from pathlib import Path
import gzip
import hashlib
import socket

import duckdb
import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_distal_mutation_workflow as workflow,mc3_sources,native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.distal_topology import topology_lineage,feature_topology_lineage
from test_v32_current_state_context import current_sources

CHAIN=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_distal_regulatory_mutation_20260830_r1/mc3_grch38_coordinate_authority_20260830_r1/hg19ToHg38.over.chain.gz')


def forbidden(*a,**k):raise AssertionError('No learned-model optimizer')


@pytest.fixture(scope='module',autouse=True)
def no_fit():
    assert socket.gethostname()=='149'
    with pytest.MonkeyPatch.context() as patch:
        for name in ('Adam','AdamW','SGD'):patch.setattr(torch.optim,name,forbidden)
        yield


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    from scripts.build_v32_independent_distal_topology import main as build
    from scripts import materialize_v32_distal_mutation_lncrna as producer
    root=tmp_path_factory.mktemp('native_distal_mutation');blocks=mc3_sources.chain_blocks(CHAIN)
    block=blocks.loc[blocks.chromosome.eq('1') & blocks.strand.eq('+') & (blocks.block_end-blocks.block_start).gt(1000)].iloc[0]
    start=int(block.block_start);target=int(block.target_start);i=np.arange(60);patients=[f'TCGA-AA-{n:04d}' for n in i]
    folds=root/'folds.tsv';pd.DataFrame({'cancer_id':'BRCA','patient_id':patients,'patient_fold_id':i%5}).to_csv(folds,sep='\t',index=False)
    pd.DataFrame({'#chrom':['chr'+block.target_chromosome],'start':[target+10],'end':[target+20],'gene_id':['LNC:Z']}).to_csv(root/'promoters.tsv',sep='\t',index=False)
    pd.DataFrame({'seqnames':['chr'+block.target_chromosome],'start':[target+101],'end':[target+106],'name':['peak-z']}).to_csv(root/'peaks.tsv',sep='\t',index=False)
    pd.DataFrame({'gene_id':['LNC:Z']}).to_csv(root/'genes.tsv',sep='\t',index=False)
    assert build(['--promoters',str(root/'promoters.tsv'),'--peak-set',str(root/'peaks.tsv'),'--required-genes',str(root/'genes.tsv'),
                  '--reference-build','GRCh38','--max-distance-bp','300','--output',str(root/'topology')])==0
    topology=root/'topology/INDEPENDENT_DISTAL_LINKS.tsv.gz';rows=[]
    for n,patient in enumerate(patients[:-1]):
        for offset in [200,*range(100,100+n%7)]:
            rows.append({'NCBI_Build':'GRCh37','Chromosome':'1','Start_Position':start+offset+1,'End_Position':start+offset+1,
                         'Reference_Allele':'A','Tumor_Seq_Allele2':'T','Tumor_Sample_Barcode':patient+'-01A','FILTER':'PASS'})
    # Duplicate aliquot/event observations cannot count as two variants.
    rows.append({**rows[2],'Tumor_Sample_Barcode':patients[1]+'-01B'})
    rows.append({**rows[0],'Chromosome':'Un_synthetic'})
    maf=root/'synthetic.maf.gz';pd.DataFrame(rows).to_csv(maf,sep='\t',index=False)
    mutation={'format':workflow.INPUT,'mc3':record(maf),'chain':record(CHAIN),'patient_folds':record(folds),
              'peak_set':record(root/'peaks.tsv'),'topology_links':record(topology)}
    manifest=root/'MUTATION.json';write(manifest,mutation)
    # Production gates must reject this file before any test-only substitution.
    official=producer.EXPECTED_MC3_SHA256
    assert official=='cb0caa0e3a80a24cc4ccaaacc95c690a4032cc87c8ac78519a5b33f3aa3f6b81'
    with pytest.raises(ValueError,match='MC3 source SHA256'):workflow.plan(manifest,root/'forbidden')
    assert not (root/'forbidden').exists()
    assert record(CHAIN)['sha256']==producer.EXPECTED_CHAIN_SHA256
    raw=root/'beta';raw.mkdir();selected=[];beta=.4+.1*np.sin(i*.8);cnv=.2*np.cos(i*.7)
    for n,patient in enumerate(patients):
        folder=raw/str(n);folder.mkdir();path=folder/'beta.txt';path.write_text(f'cg1\t{beta[n]}\n')
        selected.append({'file_id':str(n),'file_name':path.name,'file_size':path.stat().st_size,'md5sum':hashlib.md5(path.read_bytes()).hexdigest(),
                         'cancer_id':'BRCA','patient_id':patient,'sample_id':patient+'-01A'})
    pd.DataFrame(selected).to_csv(root/'selected.tsv',sep='\t',index=False)
    probe=root/'probe';probe.mkdir();pd.DataFrame({'probe_id':['cg1'],'lncrna_id':['LNC:Z'],'region_type':['promoter']}).to_parquet(probe/'map.parquet',index=False)
    write(probe/'AUDIT.json',{'output':record(probe/'map.parquet'),'topology_lineage':topology_lineage(topology),'synthetic_probe_assignment':True})
    methylation={'format':'NATIVE_METHYLATION_WORKFLOW_INPUT_V1','mode':'raw','selected_files':record(root/'selected.tsv'),
                 'probe_map':record(probe/'map.parquet'),'patient_folds':record(folds),'download_root':str(raw),'workers':1,'batch_size':30}
    expression=2*np.log1p(i%7)+.4*cnv+1.3*beta+.003*np.sin(i)
    e=root/'expression/cancer_id=BRCA';e.mkdir(parents=True)
    pd.DataFrame({'cancer_id':'BRCA','patient_id':patients,'lncrna_id':'LNC:Z','logcpm':expression}).to_parquet(e/'part-0.parquet',index=False)
    activity=root/'activity/cancer_id=BRCA';activity.mkdir(parents=True)
    pd.DataFrame({'cancer_id':'BRCA','patient_id':patients,'pathway_id':'P1','activity_score':-2*expression}).to_parquet(activity/'part-0.parquet',index=False)
    pd.DataFrame({'cancer_id':'BRCA','patient_id':patients,'lncrna_id':'LNC:Z','cnv_value':cnv,'cnv_callable':True}).to_parquet(root/'cnv.parquet',index=False)
    pd.DataFrame({'cancer_id':'BRCA','patient_id':patients,'purity':.7}).to_parquet(root/'purity.parquet',index=False)
    pd.DataFrame({'cancer_id':['BRCA'],'lncrna_id':['LNC:Z'],'pathway_id':['P1']}).to_parquet(root/'candidates.parquet',index=False)
    atac=root/'atac';atac.mkdir();write(atac/'UNAVAILABLE.json',{'reason':'Synthetic fixture without ATAC'})
    directory=lambda p:{'path':str(p),'sha256':artifact_sha256(p)}
    distal={'format':'NATIVE_DISTAL_WORKFLOW_INPUT_V1','methylation_workflow':{'stage_ref':'methylation'},
            'distal_mutation':{'stage_ref':'mutation','pointer':'/distal_mutation'},'local_cnv':record(root/'cnv.parquet'),
            'purity_covariates':record(root/'purity.parquet'),'candidate_universe':record(root/'candidates.parquet'),
            'formal_expression_root':directory(e.parent),'pathway_activity_root':directory(activity.parent),'distal_atac_root':directory(atac)}
    # The sole substituted production constant is the synthetic MAF's digest.
    # Official chain bytes/checksum, producers and all consumers remain actual.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(producer,'EXPECTED_MC3_SHA256',record(maf)['sha256'])
        chain_input=root/'CHAIN_INPUT.json';write(chain_input,{'format':chain.INPUT,'chain_id':'raw-mc3','stages':[
            {'id':'mutation','operation':'distal_mutation.run','inputs':mutation},
            {'id':'methylation','operation':'methylation.run','inputs':methylation},
            {'id':'distal','operation':'distal.run','inputs':distal}]})
        chain.initialize(chain_input,root/'chain');chain.advance(root/'chain');state=read(root/'chain/CHAIN.json')
        assert state['status']=='DECLARED_STAGES_COMPLETE',state
        yield root,mutation,state


def test_raw_mc3_to_actual_distal_closure_and_resume(completed,monkeypatch):
    root,spec,state=completed;_,binding=workflow.validate(state['results']['mutation'],root)
    actual=pd.read_parquet(binding['distal_mutation']['path']).sort_values('patient_id')
    np.testing.assert_allclose(actual.distal_mutation_burden.iloc[:-1],np.log1p(np.arange(59)%7))
    assert actual.distal_variant_count.iloc[1]==1
    assert pd.isna(actual.distal_mutation_burden.iloc[-1]) and not actual.distal_mutation_burden__available.iloc[-1]
    assert not actual.distal_mutation_locus_callable.any()
    source=read(Path(binding['path']).parent/'inputs/SOURCE_AUDIT.json') if 'path' in binding else read(Path(binding['preflight']['path']).parent/'inputs/SOURCE_AUDIT.json')
    assert source['quarantined_variants']==1 and source['assayed_patients']==59
    distal=read(state['results']['distal']['path']);assert read(distal['closure']['path'])['candidate_available_rows']==1
    monkeypatch.setattr(chain,'dispatch',forbidden);chain.advance(root/'chain')


def test_raw_mutation_sources_reach_actual_public_query(completed,current_sources,tmp_path):
    from cc_hhgt.v32.current_omics_context import materialize_current_omics_context,CurrentOmicsContextQuery
    root,spec,state=completed;distal=read(state['results']['distal']['path']);manifest=tmp_path/'INPUT.json'
    write(manifest,{'format':'CURRENT_OMICS_CONTEXT_INPUT_V1','kind':'distal','selected_analysis':record(current_sources[1]),'native_binding':distal['audit']})
    result=materialize_current_omics_context(manifest,tmp_path/'public')
    query=CurrentOmicsContextQuery(result['binding_path'],expected_sha=result['binding_sha256'],selected_analysis_sha=record(current_sources[1])['sha256'],kind='distal')
    row=pd.read_parquet(query.download('selected_member_distal')['path']).iloc[0]
    assert row.distal_regulatory_mutation_pathway_available and row.mean_oof_slope<0 and row.oof_prediction_correlation>.9
    sources={pin['path'] for pin in read(result['binding_path'])['source_artifacts']}
    assert spec['mc3']['path'] in sources and str(CHAIN) in sources


def test_rehashed_mutation_values_cannot_replace_raw_replay(completed):
    _,_,state=completed;binding=read(state['results']['mutation']['path']);path=Path(binding['distal_mutation']['path']);audit_path=path.parent/'AUDIT.json'
    workflow.validate_features(path);original={p:p.read_bytes() for p in (path,audit_path)}
    try:
        frame=pd.read_parquet(path);frame.loc[0,'distal_mutation_burden']+=.2;frame.to_parquet(path,index=False)
        audit=read(audit_path);audit['outputs'][path.name]={**record(path),'bytes':path.stat().st_size};write(audit_path,audit)
        with pytest.raises(ValueError,match='Distal mutation replay'):feature_topology_lineage(path,kind='distal_mutation')
    finally:
        for p,data in original.items():p.write_bytes(data)


def test_rehashed_lifted_input_rejected(completed):
    _,_,state=completed;binding=read(state['results']['mutation']['path']);audit_path=Path(binding['audit']['path']);audit=read(audit_path)
    path=Path(audit['inputs']['lifted_coordinates']['path']);original={p:p.read_bytes() for p in (path,audit_path)}
    try:
        frame=pd.read_parquet(path);frame.loc[frame.liftover_available,'start0_grch38']+=1;frame.to_parquet(path,index=False)
        audit['inputs']['lifted_coordinates']=record(path);write(audit_path,audit)
        with pytest.raises(ValueError,match='MC3 input replay'):workflow.validate_features(binding['distal_mutation']['path'])
    finally:
        for p,data in original.items():p.write_bytes(data)


def test_changed_raw_source_and_fractional_fold_are_rejected(completed,tmp_path):
    _,spec,state=completed;binding=read(state['results']['mutation']['path']);path=Path(spec['mc3']['path']);original=path.read_bytes()
    try:
        path.write_bytes(original+b'changed')
        with pytest.raises(ValueError):workflow.validate_features(binding['distal_mutation']['path'])
    finally:path.write_bytes(original)
    folds=pd.read_csv(spec['patient_folds']['path'],sep='\t');folds['patient_fold_id']=folds.patient_fold_id.astype(float);folds.loc[0,'patient_fold_id']=.5
    folds.to_csv(tmp_path/'folds.tsv',sep='\t',index=False)
    with pytest.raises(ValueError,match='integers'):mc3_sources.patient_authority(tmp_path/'folds.tsv')


def test_source_rows_are_chunk_invariant(completed,tmp_path):
    _,spec,_=completed
    a=mc3_sources.prepare(spec['mc3']['path'],CHAIN,spec['patient_folds']['path'],tmp_path/'a',chunksize=1)
    b=mc3_sources.prepare(spec['mc3']['path'],CHAIN,spec['patient_folds']['path'],tmp_path/'b',chunksize=41)
    assert a==b
    for name in ('variant_event.parquet','variant_coordinate_grch38.parquet','mutation_sample_crosswalk.parquet'):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'a'/name),pd.read_parquet(tmp_path/'b'/name))


def test_liftover_forward_reverse_gap_ambiguity_and_boundaries(tmp_path):
    path=tmp_path/'test.chain.gz'
    with gzip.open(path,'wt') as handle:
        handle.write('chain 1 chr1 1000 + 10 30 chrX 1000 + 100 120 1\n20\n\n'
                     'chain 1 chr2 1000 + 10 30 chrY 1000 - 100 120 2\n20\n\n'
                     'chain 1 chr3 1000 + 10 40 chrZ 1000 + 100 130 3\n10 10 10\n10\n\n'
                     'chain 1 chr4 1000 + 10 30 chrX 1000 + 100 120 4\n20\n\n'
                     'chain 1 chr4 1000 + 10 30 chrY 1000 + 200 220 5\n20\n')
    variants=pd.DataFrame({'variant_id':['plus','minus','gap','ambiguous','boundary','none'],
        'chromosome':['1','2','3','4','1','9'],'start':[11,11,20,11,31,11],'end':[12,12,31,12,31,12]})
    with duckdb.connect() as con:actual=mc3_sources.lift_variants(variants,mc3_sources.chain_blocks(path),con).set_index('variant_id')
    assert (actual.loc['plus','start0_grch38'],actual.loc['plus','end0_grch38'])==(100,102)
    assert (actual.loc['minus','start0_grch38'],actual.loc['minus','end0_grch38'])==(898,900)
    assert actual.loc['ambiguous','failure_reason']=='ambiguous_complete_block'
    assert not actual.loc[['gap','boundary','none'],'liftover_available'].any()


def test_rehashed_assembled_projection_is_rejected(completed):
    _,_,state=completed;binding=read(state['results']['distal']['path']);prepared=read(binding['prepared']['path'])
    pin=prepared['outputs'][0];path=Path(pin['path']);original=path.read_bytes()
    try:
        frame=pd.read_parquet(path);frame.loc[0,'distal_mutation_burden']+=.1;frame.to_parquet(path,index=False)
        pin.update(record(path))
        with pytest.raises(ValueError,match='projection differs'):workflow.validate_distal_input(prepared,path.parent)
    finally:path.write_bytes(original)
