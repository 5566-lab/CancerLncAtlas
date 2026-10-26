"""GTF coordinates reach original signed and genomic CNV computations; no fit."""
from pathlib import Path
import copy
import json
import shutil
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import native_genomic_annotation_workflow as annotation
from cc_hhgt.v32 import native_annotation_workflow as reference
from cc_hhgt.v32 import native_directional_cnv as directional
from cc_hhgt.v32 import native_genomic_cnv as genomic
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_annotation_workflow import inputs as reference_input, GENES, LNCS
from test_v32_native_expression_workflow import fixture_input as expression_input
from test_v32_current_core_export import export_fixture
from test_v32_native_genomic_inputs import no_fit


def inputs(root):
    source=reference_input(root/'raw');binding=reference.run(source,root/'reference')
    manifest=root/'INPUT.json';write(manifest,{'format':annotation.INPUT,'reference_annotation':record(binding)})
    return manifest


def test_all_original_coordinates_and_ids_survive_conversion(tmp_path):
    from test_v32_native_annotation_workflow import gtf_text
    source=reference_input(tmp_path/'raw',text=gtf_text(extra=True));binding=reference.run(source,tmp_path/'reference')
    manifest=tmp_path/'INPUT.json';write(manifest,{'format':annotation.INPUT,'reference_annotation':record(binding)})
    output=annotation.run(manifest,tmp_path/'intervals');value=read(output);frame=pd.read_parquet(value['intervals']['path'])
    lnc=frame.loc[frame.entity_id.eq(LNCS[0])].iloc[0]
    assert (lnc.start,lnc.end,lnc.chromosome,lnc.gtf_line)==(1,90,'chr1',4)
    assert 'LNC:'+GENES[0].split('.')[0]+'_PAR_Y' in set(frame.entity_id)
    assert set(frame.entity_type)=={'gene','lncrna'} and len(frame)==84
    assert value['metadata']['excluded_other_biotype_rows']==1
    assert frame.attrs[annotation.MARKER]==value['preflight']
    assert set(frame.coordinate_system)=={'1_BASED_CLOSED'}
    assert not value['reference_authenticity_verified']


@pytest.mark.parametrize('fault',['start','chromosome','entity_id','entity_type','omit','marker','metadata'])
def test_rehashed_coordinates_and_metadata_are_reconstructed(tmp_path,fault):
    manifest=inputs(tmp_path/'input');binding=annotation.run(manifest,tmp_path/'output');value=read(binding)
    if fault=='metadata':value['metadata']['interval_rows']-=1
    else:
        path=Path(value['intervals']['path']);frame=pd.read_parquet(path)
        if fault=='start':frame.loc[0,'start']+=1
        elif fault=='omit':frame=frame.iloc[1:].copy()
        elif fault=='marker':frame.attrs={}
        else:frame.loc[0,fault]='changed'
        frame.to_parquet(path,index=False);value['intervals']=record(path)
    write(binding,value)
    with pytest.raises(ValueError):annotation.validate(record(binding),tmp_path)


def test_copies_and_changed_raw_sources_cannot_use_cached_proof(tmp_path):
    manifest=inputs(tmp_path/'input');binding=annotation.run(manifest,tmp_path/'output');value=read(binding)
    copied=tmp_path/'copy.parquet';shutil.copyfile(value['intervals']['path'],copied)
    with pytest.raises(ValueError,match='outside'):annotation.discover(copied)
    raw=Path(read(value['preflight']['path'])['gtf']['path']);raw.write_text(raw.read_text()+'# changed\n')
    with pytest.raises(ValueError):annotation.discover(value['intervals']['path'])


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp('native_genomic_reference')
    expression=expression_input(root/'expression');spec=read(expression)
    # Preserve an explicitly unmapped pathway member, including its denominator.
    gmt=Path(read(spec['gmt_manifest']['path'])['collections'][0]['file']['path'])
    gmt.write_text(gmt.read_text().rstrip('\n')+'\tUNMAPPED\n')
    gmt_manifest=Path(spec['gmt_manifest']['path']);gmt_value=read(gmt_manifest);gmt_value['collections'][0]['file']=record(gmt)
    write(gmt_manifest,gmt_value);spec['gmt_manifest']=record(gmt_manifest)
    source=reference_input(root/'gtf');spec['annotation']={'stage_ref':'reference','pointer':'/annotation'}
    request=root/'CHAIN_INPUT.json';write(request,{'format':chain.INPUT,'chain_id':'reference-genomic-expression','stages':[
        {'id':'reference','operation':'annotation.run','inputs':read(source)},
        {'id':'intervals','operation':'genomic_annotation.run','inputs':{'format':annotation.INPUT,'reference_annotation':{'stage_ref':'reference'}}},
        {'id':'expression','operation':'expression.run','inputs':spec}]})
    chain.initialize(request,root/'chain');chain.advance(root/'chain');state=read(root/'chain/CHAIN.json')
    assert state['status']=='DECLARED_STAGES_COMPLETE',state
    expression_result=read(state['results']['expression']['path']);outputs=expression_result['outputs']
    interval=read(state['results']['intervals']['path'])['intervals']
    candidates=pd.read_parquet(outputs['candidates']['path']);pathway=candidates.pathway_id.unique().item()
    genes=['GENE:'+g.split('.')[0] for g in GENES[3:23]]+['GENE:ENSG99999999999']
    membership=root/'membership.parquet';pd.DataFrame({'pathway_id':pathway,'gene_id':genes}).to_parquet(membership,index=False)
    patients=pd.read_csv(outputs['patient_map']['path'],sep='\t');segments=[]
    for i,patient in enumerate(patients.patient_id):
        row={'cancer_id':'BRCA','patient_id':patient}
        if i==19:
            segments.append({**row,'status':'typed_unavailable','unavailable_reason':'NO_SELECTED_SEGMENT_FILE'});continue
        file=root/f'patient_{i}.seg.tsv'
        pd.DataFrame([('chr1',1,44,.9),('chr1',45,90,-.6),('chr1',91,300,0.),
            ('chr1',301,390,.6 if i%2==0 else 0.),('chr1',391,10000,0.)],
            columns=['Chromosome','Start','End','Segment_Mean']).assign(Sample=patient).to_csv(file,sep='\t',index=False)
        segments.append({**row,'status':'available','file':record(file),'study_metadata':{'dataset_id':'SYNTHETIC_SEGMENTS'}})
    segment_manifest=root/'SEGMENTS.json';write(segment_manifest,{'format':directional.INPUT,
        'candidate_definition':outputs['candidate_definition'],'candidates':outputs['candidates'],'patient_map':outputs['patient_map'],
        'intervals':interval,'membership':record(membership),'segments':segments,'event_threshold':.3,
        'segment_reference':{'genome_build':'GRCh38','coordinate_system':'1_BASED_CLOSED'}})
    from scripts.materialize_v32_directional_cnv import main as signed
    signed(['--phase','all','--native-manifest',str(segment_manifest),'--output-root',str(root/'directional')])
    assert read(root/'directional/SUCCESS.json')['status']=='SUCCESS'
    # Frozen export serialization only; original genomic CNV requires its binding
    # but does not execute an encoder, head, optimizer or fitted prediction.
    core_root=root/'core';core_root.mkdir();core_path,core=export_fixture(core_root)
    for fold in core['folds'].values():
        for kind,ids in [('lncRNA',LNCS),('pathway',[pathway])]:
            file=core_root/fold['exports'][kind]['path']
            pd.DataFrame({'node_index':range(len(ids)),'node_id':ids,'core_feature_000':np.arange(len(ids),dtype=float),'core_feature_001':1.}).to_parquet(file,index=False)
            fold['exports'][kind]={**record(file),'rows':len(ids),'features':2}
    core['fixture_no_optimizer_run']=True;write(core_path,core)
    gene=root/'mutation_gene.parquet';lnc=root/'mutation_lnc.parquet'
    pd.DataFrame([{'cancer_id':'BRCA','patient_id':p,'gene_id':g,'is_mutated':bool(i%2),'mutation_count':i%2,
                  'absence_is_wildtype':not bool(i%2)} for i,p in enumerate(patients.patient_id) for g in genes]).to_parquet(gene,index=False)
    pd.DataFrame([{'cancer_id':'BRCA','patient_id':p,'lncrna_id':g,'lncrna_any_mutation':bool(i%2),'lncrna_mutation_available':True,
                  'exonic_variant_count':i%2,'absence_is_wildtype':not bool(i%2)} for i,p in enumerate(patients.patient_id) for g in LNCS]).to_parquet(lnc,index=False)
    mc3=root/'mc3.tsv';mc3.write_text('Hugo_Symbol\tTumor_Sample_Barcode\nG1\tSYNTHETIC\n')
    genomic_manifest=root/'GENOMIC.json';write(genomic_manifest,{'format':genomic.INPUT,'segment_manifest':record(segment_manifest),
        'sample_gene_mutation':record(gene),'sample_lncrna_mutation':record(lnc),'mc3':record(mc3),
        'core_manifest':record(core_path),'core_run_id':'current-G2','max_memory_bytes':10**12})
    from scripts.materialize_native_genomic_cnv import main as compact
    assert compact(['prepare','--manifest',str(genomic_manifest),'--output',str(root/'genomic')])==0
    return root,segment_manifest,genomic_manifest,source


def test_both_original_cnv_estimands_use_gtf_coordinates_and_keep_missing_members(completed):
    root,segments,manifest,source=completed
    directional.validate_release(record(root/'directional/SUCCESS.json'),root,replay=True)
    genomic.validate(root/'genomic/SUCCESS.json',replay=True)
    signed=root/'directional/cancer=BRCA';compact=root/'genomic/cancer=BRCA'
    ids=read(signed/'lncrna_ids.json');index=ids.index(LNCS[0])
    # Original midpoint is base 45, covered by the deletion segment, not 1..44.
    np.testing.assert_allclose(np.load(signed/'lncrna_signed_mean.npy')[:19,index],-.6)
    np.testing.assert_allclose(np.load(compact/'lncrna_burden.npy')[:19,index],.6)
    np.testing.assert_array_equal(np.load(signed/'lncrna_deletion.npy')[:19,index],1)
    even=np.arange(0,19,2);odd=np.arange(1,19,2)
    np.testing.assert_allclose(np.load(signed/'pathway_signed_mean.npy')[even,0],.6/20)
    np.testing.assert_allclose(np.load(compact/'pathway_burden.npy')[even,0],.6)
    for folder,name in [(signed,'pathway_signed_mean.npy'),(compact,'pathway_burden.npy')]:
        assert np.isnan(np.load(folder/name)[odd]).all() and np.isnan(np.load(folder/name)[19]).all()
        assert not np.load(folder/'pathway_callable.npy')[odd].any()
    context=read(root/'genomic/RUN_CONTEXT.json')['interval_reference']
    assert context['membership_genes']==21 and context['membership_genes_with_coordinates']==20
    assert context['membership_genes_without_coordinates']==1 and context['missing_members_dropped'] is False
    assert context==read(signed/'SUCCESS.json')['interval_reference']
    assert not context['segment_reference_independently_verified']


def test_actual_both_native_source_inspectors_reach_gtf(completed,monkeypatch):
    root,_,_,source=completed
    from cc_hhgt.v32.current_model_use import Inventory
    from cc_hhgt.v32.current_cnv_model_use import inspect_bound_sources
    from cc_hhgt.v32.current_auxiliary_model_use import input_source
    for mode in ['directional','genomic']:
        inventory=Inventory([],root)
        if mode=='directional':inspect_bound_sources(inventory,[record(root/'directional/cancer=BRCA/SUCCESS.json')],root,'core','cnv',0)
        else:input_source(inventory,record(root/'genomic/SUCCESS.json'),root,'genomic','cnv',0)
        assert read(source)['gtf']['path'] in {p['path'] for p in inventory.dependencies}
        assert any(p['format']==annotation.FORMAT for p in inventory.transform_replays)
        reasons={p['reason'] for p in inventory.gaps if p['role'].endswith('.intervals')}
        assert reasons=={'GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED'}
    before={str(p):p.stat().st_mtime_ns for p in (root/'chain/stages').rglob('*') if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('Completed scientific stage rerun')
    monkeypatch.setattr(chain,'dispatch',forbidden);chain.advance(root/'chain')
    assert before=={str(p):p.stat().st_mtime_ns for p in (root/'chain/stages').rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['missing_reference','build','half_open','zero_start','different_annotation'])
def test_reference_or_coordinate_mismatch_cannot_publish_partition(completed,tmp_path,fault):
    root,manifest,_,_=completed;spec=copy.deepcopy(read(manifest))
    if fault=='missing_reference':spec.pop('segment_reference')
    elif fault=='build':spec['segment_reference']['genome_build']='GRCh37'
    elif fault=='half_open':spec['segment_reference']['coordinate_system']='0_BASED_HALF_OPEN'
    elif fault=='zero_start':
        raw=tmp_path/'zero.tsv';frame=pd.read_csv(spec['segments'][0]['file']['path'],sep='\t');frame.loc[0,'Start']=0
        frame.to_csv(raw,sep='\t',index=False);spec['segments'][0]['file']=record(raw)
    else:
        from test_v32_native_annotation_workflow import gtf_text
        other_source=reference_input(tmp_path/'other',text=gtf_text().replace('\t1\t90\t','\t2\t90\t',1))
        other_reference=reference.run(other_source,tmp_path/'other_reference')
        other=tmp_path/'other.json';write(other,{'format':annotation.INPUT,'reference_annotation':record(other_reference)})
        binding=annotation.run(other,tmp_path/'other_intervals')
        spec['intervals']=read(binding)['intervals']
    request=tmp_path/'SEGMENTS.json';write(request,spec)
    with pytest.raises(ValueError):directional.materialize(record(request),tmp_path,tmp_path/'failed','BRCA')
    assert not (tmp_path/'failed/cancer=BRCA/SUCCESS.json').exists()


def test_same_reference_bytes_in_a_new_bound_location_are_accepted(completed,tmp_path):
    manifest=inputs(tmp_path/'same');binding=annotation.run(manifest,tmp_path/'intervals')
    spec=copy.deepcopy(read(completed[1]));spec['intervals']=read(binding)['intervals']
    request=tmp_path/'SEGMENTS.json';write(request,spec)
    loaded=directional.load_manifest(record(request),tmp_path)
    assert directional.reference_context(loaded)['membership_genes_without_coordinates']==1


def test_host_and_existing_output_guards(tmp_path,monkeypatch):
    manifest=inputs(tmp_path/'input')
    command=[sys.executable,'-m','scripts.run_native_genomic_annotation_workflow','--manifest',str(manifest),
             '--output',str(tmp_path/'cli')]
    planned=subprocess.run([*command,'--plan'],cwd=annotation.ROOT,capture_output=True,text=True,timeout=30)
    assert planned.returncode==0,planned.stderr
    assert json.loads(planned.stdout)['target_host']=='149' and not (tmp_path/'cli').exists()
    result=subprocess.run(command,cwd=annotation.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    annotation.validate(json.loads(result.stdout)['binding'],tmp_path)
    with pytest.raises(FileExistsError):annotation.run(manifest,tmp_path/'input')
    monkeypatch.setattr(socket,'gethostname',lambda:'Windows')
    with pytest.raises(RuntimeError,match='149'):annotation.plan(tmp_path/'missing',tmp_path/'new')
