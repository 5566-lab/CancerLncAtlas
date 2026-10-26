"""Original GTF cache reaches actual H5 mapping; no cell or model fitting."""
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pandas as pd
import pytest

from cc_hhgt.v32 import native_annotation_workflow as reference
from cc_hhgt.v32 import native_single_cell_annotation_workflow as workflow
from cc_hhgt.v32 import native_single_cell_mapping_workflow as mapping
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_single_cell_mapping_workflow import source, no_fit, forbidden


def reference_input(root, *, extras=False):
    root.mkdir(parents=True, exist_ok=True)
    rows = ['##description: evidence-based annotation of the human genome (GRCh38.p13), version 36 (Ensembl 102)']
    for i in range(7):
        kind = 'protein_coding' if i < 5 else 'lncRNA'
        rows.append(f'chr1\tHAVANA\tgene\t{i*100+1}\t{i*100+90}\t.\t+\t.\tgene_id "ENSG{i+1:06d}.1"; gene_type "{kind}"; gene_name "S{i}"; level 2;')
    if extras:
        for chromosome, gene, kind, symbol in [('chrY','ENSG000006.1_PAR_Y','lncRNA','S5'),
                ('chrM','ENSG800001.1','protein_coding','M'), ('KI270728.1','ENSG800002.1','protein_coding','ALT'),
                ('chr1','ENSG800003.1','pseudogene','PSEUDO')]:
            rows.append(f'{chromosome}\tHAVANA\tgene\t1\t30\t.\t+\t.\tgene_id "{gene}"; gene_type "{kind}"; gene_name "{symbol}";')
    raw = root / 'reference.gtf'; raw.write_text('\n'.join(rows)+'\n')
    manifest = root / 'REFERENCE_INPUT.json'
    write(manifest, {'format':reference.INPUT,'gtf':record(raw),'gencode_release':36,'genome_build':'GRCh38'})
    return manifest


def inputs(root, *, extras=False):
    raw = reference_input(root/'raw',extras=extras)
    binding = reference.run(raw,root/'reference')
    manifest = root/'SC_INPUT.json'
    write(manifest,{'format':workflow.INPUT,'reference_annotation':record(binding)})
    return manifest


def attach_reference(manifest, root):
    """Replace a synthetic hand-authored gene table with original GTF production."""
    spec = read(manifest); input_file = inputs(root)
    binding = workflow.run(input_file,root/'single_cell')
    pin = read(binding)['annotation']; status_path = Path(spec['run_status']['path'])
    status = read(status_path); row = status['per_cancer']['BRCA']
    row['annotation_path'] = pin['path']; row['annotation_sha256'] = pin['sha256']
    write(status_path,status); spec['run_status']=record(status_path); write(manifest,spec)
    return binding


def test_original_gene_cache_rules_and_par_y_are_unchanged(tmp_path):
    manifest=inputs(tmp_path/'input',extras=True); binding=workflow.run(manifest,tmp_path/'output')
    _,value=workflow.validate(record(binding),tmp_path)
    table=pd.read_parquet(value['annotation']['path'])
    assert len(table)==8 and set(table.gene_class)=={'lncRNA','protein_coding'}
    assert set(table.chromosome)=={'chr1','chrY'} and 'ENSG000006.1_PAR_Y' in set(table.gene_id)
    assert not table.loc[table.gene_symbol.eq('S5'),'unique_symbol'].any()
    assert table.loc[table.gene_symbol.eq('S0'),'unique_symbol'].all()
    assert table.attrs[workflow.MARKER]==value['preflight']
    legacy=tmp_path/'original.parquet'; original=tmp_path/'original.json'
    from cc_hhgt.v32.single_cell_partition_builder import prepare_gencode_annotation
    launch=read(value['preflight']['path'])
    prepare_gencode_annotation(gtf_path=launch['gtf']['path'],output_parquet=legacy,provenance_path=original)
    pd.testing.assert_frame_equal(table,pd.read_parquet(legacy),check_exact=True)
    assert not pd.read_parquet(legacy).attrs and workflow.discover(legacy) is None


@pytest.mark.parametrize('fault',['gene_class','gene_symbol','unique_symbol','drop_row','provenance','marker'])
def test_rehashed_annotation_cannot_replace_original_reconstruction(tmp_path,fault):
    manifest=inputs(tmp_path/'input'); binding=workflow.run(manifest,tmp_path/'output'); value=read(binding)
    path=Path(value['annotation']['path']); provenance=Path(value['provenance']['path']); payload=read(provenance)
    if fault=='provenance': payload['primary_chromosomes_only']=False
    else:
        frame=pd.read_parquet(path)
        if fault=='drop_row': frame=frame.iloc[1:].copy()
        elif fault=='marker': frame.attrs={}
        elif fault=='unique_symbol': frame.loc[0,fault]=False
        else: frame.loc[0,fault]='changed'
        frame.to_parquet(path,index=False); value['annotation']=record(path); payload['annotation_sha256']=value['annotation']['sha256']
    from cc_hhgt.v32.single_cell_partition_builder import _canonical_sha256
    payload.pop('contract_sha256');payload['contract_sha256']=_canonical_sha256(payload)
    write(provenance,payload);value['provenance']=record(provenance);write(binding,value)
    with pytest.raises(ValueError):workflow.validate(record(binding),tmp_path)


def test_marked_copy_missing_reference_and_cached_raw_changes_are_rejected(tmp_path):
    manifest=inputs(tmp_path/'input');binding=workflow.run(manifest,tmp_path/'output');value=read(binding)
    copied=tmp_path/'copy.parquet';shutil.copyfile(value['annotation']['path'],copied)
    with pytest.raises(ValueError,match='outside'):workflow.discover(copied)
    launch=read(value['preflight']['path']);raw=Path(launch['gtf']['path']);raw.write_text(raw.read_text()+'# changed\n')
    with pytest.raises(ValueError):workflow.discover(value['annotation']['path'])


def test_actual_three_stage_chain_and_mapping_contract(tmp_path,monkeypatch):
    manifest,spec,values,_=source(tmp_path/'raw')
    raw=reference_input(tmp_path/'gencode'); status=read(spec['run_status']['path'])
    row=status['per_cancer']['BRCA']
    row['annotation_path']={'stage_ref':'annotation','pointer':'/annotation/path'}
    row['annotation_sha256']={'stage_ref':'annotation','pointer':'/annotation/sha256'}
    spec['run_status']={'json_file':status}
    request=tmp_path/'CHAIN_INPUT.json'
    write(request,{'format':chain.INPUT,'chain_id':'raw-reference-sc-mapping','stages':[
        {'id':'reference','operation':'annotation.run','inputs':read(raw)},
        {'id':'annotation','operation':'single_cell_annotation.run','inputs':{'format':workflow.INPUT,
            'reference_annotation':{'stage_ref':'reference'}}},
        {'id':'mapping','operation':'single_cell_mapping.run','inputs':spec}]})
    output=tmp_path/'chain';chain.initialize(request,output);chain.advance(output);state=read(output/'CHAIN.json')
    assert state['status']=='DECLARED_STAGES_COMPLETE',state
    result=read(state['results']['mapping']['path']);launch=read(result['preflight']['path'])
    expected=state['results']['annotation']; assert launch['contract']['gene_annotation_preparation']==expected
    assert read(raw)['gtf']['path'] in {pin['path'] for pin in launch['source_files']}
    from scipy import sparse
    import numpy as np
    mapped=Path(result['mapped_success']['path']).parent
    np.testing.assert_array_equal(sparse.load_npz(mapped/'CODING_MEASURED.npz').toarray(),np.vstack((values[0]+values[1],values[2:6]))[:,3:].T)
    np.testing.assert_array_equal(sparse.load_npz(mapped/'LNC_MEASURED.npz').toarray(),values[6:8,3:].T)
    from cc_hhgt.v32.current_model_use import Inventory
    from cc_hhgt.v32.current_auxiliary_model_use import input_source
    inventory=Inventory([],tmp_path);pin=read(expected['path'])['annotation']
    input_source(inventory,pin,tmp_path,'sc-model','gene_annotation')
    assert read(raw)['gtf']['path'] in {p['path'] for p in inventory.dependencies}
    assert {p['format'] for p in inventory.transform_replays}=={workflow.FORMAT}
    assert not inventory.usage  # A reference is not a patient cohort/study.
    before={str(p):p.stat().st_mtime_ns for p in (output/'stages').rglob('*') if p.is_file()}
    monkeypatch.setattr(chain,'dispatch',forbidden);chain.advance(output)
    assert before=={str(p):p.stat().st_mtime_ns for p in (output/'stages').rglob('*') if p.is_file()}


def test_native_h5_loader_rejects_changed_reference_even_with_same_table(tmp_path):
    manifest,_,_,_=source(tmp_path/'raw');binding=attach_reference(manifest,tmp_path/'gencode')
    plan=mapping.plan(manifest,tmp_path/'mapping');mapping.upstream(plan)
    original=Path(read(read(binding)['preflight']['path'])['gtf']['path']);original.write_text(original.read_text()+'# changed\n')
    with pytest.raises(ValueError):mapping.upstream(plan)
    assert not (tmp_path/'mapping').exists()


def test_unmarked_external_table_retains_legacy_source_contract(tmp_path):
    manifest,spec,_,_=source(tmp_path/'raw');status=read(spec['run_status']['path']);row=status['per_cancer']['BRCA']
    copied=tmp_path/'arbitrary.parquet';shutil.copyfile(row['annotation_path'],copied)
    row['annotation_path']=str(copied);row['annotation_sha256']=record(copied)['sha256']
    write(Path(spec['run_status']['path']),status);spec['run_status']=record(spec['run_status']['path']);write(manifest,spec)
    from cc_hhgt.v32.single_cell_r7_streaming import StreamingContractError
    with pytest.raises(StreamingContractError,match='outside the pinned r7 contract'):mapping.plan(manifest,tmp_path/'new')


def test_cli_and_host_guard(tmp_path,monkeypatch):
    manifest=inputs(tmp_path/'input')
    command=[sys.executable,'-m','scripts.run_native_single_cell_annotation_workflow','--manifest',str(manifest),'--output',str(tmp_path/'output')]
    result=subprocess.run(command,cwd=workflow.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    workflow.validate(record(tmp_path/'output'/workflow.BINDING),tmp_path)
    with pytest.raises(FileExistsError):workflow.run(manifest,tmp_path/'output')
    monkeypatch.setattr(socket,'gethostname',lambda:'Windows')
    with pytest.raises(RuntimeError,match='149'):workflow.plan(tmp_path/'missing',tmp_path/'new')
