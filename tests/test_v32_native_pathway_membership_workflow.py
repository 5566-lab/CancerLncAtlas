"""Actual GMT/GENCODE mapping and immutable handoffs; no scientific fitting."""
from pathlib import Path
import json
import shutil
import socket
import subprocess
import sys

import pandas as pd
import pytest

from cc_hhgt.v32 import native_annotation_workflow as reference
from cc_hhgt.v32 import native_pathway_membership_workflow as members
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_annotation_workflow import inputs as reference_input, gtf_text, GENES


def inputs(root):
    root.mkdir(parents=True, exist_ok=True)
    text=gtf_text()+('chrY\tHAVANA\tgene\t1\t90\t.\t+\t.\tgene_id "ENSG99999999990.1_PAR_Y"; gene_type "protein_coding"; gene_name "G1";\n'
        'chr1\tHAVANA\tgene\t10001\t10090\t.\t+\t.\tgene_id "ENSG99999999991.1"; gene_type "pseudogene"; gene_name "G1";\n'
        'chr1\tHAVANA\tgene\t10101\t10190\t.\t+\t.\tgene_id "ENSG99999999992.1"; gene_type "pseudogene"; gene_name "ONLY_NONCODING";\n')
    raw=reference_input(root/'raw',text=text);ref=reference.run(raw,root/'reference')
    gmt=root/'sets.gmt';gmt.write_text('P1\tfixture\tG1\tG1\tG2\tUNKNOWN\tg1\tONLY_NONCODING\nP2\tfixture\tUNKNOWN\n')
    gmt_manifest=root/'GMT.json';pin=record(gmt);pin['path']='sets.gmt'
    write(gmt_manifest,{'format':'SSGSEA_GMT_INPUT_V1','collections':[{'collection':'HALLMARK','file':pin}]})
    request=root/'INPUT.json';write(request,{'format':members.INPUT,'reference_annotation':record(ref),'gmt_manifest':record(gmt_manifest)})
    return request


def test_exact_original_mapping_and_all_unavailable_members_are_preserved(tmp_path):
    request=inputs(tmp_path/'input');binding=members.run(request,tmp_path/'output');value=read(binding)
    frame=pd.read_parquet(value['membership']['path']);p1=frame.loc[frame.pathway_id.eq('MSIGDB:HALLMARK:P1')]
    assert set(p1.loc[p1.gene_symbol.eq('G1'),'gene_id'])=={'GENE:'+GENES[3].split('.')[0],'GENE:ENSG99999999990_PAR_Y'}
    assert 'GENE:ENSG99999999991' not in set(frame.gene_id)
    assert p1.loc[p1.gene_symbol.eq('ONLY_NONCODING'),'gene_id'].item()=='GENE:ENSG99999999992'
    assert set(p1.loc[p1.mapping_status.eq('UNMAPPED_SYMBOL'),'gene_symbol'])=={'UNKNOWN','g1'}
    assert frame.loc[frame.gene_symbol.eq('UNKNOWN'),'gene_id'].nunique()==1
    assert frame.loc[frame.mapping_status.eq('UNMAPPED_SYMBOL'),'gene_id'].str.startswith('UNMAPPED_GMT:').all()
    assert value['metadata']['source_symbol_memberships']==6 and len(frame)==7
    assert value['metadata']['one_to_many_symbol_memberships']==1
    assert value['metadata']['unmapped_symbol_memberships']==3 and value['metadata']['noncoding_fallback_rows']==1
    # The original standalone builder retains its historical mapping output.
    from scripts.build_v32_genomic_static_assets import build
    launch=read(value['preflight']['path']);annotation=reference.annotation_table(launch['annotation'],tmp_path)
    dim=tmp_path/'dim.parquet';annotation.to_parquet(dim,index=False)
    raw_members=tmp_path/'symbols.parquet';frame[['pathway_id','gene_symbol']].drop_duplicates().to_parquet(raw_members,index=False)
    lnc=tmp_path/'lnc.parquet';pd.DataFrame([{'lncrna_id':'LNC:fixture','feature':'gene','chromosome':'chr1','start0':0,'end0':90}]).to_parquet(lnc,index=False)
    result=build(raw_members,dim,lnc,tmp_path/'legacy')
    old=pd.read_parquet(result['membership']);expected=frame.loc[frame.mapping_status.eq('MAPPED'),['pathway_id','ensembl_gene_id']]
    expected=expected.rename(columns={'ensembl_gene_id':'gene_id'}).drop_duplicates()
    pd.testing.assert_frame_equal(old.sort_values(['pathway_id','gene_id']).reset_index(drop=True),
        expected.sort_values(['pathway_id','gene_id']).reset_index(drop=True))
    assert set(pd.read_parquet(result['intervals']).query("entity_type=='lncrna'").start)=={1}


@pytest.mark.parametrize('fault',['gene_id','symbol','status','drop_unmapped','marker','metadata'])
def test_repinning_cannot_forge_mapping_or_missingness(tmp_path,fault):
    request=inputs(tmp_path/'input');binding=members.run(request,tmp_path/'output');value=read(binding)
    if fault=='metadata':value['metadata']['unmapped_symbol_memberships']=0
    else:
        path=Path(value['membership']['path']);frame=pd.read_parquet(path)
        if fault=='drop_unmapped':frame=frame.loc[frame.mapping_status.eq('MAPPED')].copy()
        elif fault=='marker':frame.attrs={}
        else:frame.loc[0,{'gene_id':'gene_id','symbol':'gene_symbol','status':'mapping_status'}[fault]]='forged'
        frame.to_parquet(path,index=False);value['membership']=record(path)
    write(binding,value)
    with pytest.raises(ValueError):members.validate(record(binding),tmp_path)


@pytest.mark.parametrize('fault',['gmt','gtf','copy','missing_binding'])
def test_cached_proof_rechecks_original_sources_and_location(tmp_path,fault):
    request=inputs(tmp_path/'input');binding=members.run(request,tmp_path/'output');value=read(binding);launch=read(value['preflight']['path'])
    table=Path(value['membership']['path'])
    if fault=='copy':
        copy=tmp_path/'copy.parquet';shutil.copyfile(table,copy);table=copy
    elif fault=='missing_binding':binding.unlink()
    else:
        source=Path(launch['gtf']['path'] if fault=='gtf' else launch['gmt_sources']['HALLMARK']['path'])
        source.write_text(source.read_text()+'#changed\n')
    with pytest.raises((ValueError,FileNotFoundError)):members.discover(table)


def test_consumer_accepts_equivalent_bound_references_and_rejects_other_resources(tmp_path):
    request=inputs(tmp_path/'input');binding=members.run(request,tmp_path/'output');value=read(binding);launch=read(value['preflight']['path'])
    table=value['membership']['path'];definition=tmp_path/'CANDIDATE.json'
    gmt_copy=tmp_path/'copied.gmt';shutil.copyfile(launch['gmt_sources']['HALLMARK']['path'],gmt_copy)
    new_gmt=tmp_path/'GMT.json';write(new_gmt,{'format':'SSGSEA_GMT_INPUT_V1','collections':[{'collection':'HALLMARK','file':record(gmt_copy)}]})
    annotation=tmp_path/'annotation.tsv';shutil.copyfile(launch['annotation']['path'],annotation)
    write(definition,{'annotation':record(annotation),'gmt_manifest':record(new_gmt)})
    assert members.consumer(table,record(definition),tmp_path)['missing_members_dropped'] is False
    gmt_copy.write_text(gmt_copy.read_text().replace('G2','G3'))
    v=read(new_gmt);v['collections'][0]['file']=record(gmt_copy);write(new_gmt,v)
    write(definition,{'annotation':record(annotation),'gmt_manifest':record(new_gmt)})
    with pytest.raises(ValueError,match='GMT'):members.consumer(table,record(definition),tmp_path)


def test_actual_cli_plan_run_and_host_guards(tmp_path,monkeypatch):
    request=inputs(tmp_path/'input');output=tmp_path/'cli'
    cmd=[sys.executable,'-m','scripts.run_native_pathway_membership_workflow','--manifest',str(request),'--output',str(output)]
    plan=subprocess.run([*cmd,'--plan'],cwd=members.ROOT,capture_output=True,text=True,timeout=30)
    assert plan.returncode==0,plan.stderr
    assert json.loads(plan.stdout)['target_host']=='149' and not output.exists()
    result=subprocess.run(cmd,cwd=members.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    members.validate(json.loads(result.stdout)['binding'],tmp_path)
    with pytest.raises(FileExistsError):members.run(request,output)
    monkeypatch.setattr(socket,'gethostname',lambda:'Windows')
    with pytest.raises(RuntimeError,match='149'):members.plan(request,tmp_path/'wrong_host')
