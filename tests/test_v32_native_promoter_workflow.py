"""Original strand-aware geometry, stable IDs and actual probe consumers; no fit."""
from pathlib import Path
import gzip
import hashlib
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from cc_hhgt.v32 import native_promoter_workflow as promoter,native_annotation_workflow as reference,native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from test_v32_native_annotation_workflow import inputs as reference_input,gtf_text
from test_v32_native_probe_workflow import REFERENCE

PLUS='ENSG00000900001';MINUS='ENSG00000900002';PAR=PLUS+'_PAR_Y'


def row(chrom,feature,start,end,strand,gene,symbol,transcript=None):
    attrs=f'gene_id "{gene}"; gene_type "lncRNA"; gene_name "{symbol}";'
    if transcript is not None:attrs+=f' transcript_id "{transcript}";'
    return f'{chrom}\tHAVANA\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}'


def text():
    return '\n'.join(gtf_text().splitlines()[:3]+[
        row('chr1','gene',101,10000,'+',PLUS+'.1','A'),
        row('chr1','transcript',101,500,'+',PLUS+'.1','A','ENST00000900001.1'),
        row('chr1','transcript',2001,3000,'+',PLUS+'.1','A','ENST00000900002.1'),
        row('chr1','transcript',2001,3000,'+',PLUS+'.1','A','ENST00000900003.1'),
        row('chr2','gene',8001,10000,'-',MINUS+'.2','B'),
        row('chr2','transcript',9001,10000,'-',MINUS+'.2','B','ENST00000900004.1'),
        row('chrY','gene',1,90,'+',PLUS+'.1_PAR_Y','A'),
        row('chrY','transcript',1,90,'+',PLUS+'.1_PAR_Y','A','ENST00000900001.1_PAR_Y'),
        row('KI270728.1','gene',1,90,'+','ENSG00000900003.1','ALT'),
        row('KI270728.1','transcript',1,90,'+','ENSG00000900003.1','ALT','ENST00000900005.1')])+'\n'


def inputs(root,*,payload=None,compressed=False):
    source=reference_input(root/'raw',text=payload or text(),compressed=compressed)
    binding=reference.run(source,root/'reference');manifest=root/'PROMOTER.json'
    write(manifest,{'format':promoter.INPUT,'reference_annotation':record(binding)});return manifest


@pytest.mark.parametrize('compressed',[False,True])
def test_original_both_strands_clipping_all_transcripts_and_par_identity(tmp_path,compressed):
    manifest=inputs(tmp_path/'input',compressed=compressed);binding=promoter.run(manifest,tmp_path/'output');value=read(binding)
    frame=pd.read_csv(value['promoters']['path'],sep='\t');original=pd.read_csv(value['original_promoters']['path'],sep='\t')
    assert frame.start.tolist()==[0,1000,1000,9899,0] and frame.end.tolist()==[201,2101,2101,11000,101]
    assert frame.gene_id.tolist()==[PLUS,PLUS,PLUS,MINUS,PAR]
    assert frame.versioned_gene_id.tolist()==original.gene_id.tolist()
    pd.testing.assert_frame_equal(frame.iloc[:,:8].drop(columns='gene_id'),original.iloc[:,:8].drop(columns='gene_id'))
    assert value['metadata']['transcript_records']==6 and value['metadata']['excluded_nonreference_transcripts']==1
    assert value['metadata']['n_transcript_promoters']==5 and not value['metadata']['transcript_rows_collapsed']
    assert list(frame.gtf_line)==[5,6,7,9,11]
    assert promoter.discover(value['promoters']['path'])==value


@pytest.mark.parametrize('fault',['coordinate','gene_id','versioned','omit','metadata','original','marker'])
def test_rehashed_outputs_cannot_replace_original_transcript_replay(tmp_path,fault):
    manifest=inputs(tmp_path/'input');binding=promoter.run(manifest,tmp_path/'output');value=read(binding)
    if fault=='metadata':value['metadata']['n_transcript_promoters']=999
    elif fault=='original':
        path=Path(value['original_promoters']['path']);path.write_bytes(promoter.compressed(b'fake\n'));value['original_promoters']=record(path)
    else:
        path=Path(value['promoters']['path']);frame=pd.read_csv(path,sep='\t',keep_default_na=False)
        if fault=='coordinate':frame.loc[0,'start']=1
        elif fault=='gene_id':frame.loc[4,'gene_id']=PLUS
        elif fault=='versioned':frame.loc[0,'versioned_gene_id']=PLUS+'.9'
        elif fault=='omit':frame=frame.iloc[1:]
        else:frame=frame[[c for c in frame if not c.startswith(promoter.MARKER)]]
        path.write_bytes(promoter.compressed(frame.to_csv(sep='\t',index=False).encode()));value['promoters']=record(path)
    write(binding,value)
    with pytest.raises(ValueError):promoter.validate(record(binding),tmp_path)


@pytest.mark.parametrize('fault',['unknown_parent','outside_gene','strand','duplicate','attribute','empty'])
def test_bad_transcript_contract_rejected_before_output(tmp_path,fault):
    payload=text()
    if fault=='unknown_parent':payload=payload.replace('gene_id "'+PLUS+'.1"','gene_id "ENSG99999999999.1"',1)
    elif fault=='outside_gene':payload=payload.replace('transcript\t101\t500','transcript\t100\t500',1)
    elif fault=='strand':payload=payload.replace('transcript\t101\t500\t.\t+','transcript\t101\t500\t.\t-',1)
    elif fault=='duplicate':payload=payload.replace('ENST00000900002.1','ENST00000900001.1')
    elif fault=='attribute':payload=payload.replace('transcript_id "ENST00000900001.1";','transcript_id "ENST00000900001.1"; transcript_id "other";',1)
    else:payload='\n'.join(l for l in payload.splitlines() if '\ttranscript\t' not in l)+'\n'
    manifest=inputs(tmp_path/'input',payload=payload)
    with pytest.raises(ValueError):promoter.run(manifest,tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_actual_cli_source_tamper_and_copied_marker(tmp_path):
    manifest=inputs(tmp_path/'input');command=[sys.executable,'-m','scripts.run_native_promoter_workflow',
        '--manifest',str(manifest),'--output',str(tmp_path/'cli')]
    planned=subprocess.run([*command,'--plan'],cwd=promoter.ROOT,capture_output=True,text=True,timeout=30)
    assert planned.returncode==0,planned.stderr
    assert not (tmp_path/'cli').exists()
    result=subprocess.run(command,cwd=promoter.ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    value=promoter.discover(tmp_path/'cli'/promoter.TABLE)
    copied=tmp_path/'copy.bed.gz';shutil.copyfile(value['promoters']['path'],copied)
    with pytest.raises(ValueError,match='outside'):promoter.discover(copied)
    raw=Path(read(value['preflight']['path'])['gtf']['path']);raw.write_text(raw.read_text()+'# change\n')
    with pytest.raises(ValueError):promoter.discover(value['promoters']['path'])


@pytest.mark.parametrize('mode',['same_content','different_annotation','unscoped'])
def test_ATAC_promoter_reference_identity_is_checked_independently_of_membership(tmp_path,mode):
    from cc_hhgt.v32.atac_cohort import promoter_reference
    manifest=inputs(tmp_path/'input');binding=promoter.run(manifest,tmp_path/'output');table=read(binding)['promoters']['path']
    if mode=='unscoped':
        with pytest.raises(ValueError,match='candidate definition'):promoter_reference(table,None,tmp_path)
        return
    payload=text() if mode=='same_content' else text().replace('gene_name "A"','gene_name "CHANGED"')
    raw=reference_input(tmp_path/'other_source',text=payload);other=reference.run(raw,tmp_path/'other_reference')
    candidate=tmp_path/'CANDIDATE.json';write(candidate,{'annotation':read(other)['annotation']})
    if mode=='different_annotation':
        with pytest.raises(ValueError,match='different gene annotations'):promoter_reference(table,record(candidate),tmp_path)
    else:
        value=promoter_reference(table,record(candidate),tmp_path)
        assert value['candidate_annotation']==read(other)['annotation']


def test_actual_reference_promoter_probe_methylation_chain_preserves_sources(tmp_path):
    from cc_hhgt.v32 import native_probe_workflow as probes,native_methylation_workflow as methylation
    static=pd.read_csv(REFERENCE,sep='\t',usecols=['CpG_chrm','CpG_beg','CpG_end','probeID'])
    local=static.loc[static.CpG_chrm.eq('chr1') & static.CpG_beg.gt(2000)].sort_values('CpG_beg')
    first=local.iloc[0];second=local.loc[local.CpG_beg.gt(first.CpG_end+2000)].iloc[0]
    tss=int(first.CpG_beg)+1
    payload='\n'.join(gtf_text().splitlines()[:3]+[row('chr1','gene',tss,tss+200,'+',PLUS+'.1','A'),
        row('chr1','transcript',tss,tss+200,'+',PLUS+'.1','A','ENST00000900001.1')])+'\n'
    source=reference_input(tmp_path/'raw_reference',text=payload,compressed=True)
    genes=tmp_path/'genes.tsv';pd.DataFrame({'gene_id':['LNC:'+PLUS,'LNC:UNLINKED']}).to_csv(genes,sep='\t',index=False)
    peaks=tmp_path/'peaks.tsv';pd.DataFrame({'seqnames':['chr1'],'start':[int(second.CpG_beg)+1],
        'end':[int(second.CpG_end)],'name':['distal']}).to_csv(peaks,sep='\t',index=False)
    raw=tmp_path/'beta/F1';raw.mkdir(parents=True);beta=raw/'beta.txt';beta.write_text(f'{first.probeID}\t0.4\n{second.probeID}\t0.8\n')
    selected=tmp_path/'selected.tsv';pd.DataFrame([{'file_id':'F1','file_name':'beta.txt','file_size':beta.stat().st_size,
        'md5sum':hashlib.md5(beta.read_bytes()).hexdigest(),'cancer_id':'BRCA','patient_id':'TCGA-AA-0001','sample_id':'S1'}]).to_csv(selected,sep='\t',index=False)
    folds=tmp_path/'folds.tsv';pd.DataFrame({'cancer_id':['BRCA'],'patient_id':['TCGA-AA-0001'],'patient_fold_id':[0]}).to_csv(folds,sep='\t',index=False)
    request=tmp_path/'CHAIN.json';write(request,{'format':chain.INPUT,'chain_id':'promoter-probe','stages':[
        {'id':'reference','operation':'annotation.run','inputs':read(source)},
        {'id':'promoter','operation':'promoter.run','inputs':{'format':promoter.INPUT,'reference_annotation':{'stage_ref':'reference'}}},
        {'id':'probe','operation':'probe.run','inputs':{'format':probes.INPUT,'hm450_manifest':record(REFERENCE),
            'promoters':{'stage_ref':'promoter','pointer':'/promoters'},'required_genes':record(genes),'peak_set':record(peaks),
            'max_distance_bp':int(second.CpG_end-first.CpG_beg)+100}},
        {'id':'methylation','operation':'methylation.run','inputs':{'format':methylation.INPUT,'mode':'raw',
            'selected_files':record(selected),'patient_folds':record(folds),'probe_map':{'stage_ref':'probe','pointer':'/probe_map'},
            'download_root':str(raw.parent),'workers':1,'batch_size':1}}]})
    chain.initialize(request,tmp_path/'chain');chain.advance(tmp_path/'chain');state=read(tmp_path/'chain/CHAIN.json')
    assert state['status']=='DECLARED_STAGES_COMPLETE',state
    probe=read(state['results']['probe']['path']);probe_map=pd.read_parquet(probe['probe_map']['path'])
    assert set(probe_map.region_type)=={'promoter','distal'}
    value=read(state['results']['methylation']['path']);features=pd.read_parquet(value['outputs']['patient_features']['path'])
    observed=features.loc[features.lncrna_id.eq('LNC:'+PLUS)].iloc[0]
    assert observed.promoter_methylation_beta==pytest.approx(.4) and observed.distal_methylation_beta==pytest.approx(.8)
    pins=read(value['preflight']['path'])['source_files'];assert read(source)['gtf']['path'] in {p['path'] for p in pins}
    assert read(probe['preflight']['path'])['promoter_reference']['genome_build']=='GRCh38'
    before=record(tmp_path/'chain/CHAIN.json');chain.advance(tmp_path/'chain');assert record(tmp_path/'chain/CHAIN.json')==before
