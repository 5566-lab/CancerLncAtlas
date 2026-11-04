"""Run the CPTAC integration stages on server 149, including all native score jobs."""
from __future__ import annotations
import argparse,concurrent.futures,os,socket,subprocess,sys
from pathlib import Path
from .common import read_json,atomic_json,sha,fail
from .cptac_scope import select_cohorts
from .cptac_download import run as download
from .cptac_materialize import materialize
from .cptac_omics_prepare import run as prepare_omics
from .cptac_native_states import run as native_states
from .cptac_bundle import assemble
from .state_protein_map import pinned

def run(spec,output,cancer_type='all',cancer_list=None,workers=3):
    if socket.gethostname().split('.')[0]!='149':fail('CPTAC_EXECUTION_HOST','All CPTAC stages must execute on 149')
    if not 1<=workers<=10:fail('CPTAC_WORKERS','Use 1-10 CPU jobs')
    out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):fail('CPTAC_OUTPUT_EXISTS','Use a new versioned output directory')
    out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir()
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'CPTAC_COMPLETE_PIPELINE','cancer_type':cancer_type,
                                          'cancer_list':cancer_list,'CPU_workers':workers,'paid_compute':False})
    for source in spec['references'].values():pinned(source)
    repo=Path(__file__).resolve().parents[1]
    sources=spec['references'];pin=lambda p:{'path':str(p),'sha256':sha(p)}
    if spec.get('pre_downloaded_complete'):
        dc=pinned(spec['pre_downloaded_complete'])
    else:
        manifest=pinned(spec['download_manifest'])
        download(manifest,out/'raw',cancer_type,cancer_list,min(workers,4))
        dc=out/'raw/DOWNLOAD_COMPLETE.json'
    mc={**{k:sources[k] for k in ['gencode','hgnc','protein_gene','tcga_patient_map']},'download_complete':pin(dc)}
    materialize(mc,out/'aligned',cancer_type,cancer_list)
    ac=out/'aligned/COMPLETE.json';cohorts=read_json(ac)['cohorts']
    oc={'aligned_complete':pin(ac),'pathway_gmt':sources['pathway_gmt'],'methylation_definition':sources['methylation_definition']}
    prepare_omics(oc,out/'omics_inputs')
    native_states({'aligned_complete':pin(ac),'malta':sources['malta']},out/'native_states')
    resource_keys=['pathway_gmt','historical_networks','progeny','dorothea','mcp','senmayo']
    resources={k:str(pinned(sources[k])) for k in resource_keys}
    from .cptac_bundle import md5
    resources['md5']={k:md5(v) for k,v in resources.items()}
    atomic_json(out/'SCORING_RESOURCES.json',resources)
    env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    rscript=spec.get('Rscript','/usr/bin/Rscript');libs=':'.join(spec['R_libraries'])
    # All dependencies must exist before R is started; Broad source must match its pinned commit.
    broad=Path(spec['broad_source']['path'])
    commit=subprocess.check_output(['git','-C',str(broad),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(broad),'status','--porcelain'],text=True).strip()
    if commit!=spec['broad_source']['commit'] or dirty:fail('CPTAC_BROAD_SOURCE','Unexpected or modified Broad source checkout')
    def execute(name,script,args,timeout=7200):
        cmd=[rscript,'--vanilla',str(repo/'scripts'/script),'--r-library',libs]+args
        with (logs/(name+'.log')).open('w') as log:
            p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if p.returncode:fail('CPTAC_SCORING_FAILED',name,exit_code=p.returncode,log=str(logs/(name+'.log')))
        return name
    def rna_job(c):
        return execute('RNA_'+c,'score_cptac_rna.R',['--input-root',str(out/'aligned'),'--output-root',str(out/'rna'),
          '--senmayo-json',resources['senmayo'],'--resources',str(out/'SCORING_RESOURCES.json'),'--cancer-type','list','--cancer-list',c])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:list(ex.map(rna_job,cohorts))
    execute('EXTRA_STATE','score_cptac_extra_states.R',['--input-root',str(out/'aligned'),'--output',str(out/'extra_states')])
    def omics_job(job):
        c,role=job
        folder=out/'omics_inputs'/c
        if role=='protein':
            script='score_cptac_protein.R';inp=folder/'PROTEIN_SYMBOL.tsv.gz';gmt=sources['pathway_gmt'];extra=[]
        else:
            script='score_cptac_ptm.R';inp=folder/'PHOSPHO_SITE_CENTERED.gct';gmt=sources['ptm_gmt'];extra=['--broad-source',str(broad)]
        return execute(role+'_'+c,script,['--input',str(inp),'--output',str(out/'omics_scores'/c/role),'--gmt',str(pinned(gmt))]+extra)
    jobs=[(c,role) for role in ['protein','phosphorylation'] for c in cohorts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:list(ex.map(omics_job,jobs))
    bc={k:sources[k] for k in ['pathway_gmt','ptm_gmt','state_protein_map_complete','tcga_state','methylation_definition']}
    atomic_json(out/'SOURCE_SPEC.json',spec)
    bc['source_spec']=pin(out/'SOURCE_SPEC.json')
    bc.update(aligned_complete=pin(ac),omics_input_complete=pin(out/'omics_inputs/COMPLETE.json'),
       native_state_complete=pin(out/'native_states/COMPLETE.json'),extra_state_complete=pin(out/'extra_states/COMPLETE.json'),
       rna_scores_root=str(out/'rna'),omics_scores_root=str(out/'omics_scores'))
    atomic_json(out/'BUNDLE_CONFIG.json',bc);result=assemble(bc,out/'bundle')
    from .cptac_binding import validate_binding,ROLE
    from .state_catalog import expanded_targets,AUGMENTED_CATALOG_ID
    binding={'format':result['format'],'role':ROLE,'training_pooling':False,'complete':pin(out/'bundle/COMPLETE.json')}
    checked=validate_binding(binding,expanded_targets(AUGMENTED_CATALOG_ID),sources['tcga_state']['sha256'])
    atomic_json(out/'CPTAC_BINDING.json',binding);atomic_json(out/'VALIDATION.json',checked)
    return checked

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-spec',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default='all')
    p.add_argument('--cancer-list','--cancer_list',nargs='+');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();print(run(read_json(a.source_spec),a.output,a.cancer_type,a.cancer_list,a.workers))
if __name__=='__main__':main()
