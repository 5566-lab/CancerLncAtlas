"""Source-patient-fitted raw-omics pathway targets, never global-fitted labels."""
from pathlib import Path
import os,socket,subprocess,shutil
import numpy as np
import pandas as pd
from .common import atomic_json,digest,fail,lock,read_json,sha

FORMAT='CLA_SOURCE_FITTED_FIVE_ASSAY_PATHWAY_FUSION_V2'
ASSAYS=frozenset(['RNA','CNV','METHYLATION','PROTEIN','PHOSPHORYLATION'])
SCOPE='PREPARED_SOURCE_PATIENTS_ONLY'

def validate_fusion_config(cfg):
    c=cfg.get('pathway_fusion',{})
    if not isinstance(c,dict) or type(c.get('enabled',False)) is not bool:fail('FUSION_CONFIG','Explicit boolean enabled required')
    if not c.get('enabled'):return
    if set(c.get('assays',[]))!=ASSAYS:fail('FUSION_ASSAY_SCOPE','RNA, protein, phosphorylation, gene CNV and promoter methylation are all required; no substitution')
    from .state_catalog import ROUTED_CATALOG_ID
    if cfg.get('state_catalog')!=ROUTED_CATALOG_ID:fail('FUSION_STATE_ROUTING','PROGENY and DOROTHEA must be registered as State first')
    if c.get('format')!=FORMAT or c.get('fit_scope')!=SCOPE:fail('FUSION_FIT_SCOPE','Only source-patient fitting is accepted for prepared targets')
    if type(c.get('nf')) is not int or c['nf']<1:fail('FUSION_COMPONENTS','Positive fixed component count required')
    if not c.get('r_libraries') or not c.get('cache_root'):fail('FUSION_RUNTIME','R library and cache paths required')
    complete=c.get('complete',{})
    if not complete.get('path') or len(complete.get('sha256',''))!=64:fail('FUSION_INPUT_BINDING','Verified raw input COMPLETE required')

class FittedPathwayScores:
    def __init__(self,cfg,authority):
        if socket.gethostname().split('.')[0]!='149':fail('FUSION_HOST','Pathway scoring must execute on 149')
        validate_fusion_config(cfg);self.cfg=cfg['pathway_fusion'];self.authority=authority
        p=Path(self.cfg['complete']['path']);self.root=p.parent
        if sha(p)!=self.cfg['complete']['sha256']:fail('FUSION_INPUT_HASH','Raw input manifest changed')
        self.complete=read_json(p)
        if self.complete.get('target_host')!='149' or self.complete.get('status')!='COMPLETE':fail('FUSION_INPUT_CONTRACT','Incomplete or wrong-host raw assays')
        if self.complete.get('format')!='CLA_FIVE_ASSAY_RAW_INPUTS_V1' or set(self.complete.get('assays',[]))!=ASSAYS:fail('FUSION_RAW_ASSAY_DESIGN','The old three-assay artifact cannot be relabelled as five-assay input')
        self.parts={r['cancer_id']:r for r in self.complete['partitions']};self.meta={}
        self.catalog_path=self.root/'CATALOG.json'
        if sha(self.catalog_path)!=self.complete['catalog_sha256']:fail('FUSION_CATALOG_HASH','Gene sets or hierarchy changed')
        self.catalog=read_json(self.catalog_path)
        if self.catalog.get('source_gene_membership',{}).get('sha256')!=cfg['sources']['exact_membership'].get('sha256'):
            fail('FUSION_GENE_SET_LINEAGE','Scoring and graph gene memberships must bind the same reviewed source')
        for parent,children in self.catalog['parent_members'].items():
            if parent not in self.catalog['gene_sets'] or len(children)<3 or len(children)!=len(set(children)) or set(children)-set(self.catalog['gene_sets']):
                fail('FUSION_PARENT_CATALOG','Parent and distinct children must be registered gene sets')
        annotation=self.catalog.get('rppa_feature_mapping',{})
        if not annotation.get('path') or sha(annotation['path'])!=annotation.get('sha256'):fail('FUSION_PROTEIN_ANNOTATION','Verified antibody-to-gene identity required')
        frame=pd.read_csv(annotation['path'],sep='\t',keep_default_na=False)
        frame=frame.loc[frame.mapping_status.eq('ANNOTATED')]
        expected={layer:dict(zip(f.feature_id,f.gene_ids.str.split('|'))) for layer,f in frame.groupby('molecular_layer')}
        if self.catalog.get('assay_feature_genes')!=expected:fail('FUSION_PROTEIN_ANNOTATION','Catalog antibody identities differ from reviewed mapping')
        self.annotation=annotation
        if any(not x.startswith('MSIGDB:') for x in self.catalog['gene_sets']):fail('FUSION_STATE_CONTAMINATION','Only reviewed RNA gene sets enter MOGSA')
        repo=Path(__file__).resolve().parents[1]
        self.core=repo/'scripts/pathway_fusion_core.R';self.runner=repo/'scripts/pathway_fusion_run.R'

    def inputs(self,cancer):
        if cancer in self.meta:return self.meta[cancer]
        if cancer not in self.parts:fail('FUSION_CANCER_MISSING',cancer)
        entry=self.parts[cancer];p=Path(entry['path'])
        if p.resolve()!= (self.root/cancer/'COMPLETE.json').resolve() or sha(p)!=entry['sha256']:fail('FUSION_PARTITION_HASH',cancer)
        m=read_json(p)
        authority=self.authority.loc[self.authority.cancer_id.eq(cancer),['patient_id','patient_fold_id']].drop_duplicates().sort_values('patient_id')
        if m['patient_ids']!=authority.patient_id.tolist() or m['patient_fold_ids']!=authority.patient_fold_id.tolist():fail('FUSION_PATIENT_AUTHORITY',cancer)
        if m['catalog_sha256']!=self.complete['catalog_sha256'] or set(m['blocks'])!=ASSAYS:fail('FUSION_RAW_ASSAYS','Expected all five verified molecular assays')
        for name,b in m['blocks'].items():
            q=p.parent/b['file']
            if not q.resolve().is_relative_to(p.parent.resolve()) or sha(q)!=b['sha256']:fail('FUSION_RAW_BLOCK_HASH',name)
            features=b.get('feature_ids',[])
            if not features or len(features)!=len(set(features)) or b.get('shape')!=[len(features),len(m['patient_ids'])]:fail('FUSION_RAW_BLOCK_AXES',name)
            if b.get('dtype')!='little_endian_float64' or b.get('orientation') not in ('R_column_major_gene_by_patient','R_column_major_feature_by_patient') or q.stat().st_size!=8*len(features)*len(m['patient_ids']):fail('FUSION_RAW_BLOCK_ENCODING',name)
            if name in ('PROTEIN','PHOSPHORYLATION') and set(features)!=set(self.catalog['assay_feature_genes'].get(name,{})):fail('FUSION_RAW_ANTIBODY_AXIS',name)

        self.meta[cancer]=m;return m

    def input_hashes(self):
        result={'pathway_fusion::complete':dict(self.cfg['complete']),
                'pathway_fusion::catalog':{'path':str(self.catalog_path),'sha256':self.complete['catalog_sha256']},
                'pathway_fusion::antibody_annotation':dict(self.annotation)}
        for cancer,m in self.meta.items():
            entry=self.parts[cancer]
            result['pathway_fusion::'+cancer+'::partition']={'path':entry['path'],'sha256':entry['sha256']}
            for name,b in m['blocks'].items():
                result['pathway_fusion::'+cancer+'::'+name]={'path':str(self.root/cancer/b['file']),'sha256':b['sha256']}
        return result

    def view(self,cancer,source_patients,patients,pathways):
        m=self.inputs(cancer)
        if list(patients)!=m['patient_ids'] or not source_patients or len(source_patients)!=len(set(source_patients)) or set(source_patients)-set(patients):fail('FUSION_VIEW_PATIENTS',cancer)
        if set(pathways)-set(self.catalog['gene_sets']):fail('FUSION_TARGET_CATALOG','Unregistered pathway target')
        key={'cancer_id':cancer,'source_patients':list(source_patients),'raw_complete':self.cfg['complete']['sha256'],
             'partition_sha256':self.parts[cancer]['sha256'],'catalog_sha256':self.complete['catalog_sha256'],
             'core_sha256':sha(self.core),'runner_sha256':sha(self.runner),'nf':self.cfg['nf'],'scope':SCOPE}
        out=Path(self.cfg['cache_root'])/cancer/digest(key);out.mkdir(parents=True,exist_ok=True)
        with lock(out):
            seal=out/'COMPLETE.json'
            if not seal.exists():
                if (out/'REQUEST.json').exists():fail('FUSION_INCOMPLETE_CACHE','An interrupted fit must be inspected before resuming',path=str(out))
                # Rscript reads source incrementally. Freeze bytes before launching;
                # a source edit must never alter an already running task.
                core_snapshot=out/'FITTED_CORE.R';runner_snapshot=out/'FITTED_RUNNER.R'
                shutil.copyfile(self.core,core_snapshot);shutil.copyfile(self.runner,runner_snapshot)
                if sha(core_snapshot)!=key['core_sha256'] or sha(runner_snapshot)!=key['runner_sha256']:fail('FUSION_SCRIPT_COPY_DRIFT','Scoring code changed while creating immutable task inputs')
                core_snapshot.chmod(0o444);runner_snapshot.chmod(0o444)
                request={'target_host':'149','scope':SCOPE,'cancer_id':cancer,'input_dir':str(self.root/cancer),'catalog':str(self.catalog_path),
                         'fit_patients':list(source_patients),'output_dir':str(out),'nf':self.cfg['nf'],'r_libraries':self.cfg['r_libraries'],
                         'core_script':str(core_snapshot),'binding':key}
                atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','scope':SCOPE,'source_patients_sha256':digest(source_patients)})
                atomic_json(out/'REQUEST.json',request)
                env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
                with (out/'run.log').open('w') as log:
                    try:r=subprocess.run([self.cfg.get('rscript','Rscript'),str(runner_snapshot),str(out/'REQUEST.json')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
                    except subprocess.TimeoutExpired:fail('FUSION_R_TIMEOUT','Scoring stopped at two-hour limit',log=str(out/'run.log'))
                if r.returncode:fail('FUSION_R_FAILED','Package scoring failed; no fallback labels',log=str(out/'run.log'))
                receipt=read_json(out/'SCORES_RECEIPT.json')
                if receipt.get('status')!='COMPLETE' or receipt.get('scope')!=SCOPE or receipt.get('fit_patients')!=list(source_patients):fail('FUSION_FIT_RECEIPT','Wrong fitted patient scope')
                files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='.lock'}
                atomic_json(seal,{'format':FORMAT,'target_host':'149','binding':key,'files':files})
            sealed=read_json(seal)
            if sealed.get('format')!=FORMAT or sealed.get('binding')!=key:fail('FUSION_CACHE_BINDING','Fit cache belongs to another source cohort')
            for name,h in sealed['files'].items():
                if sha(out/name)!=h:fail('FUSION_CACHE_HASH',name)
            frame=pd.read_csv(out/'FINAL_PATHWAY_SCORES.tsv.gz',sep='\t').set_index('patient_id')
            if frame.index.tolist()!=list(patients) or set(frame.columns)!=set(self.catalog['gene_sets']):fail('FUSION_SCORE_AXIS','Scored axes changed')
            result=frame.loc[patients,pathways].to_numpy(float)
            if np.isinf(result).any():fail('FUSION_SCORE_NONFINITE','Infinite score')
            r=read_json(out/'SCORES_RECEIPT.json')
            report={'format':FORMAT,'target_host':'149','scope':SCOPE,'cancer_id':cancer,'source_patients':list(source_patients),
                    'source_patients_sha256':digest(source_patients),'input_complete_sha256':self.cfg['complete']['sha256'],
                    'cache_manifest_sha256':sha(seal),'score_sha256':sealed['files']['FINAL_PATHWAY_SCORES.tsv.gz'],
                    'core_sha256':key['core_sha256'],'runner_sha256':key['runner_sha256'],'nf':self.cfg['nf'],
                    'mogsa_status':r['mogsa_status'],'mogsa_fit_patients':r.get('mogsa_fit_patients',[]),
                    'mogsa_available_patients':r['mogsa_available_patients'],'parents_supported':r['parents_supported'],
                    'assays':r['assays'],'projection_policy':r.get('projection_policy',{}),
                    'parent_policy':r['parent_policy'],'target_ids_sha256':digest(pathways),
                    'scores_used':'FINAL_PATHWAY_SCORES; RAW_ASSAYS_FITTED_ON_SOURCE_PATIENTS_ONLY',
                    'global_RNA_consensus_used_as_training_label':False}
            return result,report

def verify_fusion_view(folder,cfg,source_patients,pathways):
    if not cfg.get('pathway_fusion',{}).get('enabled'):return
    p=Path(folder)/'PATHWAY_FUSION.json'
    if not p.is_file():fail('FUSION_PREPARED_RECEIPT','Source-fitted scoring receipt required')
    r=read_json(p);c=cfg['pathway_fusion']
    if r.get('format')!=FORMAT or r.get('target_host')!='149' or r.get('scope')!=SCOPE or r.get('source_patients')!=source_patients or r.get('source_patients_sha256')!=digest(source_patients):fail('FUSION_PREPARED_PATIENT_LEAK','Scoring was not fitted on the exact feature-patient cohort')
    if r.get('input_complete_sha256')!=c['complete']['sha256'] or r.get('nf')!=c['nf'] or r.get('target_ids_sha256')!=digest(pathways):fail('FUSION_PREPARED_BINDING','Raw input, components or target scope drifted')
    if set(r.get('assays',[]))!=ASSAYS:fail('FUSION_PREPARED_ASSAY_SCOPE','A prepared view lost or substituted an assay')
    if r.get('mogsa_status')=='OK' and r.get('projection_policy',{}).get('fixed_coverage_percentage_applied') is not False:fail('FUSION_UNREVIEWED_COVERAGE_GATE','Fixed per-assay coverage percentage gates are withdrawn')
    if set(r.get('mogsa_fit_patients',[]))-set(source_patients) or r.get('global_RNA_consensus_used_as_training_label') is not False:fail('FUSION_GLOBAL_FIT_FORBIDDEN','Holdout/global-fitted labels are forbidden')
    if any(p.startswith(('PROGENY:','DOROTHEA:')) for p in pathways):fail('FUSION_STATE_CONTAMINATION','Native State programs remained in Pathway')
