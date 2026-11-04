"""Bound, read-only inputs and Windows-to-149 measurement transfer. No source discovery loops."""
from __future__ import annotations
import json,os,zipfile,shutil,importlib.util
from pathlib import Path
from .common import fail,sha,atomic_json,read_json,status,digest
from .tables import table,patient_map,state_wide

LOCAL_ROOT=Path(r'D:\model\CC_HHGT_v3_2_ranked_subtypes_dev')

def export_local(out,root=None):
    root=Path(root) if root else LOCAL_ROOT;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sc=root/'artifacts'/'single_cell_training_inputs_20260826_r1';gen=root/'inputs'/'v32_full_multitask'/'genomic'
    selected=[]
    for source,rel in [(sc/'dataset_manifest_33c.parquet','singlecell/dataset_manifest_33c.parquet'),(gen/'sample_lncrna_mutation_mc3_assay_callable.parquet','genomic/sample_lncrna_mutation_mc3_assay_callable.parquet'),(gen/'sample_gene_mutation_mc3_assay_callable.parquet','genomic/sample_gene_mutation_mc3_assay_callable.parquet')]:
        if source.exists():selected.append((source,rel))
    for source in sorted((sc/'lnc_celltype').glob('*.parquet')):selected.append((source,'singlecell/lnc_celltype/'+source.name))
    for sub in ['activity','association','sc_association','lnc_exact_pathway']:
        for source in sorted((sc/sub).glob('*.parquet')):selected.append((source,'singlecell/'+sub+'/'+source.name))
    for module in ['clinical','drug','genomic','atac','epigenetic','evidence']:
        base=root/'inputs'/'v32_full_multitask'/module
        if base.exists():
            files=sorted(p for p in base.rglob('*') if p.is_file())
            if len(files)>10000:fail('LOCAL_MODULE_INPUT_SCOPE_TOO_LARGE','Bind narrower raw input directories',module=module)
            for source in files:
                name=source.name.lower()
                if any(t in name for t in ['prediction','checkpoint','ranking','best_model','typed_probability']):continue
                if source.suffix.lower() in ['.parquet','.tsv','.csv','.json','.gz','.xlsx','.npy']:
                    selected.append((source,'modules/'+module+'/'+source.relative_to(base).as_posix()))
    # Duplicate primary genomic files are avoided; each archive path is unique.
    selected=list({rel:(path,rel) for path,rel in selected}.values())
    required=['singlecell/dataset_manifest_33c.parquet','genomic/sample_lncrna_mutation_mc3_assay_callable.parquet','genomic/sample_gene_mutation_mc3_assay_callable.parquet']
    missing=[r for r in required if r not in [b for a,b in selected]]
    if not list((sc/'lnc_celltype').glob('*.parquet')):missing.append('singlecell/lnc_celltype/*.parquet')
    if missing:fail('LOCAL_MEASUREMENTS_MISSING','Known local measurement files missing; no prediction files substituted',missing=missing,root=str(root))
    manifest={'format':'CLA_FULL_PYG_R62_LOCAL_MEASUREMENTS_V1','files':{rel:{'sha256':sha(p),'bytes':p.stat().st_size,'original_path':str(p)} for p,rel in selected},'contents':'measured cell-type aggregates and patient mutation events; not model predictions','raw_patient_values_included':True}
    target=out/'UPLOAD_TO_149_LOCAL_INPUTS.zip';tmp=target.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p,rel in selected:z.write(p,rel)
        z.writestr('LOCAL_INPUT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    os.replace(tmp,target);atomic_json(out/'LOCAL_EXPORT_RECEIPT.json',{'path':str(target),'sha256':sha(target),'files':len(selected),'private_measurements':True,'upload_to_chat_required':False})
    return str(target)

def import_local(archive,cfg):
    from .controller import assert_controller
    assert_controller(cfg)
    root=Path(cfg['imports_root']);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('LOCAL_INPUT_MANIFEST.json'));names=set(z.namelist())
        if manifest.get('format') not in ('CLA_FULL_PYG_R62_LOCAL_MEASUREMENTS_V1','CLA_FULL_PYG_R61_LOCAL_MEASUREMENTS_V1','CLA_FULL_PYG_R6_LOCAL_MEASUREMENTS_V1'):fail('OLD_IMPORT_GENERATION','Use a verified R6/R6.1/R6.2 raw-measurement export; never relabel old prepared/checkpoint/results')
        if names!=set(manifest['files'])|{'LOCAL_INPUT_MANIFEST.json'}:fail('LOCAL_ARCHIVE_MISMATCH','Unlisted files in local archive')
        for rel,rec in manifest['files'].items():
            dest=(root/rel).resolve()
            if not dest.is_relative_to(root.resolve()):fail('PATH_TRAVERSAL',rel)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha(dest)==rec['sha256']:continue
                fail('LOCAL_IMPORT_DRIFT','Existing import differs. Use a new versioned imports_root, not overwrite.',path=str(dest))
            tmp=dest.with_suffix('.partial')
            with z.open(rel) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
            if tmp.stat().st_size!=rec['bytes'] or sha(tmp)!=rec['sha256']:tmp.unlink(missing_ok=True);fail('LOCAL_IMPORT_HASH_MISMATCH',rel)
            os.replace(tmp,dest)
    atomic_json(root/'LOCAL_INPUT_MANIFEST.json',manifest)
    commit_import_authority(cfg,manifest,archive_sha256=sha(archive))
    return {'state':'LOCAL_INPUTS_IMPORTED','path':str(root),'files':len(manifest['files'])}

def preflight(cfg,out,*,allow_unbound_optional=False,profile='diagnostic'):
    from .controller import assert_controller
    from .execution_contract import validate_config_contract
    from .cancer_scope import selection,resolve_cancer_scope
    selection(cfg,profile)
    validate_config_contract(cfg);assert_controller(cfg)
    import_authority=verify_import_authority(cfg)
    from .rppa500 import validate_binding
    rppa_input=validate_binding(cfg.get('rppa_input'), sha(Path(cfg['sources']['patient_map']['path'])))
    out=Path(out);out.mkdir(parents=True,exist_ok=True);checks=[];bad=[]
    for role,rec in cfg['sources'].items():
        p=Path(rec['path']);actual=p
        if not p.exists() and role=='state' and rec.get('mirror'):actual=Path(rec['mirror'])
        exists=actual.exists();h=sha(actual) if actual.is_file() else None
        ok=exists and (not rec.get('sha256') or h==rec['sha256'])
        checks.append({'role':role,'path':str(actual),'exists':exists,'sha256':h,'pinned_sha256':rec.get('sha256'),'pass':ok})
        if not ok:bad.append(role)
    manifest_path=Path(cfg['imports_root'])/'LOCAL_INPUT_MANIFEST.json'
    if manifest_path.exists():
        imported=read_json(manifest_path)
        if imported.get('format') not in ('CLA_FULL_PYG_R62_LOCAL_MEASUREMENTS_V1','CLA_FULL_PYG_R61_LOCAL_MEASUREMENTS_V1','CLA_FULL_PYG_R6_LOCAL_MEASUREMENTS_V1'):fail('OLD_IMPORT_GENERATION','Run the R6.2 importer for a verified raw-input archive; do not edit manifest generation')
        for rel,record in imported['files'].items():
            q=(Path(cfg['imports_root'])/rel).resolve()
            if not q.is_relative_to(Path(cfg['imports_root']).resolve()):fail('PATH_TRAVERSAL',rel)
            if not q.exists() or sha(q)!=record['sha256']:bad.append('local_import_changed:'+rel)
    disabled=cfg.get('disabled_modalities',[])
    roles=[('mutation','genomic/sample_lncrna_mutation_mc3_assay_callable.parquet')]
    if cfg.get('genomics',{}).get('enabled'):roles.append(('coding_gene_mutation','genomic/sample_gene_mutation_mc3_assay_callable.parquet'))
    if cfg.get('singlecell_role')!='interpretation_only':roles.append(('singlecell','singlecell/dataset_manifest_33c.parquet'))
    if cfg.get('singlecell_association_role')=='input_context':
        roles.extend([('singlecell_association_manifest','singlecell/dataset_manifest_33c.parquet'),
                      ('singlecell_association_directory','singlecell/association')])
    for role,rel in roles:
        p=Path(cfg['imports_root'])/rel;ok=(p.is_dir() if role.endswith('_directory') else p.is_file()) or role in disabled
        checks.append({'role':role,'path':str(p),'pass':ok,'state':'DISABLED_BY_CONFIG' if role in disabled else ('BOUND' if ok else 'MATERIAL_NOT_BOUND')})
        if not ok and not allow_unbound_optional:bad.append(role)
    dependencies={m:importlib.util.find_spec(m) is not None for m in ['numpy','pandas','scipy','sklearn','torch','pyarrow']}
    if not dependencies['pyarrow'] and any(str(r['path']).endswith('.parquet') for r in cfg['sources'].values()):bad.append('pyarrow')
    report={'state':'INPUT_CHECK_FAIL' if bad else 'INPUT_CHECK_PASS_NOT_TRAINED','config_sha256':digest(cfg),'import_authority':import_authority,'checks':checks,'dependencies':dependencies,'blockers':bad,'declared_target_roles':cfg['targets'],'unbound_independent_modules':{k:v for k,v in cfg['modality_policy'].items() if k in ['atac','methylation','drug','clinical_survival']},'all_multimodal_modules_implemented_and_bound':False}
    atomic_json(out/'INPUT_CHECK.json',report)
    if bad:fail('INPUT_CHECK_FAIL','Required actual paths/hash/readers not ready. No GPU launch.',blockers=bad,report=str(out/'INPUT_CHECK.json'))
    # Do not merely test names: execute real State join before paid work.
    if rppa_input is not None:
        atomic_json(out/'RPPA_INPUT_VALIDATION.json',rppa_input)
    fm=patient_map(cfg['sources']['patient_map']['path'])
    cancer_scope=resolve_cancer_scope(cfg,fm.cancer_id.unique(),profile)
    atomic_json(out/'CANCER_SCOPE.json',cancer_scope)
    fm=fm.loc[fm.cancer_id.isin(cancer_scope['selected_cancers'])].copy()
    measurements,_=state_wide(cfg['sources']['state'],fm,cfg['targets'])
    report['cancer_scope']=cancer_scope
    atomic_json(out/'INPUT_CHECK.json',report)
    coverage=measurements.groupby(['cancer_id','target_id'],observed=True).value.agg(['count','size']).reset_index()
    coverage.to_csv(out/'STATE_SOURCE_COVERAGE.tsv',sep='\t',index=False)
    status(out,'INPUT_CHECK_PASS_NOT_TRAINED',states=len(measurements.target_id.unique()),patient_rows=len(fm))
    return report


def _import_signing_config(cfg):
    # A data import is shared by G0/G1/G2 and five folds. Its authorization is
    # bound to the source controller, not an arbitrary graph-variant config.
    return {k:cfg[k] for k in ('work_root','imports_root','controller_contract','synthetic_fixture') if k in cfg}


def _required_imports(cfg):
    files=[]
    if 'mutation' not in cfg.get('disabled_modalities',[]):
        files.append('genomic/sample_lncrna_mutation_mc3_assay_callable.parquet')
    if cfg.get('genomics',{}).get('enabled'):
        files.append('genomic/sample_gene_mutation_mc3_assay_callable.parquet')
    if cfg.get('singlecell_role')!='interpretation_only':files.append('singlecell/dataset_manifest_33c.parquet')
    if cfg.get('singlecell_association_role')=='input_context':files.append('singlecell/dataset_manifest_33c.parquet')
    return files


def commit_import_authority(cfg, manifest, *, archive_sha256=None, fixture=False):
    from .controller import assert_controller
    from .lineage import sign_file
    assert_controller(cfg)
    if fixture and cfg.get('synthetic_fixture') is not True:fail('SYNTHETIC_AUTHORITY_FORBIDDEN','Fixture receipts cannot authorize real inputs')
    if not fixture and not archive_sha256:fail('IMPORT_ARCHIVE_BINDING_MISSING','Use import-local with the exported measurement archive')
    root=Path(cfg['imports_root']);path=root/'LOCAL_IMPORT_RECEIPT.json'
    body={'format':'CLA_R62_VERIFIED_MEASUREMENT_IMPORT_V1','manifest_sha256':sha(root/'LOCAL_INPUT_MANIFEST.json'),
        'archive_sha256':archive_sha256,'files':manifest['files'],'synthetic_fixture':bool(fixture),
        'source_role':'EXPORTED_RAW_MEASUREMENTS_NOT_PREDICTIONS','generated_by':'cla.py import-local',
        'biological_callability_certified':False,'original_paths_preserved':True}
    if path.exists() and read_json(path)!=body:fail('IMPORT_AUTHORITY_DRIFT','An import root cannot change authority; use a new versioned root')
    atomic_json(path,body);sign_file(path,_import_signing_config(cfg),'MEASUREMENT_IMPORT')
    return body


def verify_import_authority(cfg):
    """Require a controller-verified byte-origin receipt, not mere file presence.

    This proves the imported bytes and original declared paths. Per-locus assay
    semantics still require the genomic normalizer; a signed receipt is no
    substitute for correct biological provenance.
    """
    from .lineage import verify_signed_file
    import re
    root=Path(cfg['imports_root']);mp=root/'LOCAL_INPUT_MANIFEST.json';rp=root/'LOCAL_IMPORT_RECEIPT.json'
    if not mp.is_file() or not rp.is_file():
        fail('IMPORT_PROVENANCE_REQUIRED','Run import-local; manually copied files do not authorize primary training')
    verify_signed_file(rp,_import_signing_config(cfg),'MEASUREMENT_IMPORT')
    m=read_json(mp);r=read_json(rp)
    if r.get('format')!='CLA_R62_VERIFIED_MEASUREMENT_IMPORT_V1' or r.get('manifest_sha256')!=sha(mp) or r.get('files')!=m.get('files'):
        fail('IMPORT_PROVENANCE_MISMATCH','Source manifest and verified import receipt differ')
    if bool(r.get('synthetic_fixture'))!=bool(cfg.get('synthetic_fixture')):
        fail('IMPORT_FIXTURE_ROLE_MISMATCH','Synthetic imports cannot be used in real preparation')
    if not r.get('synthetic_fixture') and not re.fullmatch(r'[a-f0-9]{64}',str(r.get('archive_sha256',''))):
        fail('IMPORT_ARCHIVE_BINDING_MISSING','No verified measurement archive')
    bound={}
    for rel in _required_imports(cfg):
        if cfg.get('synthetic_fixture') and rel not in m['files']:rel=str(Path(rel).with_suffix('.tsv')).replace('\\','/')
        rec=m['files'].get(rel)
        if not isinstance(rec,dict) or not rec.get('original_path') or not re.fullmatch(r'[a-f0-9]{64}',str(rec.get('sha256',''))):
            fail('IMPORT_ROLE_NOT_ATTESTED','Required measured role missing provenance',role=rel)
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()) or p.is_symlink() or not p.is_file():fail('IMPORT_ROLE_PATH_INVALID',rel)
        if p.stat().st_size!=rec.get('bytes') or sha(p)!=rec['sha256']:fail('IMPORT_MEASUREMENT_CHANGED',rel)
        bound['import:'+rel]={'path':str(p),'sha256':rec['sha256']}
    for p in (mp,rp,rp.with_name(rp.name+'.attestation.json')):bound['import_authority:'+p.name]={'path':str(p),'sha256':sha(p)}
    return {'format':'CLA_R62_IMPORT_BINDING_V1','manifest_sha256':sha(mp),'import_receipt_sha256':sha(rp),
        'archive_sha256':r.get('archive_sha256'),'synthetic_fixture':r['synthetic_fixture'],
        'source_role':r['source_role'],'biological_callability_certified':False,'bound_files':bound}
