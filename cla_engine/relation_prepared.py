"""Assemble the continuous relation task for the original joint HGT worker."""
from __future__ import annotations
import copy,json,shutil
from pathlib import Path
import numpy as np
from .common import atomic_json,atomic_npz,sha,digest,code_digest,commit_manifest,verify_manifest,read_json
from .model import ARCHITECTURE
from .relation_baseline import load_relation_baseline
from .relation_components import CONTEXTS
from .relation_graph_view import load_relation_graph_view,validate_graph_arrays

FORMAT='CLA_CONTINUOUS_RELATION_PREPARED_V1'


def reindex_target_axis(arrays,old_axis,new_axis):
    if len(old_axis)!=len(set(old_axis)) or len(new_axis)!=len(set(new_axis)) or set(old_axis)!=set(new_axis):
        raise ValueError('RELATION_PREPARED_TARGET_IDENTITY_MISMATCH')
    lookup={t:i for i,t in enumerate(new_axis)}
    remap=np.asarray([lookup[t] for t in old_axis],np.int64)
    result={k:v.copy() for k,v in arrays.items()}
    result['target']=remap[arrays['target']]
    return result


def modality_groups(feature_names,numeric_names):
    context={'cnv':{},'lncrna_cnv':{},'protein':{},'phospho':{}}
    for kind,names in feature_names.items():
        cnv=[i for i,n in enumerate(names) if 'local_cnv_' in n]
        if cnv:context['cnv'][kind]=cnv;context['lncrna_cnv'][kind]=cnv
        if kind.endswith('_protein_assay'):context['protein'][kind]=list(range(len(names)))
        if kind.endswith('_phospho_assay'):context['phospho'][kind]=list(range(len(names)))
    numeric={name:[] for name in ('cnv','lncrna_cnv','protein','phospho','methylation')}
    for i,name in enumerate(numeric_names):
        if 'local_cnv_' in name:numeric['cnv'].append(i);numeric['lncrna_cnv'].append(i)
        for marker,modality in (('/PROTEIN/','protein'),('/PHOSPHO/','phospho'),('/METHYLATION/','methylation')):
            if marker in name:numeric[modality].append(i)
    return {'context_feature_groups':context,'numeric_feature_groups':numeric,
            'node_input_groups':{'cnv':{'gene':[2,3]},'gene_cnv':{'gene':[2,3]},'methylation':{'gene':[4,5]}}}


def _lineage(root):
    return {'format':FORMAT,'code_sha256':code_digest(),
        **{name:sha(root/name) for name in ('SPEC.json','CONFIG.json','SOURCE_BINDINGS.json','TASK_CONTRACT.json','VIEWS.json','FEATURE_NAMES.json')}}


def prepare_relation_capsule(plan,out):
    from .local_execution import require_host_149
    require_host_149()
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('RELATION_PREPARED_OUTPUT_NOT_EMPTY')
    if set(plan['graph_views'])!=set(CONTEXTS):raise ValueError('RELATION_PREPARED_VIEW_GRID')
    baseline_root=Path(plan['baseline']);baseline=load_relation_baseline(baseline_root)
    cancer=baseline['receipt']['cancer'];static=Path(plan['static'])
    graph_axis=read_json(static/'PATHWAY_AXIS.json')['canonical_ids']
    if set(graph_axis)!=set(baseline['axes']['targets']['pathway']):raise ValueError('RELATION_PREPARED_PATHWAY_SCOPE')
    cfg={'seed':baseline['preflight']['config']['seed'],'model':copy.deepcopy(plan['model']),
        'training':copy.deepcopy(plan['training']),'gates':copy.deepcopy(plan['gates']),
        'baseline':baseline['preflight']['config']['baseline'],'relation_objective':'observed_fused_rms_effect',
        'execution_contract':FORMAT,'metrics_contract':'CANCER_EXACT_PATHWAY_MSE_STATE_TARGET_THEN_FAMILY_MSE_V1'}
    if cfg['model']['task_weights']!={'pathway':1.,'state':.5}:raise ValueError('RELATION_PREPARED_TASK_WEIGHTS')
    if cfg['model'].get('backend')!='original_pyg_cc_hhgt':raise ValueError('RELATION_PREPARED_BACKEND')
    if not np.isfinite(cfg['gates']['diagnostic_min_delta_mse']) or cfg['gates']['diagnostic_min_delta_mse']<0:
        raise ValueError('RELATION_PREPARED_DIAGNOSTIC_THRESHOLD')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','plan':plan,'sealed_test_opened':False})
    sources={'baseline':{'path':str(baseline_root),'sha256':sha(baseline_root/'COMPLETE.json')},
        'static':{'path':str(static),'sha256':sha(static/'COMPLETE.json')},'graphs':{},
        'components':baseline['preflight']['components'],'policy':baseline['preflight']['policy']}
    views=[];spec=None
    for context in CONTEXTS:
        component_path=baseline['preflight']['components'][context]['path']
        source=Path(plan['graph_views'][context])
        view=load_relation_graph_view(source,component_path,static)
        ids=copy.deepcopy(view['ids']);original_pathways=ids['pathway'];ids['pathway']=graph_axis
        if spec is None:
            spec={'prepared_format':FORMAT,'architecture':ARCHITECTURE,'execution_contract':FORMAT,
                'cancers':[cancer],'nodes':ids,'relations':view['relations'],
                'pathway_source_ids':original_pathways,'pathway_canonical_ids':graph_axis,
                'states':baseline['axes']['targets']['state'],'state_families':baseline['axes']['state_families'],
                'numeric_features':len(baseline['axes']['feature_names']),
                'feature_dims':{k:len(v) for k,v in view['feature_names'].items()},
                'relation_objective':'observed_fused_rms_effect','metrics_contract':cfg['metrics_contract'],
                'fold':0,'validation_fold':1,'train_folds':[2,3,4],'graph_variant':'G2',
                'sc_dim':1,'singlecell_role':'interpretation_only','genomics_enabled':False,
                'gene_node_omics_enabled':True,'node_input_dim':6,
                'node_input_feature_order':['log1p_degree','constant','cnv_signed_value','cnv_signed_available','methylation_promoter_beta_value','methylation_promoter_beta_available'],
                'conservation_context_indices':list(range(len(baseline['axes']['feature_names']))),
                **modality_groups(view['feature_names'],baseline['axes']['feature_names'])}
            atomic_json(out/'FEATURE_NAMES.json',view['feature_names'])
        elif ids!=spec['nodes'] or view['relations']!=spec['relations'] or {k:len(v) for k,v in view['feature_names'].items()}!=spec['feature_dims']:
            raise ValueError('RELATION_PREPARED_GRAPH_AXIS_DRIFT')
        view_id=cancer+'__'+('validation' if context=='validation' else 'train'+context[-1])
        folder=out/'views'/view_id;folder.mkdir(parents=True)
        shutil.copyfile(source/'graph.npz',folder/'graph.npz')
        for task in ('pathway','state'):
            target_axis=graph_axis if task=='pathway' else spec['states']
            arrays=reindex_target_axis(baseline['arrays'][context][task],baseline['axes']['targets'][task],target_axis)
            atomic_npz(folder/(task+'.npz'),**arrays)
        atomic_json(folder/'PROVENANCE.json',{'cancer':cancer,'context':context,
            'cohort_splits':baseline['components'][context]['splits'],'graph_audit':view['audit']})
        sources['graphs'][context]={'path':str(source),'sha256':sha(source/'COMPLETE.json')}
        views.append({'id':view_id,'split':'validation' if context=='validation' else 'train','cancer_index':0,'context':context})
    shutil.copyfile(baseline['preflight']['policy']['path'],out/'TARGET_POLICY.tsv')
    atomic_json(out/'CONFIG.json',cfg);atomic_json(out/'SPEC.json',spec)
    atomic_json(out/'VIEWS.json',views);atomic_json(out/'SOURCE_BINDINGS.json',sources)
    atomic_json(out/'TASK_CONTRACT.json',{'unit':'cancer x lncRNA x exact pathway',
        'label':'observed fused RMS relationship effect','missing':'unavailable, not negative',
        'task_weights':{'pathway':1.,'state':.5},'display_category_adds_loss':False,
        'outer_test_opened':False,'baseline':'source-only numeric Lasso per cancer and task'})
    atomic_json(out/'LINEAGE.json',_lineage(out))
    commit_manifest(out,{'prepared_format':FORMAT,'profile':'diagnostic','fold':0,'config_sha256':digest(cfg)})
    validate_relation_prepared(out)
    return {'prepared':str(out),'manifest_sha256':sha(out/'MANIFEST.json'),'cancer':cancer}


def validate_relation_prepared(root):
    root=Path(root);manifest=verify_manifest(root)
    spec,cfg,sources,views=(read_json(root/name) for name in ('SPEC.json','CONFIG.json','SOURCE_BINDINGS.json','VIEWS.json'))
    if (spec.get('prepared_format')!=FORMAT or manifest['metadata'].get('prepared_format')!=FORMAT
            or manifest['metadata']['profile']!='diagnostic' or spec['fold']!=0
            or spec['validation_fold']!=1 or spec['train_folds']!=[2,3,4]
            or cfg['model']['task_weights']!={'pathway':1.,'state':.5}
            or spec['relation_objective']!='observed_fused_rms_effect'
            or manifest['metadata']['config_sha256']!=digest(cfg) or read_json(root/'LINEAGE.json')!=_lineage(root)):
        raise ValueError('RELATION_PREPARED_CONTRACT_DRIFT')
    for item in [sources['baseline'],sources['static'],*sources['graphs'].values(),*sources['components'].values()]:
        if sha(Path(item['path'])/'COMPLETE.json')!=item['sha256']:
            raise ValueError('RELATION_PREPARED_COMPONENT_DRIFT')
    if sha(root/'TARGET_POLICY.tsv')!=sources['policy']['sha256']:
        raise ValueError('RELATION_PREPARED_POLICY_DRIFT')
    baseline=load_relation_baseline(sources['baseline']['path'])
    static_ids=read_json(Path(sources['static']['path'])/'NODES.json')
    static_axis=read_json(Path(sources['static']['path'])/'PATHWAY_AXIS.json')['canonical_ids']
    if (spec['nodes']['pathway']!=static_axis or spec['pathway_source_ids']!=static_ids['pathway']
            or spec['pathway_canonical_ids']!=static_axis or spec['architecture']!=ARCHITECTURE
            or cfg['relation_objective']!=spec['relation_objective']
            or spec['numeric_features']!=len(baseline['axes']['feature_names'])
            or spec['conservation_context_indices']!=list(range(spec['numeric_features']))):
        raise ValueError('RELATION_PREPARED_GRAPH_TARGET_CONTRACT')
    if spec['cancers']!=[baseline['receipt']['cancer']] or spec['states']!=baseline['axes']['targets']['state'] or spec['state_families']!=baseline['axes']['state_families']:
        raise ValueError('RELATION_PREPARED_AXIS_DRIFT')
    if len(views)!=4 or {v['context'] for v in views}!=set(CONTEXTS):
        raise ValueError('RELATION_PREPARED_VIEW_GRID')
    names=read_json(root/'FEATURE_NAMES.json')
    if any(spec[k]!=v for k,v in modality_groups(names,baseline['axes']['feature_names']).items()):
        raise ValueError('RELATION_PREPARED_MODALITY_CHANNEL_DRIFT')
    from .readiness import validate_task_arrays
    for v in views:
        context=v['context'];cancer=spec['cancers'][0]
        expected_id=cancer+'__'+('validation' if context=='validation' else 'train'+context[-1])
        if v['id']!=expected_id or v['cancer_index']!=0 or v['split']!=('validation' if context=='validation' else 'train'):
            raise ValueError('RELATION_PREPARED_VIEW_ROUTING')
        folder=root/'views'/v['id'];source=Path(sources['graphs'][context]['path'])
        if sha(folder/'graph.npz')!=sha(source/'graph.npz'):
            raise ValueError('RELATION_PREPARED_GRAPH_DRIFT')
        with np.load(folder/'graph.npz',allow_pickle=False) as f:
            if validate_graph_arrays(spec['nodes'],dict(f),names)!=spec['relations']:
                raise ValueError('RELATION_PREPARED_RELATIONS')
        if read_json(folder/'PROVENANCE.json')['cohort_splits']!=baseline['components'][context]['splits']:
            raise ValueError('RELATION_PREPARED_PATIENT_SPLIT')
        for task in ('pathway','state'):
            targets=spec['nodes']['pathway'] if task=='pathway' else spec['states']
            expected=reindex_target_axis(baseline['arrays'][context][task],baseline['axes']['targets'][task],targets)
            with np.load(folder/(task+'.npz'),allow_pickle=False) as f:arrays=dict(f)
            if set(arrays)!=set(expected) or any(not np.array_equal(arrays[k],value) for k,value in expected.items()):
                raise ValueError('RELATION_PREPARED_MATCHED_BASELINE_DRIFT')
            validate_task_arrays(arrays,v['id'],task,spec['numeric_features'],allow_empty=True,objective='observed_fused_rms_effect')
    return manifest,spec,cfg,read_json(root/'LINEAGE.json')
