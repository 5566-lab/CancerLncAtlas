"""Bring actual native SC source reconstruction into the model-use inventory.

The supplied geometry remains an explicit gap, and the recorded LASSO evidence is
rebuilt into the model-facing nodes, pair features and sparse edges here. The
recorded coefficients are still never re-solved against the L1 objective. This
module does not fit, construct neural models, replay predictions or attest
independence.
"""
from pathlib import Path
import time

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from . import native_single_cell_mapping_workflow as mapping
from . import native_single_cell_static_workflow as static
from . import native_single_cell_graph_workflow as graph
from . import native_single_cell_donors_workflow as donors
from . import single_cell_nested_numeric_replay as nested
from . import single_cell_outer_numeric_replay as outer

FORMAT = 'NATIVE_SC_NESTED_FROZEN_STATISTICS_REPLAY_V1'
ASSEMBLY = 'NATIVE_SC_EPISODE_MANIFEST_RECONSTRUCTION_V1'
CODE = tuple(dict.fromkeys((*mapping.CODE, *static.CODE, *graph.CODE, *donors.CODE, *nested.CODE,
    'cc_hhgt/v32/native_single_cell_model_sources.py', 'cc_hhgt/v32/native_single_cell_episode_sources.py',
    'cc_hhgt/v32/single_cell_hhgt_episodes.py', 'cc_hhgt/v32/single_cell_batched_lasso.py',
    'cc_hhgt/v32/single_cell_lasso_replay.py')))


def assembler_settings(preflight):
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
    from .single_cell_batched_lasso import BatchedLassoPolicy
    argv = ['--run-id', preflight['run_id'], '--finalize', '--deadline', str(time.time()+3600),
            '--native-graph-binding', preflight['bindings']['native_graph_binding']['path']]
    for key in ('source_result_root', 'graph_asset_root', 'nested_asset_root', 'output_root'):
        argv += ['--'+key.replace('_','-'), preflight[key]]
    settings = native.settings_from_args(native.parser().parse_args(argv))
    policy = dict(preflight['lasso']['policy']); policy['alphas'] = tuple(policy['alphas'])
    settings.lasso_policy = BatchedLassoPolicy(**policy).validated()
    settings.lasso_stability_threshold = preflight['lasso_stability_threshold']
    return settings


def inspect(inventory, episode_path, graph_path, model, role):
    """Return only identities explicitly handled by native reconstruction.

    Legacy manifests keep the existing generic inspection. A native marker
    without its complete source binding is an error, never a legacy fallback.
    """
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as assembler
    from .current_auxiliary_model_use import source_metadata
    from .preparation_sources import metadata_tokens
    require_host(); episode_path=Path(episode_path).resolve(); graph_path=Path(graph_path).resolve()
    manifest=read(episode_path); graph_manifest=read(graph_path)
    launch=manifest.get('launch_preflight')
    preflight_path=verified_path(launch,episode_path.parent) if launch else None
    preflight=read(preflight_path) if preflight_path else {}
    graph_native='native_graph_preparation' in graph_manifest
    if preflight.get('native_episode_sources') is not True:
        if graph_native or 'native_graph_binding' in preflight.get('bindings',{}):
            raise ValueError('Native SC model sources lack their episode preflight marker')
        return set()
    if (not graph_native or preflight_path!=episode_path.parent/'LAUNCH_PREFLIGHT.json'
            or Path(preflight['output_root']).resolve()!=episode_path.parent
            or Path(preflight['graph_asset_root']).resolve()!=graph_path.parent
            or verified_path(manifest['graph_asset_manifest'],episode_path.parent)!=graph_path):
        raise ValueError('Native SC model source roots differ from consumed manifests')
    settings=assembler_settings(preflight)
    expected=assembler.finalize(settings,read_only=True)
    if {k:v for k,v in manifest.items() if k!='finished_unix'}!={k:v for k,v in expected.items() if k!='finished_unix'}:
        raise ValueError('Native SC consumed episode manifest differs from reconstruction')
    bindings=preflight['bindings']
    graph_binding_path,graph_binding=graph.validate(bindings['native_graph_binding'],preflight_path.parent)
    graph_value=read(graph_binding['preflight']['path'])
    donor_path=verified_path(bindings['native_donor_preparation'],preflight_path.parent)
    donor_binding=read(donor_path); donor_value=read(donor_binding['preflight']['path'])
    static_path,static_binding=static.validate(graph_value['static_workflow'],graph_binding_path.parent)
    static_value=read(static_binding['preflight']['path'])
    mapping_path,mapping_binding=mapping.validate(graph_value['mapping'],graph_binding_path.parent)
    mapping_value=read(mapping_binding['preflight']['path'])
    _,bundle,_=mapping.upstream(mapping_value)
    mapped_path=Path(mapping_binding['mapped_success']['path']); mapped=read(mapped_path)
    handled=set()

    def bind(pin,label):
        path=inventory.bind(pin,episode_path.parent,model,role+'.native.'+label)
        return path

    def derived(pins,label):
        """Only verified leaf arrays/tables bypass generic unknown transforms."""
        leaves=0
        for number,pin in enumerate(pins):
            path=bind(pin,label+'.'+str(number))
            if path.suffix in {'.npz','.npy','.parquet','.tsv','.csv'}:
                handled.add((str(path),pin['sha256'])); leaves+=1
        return leaves

    def replay(source,label,format,folds,count):
        inventory.transform_replays.append({'model_id':model,'dependency_context':inventory.context_id,
            'patient_fold_id':-1,'role':role+'.native.'+label,'format':format,
            'source_release_sha256':record(source)['sha256'],'normalization_sha256':'','activity_sha256':'',
            'folds_recomputed':folds,'outputs_compared':count,'model_training_started':False})

    for label,path in (('episode_manifest',episode_path),('episode_preflight',preflight_path),
                       ('graph_binding',graph_binding_path),('donor_binding',donor_path),
                       ('static_binding',static_path),('mapping_binding',mapping_path)):
        bind(record(path),label)
    for index,name in enumerate(CODE):
        bind(record(Path(__file__).resolve().parents[2]/name),'implementation.'+str(index))
    for label,value in (('mapping',mapping_value),('static',static_value),('graph',graph_value),('donors',donor_value)):
        for number,pin in enumerate([value['input'],*value['implementations']]):bind(pin,label+'.control.'+str(number))
    count=derived([record(mapped_path),*mapped['files'].values()],'mapping')
    replay(mapping_path,'mapping',mapping.FORMAT,0,count)
    _,pins=static.inventories(static_value,static_path.parent)
    replay(static_path,'static',static.FORMAT,0,derived(pins,'static'))
    replay(graph_binding_path,'fold_graph',graph.FORMAT,len(graph_value['folds']),
           derived(graph.inventory(graph_value,graph_path.parent),'fold_graph'))
    replay(Path(graph_value['metacell_success']['path']),'outer_statistics',outer.FORMAT,len(graph_value['folds']),
           derived(graph_value['source_files'],'outer_statistics'))
    derived(donor_binding['files'].values(),'canonical_donors')
    replay(Path(bindings['nested_preflight']['path']),'nested_statistics',FORMAT,len(graph_value['folds']),
           derived(bindings['nested_sources'],'nested_statistics'))
    replay(episode_path,'episode_manifest',ASSEMBLY,len(manifest['folds']),manifest['episode_count'])

    from .single_cell_geometry_replay import FORMAT as GEOMETRY_FORMAT
    # The original numeric consumers above already verified these parameters,
    # coordinates, radius and assignments. PCA fit provenance remains a gap.
    geometry_pins = {p['path']: p for p in [*graph_value['source_files'], *bindings['nested_sources']]
                     if Path(p['path']).name == 'GEOMETRY_AUDIT.json'}
    for index, pin in enumerate(geometry_pins.values()):
        path = Path(pin['path'])
        if read(path).get('local_space_format') == GEOMETRY_FORMAT:
            replay(path, 'geometry.'+str(index), GEOMETRY_FORMAT, 1, 3)

    from . import single_cell_lasso_replay as lasso_replay
    # Rebuild the model-facing LASSO inputs from the evidence recorded inside each
    # episode. The recorded coefficients themselves are not re-solved, so their
    # fit provenance and L1 objective consistency stay declared gaps.
    lasso_replay_result = lasso_replay.verify_episodes(episode_path, settings, manifest)
    if lasso_replay_result['episode_count'] != manifest['episode_count']:
        raise ValueError('SC-HHGT LASSO replay did not cover every recorded episode')
    replay(episode_path, 'lasso_evidence', lasso_replay.FORMAT, len(manifest['folds']),
           int(lasso_replay_result['episode_count']))

    # Raw sources are bound and described separately from derived outputs.
    # Their mapping is replayed, while acquisition/QC/reference claims remain
    # unresolved. Patient identities remain inside PatientUse's private map.
    tokens=metadata_tokens({'dataset_id':bundle['dataset_id']})
    raw_paths={Path(p).resolve() for p in bundle['input_paths'].values()}
    from .native_single_cell_annotation_workflow import inspect as inspect_annotation
    annotation_path=Path(bundle['input_paths']['annotation']).resolve()
    annotation_verified=inspect_annotation(inventory,annotation_path,model,role+'.native.gene_annotation')
    reference_paths={Path(pin['path']).resolve() for pin in bundle.get('gene_annotation_sources',[])}
    for index,pin in enumerate(mapping_value['source_files']):
        path=bind(pin,'raw_source.'+str(index))
        if annotation_verified and path in reference_paths:
            handled.add((str(path),pin['sha256']))
            continue
        if path.suffix in {'.parquet','.tsv','.csv'}:
            source_metadata(inventory,path,model,role+'.native.raw_source.'+str(index),upstream_tokens=tokens)
        if path in raw_paths and path.suffix!='.json':handled.add((str(path),pin['sha256']))
    for label,pin in (('pathway_family',static_value['pathway_family']),('external_evidence',static_value['external_evidence']),
                      ('bulk_candidates',donor_value['candidates']),('donor_crosswalk',donor_value['donor_crosswalk'])):
        path=bind(pin,label)
        source_metadata(inventory,path,model,role+'.native.'+label)
        handled.add((str(path),pin['sha256']))
    for reason in ('SC_RAW_ACQUISITION_QC_AND_REFERENCE_ANCESTRY_UNVERIFIED',
                   'SC_FITTED_GEOMETRY_NOT_INDEPENDENTLY_VERIFIED',
                   'SC_FITTED_LASSO_COEFFICIENTS_NOT_RECONSTRUCTED',
                   'SC_LASSO_COEFFICIENT_OBJECTIVE_NOT_REFIT',
                   'SC_BULK_CANDIDATE_AND_EXTERNAL_REFERENCE_ANCESTRY_UNVERIFIED'):
        inventory.gap(model,role+'.native',reason,digest=record(preflight_path)['sha256'])
    return handled
