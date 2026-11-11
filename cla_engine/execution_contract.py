"""One active preparation/profile contract. Retained history is not another runner."""
from __future__ import annotations
import sys
from pathlib import Path
from .common import fail,digest
_CODE=str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:sys.path.insert(0,_CODE)
from cc_hhgt.v32.adjustment_policy import validate_adjustment_contract, effective_policy

ENGINE_ID='CLA_FULL_PYG_R62_SINGLE_ENTRYPOINT_V1'
METRIC_CONTRACT='CANCER_PATHWAY_AP_STATE_TARGET_THEN_FAMILY_AP_AND_PRECISION_V1'


def baseline_contract(cfg):
    from .exact_baseline import ENCODING
    b=cfg.get('baseline',{})
    if b.get('identity_encoding')!=ENCODING or 'hash_dimensions' in b:
        fail('BASELINE_FEATURE_CONTRACT_INVALID','R6.2 requires exact sparse identity terms; no unsigned hash fallback')
    for key in ('batch_rows','max_identity_features'):
        if not isinstance(b.get(key),int) or isinstance(b.get(key),bool) or b[key]<1:
            fail('BASELINE_RESOURCE_CONTRACT_MISSING',key)
    for key in ('max_working_gib','max_vocabulary_gib'):
        if not isinstance(b.get(key),(int,float)) or not 0<float(b[key])<float('inf'):
            fail('BASELINE_RESOURCE_CONTRACT_MISSING',key)
    if not 0<float(b.get('max_available_memory_fraction',0))<=.9:
        fail('BASELINE_RESOURCE_CONTRACT_MISSING','max_available_memory_fraction must be in (0,.9]')
    result={'contract_id':'CLA_R62_MATCHED_BASELINE_V1','identity_encoding':ENCODING,
        'numeric_scaler_fit':'FEATURE_TRAIN_ONLY','identity_fit_scope':'AVAILABLE_TRAIN_ROWS_ONLY',
        'C':b.get('C'),'max_iter':b.get('max_iter'),'collision_free_claimed':True,
        'old_prepare_v32_formal_executable':False,**{k:b[k] for k in ('batch_rows','max_identity_features','max_working_gib','max_vocabulary_gib','max_available_memory_fraction')}}
    if 'identity_mode' in b:
        if b['identity_mode'] not in {'exact','numeric_only'}:
            fail('BASELINE_IDENTITY_MODE_INVALID','Expected exact or numeric_only')
        result['identity_mode']=b['identity_mode']
        if b['identity_mode']=='numeric_only':
            result['identity_fit_scope']='NONE_NUMERIC_ONLY'
    return result


def validate_config_contract(cfg):
    if cfg.get('execution_contract')!=ENGINE_ID:
        fail('EXECUTION_CONTRACT_MISSING','Only cla.py can produce current primary artifacts')
    from .state_catalog import validate_targets
    validate_targets(cfg)
    try:validate_adjustment_contract(cfg.get('adjustment_contract'),[t['id'] for t in cfg['targets']])
    except ValueError as exc:fail('ADJUSTMENT_CONTRACT_INVALID',str(exc))
    if cfg.get('statistics',{}).get('cnv_transform')!='raw_signed':
        fail('PRIMARY_CNV_ESTIMAND_DRIFT','This patch keeps the existing raw-signed A1 primary; ranked A2 is a separate sensitivity')
    if cfg.get('metrics_contract')!=METRIC_CONTRACT:
        fail('METRICS_CONTRACT_MISSING','Explicit selection/evaluation units required')
    from .pathway_fusion import validate_fusion_config
    validate_fusion_config(cfg)
    baseline_contract(cfg)
    return cfg


def verify_capsule_contract(root,cfg,spec):
    from .common import read_json
    root=Path(root);validate_config_contract(cfg)
    if spec.get('state_catalog')!=cfg.get('state_catalog'):
        fail('STATE_CATALOG_GENERATION','Prepared State catalog generation differs')
    if cfg.get('relation_target_policy') is not None:
        from .prepared_target_policy import validate_prepared_policy
        state_targets=validate_prepared_policy(root,cfg,spec)['state_targets']
    else:
        state_targets=[t for t in cfg['targets'] if t['role']=='primary']
    if spec.get('state_families')!={t['id']:t['family'] for t in state_targets}:
        fail('STATE_FAMILY_DRIFT','Prepared State families differ from target definitions')
    if cfg['sources']['state'].get('format')=='CLA_PATIENT_STATE_V1':
        from .common import sha
        materialization=root/'STATE_MATERIALIZATION.json'
        if not materialization.is_file() or sha(materialization)!=cfg['sources']['state']['materialization_manifest_sha256']:
            fail('STATE_MATERIALIZATION_HASH','Prepared source receipt differs from verified patient matrix')
        if read_json(materialization).get('target_catalog_sha256')!=digest(cfg['targets']):
            fail('STATE_MATERIALIZATION_CATALOG','Prepared source receipt has different targets')
    if read_json(root/'ADJUSTMENT_CONTRACT.json')!=cfg['adjustment_contract']:
        fail('ADJUSTMENT_CONTRACT_DRIFT','Capsule covariate policy differs from config')
    if read_json(root/'BASELINE_CONTRACT.json')!=baseline_contract(cfg):
        fail('BASELINE_FEATURE_CONTRACT_DRIFT','Capsule baseline differs from config')
    if spec.get('execution_contract')!=ENGINE_ID:
        fail('LEGACY_PRIMARY_INPUT_REJECTED','Historical prepared folds cannot enter this executor')
    audit=read_json(root/'COVARIATE_AUDIT.json')
    targets={'__pathway__'}|set(spec['states']);views=read_json(root/'VIEWS.json')
    expected={(v['id'],t) for v in views for t in targets}
    actual=set()
    for row in audit:
        if row.get('role')=='graph':continue
        key=(row.get('view'),row.get('target_id'))
        if key in actual or key not in expected:fail('ADJUSTMENT_AUDIT_GRID_INVALID',str(key))
        actual.add(key)
        try:_,policy=effective_policy(key[1],contract=cfg['adjustment_contract'],purity_families=row.get('purity_source_families',[]))
        except ValueError as exc:fail('ADJUSTMENT_AUDIT_SOURCE_INVALID',str(exc))
        if row.get('effective_policy')!=policy:fail('ADJUSTMENT_EFFECTIVE_POLICY_DRIFT',str(key))
        # CovariateEncoder has one explicit estimability fallback: when purity
        # is required by the target policy but its train-cohort coverage is
        # below the frozen minimum, it is omitted without imputation and the
        # audit records OPTIONAL_SOURCE_UNAVAILABLE_EXPLICIT_NO_IMPUTATION.
        # Accept only that exact, data-audited reduction.  All other role
        # changes remain fatal.
        audited_required=list(policy['required'])
        purity_unavailable=(
            'purity' in audited_required
            and row.get('dropped',{}).get('purity') in {
                'OPTIONAL_SOURCE_UNAVAILABLE_EXPLICIT_NO_IMPUTATION',
                # The generic per-column pass overwrites the explicit marker
                # when fewer than minimum_observed_count purity values exist.
                'INSUFFICIENT_TRAIN_MEASUREMENTS',
            }
            and isinstance(row.get('fit_counts',{}).get('purity'),dict)
            and row['fit_counts']['purity'].get('observed_fraction',1.0)
                < policy['minimum_observed_fraction']
        )
        if purity_unavailable:
            audited_required=[column for column in audited_required if column!='purity']
        if row.get('required')!=audited_required or row.get('requested')!=policy['requested'] or row.get('excluded')!=policy['excluded']:
            fail('ADJUSTMENT_AUDIT_ROLE_DRIFT',str(key))
        if row.get('admission_passed') is not True or set(row.get('cohorts',{}))!={'source','label'}:
            fail('ADJUSTMENT_ADMISSION_MISSING',str(key))
        for column in audited_required:
            entry=row.get('fit_counts',{}).get(column,{})
            if entry.get('observed',0)<policy['minimum_observed_count'] or entry.get('observed_fraction',0)<policy['minimum_observed_fraction']:
                fail('REQUIRED_COVARIATES_UNAVAILABLE','Audit lacks actual required measurements',view=key,column=column)
            if column not in row.get('encoded',[]) and column not in row.get('redundant',{}):
                fail('REQUIRED_COVARIATES_NOT_ACCOUNTED','Neither encoded nor observed redundant',view=key,column=column)
        for cohort,record in row['cohorts'].items():
            if record.get('valid',0)<cfg['minimum_pairs'] or record.get('valid_fraction',0)<policy['minimum_joint_valid_fraction']:
                fail('COVARIATE_COHORT_NOT_ADMISSIBLE','Stored admission is inconsistent',view=key,cohort=cohort)
    if actual!=expected:fail('ADJUSTMENT_AUDIT_GRID_INCOMPLETE','Missing target/view adjustment receipts')


def verify_graph_view_audit(root,spec,view,cfg):
    """Graph mask/support audit is part of the worker's input contract."""
    import numpy as np
    from .common import read_json
    folder=Path(root)/'views'/view['id'];a=read_json(folder/'COEXPRESSION_AUDIT.json');pr=read_json(folder/'PROVENANCE.json')
    if a.get('contract')!='PAIRWISE_COMMON_MASK_COEXPRESSION_R62_V1' or a.get('source_patient_hash')!=pr.get('feature_patient_hash'):
        fail('COEXPRESSION_AUDIT_DRIFT','Wrong coexpression algorithm/source cohort',view=view['id'])
    if a.get('minimum_pairs')!=cfg['minimum_pairs'] or a.get('topk_per_lncrna')!=cfg['graph']['coexpression_topk'] or a.get('estimable_pairs',0)<1:
        fail('COEXPRESSION_AUDIT_DRIFT','Coexpression admission differs from frozen settings',view=view['id'])
    with np.load(folder/'graph.npz',allow_pickle=False) as graph, np.load(folder/'coexpression_support.npz',allow_pickle=False) as support:
        for sign in ('positive','negative'):
            key='lncRNA|coexpression_'+sign+'|gene';edge=graph['edge::'+key];weight=graph['weight::'+key]
            n=support[sign+'_n'];rank=support[sign+'_rank']
            if edge.ndim!=2 or edge.shape[0]!=2 or len(weight)!=edge.shape[1] or n.shape!=weight.shape or rank.shape!=weight.shape:
                fail('COEXPRESSION_SUPPORT_SHAPE','Pairwise support must align with the actual dynamic edges')
            if edge.shape[1]!=a['selected_edges'][sign] or np.any(n<cfg['minimum_pairs']) or np.any(n>a['source_patients']) or np.any(n-rank-1<=0):
                fail('COEXPRESSION_SUPPORT_INVALID','Invalid common-mask N or rank',view=view['id'])
            if not np.isfinite(weight).all() or np.any(weight<.15-1e-7) or np.any(weight>1+1e-7):
                fail('COEXPRESSION_SUPPORT_INVALID','Invalid signed-correlation magnitude')
    if cfg.get('genomics',{}).get('enabled'):
        g=read_json(folder/'GENOMIC_CONTEXT.json');b=g.get('binding_context',{})
        context_variant=cfg['genomics'].get('context_graph_variant',spec['graph_variant'])
        if context_variant not in ('G0','G1','G2'):
            fail('GENOMIC_CONTEXT_VARIANT_INVALID','Numeric genomic background must name G0/G1/G2')
        # In the registered topology-only matrix every arm explicitly uses the
        # G2 numeric genomic background. G0 still has no reported-binding graph
        # edge; retaining the same partner numeric values prevents a hidden
        # feature ablation from being confounded with the topology ablation.
        if spec['graph_variant']=='G0' and context_variant=='G0':
            if g.get('partner_context_uses_binding') is not False:fail('G0_BINDING_FEATURE_LEAK','G0 must not receive binding-derived partners')
        elif b.get('normalizer')!='normalize_binding' or b.get('cancer_id')!=pr.get('cancer'):
            fail('GENOMIC_BINDING_SCOPE_DRIFT','Partner context must share the canonical binding normalizer')
        if context_variant!=spec['graph_variant'] and context_variant!='G2':
            fail('GENOMIC_CONTEXT_VARIANT_DRIFT','Only the predeclared fixed G2 numeric background may differ from graph topology')
    from .node_omics import verify_view_node_omics
    verify_view_node_omics(root,spec,view,cfg)
