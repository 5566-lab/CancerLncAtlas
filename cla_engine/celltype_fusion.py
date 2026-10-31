"""Cell-type exact-pathway secondary fusion. Original primary remains immutable.

Reuse R6.2 nonnegative, availability-masked residual fitting. Each cell context
is an expert column, never a duplicate bulk training label. Development fitting
reads frozen primary VALIDATION only, not the sealed patient outer test.
"""
from dataclasses import asdict
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd

from .common import atomic_json, commit_manifest, digest, fail, read_json, sha, status, verify_manifest, code_digest
from cc_hhgt.v32.multimodal_fusion import (
    ANALYSIS_VERSION, ExpertSpec, ResidualFusionConfig, pair_blocked_fold,
    train_pair_blocked_crossfit_fusion, apply_residual_fusion_model,
)

KEYS=['cancer_id','lncrna_id','pathway_id']
CONTEXT_KEYS=KEYS+['cell_type']
PROB='single_cell_replication_probability'
AVAILABLE='single_cell_available'
REASON='single_cell_unavailable_reason'


def _keys(frame, columns, role):
    if set(columns)-set(frame): fail('SC_FUSION_SCHEMA',role,missing=sorted(set(columns)-set(frame)))
    result=frame.copy()
    for name in columns:
        if result[name].isna().any(): fail('SC_FUSION_KEY_NULL',name)
        result[name]=result[name].astype(str).str.strip()
        if result[name].eq('').any(): fail('SC_FUSION_KEY_EMPTY',name)
    return result


def celltype_table(candidates, predictions):
    """Aggregate studies only WITHIN the exact same cell_type (state composite retained).

    Sparse long table: absent candidate/context pairs are explicitly masked at
    the wide training join. Null placeholder 'UNAVAILABLE' is not a cell type.
    """
    primary=_keys(candidates,KEYS,'candidates')[KEYS]
    if primary.duplicated(KEYS).any(): fail('SC_FUSION_DUPLICATE_CANDIDATE','Duplicate exact primary keys')
    raw=_keys(predictions,CONTEXT_KEYS+['dataset_id'],'SC typed predictions')
    required={PROB,AVAILABLE,REASON,'analysis_version'}
    if required-set(raw): fail('SC_FUSION_SCHEMA','Require trained typed expert predictions',missing=sorted(required-set(raw)))
    if not raw.analysis_version.eq(ANALYSIS_VERSION).all(): fail('SC_FUSION_GENERATION','Wrong expert generation')
    # Never treat string "false" as True via astype(bool).
    if raw[AVAILABLE].isna().any() or not raw[AVAILABLE].map(lambda v:isinstance(v,(bool,np.bool_))).all():
        fail('SC_FUSION_AVAILABILITY','Availability must be boolean, not a string or number')
    probability=pd.to_numeric(raw[PROB],errors='coerce'); available=raw[AVAILABLE]
    if (not np.isfinite(probability[available]).all() or not probability[available].between(0,1).all()
        or raw.loc[~available,PROB].notna().any() or raw.loc[~available,REASON].isna().any()):
        fail('SC_FUSION_PROBABILITY','Require valid probabilities or null with reason; rho is not a probability')
    raw[PROB]=probability
    membership=raw.merge(primary,on=KEYS,how='left',indicator=True,validate='many_to_one')
    if membership._merge.ne('both').any(): fail('SC_FUSION_UNIVERSE','SC prediction escapes exact candidate universe')
    placeholder=raw.cell_type.eq('UNAVAILABLE')
    if raw.loc[placeholder,AVAILABLE].any(): fail('SC_FUSION_CONTEXT','An unavailable context cannot carry predictions')
    raw=raw.loc[~placeholder].copy()
    if raw.duplicated(CONTEXT_KEYS+['dataset_id']).any():
        fail('SC_FUSION_DUPLICATE_SOURCE','Expected one OOF aggregate per dataset/cell-context/exact pair')
    # cell_type from the native head may be a cell_type::cell_state composite.
    # Preserve it verbatim; never collapse distinct cell states to cell_type_major.
    result=raw.groupby(CONTEXT_KEYS,observed=True,as_index=False).agg(
        single_cell_replication_probability=(PROB,'mean'),
        single_cell_available=(AVAILABLE,'max'),
        contributing_dataset_count=(PROB,'count'),
        dataset_ids=('dataset_id',lambda values:'|'.join(sorted(set(values)))),
        single_cell_unavailable_reason=(REASON,lambda values:'|'.join(sorted(set(values.dropna().astype(str)))))
    )
    result.loc[result[AVAILABLE],REASON]=None
    result['analysis_version']=ANALYSIS_VERSION
    result['aggregation']='MEAN_WITHIN_SAME_CELL_CONTEXT_ACROSS_DATASETS_ONLY'
    result['changes_primary_ranking']=False
    return result.sort_values(CONTEXT_KEYS,kind='stable').reset_index(drop=True)


def expert_matrix(primary, long_table, cell_types):
    """One candidate = one fusion training row; no cell-type label replication."""
    primary=_keys(primary,KEYS,'primary')
    if primary.duplicated(KEYS).any(): fail('SC_FUSION_DUPLICATE_CANDIDATE','Primary duplicate')
    if not cell_types or len(set(cell_types))!=len(cell_types): fail('SC_FUSION_CONTEXT','Declare unique cell contexts')
    long_table=_keys(long_table,CONTEXT_KEYS,'cell-type fusion table')
    if long_table.duplicated(CONTEXT_KEYS).any(): fail('SC_FUSION_DUPLICATE_CONTEXT','Duplicate cell-type exact rows')
    if (not long_table[AVAILABLE].map(lambda v:isinstance(v,(bool,np.bool_))).all()
        or not np.isfinite(long_table.loc[long_table[AVAILABLE],PROB]).all()
        or not long_table.loc[long_table[AVAILABLE],PROB].between(0,1).all()):
        fail('SC_FUSION_PROBABILITY','Invalid typed expert values')
    if set(long_table.cell_type)-set(cell_types): fail('SC_FUSION_CONTEXT','Undeclared cell context would be dropped')
    if long_table.merge(primary[KEYS],on=KEYS,how='left',indicator=True)._merge.ne('both').any():
        fail('SC_FUSION_UNIVERSE','No pathway-family broadcast or extra exact pairs')
    specs=[]; catalog=[]; frame=primary.copy()
    for cell in cell_types:
        expert='sc_'+digest(cell)[:20]
        spec=ExpertSpec(expert,PROB,AVAILABLE,('discovery',),'donor_or_dataset',source_role='current_v32_oof_prediction')
        spec.validate();specs.append(spec);catalog.append({'cell_type':cell,'expert_id':expert})
        part=long_table.loc[long_table.cell_type.eq(cell),KEYS+[PROB,AVAILABLE]].rename(
            columns={PROB:expert+'_probability',AVAILABLE:expert+'_available'})
        frame=frame.merge(part,on=KEYS,how='left',validate='one_to_one',sort=False)
        frame[expert+'_available']=frame[expert+'_available'].fillna(False).astype(bool)
        if frame.loc[~frame[expert+'_available'],expert+'_probability'].notna().any():
            fail('SC_FUSION_PROBABILITY','Unavailable expert cannot have a numeric score')
    return frame,specs,catalog


def _host():
    if socket.gethostname()!='149': fail('WRONG_HOST','Single-cell preparation/fusion executes on 149 only')


def contribution_table(prediction, frame, catalog, role):
    """Per-context counterfactual score plus the jointly fused secondary score.

    Weights are jointly learned; celltype_fused_probability retains only that
    context's correction. It is not a separately fitted cell-type model.
    """
    rows=[]
    for item in catalog:
        expert=item['expert_id']
        columns=KEYS+['primary_probability','discovery_adjusted_probability',
                      expert+'_available',expert+'_fusion_weight',expert+'_logit_contribution']
        part=prediction[columns].merge(frame[KEYS+[expert+'_probability']],on=KEYS,how='left',validate='one_to_one',sort=False)
        part=part.rename(columns={expert+'_available':'available',expert+'_fusion_weight':'fusion_weight',
                                  expert+'_logit_contribution':'logit_contribution',expert+'_probability':'sc_probability'})
        part['cell_type']=item['cell_type']
        baseline=part.primary_probability.to_numpy(float)
        clipped=np.clip(baseline,1e-6,1-1e-6)
        delta=part.logit_contribution.fillna(0).to_numpy(float)
        isolated=1/(1+np.exp(-np.clip(np.log(clipped)-np.log1p(-clipped)+delta,-700,700)))
        isolated[delta==0]=baseline[delta==0]
        part['celltype_fused_probability']=isolated
        part['unavailable_reason']=np.where(part.available,None,'NO_USABLE_CELLTYPE_EXACT_MEASUREMENT')
        part['prediction_role']=role;part['primary_ranking_unchanged']=True
        rows.append(part)
    return pd.concat(rows,ignore_index=True)


def _new_output(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()): fail('OUTPUT_EXISTS','Never overwrite a fusion artifact')
    out.mkdir(parents=True,exist_ok=True)
    return out


def _bound(record):
    path=Path(record['path'])
    if not path.is_file() or path.is_symlink() or sha(path)!=record['sha256']:
        fail('SC_FUSION_SOURCE_DRIFT','Missing or changed bound input',path=str(path))
    return path


def export_table(binding_path,out):
    """Bind fresh native SC OOF outputs. Never reinterpret association rho as an expert."""
    _host(); binding=read_json(binding_path)
    lineage_path=_bound(binding['singlecell_lineage']); lineage=read_json(lineage_path)
    expected={'analysis_version':ANALYSIS_VERSION,'execution_complete':True,
              'all_five_fold_private_heads_trained':True,'private_head_trained_from_scratch':True,
              'old_checkpoint_loaded':False,'old_predictions_used_as_features':False,'core_parameters_frozen':True}
    for key,value in expected.items():
        if lineage.get(key)!=value: fail('SC_EXPERT_NOT_READY','Fresh completed native expert required',field=key)
    predictions_path=_bound(binding['singlecell_predictions'])
    if (lineage.get('prediction_sha256')!=sha(predictions_path)
        or Path(lineage.get('prediction_path','')).resolve()!=predictions_path.resolve()):
        fail('SC_EXPERT_NOT_BOUND','Prediction does not match the completed SC lineage')
    candidates_path=_bound(binding['candidates'])
    out=_new_output(out)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','training':False})
    candidates=pd.read_parquet(candidates_path)
    result=celltype_table(candidates,pd.read_parquet(predictions_path))
    result.to_parquet(out/'SC_CELLTYPE_EXACT_FUSION.parquet',index=False)
    candidates[KEYS].to_parquet(out/'CANDIDATES.parquet',index=False)
    atomic_json(out/'CELL_TYPES.json',sorted(set(result.cell_type)))
    atomic_json(out/'SOURCE_BINDING.json',binding)
    atomic_json(out/'CONTRACT.json',{'format':'R62_SC_CELLTYPE_EXACT_V1','source_lineage_sha256':sha(lineage_path),
        'primary_ranking_unchanged':True,'cell_types_averaged_together':False,
        'sparse_absence':'NULL_WITH_REASON_AT_FUSION_JOIN','source_rows':len(result),
        'available_rows':int(result[AVAILABLE].sum()),'training_started':False})
    commit_manifest(out,{'role':'SC_CELLTYPE_EXACT_FUSION_TABLE'})
    return status(out,'SC_CELLTYPE_TABLE_READY_NOT_TRAINED')


def prepare_fusion(prepared,training,sc_table,out):
    """Build a development-only fusion dataset from frozen validation predictions."""
    _host()
    from .lineage import validate_prepared
    from .modality_report import _checkpoint_model
    from .train import evaluate
    prepared=Path(prepared);sc_table=Path(sc_table)
    _,spec,cfg,_=validate_prepared(prepared);verify_manifest(sc_table)
    if spec.get('singlecell_association_dim',0) or spec.get('singlecell_role')!='interpretation_only':
        fail('SC_PRIMARY_DOUBLE_COUNT','SC must be absent from primary LASSO and neural inputs')
    model,alpha,run=_checkpoint_model(prepared,training,spec,cfg)
    _,predictions=evaluate(model,prepared,read_json(prepared/'VIEWS.json'),spec,'cpu',cfg['training']['batch_size'],alpha)
    pieces=[]
    for cancer,arr,logit,_ in predictions['pathway']:
        # No sealed patient outer-test inputs. Same available validation candidates only.
        pieces.append(pd.DataFrame({'cancer_id':cancer,'lncrna_id':[spec['nodes']['lncRNA'][i] for i in arr['lnc']],
            'pathway_id':[spec['nodes']['pathway'][i] for i in arr['target']],
            'primary_probability':1/(1+np.exp(-np.clip(logit.astype(float),-700,700))),
            'fusion_target':arr['label'].astype(float),'target_available':arr['available'].astype(bool)}))
    primary=pd.concat(pieces,ignore_index=True)
    long_table=pd.read_parquet(sc_table/'SC_CELLTYPE_EXACT_FUSION.parquet')
    candidates=pd.read_parquet(sc_table/'CANDIDATES.parquet')
    if not primary[KEYS].sort_values(KEYS).reset_index(drop=True).equals(candidates[KEYS].sort_values(KEYS).reset_index(drop=True)):
        fail('SC_FUSION_UNIVERSE','SC table must bind the identical primary candidate universe')
    frame,specs,catalog=expert_matrix(primary,long_table,read_json(sc_table/'CELL_TYPES.json'))
    frame['fusion_pair_fold']=[pair_blocked_fold(l,p,seed=cfg['seed']) for l,p in frame[['lncrna_id','pathway_id']].itertuples(index=False,name=None)]
    trainable=frame.loc[frame.target_available]
    if set(trainable.fusion_pair_fold)!=set(range(5)): fail('SC_FUSION_FOLDS','Need five estimable pair folds')
    if not any(trainable[s.expert_id+'_available'].any() for s in specs): fail('SC_FUSION_NO_COVERAGE','No usable SC expert on labelled candidates')
    out=_new_output(out)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','training':False,'outer_test_read':False})
    frame.to_parquet(out/'FUSION_TRAINING_FRAME.parquet',index=False)
    atomic_json(out/'EXPERT_CATALOG.json',catalog)
    atomic_json(out/'PLAN.json',{'format':'R62_CELLTYPE_SECONDARY_FUSION_PLAN_V1','code_sha256':code_digest(),
        'source_prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),'sc_table_manifest_sha256':sha(sc_table/'MANIFEST.json'),
        'primary_run':run,'primary_input_sc':False,'primary_ranking_unchanged':True,'outer_test_read':False,
        'scope':'DEVELOPMENT_PRIMARY_VALIDATION_PAIR_CROSSFIT_NOT_OUTER_TEST_OR_RELEASE',
        'training_authorized':False,'fusion_config':asdict(ResidualFusionConfig(seed=cfg['seed'])),
        'experts':[asdict(s) for s in specs],'candidate_rows':len(frame),'target_available_rows':len(trainable)})
    commit_manifest(out,{'role':'SC_CELLTYPE_FUSION_TRAINING_PLAN'})
    return status(out,'FUSION_PREPARED_WAITING_TRAINING_CONFIRMATION')


def fit_fusion(plan_root,out,*,allow_training=False,deadline=None):
    _host()
    if not allow_training: fail('TRAINING_NOT_AUTHORIZED','Explicit --allow-training is required after user confirmation')
    if deadline is None or deadline<=time.time(): fail('BUDGET_NOT_BOUND','Future absolute CPU deadline required')
    plan_root=Path(plan_root);verify_manifest(plan_root);plan=read_json(plan_root/'PLAN.json')
    if plan.get('format')!='R62_CELLTYPE_SECONDARY_FUSION_PLAN_V1' or plan.get('code_sha256')!=code_digest():
        fail('SC_FUSION_PLAN_DRIFT','Rebuild against the confirmed code')
    out=_new_output(out)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','training':True,'deadline_unix':deadline,'gpu':False})
    # Independent OS deadline, including during an optimizer call; no cloud process is launched.
    import signal
    def timeout(*_): raise TimeoutError('SC fusion CPU deadline reached')
    previous=signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,deadline-time.time()))
    try:
        frame=pd.read_parquet(plan_root/'FUSION_TRAINING_FRAME.parquet')
        specs=[ExpertSpec(**s) for s in plan['experts']]
        result=train_pair_blocked_crossfit_fusion(frame.loc[frame.target_available],specs,endpoint='discovery',config=ResidualFusionConfig(**plan['fusion_config']))
        result.oof_prediction.to_parquet(out/'FUSION_OOF.parquet',index=False)
        result.fold_metrics.to_csv(out/'FUSION_METRICS.tsv',sep='\t',index=False)
        final=apply_residual_fusion_model(result.final_model,frame)
        final.to_parquet(out/'FUSION_SECONDARY_FINAL.parquet',index=False)
        catalog=read_json(plan_root/'EXPERT_CATALOG.json')
        contribution_table(final,frame,catalog,'FINAL_FIT_NOT_OOF').to_parquet(out/'SC_CELLTYPE_SCORE_CONTRIBUTIONS.parquet',index=False)
        contribution_table(result.oof_prediction,frame,catalog,'PAIR_OOF_PRIMARY_VALIDATION_ONLY').to_parquet(
            out/'SC_CELLTYPE_OOF_CONTRIBUTIONS.parquet',index=False)
        atomic_json(out/'MODELS.json',{'folds':[asdict(m) for m in result.fold_models],'final':asdict(result.final_model)})
        atomic_json(out/'RECEIPT.json',{'state':'DEVELOPMENT_SECONDARY_FUSION_COMPLETE_NOT_DEPLOYED',
            'plan_manifest_sha256':sha(plan_root/'MANIFEST.json'),'primary_ranking_unchanged':True,
            'performance_outcome':result.performance_outcome,'outer_test_read':False,'formal_admission':False,
            'optimizer_steps':[m.optimiser_steps for m in result.fold_models],
            'scope':plan['scope'],'cell_types':[x['cell_type'] for x in read_json(plan_root/'EXPERT_CATALOG.json')]})
        commit_manifest(out,{'role':'DEVELOPMENT_SECONDARY_FUSION_RESULT'})
        return status(out,'DEVELOPMENT_SECONDARY_FUSION_COMPLETE_NOT_DEPLOYED')
    except BaseException as error:
        status(out,'SC_FUSION_FAILED_NOT_DEPLOYED',message=str(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
