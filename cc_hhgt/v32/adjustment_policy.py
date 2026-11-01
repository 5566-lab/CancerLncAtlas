"""Single source of truth for R6.2 target adjustment roles.

R6.2 retains raw-signed lncRNA-only A1, but explicitly changes immune primary
targets to composition-conditional associations. Only a target itself or a
recorded same-source derivative is excluded; no whole-family exclusion. This
label-estimand revision requires new preparation and a new fitted baseline.
"""
from __future__ import annotations
import copy

CONTRACT_ID = 'V32_TARGET_ADJUSTMENT_R62_V1'
BASE = ('age_years', 'sex', 'stage', 'molecular_subtype', 'clinical_subtype', 'technical_batch')
COMPOSITION = ('purity', 'leukocyte_fraction', 'cell_fraction_Macrophages.M0',
               'cell_fraction_Macrophages.M1', 'cell_fraction_Macrophages.M2')
COLUMNS = BASE + COMPOSITION
NUMERIC = frozenset(('age_years',) + COMPOSITION)
# Exact self/alias exclusions only. Other composition measures remain required.
SELF_EXCLUSIONS = {'Leukocyte_fraction': {'leukocyte_fraction': 'TARGET_SELF_SAME_SOURCE_MEASUREMENT'},
    **{'cibersort_relative::Macrophages.'+m: {'cell_fraction_Macrophages.'+m: 'TARGET_SELF_SAME_SOURCE_MEASUREMENT'} for m in ('M0','M1','M2')}}
# Purity from ESTIMATE is a derivative involving the ESTIMATE component scores.
# It is excluded for these targets ONLY when the actual purity_source says so.
PURITY_DEPENDENCIES = {
    'ImmuneScore': ('ESTIMATE',), 'StromalScore': ('ESTIMATE',),
    'ESTIMATEScore': ('ESTIMATE',), 'TumorPurity': ('ESTIMATE',),
    'ABSOLUTE_purity': ('ABSOLUTE',),
}
PURITY_SOURCE_ALIASES = {
    'absolute': 'ABSOLUTE', 'absolute_purity': 'ABSOLUTE',
    'absolute_purity_ploidy': 'ABSOLUTE', 'absolute_purity_ploidy::purity': 'ABSOLUTE',
    'absolute purity': 'ABSOLUTE',
    'estimate': 'ESTIMATE', 'estimate::tumorpurity': 'ESTIMATE',
    'estimate_tumorpurity': 'ESTIMATE', 'estimate_purity': 'ESTIMATE',
    'estimate tumor purity': 'ESTIMATE',
    'pathology': 'PATHOLOGY', 'histology': 'PATHOLOGY', 'ihc': 'IHC',
    'lump': 'LUMP',
}

ALIASES = {'ESTIMATE': 'ESTIMATEScore'}
STATE_IDS = ('EXTEND','RNAss','EREG.EXPss','DNAss','DMPss','ENHss','EREG-METHss',
             'ImmuneScore','StromalScore','CYT','Leukocyte_fraction','ESTIMATEScore',
             'TumorPurity','ABSOLUTE_purity','ABSOLUTE_ploidy')


def default_adjustment_contract(targets=None):
    policies = {}
    for target in dict.fromkeys(('__pathway__',) + STATE_IDS + tuple(targets or ())):
        excluded = dict(SELF_EXCLUSIONS.get(target,{}))
        policies[target] = {
            'requested': list(COLUMNS),
            'required': ['age_years','sex','stage'] + [c for c in COMPOSITION if c not in excluded],
            'optional': ['molecular_subtype','clinical_subtype','technical_batch'],
            'excluded': excluded,
            'purity_dependent_source_families': list(PURITY_DEPENDENCIES.get(target,())),
            'purity_source_column':'purity_source',
            'minimum_observed_fraction': 0.8,
            'minimum_observed_count': 3,
            'minimum_joint_valid_fraction': 0.8,
            'maximum_category_levels': 32,
            'required_missing_policy': 'COMPLETE_CASE_NO_IMPUTED_REQUIRED_ADJUSTMENTS',
            'constant_policy': 'REDUNDANT_IF_OBSERVED_AND_UNCHANGED;NEW_VARIATION_UNSUPPORTED',
        }
    return {'contract_id': CONTRACT_ID, 'policies': policies,
            'scope': 'FIT_FEATURE_PATIENTS_ONLY;CHECK_SOURCE_AND_LABEL_COHORTS_BEFORE_TRAINING',
            'primary_cnv_transform': 'raw_signed',
            'local_cnv_action': 'LNC_ONLY_A1',
            'outer_covariate_admission_not_used_for_model_selection': True}


def validate_adjustment_contract(contract, targets=None):
    if not isinstance(contract, dict) or contract.get('contract_id') != CONTRACT_ID:
        raise ValueError('ADJUSTMENT_CONTRACT_MISSING_OR_WRONG_GENERATION')
    policies = contract.get('policies', {})
    necessary = {'__pathway__'} | set(targets or STATE_IDS)
    if not necessary <= set(policies):
        raise ValueError('ADJUSTMENT_TARGET_POLICY_MISSING: ' + repr(sorted(necessary-set(policies))))
    for target, p in policies.items():
        requested=p.get('requested',[]);required=p.get('required',[]);optional=p.get('optional',[]);excluded=p.get('excluded',{})
        if any(len(v)!=len(set(v)) for v in (requested,required,optional)):
            raise ValueError('ADJUSTMENT_DUPLICATE_COLUMNS: '+target)
        if set(requested)!=set(COLUMNS) or set(required)&set(optional) or (set(required)|set(optional))&set(excluded):
            raise ValueError('ADJUSTMENT_ROLE_CONFLICT: '+target)
        if set(requested) != set(required)|set(optional)|set(excluded):
            raise ValueError('ADJUSTMENT_UNCLASSIFIED_COLUMN: '+target)
        if not {'age_years','sex','stage'} <= set(required):
            raise ValueError('ADJUSTMENT_REQUIRED_BASE_REMOVED: '+target)
        expected_excluded=set(SELF_EXCLUSIONS.get(target,{}))
        if set(excluded)!=expected_excluded:
            raise ValueError('ADJUSTMENT_ESTIMAND_DRIFT: '+target)
        if p.get('purity_dependent_source_families') != list(PURITY_DEPENDENCIES.get(target,())) or p.get('purity_source_column')!='purity_source':
            raise ValueError('ADJUSTMENT_SOURCE_DEPENDENCY_DRIFT: '+target)
        if p.get('required_missing_policy') != 'COMPLETE_CASE_NO_IMPUTED_REQUIRED_ADJUSTMENTS':
            raise ValueError('ADJUSTMENT_MISSING_POLICY_DRIFT: '+target)
        for k in ('minimum_observed_fraction','minimum_joint_valid_fraction'):
            if not 0 < p.get(k,0) <= 1:
                raise ValueError('ADJUSTMENT_THRESHOLD_INVALID: '+target+':'+k)
        if not isinstance(p.get('minimum_observed_count'),int) or p['minimum_observed_count']<3:
            raise ValueError('ADJUSTMENT_MINIMUM_COUNT_INVALID: '+target)
        if not isinstance(p.get('maximum_category_levels'),int) or p['maximum_category_levels']<2:
            raise ValueError('ADJUSTMENT_CATEGORY_LIMIT_INVALID: '+target)
    return contract


def resolve_policy(target_id=None, *, contract=None):
    contract=default_adjustment_contract() if contract is None else contract
    validate_adjustment_contract(contract, targets=())
    target=ALIASES.get(target_id,target_id) if target_id else '__pathway__'
    if target not in contract['policies']:
        raise ValueError('UNKNOWN_ADJUSTMENT_TARGET: '+str(target))
    return target, copy.deepcopy(contract['policies'][target])


def purity_source_family(value):
    if value is None:return 'UNKNOWN'
    return PURITY_SOURCE_ALIASES.get(str(value).strip().lower(),'UNKNOWN')


def source_families(frame):
    """Read origin labels, never infer target identity from correlation values."""
    import pandas as pd
    if 'purity' not in frame:return []
    seen=pd.to_numeric(frame['purity'],errors='coerce').notna()
    if not seen.any():return []
    if 'purity_source' not in frame:return ['UNKNOWN']
    return sorted({purity_source_family(v) for v in frame.loc[seen,'purity_source']})


def effective_policy(target_id, *, contract=None, purity_families=()):
    target,p=resolve_policy(target_id,contract=contract)
    families=sorted(set(purity_families));dependent=set(p['purity_dependent_source_families'])
    if dependent and families:
        if 'UNKNOWN' in families:
            raise ValueError('PURITY_SOURCE_PROVENANCE_REQUIRED: '+target+'; do not guess whether purity is derived from this target')
        matches=[f in dependent for f in families]
        if any(matches) and not all(matches):
            raise ValueError('MIXED_PURITY_SOURCE_DEPENDENCY: '+target+'; one column mixes target-derived and independent estimates')
        if all(matches):
            p['excluded']['purity']='TARGET_SELF_OR_RECORDED_SAME_SOURCE_DERIVATIVE'
            p['required']=[c for c in p['required'] if c!='purity']
    p['resolved_purity_source_families']=families
    return target,p
