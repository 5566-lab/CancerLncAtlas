import numpy as np
from cla_engine.common import fail
from .adjustment_policy import default_adjustment_contract,validate_adjustment_contract,resolve_policy,effective_policy
from .covariate_contract import CovariateContractError,patient_covariate_design
def design(cov, patients, fit_patients, *, family, target_id=None, cfg=None, admission_cohorts=None):
    """Fit once on feature patients; enforce explicit roles for this target.

    admission_cohorts are source/label groups only. The outer test never selects
    an encoder or relaxes a threshold. Returned Z marks unsupported patients NaN.
    """
    fallback={'pathway':'__pathway__','stemness_telomere':'RNAss','immune':'ImmuneScore',
        'stromal':'StromalScore','cytotoxicity':'CYT','composition':'ESTIMATEScore',
        'purity':'TumorPurity','ploidy':'ABSOLUTE_ploidy'}
    target=target_id or fallback.get(family)
    if target is None:fail('ADJUSTMENT_TARGET_UNDECLARED',str(family))
    contract=cfg.get('adjustment_contract') if cfg is not None else default_adjustment_contract()
    try:
        validate_adjustment_contract(contract)
        target,policy=resolve_policy(target,contract=contract)
        if len(set(patients))!=len(patients) or len(set(fit_patients))!=len(fit_patients) or set(fit_patients)-set(patients):
            fail('COVARIATE_PATIENT_SCOPE_INVALID','Fit/transform patient identities must be unique and aligned')
        Z,report=patient_covariate_design(cov,patients=patients,fit_patients=fit_patients,
            target_id=target,columns=policy['requested'],required=policy['required'],contract=contract)
        _,policy=effective_policy(target,contract=contract,purity_families=report.get('purity_source_families',[]))
    except CovariateContractError as exc:
        fail(exc.code,str(exc),**exc.details)
    except ValueError as exc:
        fail('COVARIATE_CONTRACT_INVALID',str(exc))
    groups={'source':list(fit_patients)} if admission_cohorts is None else admission_cohorts
    index={p:i for i,p in enumerate(patients)};cohorts={}
    for name,group in groups.items():
        if set(group)-set(patients) or not group:fail('COVARIATE_COHORT_INVALID',str(name))
        ix=np.asarray([index[p] for p in group],int);valid=np.isfinite(Z[ix]).all(1)
        n=int(valid.sum());fraction=n/len(ix)
        # Cohort admission follows the adjustment contract; association pair thresholds apply later.
        minimum=int(policy.get('minimum_observed_count',3))
        cohorts[name]={'patients':len(ix),'valid':n,'valid_fraction':fraction,'minimum_observed_count':minimum}
        if n<minimum or fraction<policy['minimum_joint_valid_fraction']:
            fail('COVARIATE_COHORT_NOT_ADMISSIBLE','Required adjustments unsupported in a source/label cohort',
                target=target,cohort=name,report=cohorts[name])
    # Report full-cohort transformation counts outside the capsule only if needed;
    # the optimizer contract contains source-fit and admitted cohort counts, no outcomes.
    report.update(family=family,fit_patient_count=len(fit_patients),cohorts=cohorts,
        policy='SINGLE_TRAIN_FITTED_TARGET_CONTRACT_R62',admission_passed=True)
    return Z,report
