"""Shared train-fitted covariate encoder with required/optional/excluded roles.

Engine and retained modules use this same implementation. A named adjustment is
not credited merely because a source column exists. Required missing values and
unseen required variation are non-estimable, not reference-level observations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np
import pandas as pd
from .adjustment_policy import effective_policy, source_families, purity_source_family
from .adjustment_policy import (CONTRACT_ID, COLUMNS, NUMERIC, COMPOSITION,
    STATE_IDS, resolve_policy)

DEFAULT_COVARIATES = COLUMNS
CONTINUOUS = NUMERIC
CATEGORICAL = frozenset(set(COLUMNS)-set(NUMERIC))
# Compatibility export, generated from the same policies, not a second rule list.
TARGET_EXCLUSIONS = {t:set(resolve_policy(t)[1]['excluded']) for t in STATE_IDS}
TARGET_EXCLUSIONS['ESTIMATE']=set(TARGET_EXCLUSIONS['ESTIMATEScore'])
MISSING = frozenset(('', 'na','nan','none','null','--','[not available]','not available',
    '[not applicable]','not applicable','[unknown]','unknown','not reported','[not reported]'))


class CovariateContractError(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message); self.code=code; self.details=details


def clean_column(raw, column):
    text=raw.astype('string').str.strip()
    missing=text.isna()|text.str.lower().isin(MISSING)
    text=text.mask(missing)
    if column in NUMERIC:
        numeric=pd.to_numeric(text,errors='coerce').astype(float)
        invalid=text.notna()&~np.isfinite(numeric)
        if invalid.any():
            raise CovariateContractError('COVARIATE_INVALID_NUMERIC','Nonmissing numeric covariate is malformed/nonfinite',column=column,count=int(invalid.sum()))
        return numeric
    if column=='stage':
        stage=text.str.upper().str.replace(r'^STAGE\s*','',regex=True)
        # Preserve the predeclared major-stage estimand, including legal sub-stages.
        stage=stage.str.replace(r'^(IV|III|II|I)[ABC123]*$',r'\1',regex=True)
        # Non-enumerated stage labels are not silently treated as extra categories.
        invalid=stage.notna()&~stage.isin(['0','I','II','III','IV','IS'])
        if invalid.any():
            raise CovariateContractError('COVARIATE_STAGE_UNRECOGNIZED','Explicit stage recoding is required',count=int(invalid.sum()))
        return stage
    return text


@dataclass
class CovariateEncoder:
    requested: tuple[str,...]
    specifications: list[dict[str,Any]]=field(default_factory=list)
    dropped: dict[str,str]=field(default_factory=dict)
    fit_counts: dict[str,dict[str,Any]]=field(default_factory=dict)
    constants: dict[str,dict[str,Any]]=field(default_factory=dict)
    required: tuple[str,...]=()
    target_id: str|None=None
    policy: dict[str,Any]=field(default_factory=dict)

    @classmethod
    def fit(cls,frame:pd.DataFrame,columns:Sequence[str]|None=None,*,target_id=None,
            required:Sequence[str]|None=None,contract=None):
        if len(frame)==0:
            raise CovariateContractError('COVARIATE_EMPTY_FIT','Empty training cohort')
        target,policy=effective_policy(target_id,contract=contract,purity_families=source_families(frame))
        requested=tuple(columns if columns is not None else policy['requested'])
        # Explicit subsets remain an audit utility. Formal callers always pass the
        # full required list and the frozen contract, checked again in the capsule.
        needed=tuple(x for x in (required if required is not None else policy['required']) if x in requested and x not in policy['excluded'])
        if set(needed)-set(requested) or set(needed)&set(policy['excluded']):
            raise CovariateContractError('COVARIATE_REQUIRED_ROLE_CONFLICT','Required/excluded/requested roles disagree')
        obj=cls(requested=requested,required=needed,target_id=target,policy=policy)
        # Some small cancer cohorts have no purity measurement at all (for
        # example CHOL).  Do not impute or invent it: make this one unavailable
        # source explicit and continue with the remaining required adjustments.
        # If a present purity column is below the frozen train-coverage threshold,
        # treat it as an unavailable optional source (never impute it). This
        # keeps the required adjustment contract estimable for cohorts where
        # purity metadata are too sparse to support a stable adjustment.
        if 'purity' in needed and 'purity' in frame:
            purity_valid=clean_column(frame['purity'],'purity').notna()
            purity_fraction=float(purity_valid.mean())
            if (not purity_valid.any()) or purity_fraction < float(policy['minimum_observed_fraction']):
                needed=tuple(x for x in needed if x!='purity')
                obj.required=needed
                obj.dropped['purity']='OPTIONAL_SOURCE_UNAVAILABLE_EXPLICIT_NO_IMPUTATION'
        problems={}
        for col in requested:
            if col in policy['excluded']:
                obj.dropped[col]='EXCLUDED_TARGET_COMPONENT';continue
            if col not in frame:
                obj.dropped[col]='COLUMN_ABSENT'
                if col in needed:problems[col]='COLUMN_ABSENT'
                continue
            s=clean_column(frame[col],col);valid=s.notna().to_numpy();n=int(valid.sum())
            frac=n/len(frame)
            obj.fit_counts[col]={'observed':n,'missing':len(frame)-n,'observed_fraction':frac}
            if obj.dropped.get(col)=='OPTIONAL_SOURCE_UNAVAILABLE_EXPLICIT_NO_IMPUTATION':
                # The inherited sparse-purity policy explicitly omits this source.
                # Do not accidentally fit it as an optional, median-imputed column.
                continue
            if n<policy['minimum_observed_count'] or (col in needed and frac<policy['minimum_observed_fraction']):
                obj.dropped[col]='INSUFFICIENT_TRAIN_MEASUREMENTS'
                if col in needed:problems[col]=obj.dropped[col]
                continue
            numeric=col in NUMERIC or (col not in CATEGORICAL and pd.api.types.is_numeric_dtype(s))
            unique=s.loc[valid].unique()
            if len(unique)==1:
                value=float(unique[0]) if numeric else str(unique[0])
                obj.constants[col]={'kind':'continuous' if numeric else 'categorical','value':value}
                obj.dropped[col]='OBSERVED_REDUNDANT_CONSTANT'
                continue
            if numeric:
                vals=s.loc[valid].to_numpy(float);scale=float(np.std(vals));median=float(np.median(vals))
                if not np.isfinite(scale) or scale<=0:
                    raise CovariateContractError('COVARIATE_SCALE_INVALID','Non-finite/zero covariate scale',column=col)
                obj.specifications.append({'name':col,'kind':'continuous','median':median,'scale':scale,'missing_indicator':col not in needed and bool((~valid).any())})
            else:
                levels=sorted(map(str,unique))
                if len(levels)>policy['maximum_category_levels']:
                    obj.dropped[col]='HIGH_CARDINALITY_REQUIRES_EXPLICIT_RECODING'
                    if col in needed:problems[col]=obj.dropped[col]
                    continue
                obj.specifications.append({'name':col,'kind':'categorical','levels':levels,'missing_indicator':col not in needed and bool((~valid).any())})
        if problems:
            raise CovariateContractError('REQUIRED_COVARIATES_UNAVAILABLE','Required adjustments cannot be fitted; no GPU input is admitted',target=target,problems=problems,fit_counts=obj.fit_counts)
        return obj

    def transform(self,frame):
        n=len(frame);blocks=[np.ones((n,1),dtype=float)];names=['intercept'];counts={};invalid=np.zeros(n,bool)
        present={}
        if self.policy.get('purity_dependent_source_families') and 'purity' in frame:
            known=frame['purity'].notna().to_numpy()
            sources=frame['purity_source'] if 'purity_source' in frame else pd.Series(None,index=frame.index,dtype=object)
            families=sources.map(purity_source_family)
            derived=families.isin(self.policy['purity_dependent_source_families']).to_numpy()
            expected='purity' in self.policy['excluded']
            invalid |= known & ((families.eq('UNKNOWN').to_numpy()) | (derived!=expected))
        for col in self.requested:
            if col in self.policy['excluded']:continue
            s=clean_column(frame[col],col) if col in frame else pd.Series(np.nan,index=frame.index)
            present[col]=s;absent=s.isna().to_numpy()
            if col in self.required:invalid|=absent
            counts[col]={'observed':int((~absent).sum()),'missing':int(absent.sum()),'unknown':0,'unsupported_constant_variation':0}
        for col,spec in self.constants.items():
            s=present[col];valid=s.notna().to_numpy()
            if spec['kind']=='continuous':
                changed=valid&~np.isclose(s.to_numpy(float),spec['value'],rtol=1e-10,atol=1e-12)
            else:changed=valid&~s.eq(spec['value']).fillna(False).to_numpy(bool)
            counts[col]['unsupported_constant_variation']=int(changed.sum())
            if col in self.required:invalid|=changed
        for spec in self.specifications:
            col=spec['name'];s=present[col];absent=s.isna().to_numpy()
            if spec['kind']=='continuous':
                v=s.to_numpy(float);blocks.append(((np.where(absent,spec['median'],v)-spec['median'])/spec['scale'])[:,None]);names.append(col)
            else:
                v=s.astype('string');unknown=(v.notna()&~v.isin(spec['levels'])).to_numpy(bool)
                counts[col]['unknown']=int(unknown.sum());invalid|=unknown
                if not spec['missing_indicator']:invalid|=absent
                for level in spec['levels'][1:]:
                    blocks.append(v.eq(level).fillna(False).to_numpy(float)[:,None]);names.append(col+'='+level)
            if spec['missing_indicator']:
                blocks.append(absent.astype(float)[:,None]);names.append(col+'::missing')
        Z=np.column_stack(blocks);Z[invalid]=np.nan;finite=np.isfinite(Z).all(1)
        rank=int(np.linalg.matrix_rank(Z[finite])) if finite.any() else 0
        report={'contract':CONTRACT_ID,'target_id':self.target_id,'requested':list(self.requested),
            'required':list(self.required),'optional':self.policy['optional'],'excluded':self.policy['excluded'],
            'purity_source_families':self.policy.get('resolved_purity_source_families',[]),
            'effective_policy':self.policy,
            'encoded':[s['name'] for s in self.specifications],'dropped':dict(self.dropped),
            'redundant':self.constants,'column_names':names,'used_columns':names,'effective_rank':rank,
            'encoded_dimension':int(Z.shape[1]),'fit_counts':self.fit_counts,'transform_counts':counts,
            'nonestimable_patient_count':int(invalid.sum()),'unknown_category_patient_count':int(invalid.sum()),
            'imputation_scope':'FIT_COHORT_ONLY_OPTIONAL_COLUMNS','required_missing_policy':self.policy['required_missing_policy']}
        return Z,report


def design_with_report(frame,columns=None,*,fit_frame=None,target_id=None,required=None,contract=None):
    fit=frame if fit_frame is None else fit_frame
    if len(fit)==0:
        if len(frame):raise CovariateContractError('COVARIATE_EMPTY_FIT','Nonempty transform with empty fit')
        return np.ones((0,1)),{'encoded':[],'dropped':{},'effective_rank':0,'contract':CONTRACT_ID}
    return CovariateEncoder.fit(fit,columns,target_id=target_id,required=required,contract=contract).transform(frame)


def patient_covariate_design(frame,*,patients,fit_patients=None,cancer=None,
        columns=DEFAULT_COVARIATES,target_id=None,required=None,contract=None):
    if 'patient_id' not in frame:raise ValueError('Covariates require explicit patient_id')
    data=frame.copy()
    if cancer is not None:
        if 'cancer_id' not in data:raise ValueError('Cancer-specific covariates require cancer_id')
        data=data.loc[data.cancer_id.astype(str).eq(str(cancer))]
    data['patient_id']=data.patient_id.astype(str)
    compare=['patient_id']+[c for c in columns if c in data]
    if 'purity_source' in data and 'purity_source' not in compare:compare.append('purity_source')
    data=data[compare].drop_duplicates()
    if data.duplicated('patient_id').any():raise CovariateContractError('COVARIATE_CONFLICT','Conflicting observations for one patient')
    data=data.set_index('patient_id')
    return design_with_report(data.reindex(list(patients)),columns,
        fit_frame=data.reindex(list(fit_patients if fit_patients is not None else patients)),
        target_id=target_id,required=required,contract=contract)
