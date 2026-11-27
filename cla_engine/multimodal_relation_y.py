"""Materialize measured relation evidence after structural readout binding.

Equivalent pathway readouts and regulator-member states retain separate strata.
No model is fitted and no source association summary is constructed as input.
"""
from __future__ import annotations
import collections
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .relation_evidence_bindings import FORMAT as BINDING_FORMAT, file_sha, read_table
from .functional_fusion import FORMAT, validate_fused_arrays
from .fused_adjustment.design import design
from .fused_adjustment.adjustment_policy import resolve_policy

ASSAYS={'RNA':1,'PROTEIN':2,'PHOSPHO':4,'METHYLATION':8,'CNV':16,'VARIANT':32,'OTHER':64}


def assay_bit(value):
    value=str(value).upper()
    for names,bit in [(('PHOSPH',),4),(('PROTEIN','PROTEOM'),2),(('RNA','TRANSCRIPT'),1),
                      (('METHYL',),8),(('COPY','CNV'),16),(('VARIANT','SNV','MUTATION','SEQUENC'),32)]:
        if any(s in value for s in names):return bit
    return 64


def summarize_effect(rho,available,duplicate_groups):
    rho=np.asarray(rho,float);available=np.asarray(available,bool)
    if rho.shape!=available.shape or rho.ndim!=2 or not np.isfinite(rho[available]).all() or np.any(abs(rho[available])>1+1e-6):
        raise ValueError('MULTIMODAL_EFFECT_AXIS_OR_VALUE')
    if sorted(i for group in duplicate_groups for i in group)!=list(range(rho.shape[1])):
        raise ValueError('MULTIMODAL_EFFECT_DUPLICATE_PARTITION')
    weight=available.astype(float);distinct=np.zeros(len(rho),np.uint16)
    for group in duplicate_groups:
        n=available[:,group].sum(1);distinct+=n>0
        weight[:,group]=np.divide(weight[:,group],n[:,None],out=np.zeros((len(rho),len(group))),where=n[:,None]>0)
    clean=np.where(available,rho,0.);den=weight.sum(1)
    rms=np.sqrt(np.divide((weight*clean**2).sum(1),den,out=np.full(len(rho),np.nan),where=den>0))
    signed=np.divide((weight*clean).sum(1),den,out=np.full(len(rho),np.nan),where=den>0)
    return rms,signed,weight,distinct


def _json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    temp.replace(path)


def _npz(path,arrays):
    path=Path(path);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)


def materialize_context(*, root,catalog,binding_dir,output_root,cancer,context,empirical_fisher,bh,basis,algorithm_path,progress=print):
    raise RuntimeError('ASSOCIATION_STATISTICS_ARE_NOT_MEASURED_Y: use measured_readouts.materialize_measured_readouts; define Y before lncRNA modeling')
