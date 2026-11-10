"""Official PROGENy weight matrix with source-patient-only score scaling."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def score_progeny(expression, model, fit_patients, *, minimum_genes=5):
    if not expression.index.is_unique or not expression.columns.is_unique or not model.index.is_unique:
        raise ValueError("PROGENY_DUPLICATE_AXIS")
    fit=list(fit_patients)
    if len(set(fit))!=len(fit) or not set(fit)<=set(expression.index) or len(fit)<2:
        raise ValueError("PROGENY_INVALID_FIT_PATIENTS")
    genes=[g for g in model.index if g in expression.columns]
    weights=model.loc[genes].to_numpy(float)
    if not np.isfinite(weights).all():raise ValueError("PROGENY_NONFINITE_MODEL")
    values=expression[genes].to_numpy(float)
    # The official model is fixed before expression-gene intersection.
    raw=np.asarray(csr_matrix(weights).T@values.T).T
    count=(weights!=0).sum(axis=0);raw[:,count<minimum_genes]=np.nan
    raw=pd.DataFrame(raw,index=expression.index,columns=model.columns)
    mu=raw.loc[fit].mean(axis=0);sd=raw.loc[fit].std(axis=0,ddof=1)
    sd=sd.mask(~np.isfinite(sd)|(sd<=1e-12))
    scaled=(raw-mu)/sd
    receipt={"method":"PROGENY_OFFICIAL_TOP100_LINEAR_MODEL","input":"log2(TPM+1)","fit_patients":fit,
        "scaling":"PATHWAY_SCORE_Z_USING_SOURCE_PATIENTS_ONLY","model_gene_counts":(model!=0).sum().to_dict(),
        "measured_gene_counts":dict(zip(model.columns,count.astype(int).tolist())),"mean":mu.to_dict(),"sd":sd.to_dict()}
    return raw,scaled,receipt
