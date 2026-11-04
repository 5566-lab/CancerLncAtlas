"""CPTAC publication scope with explicit TCGA cancer-name aliases."""
from .common import fail
CANCER_MAP={'BRCA':'BRCA','CCRCC':'KIRC','COAD':'COAD','GBM':'GBM','HNSCC':'HNSC','LSCC':'LUSC','LUAD':'LUAD','OV':'OV','PDAC':'PAAD','UCEC':'UCEC'}
def select_cohorts(mode, requested, available):
    available=set(available)
    if mode=='all':return available
    if mode!='list':fail('CPTAC_CANCER_SCOPE','cancer_type must be all or list')
    if isinstance(requested,str):requested=requested.split(',')
    tokens=[v.strip().upper() for item in (requested or []) for v in str(item).split(',') if v.strip()]
    aliases={v:k for k,v in CANCER_MAP.items()}
    selected={x if x in CANCER_MAP else aliases.get(x,x) for x in tokens}
    if not selected or not selected<=available:
        fail('CPTAC_CANCER_SCOPE','Unknown/empty or unavailable CPTAC cancer list',requested=tokens,available=sorted(available))
    return selected
