"""Resolve an explicit cancer scope against the frozen patient authority."""
from copy import deepcopy
from .common import fail


def selection(cfg, profile='diagnostic'):
    if 'cancer_type' in cfg:
        mode=cfg['cancer_type']; requested=cfg.get('cancer_list',[])
    else:
        # Old bound configurations retain their declared scope until overridden.
        requested=cfg.get('profiles',{}).get(profile,{}).get('cancers',[])
        mode='list' if requested else 'all'
        if 'cancer_list' in cfg:
            fail('CANCER_TYPE_REQUIRED','cancer_list requires cancer_type=list')
    if mode not in ('all','list'):
        fail('CANCER_TYPE_INVALID','cancer_type must be all or list')
    if not isinstance(requested,list) or any(not isinstance(c,str) or not c.strip() for c in requested):
        fail('CANCER_LIST_INVALID','cancer_list must be a JSON list of nonempty cancer IDs')
    requested=sorted(set(c.strip().upper() for c in requested))
    if mode=='list' and not requested:
        fail('CANCER_LIST_EMPTY','cancer_type=list requires at least one cancer ID')
    if mode=='all' and requested:
        fail('CANCER_LIST_WITH_ALL','Use an empty cancer_list with cancer_type=all')
    return mode,requested


def override_selection(cfg, *, cancer_type=None, cancer_list=None):
    result=deepcopy(cfg)
    if cancer_type is not None:
        result['cancer_type']=cancer_type
        if cancer_type=='all':result['cancer_list']=[]
    if cancer_list is not None:
        result['cancer_list']=cancer_list
    for profile in ('diagnostic','formal'):selection(result,profile)
    return result


def resolve_cancer_scope(cfg, available, profile='diagnostic'):
    mode,requested=selection(cfg,profile)
    available=sorted(set(map(str,available)))
    if not available:fail('CANCER_AUTHORITY_EMPTY','Patient authority has no cancers')
    unknown=sorted(set(requested)-set(available))
    if unknown:
        fail('CANCER_SCOPE','Requested cancer absent in patient authority',unknown=unknown,available=available)
    selected=available if mode=='all' else requested
    return {'format':'CLA_CANCER_SCOPE_V1','profile':profile,'cancer_type':mode,
            'cancer_list':requested,'available_cancers':available,'selected_cancers':selected}
