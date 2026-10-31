"""Read-only query contracts shared by the R6 interpretation modules.

These modules do not import the optimizer, alter candidate ranks, or generate
training labels. Existing scores are carried through unchanged and hashed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import fail,sha,atomic_json,commit_manifest
from .tables import table,require
from .identity import frame_ids,unique_values

KEYS=['cancer_id','task','lncrna_id','target_id']

def queries(path,maximum=5000):
    p=Path(path);f=frame_ids(table(p))
    if 'task' not in f:f['task']='state' if 'state_id' in f and 'pathway_id' not in f else 'pathway'
    if 'target_id' not in f:
        if 'pathway_id' in f and 'state_id' in f:fail('INTERPRETATION_QUERY_AMBIGUOUS','Provide target_id and task when both types are present')
        if 'pathway_id' in f:f=f.rename(columns={'pathway_id':'target_id'})
        elif 'state_id' in f:f=f.rename(columns={'state_id':'target_id'})
    require(f,KEYS,'interpretation queries')
    if f[KEYS].isna().any().any() or not f.task.isin(['pathway','state']).all():fail('INTERPRETATION_QUERY_INVALID','Expected explicit cancer/lncRNA/target/task')
    if f.duplicated(KEYS).any():fail('INTERPRETATION_QUERY_DUPLICATE','Supply one frozen query per relation, not repeated patient folds')
    if len(f)>maximum:fail('INTERPRETATION_QUERY_CAP','Select a fixed output subset before interpretation, not by explanation score',rows=len(f),cap=maximum)
    f=f.sort_values(KEYS,kind='stable').reset_index(drop=True)
    return f,{'path':str(p.resolve()),'sha256':sha(p),'rows':len(f),'primary_rank_modified':False}

def output_directory(path):
    p=Path(path)
    if p.exists() and any(p.iterdir()):fail('OUTPUT_EXISTS','Interpretation results are immutable; choose a new directory')
    p.mkdir(parents=True,exist_ok=True);return p

def finish(out,receipt):
    receipt.update(primary_score_changed=False,primary_ranking_changed=False,causality_claimed=False,trained_new_model=False)
    atomic_json(out/'INTERPRETATION_RECEIPT.json',receipt)
    commit_manifest(out,{'role':'READ_ONLY_POSTHOC_NOT_TRAINING_INPUT','module':receipt['module'],'query_sha256':receipt['query']['sha256']})
    return receipt
