"""Compare previous Lasso and current ridge on identical held-out rows."""
import argparse,socket,sys
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cc_hhgt.m0_three_arm import regression_metrics,ARMS
from cla_engine.common import sha
from plot_m0_increment import classify
import json

def compare(current,previous,output):
    assert socket.gethostname()=='149';current=Path(current);out=Path(output);rows=[]
    for c in ['BRCA','LUAD','COAD']:
        cur=pd.read_parquet(current/c/'OOF_PREDICTIONS.parquet',filters=[('variant','in',['rna_only','multiomics_graph'])])
        successful=pd.read_csv(current/c/'TARGET_TRAINING_STATUS.tsv',sep='\t');good=set(successful.loc[successful.variant.eq('multiomics_graph')&successful.training_success,'target_id'])
        meta=pd.read_csv(current/c/'TARGETS.tsv',sep='\t').set_index('target_id');keys=['target_id','patient_id','phase','cohort']
        for oldroot in previous:
            d=Path(oldroot)/c;rec=json.loads((d/'COMPLETE.json').read_text());p=d/'PAIRED_OOF_PREDICTIONS.parquet'
            if sha(p)!=rec['files'][p.name]:raise ValueError('PREVIOUS_RESULT_HASH_MISMATCH')
            old=pd.read_parquet(p);old=old[old.target_id.isin(good)]
            for variant in ['rna_only','multiomics_graph']:
                for arm in ARMS:
                    n=cur[cur.variant.eq(variant)&cur.arm.eq(arm)]
                    f=old[keys+['observed','prediction_'+arm]].merge(n[keys+['observed','predicted']],on=keys,how='inner',validate='one_to_one',suffixes=('_old','_new'))
                    np.testing.assert_allclose(f.observed_old,f.observed_new,rtol=0,atol=0)
                    counts_old=old.groupby('target_id').size();counts_new=n.groupby('target_id').size()
                    for tid,g in f.groupby('target_id'):
                        if g.phase.nunique()!=5:continue
                        a=regression_metrics(g.observed_old,g['prediction_'+arm]);b=regression_metrics(g.observed_new,g.predicted);m=meta.loc[tid]
                        rows.append(dict(cancer_id=c,target_id=tid,layer=m.layer,family=m.family,arm=arm,current_variant=variant,n_paired=len(g),n_previous=int(counts_old[tid]),n_current=int(counts_new[tid]),r2_previous_lasso=a['r2'],r2_current=b['r2'],delta_r2=b['r2']-a['r2']))
        print('PREVIOUS_COMPARISON',c,flush=True)
    f=classify(pd.DataFrame(rows));f.to_csv(out/'PAIRED_PREVIOUS_LASSO.tsv',sep='\t',index=False)
    s=f.groupby(['display_layer','arm','current_variant']).agg(n_targets=('target_id','size'),median_previous_r2=('r2_previous_lasso','median'),median_current_r2=('r2_current','median'),median_paired_delta=('delta_r2','median'),improved=('delta_r2',lambda x:int((x>1e-9).sum()))).reset_index();s.to_csv(out/'PREVIOUS_LASSO_SUMMARY.tsv',sep='\t',index=False)
    print('PREVIOUS_COMPARISON_COMPLETE',len(f),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',required=True);p.add_argument('--previous',required=True,nargs='+');p.add_argument('--output',required=True);a=p.parse_args();compare(a.current,a.previous,a.output)
