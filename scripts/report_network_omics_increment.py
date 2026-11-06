"""Paired RNA increment under the same equal-modality ridge estimator."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump
from scripts.plot_network_validation import points,save,LAYERS,SHORT
from scripts.plot_m0_increment import ARMS,CANCERS,LABELS,COLORS
import matplotlib.pyplot as plt
KEYS=['cancer_id','repeat','target_id','display_layer','family']

def paired_increment(metrics):
    good=metrics[metrics.success&np.isfinite(metrics.r2)]
    baseline=good[good.variant.eq('omics_only')&good.arm.eq('shared')][KEYS+['r2','n']].rename(columns={'r2':'omics_only_r2','n':'baseline_n'})
    full=good[good.variant.eq('ridge_equal')&good.arm.isin(ARMS)][KEYS+['arm','r2','n']].rename(columns={'r2':'rna_plus_omics_r2'})
    if baseline.duplicated(KEYS).any() or full.duplicated(KEYS+['arm']).any():raise ValueError('DUPLICATE_PAIRED_METRIC')
    result=full.merge(baseline,on=KEYS,how='inner',validate='many_to_one')
    if not result.n.eq(result.baseline_n).all():raise ValueError('DIFFERENT_PATIENT_SUPPORT')
    result['delta_r2']=result.rna_plus_omics_r2-result.omics_only_r2
    return result

def verify_patient_pairing(training):
    records=[]
    for cancer in CANCERS:
        for repeat in range(2):
            for fold in range(5):
                path=Path(training)/cancer/f'repeat{repeat}_outer{fold}'/'PREDICTIONS.parquet'
                f=pd.read_parquet(path,columns=['patient_id','target_id','variant','arm','observed'],filters=[('variant','in',['ridge_equal','omics_only'])])
                b=f[f.variant.eq('omics_only')&f.arm.eq('shared')][['target_id','patient_id','observed']].rename(columns={'observed':'baseline_observed'})
                a=f[f.variant.eq('ridge_equal')&f.arm.isin(ARMS)]
                if b.duplicated(['target_id','patient_id']).any() or a.duplicated(['target_id','patient_id','arm']).any():raise ValueError('DUPLICATE_PATIENT_PREDICTION')
                paired=a.merge(b,on=['target_id','patient_id'],how='left',validate='many_to_one')
                if len(paired)!=len(a) or paired.baseline_observed.isna().any():raise ValueError('UNPAIRED_PATIENT')
                if not np.array_equal(paired.observed.to_numpy(),paired.baseline_observed.to_numpy()):raise ValueError('DIFFERENT_PATIENT_TARGET')
                count=a.groupby(['target_id','arm']).size();expected=b.groupby('target_id').size().reindex(count.index.get_level_values('target_id')).to_numpy()
                if not np.array_equal(count.to_numpy(),expected):raise ValueError('UNEQUAL_PATIENT_SETS')
                records.append(dict(cancer=cancer,repeat=repeat,fold=fold,paired_prediction_rows=len(paired),patients_matched=True,targets_matched=True))
    return records

def report(summary,training):
    assert socket.gethostname()=='149';out=Path(summary)
    if not (out/'SUMMARY_COMPLETE.json').exists():raise ValueError('SUMMARY_REQUIRED')
    dump(out/'RNA_INCREMENT_PREFLIGHT.json',dict(target_host='149',code=sha(Path(__file__)),summary_sha256=sha(out/'SUMMARY_COMPLETE.json'),comparison='ridge_equal RNA+eligible omics+clinical vs shared omics_only ridge+clinical',network_regularization_changed=False,role='RNA increment with matched estimator; does not attribute module-vs-omics gain solely to RNA'))
    records=verify_patient_pairing(training);p=paired_increment(pd.read_parquet(out/'ALL_TARGET_METRICS.parquet'));p.to_parquet(out/'RNA_INCREMENT_OVER_OMICS.parquet',index=False)
    p.groupby(['cancer_id','repeat','arm','display_layer']).agg(targets=('target_id','size'),mean_delta_r2=('delta_r2','mean'),median_delta_r2=('delta_r2','median'),improved=('delta_r2',lambda s:int(s.gt(1e-9).sum())),median_patients=('n','median')).reset_index().to_csv(out/'RNA_INCREMENT_OVER_OMICS_SUMMARY.tsv',sep='\t',index=False)
    pd.DataFrame(records).to_csv(out/'RNA_INCREMENT_PATIENT_PAIRING.tsv',sep='\t',index=False)
    counts=p.groupby(['cancer_id','target_id','arm']).repeat.nunique();keep=counts[counts.eq(2)].index;avg=p.set_index(['cancer_id','target_id','arm']).loc[keep].reset_index().groupby(['cancer_id','target_id','arm','display_layer']).delta_r2.mean().reset_index()
    fig,axs=plt.subplots(1,3,figsize=(16,5.5),sharey=True)
    for ax,arm in zip(axs,ARMS):points(ax,avg[avg.arm.eq(arm)],'delta_r2',COLORS[arm]);ax.set_title(LABELS[arm])
    axs[0].set_ylabel('Paired Delta R2 over omics + clinical baseline');fig.suptitle('RNA increment with the same ridge estimator\nPooled cancer-target pairs; mean of two outer CV repeats; no graph constraint');save(fig,out,'11_rna_increment_over_omics')
    names=['RNA_INCREMENT_PREFLIGHT.json','RNA_INCREMENT_OVER_OMICS.parquet','RNA_INCREMENT_OVER_OMICS_SUMMARY.tsv','RNA_INCREMENT_PATIENT_PAIRING.tsv','11_rna_increment_over_omics.png','11_rna_increment_over_omics.pdf']
    dump(out/'RNA_INCREMENT_COMPLETE.json',dict(target_host='149',status='COMPLETE',paired_rows=len(p),prediction_phases_verified=len(records),files={n:sha(out/n) for n in names}));print('RNA_INCREMENT_COMPLETE',len(p),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--training',required=True);a=p.parse_args();report(a.summary,a.training)
