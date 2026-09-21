#!/usr/bin/env python3
import pandas as pd
from cc_hhgt.baselines import train_baseline_fold
from cc_hhgt.common import LOGGER, configure_logging, load_config, parse_common_args, read_table, stage_status, write_table
from cc_hhgt.gnn import train_gnn_fold

def main():
    p=parse_common_args('Run all leave-one-cancer-out folds'); p.add_argument('--models',nargs='+',default=['logistic','hist_gradient_boosting','rgcn','hgt','cc_hhgt']); p.add_argument('--folds',nargs='*'); p.add_argument('--continue-on-error',action='store_true'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    statuses=[]
    with stage_status(cfg,'15_run_loco_evaluation'):
        folds=read_table(cfg['_results']/'tables'/'fold_manifest.tsv'); folds=folds.loc[folds.fold_id.isin(a.folds)] if a.folds else folds
        for row in folds.itertuples(index=False):
            sr=pd.Series(row._asdict())
            for model in a.models:
                try:
                    if model in {'logistic','hist_gradient_boosting'}: train_baseline_fold(cfg,sr,model)
                    else: train_gnn_fold(cfg,sr,model)
                    statuses.append({'fold_id':sr.fold_id,'model_name':model,'status':'SUCCESS'})
                except Exception as exc:
                    LOGGER.exception('Failed %s/%s',sr.fold_id,model); statuses.append({'fold_id':sr.fold_id,'model_name':model,'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
                    if not a.continue_on_error: raise
        write_table(pd.DataFrame(statuses),cfg['_results']/'tables'/'loco_run_status.tsv')
if __name__=='__main__': main()
