#!/usr/bin/env python3
import pandas as pd
from cc_hhgt.common import configure_logging, load_config, parse_common_args, read_table, stage_status
from cc_hhgt.gnn import train_gnn_fold

def main():
    p=parse_common_args('Train R-GCN baseline'); p.add_argument('--fold'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'12_train_rgcn_baseline'):
        folds=read_table(cfg['_results']/'tables'/'fold_manifest.tsv'); folds=folds.loc[folds.fold_id==a.fold] if a.fold else folds
        for row in folds.itertuples(index=False): train_gnn_fold(cfg,pd.Series(row._asdict()),'rgcn')
if __name__=='__main__': main()
