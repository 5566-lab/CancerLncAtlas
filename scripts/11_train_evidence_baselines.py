#!/usr/bin/env python3
from cc_hhgt.baselines import train_baseline_fold
from cc_hhgt.common import configure_logging, load_config, parse_common_args, read_table, stage_status

def main():
    p=parse_common_args('Train logistic and histogram-gradient evidence baselines'); p.add_argument('--fold'); p.add_argument('--models',nargs='+',default=['logistic','hist_gradient_boosting']); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'11_train_evidence_baselines'):
        folds=read_table(cfg['_results']/'tables'/'fold_manifest.tsv'); folds=folds.loc[folds.fold_id==a.fold] if a.fold else folds
        for row in folds.itertuples(index=False):
            s=__import__('pandas').Series(row._asdict())
            for model in a.models: train_baseline_fold(cfg,s,model)
if __name__=='__main__': main()
