#!/usr/bin/env python3
import pandas as pd
from cc_hhgt.common import configure_logging, load_config, parse_common_args, read_table, stage_status, write_table
from cc_hhgt.scoring import score_fold_candidates

def main():
    p=parse_common_args('Score every candidate with its held-out-cancer fold model'); p.add_argument('--folds',nargs='*'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'17_score_all_candidates'):
        folds=read_table(cfg['_results']/'tables'/'fold_manifest.tsv'); folds=folds.loc[folds.fold_id.isin(a.folds)] if a.folds else folds; summary=[]
        for row in folds.itertuples(index=False):
            scored=score_fold_candidates(cfg,pd.Series(row._asdict())); summary.append({'fold_id':row.fold_id,'cancer_id':row.test_cancer,'n_scored':len(scored),'model_name':scored.model_name.iloc[0] if len(scored) else None})
        write_table(pd.DataFrame(summary),cfg['_results']/'tables'/'all_candidate_scoring_summary.tsv')
if __name__=='__main__': main()
