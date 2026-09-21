#!/usr/bin/env python3
from cc_hhgt.calibration import calibrate_fold_model
from cc_hhgt.common import configure_logging, load_config, parse_common_args, read_table, stage_status

def main():
    p=parse_common_args('Calibrate fold predictions using validation cancers'); p.add_argument('--models',nargs='*'); p.add_argument('--folds',nargs='*'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'16_calibrate_predictions'):
        folds=read_table(cfg['_results']/'tables'/'fold_manifest.tsv'); fold_ids=a.folds or folds.fold_id.tolist(); models=a.models or cfg['scoring']['model_preference']
        for model in models:
            for fold in fold_ids:
                if (cfg['_results']/'models'/model/fold/'prediction_raw.parquet').exists(): calibrate_fold_model(cfg,model,fold)
if __name__=='__main__': main()
