#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, read_table, stage_status, write_table
from cc_hhgt.explain import feature_attributions, graph_explanations

def main():
    p=parse_common_args('Generate evidence attributions and observed graph paths'); p.add_argument('--top-n',type=int,default=100000); p.add_argument('--path-top-n',type=int,default=5000); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'18_generate_explanations'):
        pred=read_table(cfg['_results']/'tables'/'all_candidate_prediction'); write_table(feature_attributions(cfg,pred,a.top_n),cfg['_results']/'tables'/'model_feature_attribution.parquet'); write_table(graph_explanations(cfg,pred,a.path_top_n),cfg['_results']/'tables'/'model_explanation_path.parquet')
if __name__=='__main__': main()
