#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.feature_builders import build_pathway_state_edges

def main():
    p=parse_common_args('Build pathway-tumor-state context edges'); p.add_argument('--cancers',nargs='*'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'04_build_pathway_state_edges'): build_pathway_state_edges(cfg,a.cancers)
if __name__=='__main__': main()
