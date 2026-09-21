#!/usr/bin/env python3
from cc_hhgt.candidates import build_candidate_universe
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status

def main():
    p=parse_common_args('Materialize cancer-detected lncRNA x pathway-family candidate universe'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'09_build_candidate_universe'): build_candidate_universe(cfg)
if __name__=='__main__': main()
