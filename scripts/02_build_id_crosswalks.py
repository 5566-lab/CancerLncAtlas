#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status, write_json
from cc_hhgt.ids import build_crosswalks

def main():
    p=parse_common_args('Build entity, gene-protein, drug-target and STRING crosswalks'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'02_build_id_crosswalks'): write_json(build_crosswalks(cfg),cfg['_results']/'tables'/'id_crosswalk_summary.json')
if __name__=='__main__': main()
