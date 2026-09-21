#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.materialize import materialize_genesets

def main():
    p=parse_common_args('Materialize default and extended lncRNA Gene Sets'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'19_materialize_model_genesets'): materialize_genesets(cfg)
if __name__=='__main__': main()
