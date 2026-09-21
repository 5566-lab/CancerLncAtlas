#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.pathways import build_pathway_similarity

def main():
    p=parse_common_args('Fuse multi-view pathway similarities'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'05_build_pathway_similarity'): build_pathway_similarity(cfg)
if __name__=='__main__': main()
