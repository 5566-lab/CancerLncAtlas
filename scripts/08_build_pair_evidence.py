#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.evidence import build_pair_evidence

def main():
    p=parse_common_args('Build cancer-context lncRNA-pathway-family pair evidence'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'08_build_pair_evidence'): build_pair_evidence(cfg)
if __name__=='__main__': main()
