#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.feature_builders import build_ucell_support

def main():
    p=parse_common_args('Build patient-blocked UCell pathway trend support'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'07_build_ucell_support'): build_ucell_support(cfg)
if __name__=='__main__': main()
