#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.graph_build import build_fold_manifests, build_graph_tables

def main():
    p=parse_common_args('Build leakage-controlled graph tables and LOCO fold manifests'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'10_build_fold_specific_graphs'): build_graph_tables(cfg); build_fold_manifests(cfg)
if __name__=='__main__': main()
