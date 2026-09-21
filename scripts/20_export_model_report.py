#!/usr/bin/env python3
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.report import export_report
from cc_hhgt.resource_audit import refresh_resource_usage_after_model

def main():
    p=parse_common_args('Export model coverage, evaluation and Gene Set report'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'20_export_model_report'): refresh_resource_usage_after_model(cfg); print(export_report(cfg))
if __name__=='__main__': main()
