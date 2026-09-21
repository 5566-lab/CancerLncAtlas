#!/usr/bin/env python3
from cc_hhgt.audit import run_audit
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_paths, stage_status, write_json, write_table

def main():
    p=parse_common_args('Audit all CC-HHGT model inputs'); p.add_argument('--allow-hard-failures', action='store_true'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'00_audit_model_inputs'):
        frame,summary=run_audit(cfg); paths=stage_paths(cfg); write_table(frame,paths.tables/'model_input_audit.tsv'); write_json(summary,paths.tables/'model_input_audit_summary.json')
        if summary['n_hard_failures'] and not a.allow_hard_failures: raise SystemExit(f"Hard input failures: {summary['hard_failure_keys']}")
if __name__=='__main__': main()
