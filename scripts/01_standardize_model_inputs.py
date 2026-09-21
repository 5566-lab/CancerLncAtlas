#!/usr/bin/env python3
from cc_hhgt.cellline import standardize_cellline_context
from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status, write_json
from cc_hhgt.resource_audit import build_resource_usage_manifest
from cc_hhgt.standardize import standardize_dimensions, standardize_drug_tables, standardize_interactions, standardize_pathway_members, standardize_single_cell, standardize_tumor_state

def main():
    p=parse_common_args('Standardize model inputs without copying large expression matrices'); a=p.parse_args(); configure_logging(a.verbose); cfg=load_config(a.config)
    with stage_status(cfg,'01_standardize_model_inputs'):
        dims=standardize_dimensions(cfg); member=standardize_pathway_members(cfg,dims['dim_gene']); rel,events,support=standardize_interactions(cfg); state=standardize_tumor_state(cfg); sc=standardize_single_cell(cfg); drug=standardize_drug_tables(cfg)
        cellline=standardize_cellline_context(cfg); build_resource_usage_manifest(cfg); write_json({'dimensions':{k:len(v) for k,v in dims.items()},'pathway_gene_member':len(member),'interaction_relation':len(rel),'evidence_event':len(events),'interaction_pathway_support':len(support),'tumor_state':len(state),'single_cell':sc,'drug':drug,'cellline':cellline},cfg['_results']/ 'tables'/'standardization_summary.json')
if __name__=='__main__': main()
