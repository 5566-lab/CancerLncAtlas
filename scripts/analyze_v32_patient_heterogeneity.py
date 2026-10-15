#!/usr/bin/env python3
"""Analyze prespecified patient subgroups for exact lncRNA--pathway candidates.

This command performs statistical analysis when explicitly executed. It is
provided as code in this task; no real-data analysis or training is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cc_hhgt.v32.distal_topology import sha256
from cc_hhgt.v32.patient_heterogeneity import (HeterogeneityConfig, add_heterogeneity_fdr,
    analyze_patient_heterogeneity, assemble_patient_pair, mutation_subgroups)


def _read_cancer(path: Path, cancer: str):
    if path.is_file() and path.name.endswith((".tsv",".tsv.gz")):
        table=pd.read_csv(path,sep="\t")
        return table.loc[table.cancer_id.astype(str).str.upper().eq(cancer)].copy()
    return pd.read_parquet(path,filters=[("cancer_id","==",cancer)])


def _source_record(path: Path):
    if path.is_file():return {"path":str(path.resolve()),"sha256":sha256(path)}
    files=sorted(path.rglob("*.parquet"))
    if not files:raise FileNotFoundError(f"No Parquet dataset at {path}")
    entries=[{"path":p.relative_to(path).as_posix(),"sha256":sha256(p),"bytes":p.stat().st_size} for p in files]
    digest=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"path":str(path.resolve()),"dataset_manifest_sha256":digest,"files":entries}


def _flush_patient_frames(parts,writer,path):
    if not parts:return writer
    frame=pd.concat(parts,ignore_index=True)
    frame["prediction_unavailable_reason"]=frame.prediction_unavailable_reason.astype("string")
    # cancer_id is supplied by the Hive partition, avoiding string/dictionary
    # conflicts when downstream consumers read the complete Parquet dataset.
    frame=frame.drop(columns="cancer_id")
    table=pa.Table.from_pandas(frame,preserve_index=False)
    if writer is None:writer=pq.ParquetWriter(path,table.schema,compression="zstd")
    writer.write_table(table.cast(writer.schema));parts.clear()
    return writer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["expression","activity","groups","covariates","folds","candidates","output"]:
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--group-id",required=True)
    parser.add_argument("--groups-are-mutation-calls",action="store_true")
    parser.add_argument("--mutation-count-column",default="reported_variant_count")
    parser.add_argument("--mutation-assay-column",default="mutation_assay_available")
    parser.add_argument("--locus-callable-column",default="locus_callable")
    parser.add_argument("--nuisance-columns",default="local_cnv_log2,tumor_purity")
    parser.add_argument("--min-train-patients",type=int,default=20)
    parser.add_argument("--min-group-patients",type=int,default=8)
    parser.add_argument("--confidence",type=float,default=.95)
    parser.add_argument("--ridge-alpha",type=float,default=1.)
    parser.add_argument("--candidate-selection",choices=["prespecified_independent","exploratory_current_core"],required=True,
                        help="Current-core selections remain exploratory; their selection is not external validation")
    args=parser.parse_args(argv)
    if socket.gethostname()!="149":raise RuntimeError("Patient analyses must execute on target host 149")
    config=HeterogeneityConfig(min_train_patients=args.min_train_patients,min_group_patients=args.min_group_patients,
        confidence=args.confidence,ridge_alpha=args.ridge_alpha,covariates=tuple(filter(None,args.nuisance_columns.split(","))))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","training_started":False,
        "stage":"PATIENT_SUBGROUP_STATISTICAL_ANALYSIS","neural_optimizer_started":False},indent=2))
    source_paths={name:getattr(args,name) for name in ["expression","activity","groups","covariates","folds","candidates"]}
    sources={name:_source_record(path) for name,path in source_paths.items()}
    candidates=pd.read_parquet(args.candidates)
    keys=["cancer_id","lncrna_id","pathway_id"]
    if candidates.empty or not set(keys).issubset(candidates) or candidates[keys].isna().any().any() or candidates.duplicated(keys).any():
        raise ValueError("Candidate universe must be non-empty with unique complete keys")
    candidates=candidates[keys].sort_values(keys)
    if not candidates.cancer_id.astype(str).str.fullmatch(r"[A-Z0-9_]+").all():
        raise ValueError("Candidate cancer IDs must be canonical safe partition identifiers")
    summary_writer=None;metrics_writer=None;patient_writer=None
    patient_root=args.output/"patient_oof";patient_root.mkdir()
    all_rows=0;completed=0
    try:
        with (args.output/"FIT_AUDIT.jsonl").open("w",encoding="utf-8") as audits:
            for cancer,local_candidates in candidates.groupby("cancer_id",sort=True):
                data={name:_read_cancer(path,cancer) for name,path in source_paths.items() if name!="candidates"}
                if "lncrna_expression" not in data["expression"] and "logcpm" in data["expression"]:
                    data["expression"]=data["expression"].rename(columns={"logcpm":"lncrna_expression"})
                if args.groups_are_mutation_calls:
                    calls=data["groups"].rename(columns={args.mutation_count_column:"reported_variant_count",
                        args.mutation_assay_column:"mutation_assay_available",args.locus_callable_column:"locus_callable"})
                    data["groups"]=mutation_subgroups(calls,group_id=args.group_id)
                partition=patient_root/f"cancer_id={cancer}";partition.mkdir()
                patient_parts=[];patient_buffer_rows=0
                # Stream per candidate to avoid retaining a candidates x patients cube.
                for candidate in local_candidates.itertuples(index=False):
                    frame=assemble_patient_pair(data["expression"],data["activity"],data["groups"],
                        data["covariates"],data["folds"],cancer=cancer,lncrna=candidate.lncrna_id,
                        pathway=candidate.pathway_id,group_id=args.group_id,config=config)
                    patient,summary,metrics,audit=analyze_patient_heterogeneity(frame,config=config)
                    summary["candidate_selection"]=args.candidate_selection
                    patient_parts.append(patient);patient_buffer_rows+=len(patient)
                    if patient_buffer_rows>=100_000:
                        patient_writer=_flush_patient_frames(patient_parts,patient_writer,partition/"part-0.parquet")
                        patient_buffer_rows=0
                    # Fix optional numeric fields so an unavailable first pair
                    # cannot turn later finite statistics into an Arrow null type.
                    summary["inference_patients"]=summary.get("inference_patients",0)
                    summary["inference_method"]=summary.get("inference_method","UNAVAILABLE")
                    summary["confidence_level"]=config.confidence
                    summary["inference_reason"]=summary.inference_reason.astype("string")
                    for column in summary:
                        if column.endswith(("_slope","_difference","_se","_ci_low","_ci_high","_p_value")):
                            summary[column]=pd.to_numeric(summary[column],errors="raise").astype(float)
                    table=pa.Table.from_pandas(summary,preserve_index=False)
                    metric_table=pa.Table.from_pandas(metrics,preserve_index=False)
                    if summary_writer is None:
                        summary_writer=pq.ParquetWriter(args.output/".INTERACTION_SUMMARY.tmp.parquet",table.schema,compression="zstd")
                        metrics_writer=pq.ParquetWriter(args.output/"SUBGROUP_OOF_METRICS.parquet",metric_table.schema,compression="zstd")
                    summary_writer.write_table(table.select(summary_writer.schema.names).cast(summary_writer.schema))
                    metrics_writer.write_table(metric_table.cast(metrics_writer.schema))
                    audits.write(json.dumps({**candidate._asdict(),"group_id":args.group_id,**audit},allow_nan=False)+"\n")
                    all_rows+=len(patient);completed+=1
                patient_writer=_flush_patient_frames(patient_parts,patient_writer,partition/"part-0.parquet")
                if patient_writer is not None:patient_writer.close();patient_writer=None
    finally:
        if patient_writer is not None:patient_writer.close()
        if summary_writer is not None:summary_writer.close()
        if metrics_writer is not None:metrics_writer.close()
    # Summary is only one row per pair; family-wide BH is computed AFTER every
    # prespecified candidate was assessed, never after filtering significant rows.
    summary=add_heterogeneity_fdr(pd.read_parquet(args.output/".INTERACTION_SUMMARY.tmp.parquet"))
    summary.to_parquet(args.output/"INTERACTION_SUMMARY.parquet",index=False,compression="zstd")
    outputs={name:{"path":str((args.output/name).resolve()),"sha256":sha256(args.output/name)} for name in
             ["INTERACTION_SUMMARY.parquet","SUBGROUP_OOF_METRICS.parquet","FIT_AUDIT.jsonl"]}
    outputs["patient_oof"]=_source_record(patient_root)
    written_rows=sum(pq.ParquetFile(path).metadata.num_rows for path in patient_root.rglob("*.parquet"))
    if written_rows!=all_rows:raise RuntimeError("Patient prediction row-count drift during materialization")
    for name,path in source_paths.items():
        if _source_record(path)!=sources[name]:raise ValueError("Patient analysis source changed during execution: "+name)
    from cc_hhgt.v32 import patient_heterogeneity as implementation
    audit={"format":"PATIENT_SUBGROUP_HETEROGENEITY_V1","status":"PASS","target_host":"149",
        "training_started":False,"statistical_analysis_executed":True,"neural_optimizer_started":False,
        "sources":sources,"outputs":outputs,"config":asdict(config),"candidate_selection":args.candidate_selection,
        "group_source_policy":{"kind":"MUTATION_CALLS" if args.groups_are_mutation_calls else "DECLARED_GROUPS",
            "mutation_count_column":args.mutation_count_column,"mutation_assay_column":args.mutation_assay_column,
            "locus_callable_column":args.locus_callable_column},
        "implementation_sha256":{"scripts/analyze_v32_patient_heterogeneity.py":sha256(Path(__file__)),
            "cc_hhgt/v32/patient_heterogeneity.py":sha256(Path(implementation.__file__))},
        "candidate_rows":completed,"patient_prediction_rows":all_rows,"group_id":args.group_id,
        "inference_available_pairs":int(summary.inference_available.sum()),"heldout_outcomes_used_for_own_fit":False,
        "missing_mutation_assumed_wildtype":False,"causal_claimed":False,"external_validation_claimed":False,
        "patient_predictions_root":str(patient_root.resolve()),"fdr_family":"ALL_DECLARED_CANDIDATES_WITHIN_CANCER_AND_GROUP"}
    (args.output/"AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    (args.output/"SUCCESS.json").write_text(json.dumps({"status":"PASS","audit_sha256":sha256(args.output/"AUDIT.json")}))
    print(json.dumps(audit,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
