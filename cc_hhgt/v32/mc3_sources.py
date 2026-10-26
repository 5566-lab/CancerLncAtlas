"""Raw MC3 event preparation for distal analysis, with explicit interval lift.

MAF identity and sample-observed assay semantics follow the historical v2.7
variants/context producers. Entire intervals must map within one chain block;
endpoint-only bridging of gaps is deliberately not used.
"""
import gzip
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .methylation_recovery import read_patient_folds

REQUIRED=("NCBI_Build","Chromosome","Start_Position","End_Position",
          "Reference_Allele","Tumor_Seq_Allele2","Tumor_Sample_Barcode")
EVENT_SCHEMA=pa.schema([(key,pa.string()) for key in (
    "variant_id","cancer_id","patient_id","sample_id","dna_aliquot_id","chromosome")]+
    [("start",pa.int64()),("end",pa.int64()),("reference",pa.string()),("alternate",pa.string()),
     ("source_reference_build",pa.string()),("source_row",pa.int64())])
COORDINATE_SCHEMA=pa.schema([("variant_id",pa.string()),("chromosome_grch38",pa.string()),
    ("start0_grch38",pa.int64()),("end0_grch38",pa.int64()),("strand_liftover",pa.string()),
    ("liftover_available",pa.bool_()),("failure_reason",pa.string()),("chain_id",pa.int64())])


def patient_authority(path):
    folds=read_patient_folds(Path(path))
    if (folds.empty or not folds.patient_id.str.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}").all()
            or folds.patient_id.duplicated().any() or set(folds.patient_fold_id)!=set(range(5))):
        raise ValueError("MC3 requires unique cross-cancer patients and exact folds 0..4")
    return folds


def chain_blocks(path):
    """Parse UCSC t=hg19 to q=hg38 aligned blocks, retaining reverse strands."""
    rows=[];header=None;ids=set()
    with gzip.open(path,"rt") as handle:
        for line in handle:
            parts=line.split()
            if not parts:continue
            if parts[0]=="chain":
                if header is not None or len(parts)!=13:raise ValueError("Incomplete chain record")
                _,score,tname,tsize,tstrand,tstart,tend,qname,qsize,qstrand,qstart,qend,identity=parts
                tsize,tstart,tend,qsize,qstart,qend,identity=map(int,(tsize,tstart,tend,qsize,qstart,qend,identity))
                if (tstrand!="+" or qstrand not in {"+","-"} or identity in ids
                        or not 0<=tstart<tend<=tsize or not 0<=qstart<qend<=qsize):
                    raise ValueError("Invalid chain coordinate contract")
                ids.add(identity);header=(tname,qname,qsize,qstrand,identity,tend,qend)
                tpos,qpos=tstart,qstart
            else:
                if header is None or len(parts) not in (1,3):raise ValueError("Invalid chain block")
                size=int(parts[0]);dt,dq=(0,0) if len(parts)==1 else map(int,parts[1:])
                tname,qname,qsize,qstrand,identity,tend,qend=header
                if size<1 or min(dt,dq)<0 or tpos+size>tend or qpos+size>qend:raise ValueError("Invalid chain block size")
                rows.append((tname.removeprefix("chr"),tpos,tpos+size,qname.removeprefix("chr"),qpos,qsize,qstrand,identity))
                tpos+=size+dt;qpos+=size+dq
                if len(parts)==1:
                    if (tpos,qpos)!=(tend,qend):raise ValueError("Chain end differs from blocks")
                    header=None
    if header is not None or not rows:raise ValueError("Truncated or empty chain")
    return pd.DataFrame(rows,columns=["chromosome","block_start","block_end","target_chromosome",
        "target_start","target_size","strand","chain_id"])


def lift_variants(variants,blocks,con):
    """Quarantine zero/multiple complete-block matches; no silent truncation."""
    con.register("source_variants",variants);con.register("chain_blocks",blocks)
    result=con.execute("""
      WITH hits AS (
        SELECT v.variant_id,b.target_chromosome,b.strand,b.chain_id,
          CASE WHEN b.strand='+' THEN b.target_start+(v.start-1-b.block_start)
               ELSE b.target_size-(b.target_start+v.end-b.block_start) END AS mapped_start,
          CASE WHEN b.strand='+' THEN b.target_start+(v.end-b.block_start)
               ELSE b.target_size-(b.target_start+v.start-1-b.block_start) END AS mapped_end
        FROM source_variants v JOIN chain_blocks b ON v.chromosome=b.chromosome
          AND v.start-1>=b.block_start AND v.end<=b.block_end
      ), counts AS (
        SELECT variant_id,count(*) AS n FROM hits GROUP BY variant_id
      )
      SELECT v.variant_id, CASE WHEN c.n=1 THEN h.target_chromosome END AS chromosome_grch38,
        CASE WHEN c.n=1 THEN h.mapped_start END AS start0_grch38,
        CASE WHEN c.n=1 THEN h.mapped_end END AS end0_grch38,
        CASE WHEN c.n=1 THEN h.strand END AS strand_liftover,
        coalesce(c.n=1,false) AS liftover_available,
        CASE WHEN c.n=1 THEN NULL WHEN c.n>1 THEN 'ambiguous_complete_block'
             ELSE 'no_complete_aligned_block' END AS failure_reason,
        CASE WHEN c.n=1 THEN h.chain_id END AS chain_id
      FROM source_variants v LEFT JOIN counts c USING(variant_id)
      LEFT JOIN hits h ON v.variant_id=h.variant_id AND c.n=1 ORDER BY v.variant_id
    """).fetchdf()
    return result


def prepare(maf,chain,folds_path,output,*,chunksize=150_000):
    """Stream raw records into the declared cohort; keep source row identities."""
    folds=patient_authority(folds_path);mapping=folds.set_index("patient_id").cancer_id.to_dict()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    writer=pq.ParquetWriter(output/"variant_event.parquet",EVENT_SCHEMA,compression="zstd")
    total=selected=0
    try:
        for chunk in pd.read_csv(maf,sep="\t",comment="#",dtype=str,chunksize=chunksize):
            if not set(REQUIRED)<=set(chunk) or chunk[list(REQUIRED)].isna().any().any():
                raise ValueError("MC3 lacks complete coordinate/sample/allele fields")
            if not chunk.NCBI_Build.eq("GRCh37").all():raise ValueError("MC3 source build must be GRCh37")
            start=pd.to_numeric(chunk.Start_Position,errors="raise");end=pd.to_numeric(chunk.End_Position,errors="raise")
            if (not np.isfinite(start).all() or not np.isfinite(end).all() or not start.eq(np.floor(start)).all()
                    or not end.eq(np.floor(end)).all() or start.lt(1).any() or end.lt(start).any()):
                raise ValueError("Invalid MAF 1-based inclusive interval")
            aliquot=chunk.Tumor_Sample_Barcode.str.strip().str.upper()
            if not aliquot.str.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z0-9](?:-[A-Z0-9]+)*").all():
                raise ValueError("Invalid TCGA tumor aliquot")
            patient=aliquot.str[:12];chromosome=chunk.Chromosome.str.removeprefix("chr")
            if chromosome.str.strip().eq("").any():raise ValueError("Empty MC3 chromosome")
            # Retain the v2.7 genomic/allele identity; canonical integers make
            # identities invariant to chunk boundaries and numeric formatting.
            key=chromosome+":"+start.astype("int64").astype(str)+":"+end.astype("int64").astype(str)+":"+chunk.Reference_Allele+":"+chunk.Tumor_Seq_Allele2
            identities={k:"VAR:"+hashlib.sha256(k.encode()).hexdigest()[:20] for k in key.unique()}
            frame=pd.DataFrame({"variant_id":key.map(identities),"cancer_id":patient.map(mapping),"patient_id":patient,
                "sample_id":aliquot.str[:16],"dna_aliquot_id":aliquot,"chromosome":chromosome,
                "start":start.astype("int64"),"end":end.astype("int64"),"reference":chunk.Reference_Allele,
                "alternate":chunk.Tumor_Seq_Allele2,"source_reference_build":"GRCh37",
                "source_row":np.arange(total,total+len(chunk),dtype=np.int64)})
            total+=len(chunk);frame=frame.loc[frame.cancer_id.notna()];selected+=len(frame)
            writer.write_table(pa.Table.from_pandas(frame,schema=EVENT_SCHEMA,preserve_index=False))
    finally:writer.close()
    if not selected:raise ValueError("No MC3 events in declared patient authority")
    blocks=chain_blocks(chain)
    with duckdb.connect() as con:
        con.execute("PRAGMA threads=8");con.execute("PRAGMA memory_limit='8GB'")
        source=str((output/"variant_event.parquet").resolve()).replace("'","''")
        variants=con.execute(f"SELECT DISTINCT variant_id,chromosome,start,\"end\" FROM read_parquet('{source}') ORDER BY variant_id").fetchdf()
        if variants.variant_id.duplicated().any():raise ValueError("Conflicting MC3 variant identity")
        lifted=lift_variants(variants,blocks,con)
        pq.write_table(pa.Table.from_pandas(lifted,schema=COORDINATE_SCHEMA,preserve_index=False),
                       output/"variant_coordinate_grch38.parquet",compression="zstd")
        observed=con.execute(f"SELECT DISTINCT patient_id FROM read_parquet('{source}')").fetchdf().patient_id
    crosswalk=folds.assign(mutation_assay_available=folds.patient_id.isin(observed),locus_coverage_available=False)
    crosswalk.to_parquet(output/"mutation_sample_crosswalk.parquet",index=False,compression="zstd")
    return {"raw_rows":total,"selected_event_rows":selected,"outside_declared_patient_rows":total-selected,
            "unique_variants":len(variants),"mapped_variants":int(lifted.liftover_available.sum()),
            "quarantined_variants":int((~lifted.liftover_available).sum()),"formal_patients":len(folds),
            "assayed_patients":int(crosswalk.mutation_assay_available.sum()),
            "liftover_policy":"UNIQUE_COMPLETE_ALIGNED_BLOCK_GRCH37_TO_GRCH38",
            "assay_policy":"PATIENT_OBSERVED_IN_MAF_NOT_INDEPENDENT_ASSAY_COVERAGE",
            "all_maf_filter_values_retained":True,"locus_callability_claimed":False}
