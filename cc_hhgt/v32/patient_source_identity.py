"""Explicit file-scoped patient identities, private usage and public overlap counts.

Raw identifiers are never guessed across cohorts. A mapping asserts identity;
it does not assert independence, actual optimizer use, or comparable fold schemes.
"""
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .evidence_rule_summary import stable_id,text
from .current_core_export import verified_path

IDENTITY_COLUMNS = {"patient_id","sample_id","donor_id","query_donor_id","canonical_donor_id"}
PRIVATE_COLUMNS = ["model_id","dependency_context","source_file_sha256","source_role","patient_fold_id",
                   "phase","exposure_scope","patient_token"]
AUDIT_COLUMNS = PRIVATE_COLUMNS[:-1]+["observed_identity_count","mapped_identity_count","unmapped_identity_count","status"]
OVERLAP_COLUMNS = ["left_model_id","left_dependency_context","right_model_id","right_dependency_context",
                   "shared_patient_count","interpretation"]


def identities(value):
    values=value if isinstance(value,(list,tuple,np.ndarray)) else [value]
    return {text(v) for v in values if text(v)}


class IdentityMap:
    def __init__(self, declaration=None, root=None):
        self.mapping={};self.namespace=None;self.path=None
        if declaration is None:return
        namespace=declaration.get("namespace")
        if not isinstance(namespace,str) or not namespace or namespace.strip()!=namespace:
            raise ValueError("Patient identity mapping requires an explicit namespace")
        self.path=verified_path(declaration,Path(root));self.namespace=namespace
        from .evidence_rule_sources import read_table
        frame=(read_table(self.path) if self.path.suffix==".parquet" else
               pd.read_csv(self.path,sep="," if ".csv" in self.path.suffixes else "\t",dtype=str))
        columns=["source_file_sha256","identity_column","raw_id","canonical_patient_id"]
        if not set(columns)<=set(frame) or frame.empty or frame[columns].isna().any().any():
            raise ValueError("Patient identity crosswalk lacks file-scoped identities")
        for row in frame[columns].itertuples(index=False,name=None):
            digest,column,raw,canonical=row
            if (any(not isinstance(v,str) or not v or v.strip()!=v for v in row)
                    or not re.fullmatch("[0-9a-f]{64}",digest)):
                raise ValueError("Patient crosswalk identities must be explicit canonical strings")
            key=(digest,column,raw);token=stable_id("PATIENT:",[namespace,canonical])
            if key in self.mapping and self.mapping[key]!=token:
                raise ValueError("Ambiguous file-scoped patient identity mapping")
            self.mapping[key]=token

    def resolve(self,digest,column,raw):return self.mapping.get((digest,column,raw))


def identity_frames(path):
    """Read only identity columns in bounded batches; never load feature matrices."""
    path=Path(path)
    if path.suffix==".parquet":
        import pyarrow.parquet as pq
        file=pq.ParquetFile(path);columns=sorted(set(file.schema_arrow.names)&IDENTITY_COLUMNS)
        if columns:
            for batch in file.iter_batches(batch_size=65536,columns=columns):yield batch.to_pandas()
    else:
        separator="," if ".csv" in path.suffixes else "\t"
        columns=sorted(set(pd.read_csv(path,sep=separator,nrows=0).columns)&IDENTITY_COLUMNS)
        if columns:
            yield from pd.read_csv(path,sep=separator,usecols=columns,dtype=str,chunksize=65536)


class PatientUse:
    def __init__(self, mapper):
        self.mapper=mapper;self.records={}

    def observe(self,frames,*,digest,model,context,role,fold=-1,phase="FEATURE_CONSTRUCTION",scope):
        key=(model,context,digest,role,fold,phase,scope)
        observed,mapped,tokens=self.records.setdefault(key,(set(),set(),set()))
        for frame in frames:
            for column in frame:
                for value in frame[column]:
                    for raw in identities(value):
                        identity=(column,raw);observed.add(identity)
                        token=self.mapper.resolve(digest,column,raw)
                        if token is not None:mapped.add(identity);tokens.add(token)

    def table(self,path,**kwargs):self.observe(identity_frames(path),**kwargs)

    def tables(self):
        usage=[];audit=[]
        for key,(observed,mapped,tokens) in sorted(self.records.items()):
            base=dict(zip(PRIVATE_COLUMNS[:-1],key))
            usage.extend({**base,"patient_token":token} for token in sorted(tokens))
            status="NO_IDENTITY_COLUMNS" if not observed else "MAPPED" if observed==mapped else "IDENTITY_MAPPING_INCOMPLETE"
            audit.append({**base,"observed_identity_count":len(observed),"mapped_identity_count":len(mapped),
                          "unmapped_identity_count":len(observed-mapped),"status":status})
        private=pd.DataFrame(usage,columns=PRIVATE_COLUMNS)
        mapping=pd.DataFrame(audit,columns=AUDIT_COLUMNS)
        groups={key:set(frame.patient_token) for key,frame in private.groupby(["model_id","dependency_context"],sort=True)}
        overlaps=[]
        for (left,a),(right,b) in combinations(groups.items(),2):
            overlaps.append(dict(zip(OVERLAP_COLUMNS,[*left,*right,len(a&b),
                "LOWER_BOUND_FROM_MAPPED_IDENTITIES_NOT_A_FOLD_LEAKAGE_ATTESTATION"])))
        return {"patient_usage_private":private,"patient_mapping":mapping,
                "patient_context_overlap":pd.DataFrame(overlaps,columns=OVERLAP_COLUMNS)}


class ExternalPatientIsolation:
    def __init__(self,declaration,root,native_binding,native_tables):
        self.mapper=IdentityMap(declaration,root)
        native_namespace=native_binding.get("patient_identity_namespace")
        if self.mapper.namespace is not None and self.mapper.namespace!=native_namespace:
            raise ValueError("External patient identity namespace differs from native inventory")
        private=native_tables["patient_usage_private"]
        self.used=set(private.patient_token)
        audit=native_tables["patient_mapping"]
        observed=audit.loc[audit.observed_identity_count.gt(0)]
        self.complete=bool(len(observed) and observed.unmapped_identity_count.eq(0).all())

    def annotate(self,frame,spec,path=None):
        columns=spec.get("patient_identity_columns",sorted(IDENTITY_COLUMNS&set(frame)))
        if (not isinstance(columns,list) or len(columns)!=len(set(columns)) or not set(columns)<=set(frame)):
            raise ValueError("External patient identity columns are invalid")
        identity_frame=frame[columns]
        if columns and path is not None and Path(path).suffix!=".parquet":
            identity_frame=pd.read_csv(path,sep="," if ".csv" in Path(path).suffixes else "\t",usecols=columns,dtype=str)
            if len(identity_frame)!=len(frame):raise ValueError("External patient identity rows differ from observations")
        rows=[]
        for raw in identity_frame.to_dict("records"):
            observed=0;mapped=0;tokens=set()
            for column,value in raw.items():
                for identity in identities(value):
                    observed+=1;token=self.mapper.resolve(spec["table"]["sha256"],column,identity)
                    if token is not None:mapped+=1;tokens.add(token)
            rows.append({"patient_overlap":bool(tokens&self.used),"overlapping_patient_count":len(tokens&self.used),
                "external_patient_identity_complete":bool(observed and observed==mapped),
                "registered_patient_mapping_complete":self.complete,
                "patient_identity_status":"NO_RECORDED_IDENTITIES" if not observed else "MAPPED" if observed==mapped else "MAPPING_INCOMPLETE"})
        # pandas to_dict on an empty-column frame may yield no rows.
        if not columns:
            rows=[{"patient_overlap":False,"overlapping_patient_count":0,"external_patient_identity_complete":False,
                "registered_patient_mapping_complete":self.complete,"patient_identity_status":"NO_RECORDED_IDENTITIES"} for _ in range(len(frame))]
        return pd.DataFrame(rows,index=frame.index)
