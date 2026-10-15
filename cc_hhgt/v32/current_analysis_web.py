"""Publish/query one selected analysis and its actual version-bound contexts.

Registration never fits a model. A missing component has an explicit status;
a configured but inconsistent component is an error, never a legacy fallback.
Only registered public tables are served; input manifests/checkpoints are not
download endpoints. Evidence provenance retains source hashes without paths.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import socket

import pyarrow as pa
import pyarrow.parquet as pq

from .current_core_export import sha256, verified_path
from .drug_sparse_query import _bounded_duckdb, _parquet_file_relation
from .current_drug_sources import same_relation
from .selected_genesets import SelectedAnalysisQuery

INPUT_FORMAT = "CURRENT_ANALYSIS_WEB_INPUT_V1"
FORMAT = "CURRENT_ANALYSIS_WEB_RELEASE_V1"
COMPONENTS = ("state","clinical","drug","evidence","external")
STATUSES = {"NOT_PRODUCED","SOURCE_UNAVAILABLE","NOT_APPLICABLE"}
LABELS = {"selected":"GeneSet 与跨癌种分型","scores":"完整候选评分","state":"细胞状态",
    "clinical":"临床关联","drug":"药物","evidence":"规则证据","external":"外部评估"}
NOTES = {
    "selected":"基因集和分型由本版本选定评分生成。分型描述癌种间的基因集关系。",
    "scores":"方向来自主模型；若选定的是融合评分，方向仍保留其原始来源。",
    "state":"状态概率与实测状态相关性分开呈现。基因集汇总是成员描述性统计。",
    "clinical":"保留临床终点、事件数、方向和缺失原因；成员汇总不代表联合基因集生存效应。",
    "drug":"模型概率表示关联强度；实测药敏方向、剂量变化与模型消融分别呈现。缺失预测不填零。",
    "evidence":"保留实验背景、来源及方向冲突。规则汇总不代表因果概率。",
    "external":"展示实测指标及隔离范围。已登记来源的隔离尚不证明模型使用来源清单完整。",
}
FILTERS = {"cancer_id","cancer_a","cancer_b","lncrna_id","pathway_id","pathway_family_id","geneset_id",
    "association_direction","classification_status","state_id","gene_set_id","direction","state_association_direction",
    "pathway_association_direction","clinical_endpoint","subject_id","subject_type","clinical_effect_direction",
    "drug_id","gene_id","subtype_system","subtype_id","ablation_kind","ablation_entity_id","response_metric",
    "response_unit","dose_molar_key","low_dose_molar_key","high_dose_molar_key","exposure_hours_key",
    "source_id","source_group","endpoint","isolation_scope","model_id","baseline_model_id","metric","route_type"}


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def _no_paths(value):
    if isinstance(value,dict):
        return {k:_no_paths(v) for k,v in value.items() if k not in {"path","paths","source_repo_relative_path"}
            and not k.endswith(("_path","_paths"))}
    if isinstance(value,list): return [_no_paths(v) for v in value]
    return value


def public_evidence_file(source,target):
    """Stream the provenance projection; never rewrite its native source."""
    reader = pq.ParquetFile(source); writer = None
    try:
        for batch in reader.iter_batches(batch_size=25000):
            if "provenance_json" in batch.schema.names:
                index = batch.schema.get_field_index("provenance_json")
                values = [json.dumps(_no_paths(json.loads(v)),sort_keys=True,ensure_ascii=False) if v is not None else None
                    for v in batch.column(index).to_pylist()]
                table = pa.Table.from_batches([batch]).set_column(index,"provenance_json",pa.array(values,type=pa.string()))
            else: table = pa.Table.from_batches([batch])
            if writer is None: writer = pq.ParquetWriter(target,table.schema,compression="zstd")
            writer.write_table(table)
        if writer is None: pq.write_table(pa.Table.from_batches([],schema=reader.schema_arrow),target,compression="zstd")
    finally:
        if writer is not None: writer.close()


def source_tables(manifest_path):
    path = Path(manifest_path).resolve(); manifest = read_json(path)
    if (manifest.get("format")!=INPUT_FORMAT or not isinstance(manifest.get("release_id"),str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",manifest["release_id"])
            or set(manifest.get("components",{}))!=set(COMPONENTS)):
        raise ValueError("Current web registration requires a release ID and every component status")
    selected_path = verified_path(manifest["selected_analysis"],path.parent)
    selected_sha = manifest["selected_analysis"]["sha256"]
    selected = SelectedAnalysisQuery(selected_path,expected_sha=selected_sha)
    score = read_json(selected.score_binding_path)
    tables = {}; bindings = {"selected":{"path":str(selected_path),"sha256":selected_sha}}; state = {}

    def add(module,name,records,*,binding_sha,projection=False):
        if not re.fullmatch(r"[a-z][a-z0-9_]*",name): raise ValueError("Invalid public table ID")
        normalized = []
        for record in records:
            source = verified_path(record,path.parent)
            normalized.append({"path":str(source),"sha256":record["sha256"]})
        if not normalized: return
        tables[module+"."+name] = {"component":module,"name":name,"files":normalized,
            "source_binding_sha256":binding_sha,"project_evidence_provenance":projection}

    for name in selected.paths:
        record = selected.download(name); add("selected",name,[record],binding_sha=selected_sha)
    add("scores","candidates",[score["scores"]],binding_sha=sha256(selected.score_binding_path))
    state["selected"] = state["scores"] = {"status":"AVAILABLE","reason":None}
    selected_input_path = verified_path(selected.binding["input_manifest"],selected_path.parent)
    selected_input = read_json(selected_input_path)
    primary_path = verified_path(score["sources"]["reference_exact_producer"]["producer_binding"],selected.score_binding_path.parent)
    primary = read_json(primary_path)
    for module,spec in manifest["components"].items():
        if spec.get("status") in STATUSES:
            if (set(spec)!={"status","reason"} or not isinstance(spec["reason"],str) or not spec["reason"].strip()):
                raise ValueError("Unavailable current component needs a reason and no hidden binding")
            state[module] = dict(spec); continue
        if spec.get("status")!="AVAILABLE" or set(spec)!={"status","binding"}:
            raise ValueError("Available current component requires its exact binding")
        source = verified_path(spec["binding"],path.parent); digest=spec["binding"]["sha256"]
        bindings[module] = {"path":str(source),"sha256":digest}
        if module in {"state","clinical","drug"}:
            if module=="state":
                from .current_state_context import CurrentStateContextQuery
                reader = CurrentStateContextQuery(source,expected_sha=digest,selected_analysis_sha=selected_sha)
            elif module=="clinical":
                from .current_clinical_context import CurrentClinicalContextQuery
                reader = CurrentClinicalContextQuery(source,expected_sha=digest,selected_analysis_sha=selected_sha)
            else:
                from .current_drug_context import CurrentDrugContextQuery
                reader = CurrentDrugContextQuery(source,expected_sha=digest,selected_analysis_sha256=selected_sha)
            for name in reader.paths: add(module,name,[reader.download(name)],binding_sha=digest)
        elif module=="evidence":
            from .evidence_rule_release import RuleEvidenceQuery
            reader = RuleEvidenceQuery(source,expected_sha=digest)
            if selected_input.get("rule_evidence_binding") and selected_input["rule_evidence_binding"]["sha256"]!=digest:
                raise ValueError("Web evidence differs from the selected score's evidence annotations")
            evidence_input_path = verified_path(reader.binding["input_manifest"],source.parent)
            evidence_input = read_json(evidence_input_path)
            left = verified_path(primary["input_artifacts"][0],primary_path.parent)
            right = verified_path(evidence_input["candidates"],evidence_input_path.parent)
            same_relation("SELECT cancer_id,lncrna_id,pathway_id FROM "+_parquet_file_relation([left]),
                          "SELECT cancer_id,lncrna_id,pathway_id FROM "+_parquet_file_relation([right]))
            for name,record in reader.binding["outputs"].items(): add(module,name,[record],binding_sha=digest,projection=True)
            add(module,"events",reader.binding["event_parts"],binding_sha=digest,projection=True)
            add(module,"contexts",reader.binding["context_parts"],binding_sha=digest,projection=True)
        else:
            from .external_validation_current import CurrentExternalValidationQuery
            reader = CurrentExternalValidationQuery(source,expected_sha=digest)
            evaluated = [r for r in reader.binding["models"] if r["model_id"]==reader.binding["evaluation"]["primary_model"]]
            chosen = score["sources"]["selected_producer"]
            same_snapshot = len(evaluated)==1 and evaluated[0]["producer_binding"]["sha256"]==sha256(selected.score_binding_path) and evaluated[0]["score_column"]=="selected_score"
            same_producer = len(evaluated)==1 and evaluated[0]["producer_binding"]["sha256"]==chosen["producer_binding"]["sha256"] and evaluated[0]["score_column"]==chosen["score_column"]
            if not (same_snapshot or same_producer): raise ValueError("External primary metrics belong to a different selected model")
            for name,record in reader.binding["outputs"].items(): add(module,name,[record],binding_sha=digest)
        state[module] = {"status":"AVAILABLE","reason":None}
    return manifest,selected,score,tables,state,bindings


def materialize_current_analysis_web(manifest_path,output):
    if socket.gethostname()!="149": raise RuntimeError("Current web preparation requires target host 149")
    path=Path(manifest_path).resolve(); digest=sha256(path); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    write_json(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"input_manifest_sha256":digest})
    manifest,selected,score,tables,state,bindings=source_tables(path)
    for identity,table in tables.items():
        public=[]; total=0; columns=None
        for index,record in enumerate(table["files"]):
            source=verified_path(record,path.parent)
            if table["project_evidence_provenance"]:
                target=output/(identity+f".{index:05d}.parquet"); public_evidence_file(source,target)
            else: target=source
            if target.suffix==".gmt": kind="gmt"; rows=None; names=[]
            else:
                kind="parquet"; parquet=pq.ParquetFile(target); rows=parquet.metadata.num_rows; names=parquet.schema_arrow.names
                if columns is not None and columns!=names: raise ValueError("Public table parts have different schemas")
                if {"patient_id","canonical_model_id","cell_line_id","response_value"}&set(names):
                    raise ValueError("Individual-level measurements are not current public web tables")
                columns=names; total+=rows
            public.append({"path":str(target),"sha256":sha256(target),"rows":rows,"bytes":target.stat().st_size})
            verified_path(record,path.parent)
        table.update(files=public,kind=kind,rows=total if kind=="parquet" else None,columns=columns or [],
            filters=sorted(set(columns or [])&FILTERS),source_files=table["files"])
        table.pop("project_evidence_provenance")
    if sha256(path)!=digest: raise ValueError("Current web registration changed during preparation")
    for record in bindings.values(): verified_path(record,path.parent)
    for table in tables.values():
        for record in table["source_files"]: verified_path(record,path.parent)
    implementation = {str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2])):sha256(Path(__file__))}
    binding={"format":FORMAT,"status":"PASS","target_host":"149","release_id":manifest["release_id"],
        "selected_analysis_sha256":manifest["selected_analysis"]["sha256"],"selected_score_run_id":selected.binding["selected_score_run_id"],
        "selected_training_run_id":score["training_run_id"],"input_manifest":{"path":str(path),"sha256":digest},
        "component_status":state,"component_bindings":bindings,"tables":tables,
        "score_scientific_status":score["sources"]["source_scientific_status"],
        "score_performance_outcome":score["sources"]["source_performance_outcome"],
        "implementation_sha256":implementation,
        "independent_external_validation_attested":False,"training_started":False,"production_deployed":False}
    target=output/"CURRENT_WEB_BINDING.json"; write_json(target,binding)
    result={"status":"PASS","binding_path":str(target),"binding_sha256":sha256(target),"training_started":False,"target_host":"149"}
    write_json(output/"SUCCESS.json",result)
    try:
        CurrentAnalysisWebStore(target,expected_sha=result["binding_sha256"])
    except Exception:
        (output/"SUCCESS.json").unlink()
        raise
    return result


class CurrentWebQueryError(ValueError): pass
class CurrentWebVersionError(ValueError): pass


class CurrentAnalysisWebStore:
    """No training assets or full graph are loaded by the online reader."""
    def __init__(self,binding_path,*,expected_sha):
        self.path=verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()); self.sha=expected_sha
        self.binding=read_json(self.path)
        b=self.binding
        if (b.get("format")!=FORMAT or b.get("status")!="PASS" or b.get("target_host")!="149"
                or b.get("training_started") is not False or b.get("production_deployed") is not False
                or b.get("independent_external_validation_attested") is not False): raise ValueError("Invalid current web release")
        success=read_json(self.path.parent/"SUCCESS.json")
        if success.get("status")!="PASS" or success.get("binding_sha256")!=expected_sha: raise ValueError("Incomplete current web release")
        for table in b["tables"].values():
            for record in table["files"]: verified_path(record,self.path.parent)

    def version(self,expected):
        if expected!=self.sha: raise CurrentWebVersionError("The requested analysis version differs; reload its catalog")
        # A live process cannot silently switch to a rewritten registration.
        verified_path({"path":str(self.path),"sha256":self.sha},Path.cwd())

    def catalog(self):
        self.version(self.sha)
        tables=[]
        for identity,table in self.binding["tables"].items():
            tables.append({k:table[k] for k in ("component","name","kind","rows","columns","filters","source_binding_sha256")}|
                {"id":identity,"parts":[{"part":i,"sha256":p["sha256"],"bytes":p["bytes"],"rows":p["rows"]} for i,p in enumerate(table["files"])]})
        return {"release_id":self.binding["release_id"],"version":self.sha,
            "selected_analysis_sha256":self.binding["selected_analysis_sha256"],"selected_score_run_id":self.binding["selected_score_run_id"],
            "score_scientific_status":self.binding["score_scientific_status"],"score_performance_outcome":self.binding["score_performance_outcome"],
            "independent_external_validation_attested":False,
            "components":{key:{**value,"label":LABELS[key],"description":NOTES[key]} for key,value in self.binding["component_status"].items()},
            "tables":tables}

    def table(self,identity):
        if identity not in self.binding["tables"]: raise KeyError(identity)
        return self.binding["tables"][identity]

    def download(self,identity,part,*,version):
        self.version(version); table=self.table(identity)
        if type(part) is not int or not 0<=part<len(table["files"]): raise KeyError("Unknown table part")
        record=table["files"][part]; path=verified_path(record,self.path.parent)
        suffix="."+table["kind"]
        name=identity.removesuffix(suffix)+(f".part-{part:05d}" if len(table["files"])>1 else "")+suffix
        return {**record,"path":str(path),"filename":name,
            "version":self.sha,"selected_analysis_sha256":self.binding["selected_analysis_sha256"]}

    def query(self,identity,*,version,filters=None,limit=100,offset=0):
        self.version(version); table=self.table(identity)
        if table["kind"]!="parquet": raise CurrentWebQueryError("This artifact is a download, not a query table")
        if type(limit) is not int or type(offset) is not int or not 1<=limit<=1000 or not 0<=offset<=1000000:
            raise CurrentWebQueryError("Invalid pagination")
        filters=filters or {}
        if not set(filters).issubset(table["filters"]) or not all(isinstance(v,str) for v in filters.values()):
            raise CurrentWebQueryError("Filters must be exact identifiers from this table's filter list")
        paths=[Path(self.download(identity,i,version=version)["path"]) for i in range(len(table["files"]))]
        relation=_parquet_file_relation(paths); where=" AND ".join('"'+key+'"=?' for key in filters) or "true"
        order=','.join('"'+key.replace('"','""')+'"' for key in table["columns"])
        with _bounded_duckdb() as con:
            rows=con.execute(f"SELECT * FROM {relation} WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [*filters.values(),limit+1,offset]).df()
        has_more=len(rows)>limit
        # JSON conversion preserves nulls and native evidence JSON as strings.
        return {"version":self.sha,"selected_analysis_sha256":self.binding["selected_analysis_sha256"],"table":identity,
            "offset":offset,"limit":limit,"has_more":has_more,"rows":json.loads(rows.iloc[:limit].to_json(orient="records",double_precision=15))}

    def drug_members(self,*,geneset_id,drug_id,version,limit=100,offset=0):
        self.version(version)
        if self.binding["component_status"]["drug"]["status"]!="AVAILABLE": raise KeyError("Drug context is not available")
        from .current_drug_context import CurrentDrugContextQuery
        record=self.binding["component_bindings"]["drug"]; source=verified_path(record,self.path.parent)
        reader=CurrentDrugContextQuery(source,expected_sha=record["sha256"],selected_analysis_sha256=self.binding["selected_analysis_sha256"])
        try: result=reader.member_predictions(geneset_id=geneset_id,drug_id=drug_id,limit=limit,offset=offset)
        except ValueError as error:
            if "pagination" in str(error) or "identity" in str(error): raise CurrentWebQueryError(str(error)) from error
            raise
        return {**result,"version":self.sha}
