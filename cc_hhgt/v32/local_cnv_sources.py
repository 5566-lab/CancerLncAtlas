"""Actual local-CNV preprocessing reads bound to their produced partitions.

This records file exposure and transformations, not per-pair optimizer use or
complete raw biological ancestry. New captured outputs support numerical replay.
"""
from pathlib import Path
import json

from .cnv_input_sources import SourceCapture,require_host,IDS,ARRAYS
from .current_core_export import sha256,verified_path

FORMAT="LOCAL_CNV_TRANSFORMATION_SOURCES_V1"
ROLES={
    "association":{"source_config","covariates","lncrna_expression","pathway_activity","gene_expression",
                   "candidates","patient_map","compact","signed_membership","protein_gene_encoding"},
    "validation":{"source_config","covariates","lncrna_expression","pathway_activity","candidates","patient_map","compact"},
}
OUTPUT_ROLES={
    "pair":ROLES["association"]-{"gene_expression","signed_membership","protein_gene_encoding"},
    "graph":ROLES["association"]-{"pathway_activity","candidates"},
    "validation":ROLES["validation"],
}


def record(path):return {"path":str(Path(path).resolve()),"sha256":sha256(path)}
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight(root,cancer,producer):
    require_host()
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    value={"target_host":"149","training_started":False,"producer":producer,"cancer_id":cancer}
    path=root/("LAUNCH_PREFLIGHT_"+cancer+".json")
    if path.exists() and read(path)!=value:raise ValueError("Local-CNV preflight identity differs")
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


class TransformationCapture(SourceCapture):
    def publish(self,path,*,producer,cancer,outputs,runner):
        require_host()
        records=list(self.records.values())
        if {r["role"] for r in records}!=ROLES[producer]:raise ValueError("Local-CNV consumed input roles are incomplete")
        for item in records:
            if sha256(item["path"])!=item["sha256"]:raise ValueError("Local-CNV input changed before publication")
        from .local_cnv_source_replay import FORMAT as REPLAY_FORMAT, code_paths
        value={"format":FORMAT,"target_host":"149","producer":producer,"cancer_id":cancer,
               "records":records,"outputs":[dict(r) for r in outputs],"training_started":False,
               "numeric_replay_format":REPLAY_FORMAT,
               "exposure_scope":"WHOLE_CONSUMED_FILE_NOT_ALL_ROWS_USED",
               "implementations":[record(p) for p in code_paths(producer)]}
        if Path(runner).resolve()!=code_paths(producer)[0]:raise ValueError("Local-CNV source runner differs from producer")
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError("Local-CNV consumed source receipt is immutable")
        path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
        validate(record(path),path.parent)
        return record(path)


def validate(item,root,*,expected_inputs=None,compact_root=None):
    path=verified_path(item,root);value=read(path)
    producer=value.get("producer")
    if (value.get("format")!=FORMAT or value.get("target_host")!="149" or value.get("training_started") is not False
            or producer not in ROLES or value.get("exposure_scope")!="WHOLE_CONSUMED_FILE_NOT_ALL_ROWS_USED"):
        raise ValueError("Invalid local-CNV transformation source receipt")
    records=value.get("records",[])
    if {r["role"] for r in records}!=ROLES[producer]:raise ValueError("Local-CNV consumed input roles are incomplete")
    seen=set();compact={}
    for source in records:
        actual=verified_path(source,path.parent);scope=source.get("fold",-1)
        key=(str(actual),source["role"],scope)
        if scope not in range(-1,5):raise ValueError("Invalid local-CNV input fold")
        if source.get("upstream_expression"):
            from .formal_expression_sources import validate as validate_expression
            _,expression=validate_expression(source["upstream_expression"],path.parent,actual)
            if expression.get("format")=="FOLD_LOCAL_EXPRESSION_CONVERSION_V1":
                expected_kind={"lncrna_expression":"lncrna","gene_expression":"gene"}.get(source["role"])
                if (expression["fold"]!=scope or expression["cancer_id"]!=value["cancer_id"]
                        or expression["kind"]!=expected_kind):
                    raise ValueError("Local-CNV consumed fold-local expression scope differs")
        if source.get("upstream_activity"):
            from .fold_local_activity import validate as validate_activity
            _,activity=validate_activity(source["upstream_activity"],path.parent,actual)
            if activity["fold"]!=scope or activity["cancer_id"]!=value["cancer_id"] or source["role"]!="pathway_activity":
                raise ValueError("Local-CNV consumed fold-local activity scope differs")
        if key in seen:raise ValueError("Duplicate local-CNV input role/file")
        seen.add(key)
        if source["role"]=="compact":compact[actual.name]=actual
    for role,expected in (expected_inputs or {}).items():
        matches=[r for r in records if r["role"]==role]
        if len(matches)!=1 or Path(matches[0]["path"]).resolve()!=Path(expected).resolve():
            raise ValueError("Local-CNV reused input differs from requested authority: "+role)
    required=set(IDS+ARRAYS["cnv"])
    if not required<=set(compact) or len({p.parent for p in compact.values()})!=1:
        raise ValueError("Local-CNV compact inputs are incomplete")
    if compact_root is not None and {p.parent for p in compact.values()}!={Path(compact_root).resolve()}:
        raise ValueError("Local-CNV reused compact root differs from requested authority")
    if "SUCCESS.json" in compact:
        for name,digest in read(compact["SUCCESS.json"]).get("files",{}).items():
            if name in compact and sha256(compact[name])!=digest:
                raise ValueError("Local-CNV compact materializer hash differs")
    outputs=value.get("outputs",[])
    if not outputs or len({r["path"] for r in outputs})!=len(outputs):raise ValueError("Local-CNV produced partitions are invalid")
    expected_kinds={"pair","graph"} if producer=="association" else {"validation"}
    if (len(outputs)!=5*len(expected_kinds) or {(r.get("output_kind"),r.get("fold")) for r in outputs}
            !={(kind,fold) for kind in expected_kinds for fold in range(5)}):
        raise ValueError("Local-CNV produced partition scope is incomplete")
    for output in outputs:
        if output.get("consumed_roles")!=sorted(OUTPUT_ROLES[output["output_kind"]]):
            raise ValueError("Local-CNV output input roles differ from native transformation")
        verified_path(output,path.parent)
        for role in set(output["consumed_roles"]) & {"lncrna_expression","gene_expression","pathway_activity"}:
            selected=[r for r in records if r["role"]==role and r.get("fold",-1) in {-1,output["fold"]}]
            if len(selected)!=1:raise ValueError("Local-CNV output expression/activity fold coverage differs")
    implementations=value.get("implementations",[])
    # Historical receipts predate the explicit annotation-defined cohort module.
    from .local_cnv_source_replay import FORMAT as REPLAY_FORMAT, code_paths
    if value.get("numeric_replay_format") is not None:
        if (value["numeric_replay_format"]!=REPLAY_FORMAT or len(implementations)!=len(code_paths(producer))
                or {Path(r["path"]).name for r in implementations}!={p.name for p in code_paths(producer)}):
            raise ValueError("Local-CNV numerical replay implementation bindings are incomplete")
    elif len(implementations) not in {7,8}:raise ValueError("Local-CNV implementation bindings are incomplete")
    if len(implementations)==8 and sum(Path(r["path"]).name=="local_cnv_cohort.py" for r in implementations)!=1:
        raise ValueError("Local-CNV cohort implementation binding is incomplete")
    for implementation in value["implementations"]:verified_path(implementation,path.parent)
    return path,value


def partition_binding(path,receipt_path):
    """Use only a native producer's explicit output binding; no adjacent guesses."""
    receipt_path=Path(receipt_path)
    if not receipt_path.exists():return None
    receipt=read(receipt_path)
    item=receipt.get("consumed_sources")
    if item is None:return None  # Historical sources remain unresolved.
    source,value=validate(item,receipt_path.parent)
    actual=Path(path).resolve();digest=sha256(actual)
    matches=[r for r in value["outputs"] if Path(r["path"]).resolve()==actual and r["sha256"]==digest]
    if len(matches)!=1:raise ValueError("Local-CNV source receipt does not bind the consumed partition")
    return {"receipt":record(receipt_path),"sources":record(source),"output":record(actual)}


def inspect(inventory,binding,root,model,role,fold):
    """Follow actual captured tables and compact materialization ancestors."""
    receipt_path=inventory.bind(binding["receipt"],root,model,role+".producer",fold)
    if read(receipt_path).get("consumed_sources")!=binding["sources"]:
        raise ValueError("Local-CNV producer source binding differs")
    path,value=validate(binding["sources"],root)
    output=inventory.bind(binding["output"],root,model,role+".partition",fold)
    matches=[r for r in value["outputs"] if Path(r["path"]).resolve()==output and r["sha256"]==binding["output"]["sha256"]]
    if len(matches)!=1:
        raise ValueError("Local-CNV transformation does not bind its consumed partition")
    inventory.bind(binding["sources"],root,model,role+".source_receipt",fold)
    for index,item in enumerate(value["implementations"]):inventory.bind(item,path.parent,model,role+".implementation."+str(index),fold)
    from .current_cnv_model_use import inspect_bound_sources
    if matches[0]["fold"]!=fold:raise ValueError("Local-CNV consumed output model fold differs")
    consumed=[r for r in value["records"] if r["role"] in matches[0]["consumed_roles"] and r.get("fold",-1) in {-1,fold}]
    biological=[r for r in consumed if r["role"] not in {"source_config","patient_map","candidates"}]
    for r in consumed:inventory.bind(r,path.parent,model,role+".input."+r["role"],fold)
    phase="MODEL_SELECTION" if matches[0]["output_kind"]=="validation" else "FEATURE_CONSTRUCTION"
    inspect_bound_sources(inventory,biological,path.parent,model,role+".upstream",fold,phase=phase)
    from .local_cnv_source_replay import inspect as inspect_replay
    reconstructed = inspect_replay(inventory,binding,root,model,role,fold)
    reason = ("LOCAL_CNV_EARLIER_RAW_ANCESTRY_AND_CROSS_MODULE_PATIENT_IDENTITY_UNVERIFIED" if reconstructed else
              "LOCAL_CNV_ROW_LEVEL_USAGE_AND_EARLIER_RAW_ANCESTRY_UNVERIFIED")
    inventory.gap(model,role,reason,fold,binding["sources"]["sha256"])


def inspect_corrected_graph(inventory,receipt_path,consumed,model,role,fold):
    """Replay E1 row selection, then inspect its actual native association inputs."""
    import pandas as pd
    from .local_cnv_core_preparation import e1_coexpression_frame
    document=read(receipt_path)
    adjustment=document.get("local_cnv_adjustment")
    if adjustment is None:return
    declarations=adjustment.get("source_receipts",[])
    if not declarations:
        inventory.gap(model,role,"LOCAL_CNV_GRAPH_TRANSFORMATION_SOURCES_ABSENT",fold);return
    paths={p.resolve() for p in Path(consumed).rglob("*.parquet")} if Path(consumed).is_dir() else {Path(consumed).resolve()}
    observed=set()
    for index,item in enumerate(declarations):
        path=inventory.bind(item,Path(receipt_path).parent,model,role+".producer."+str(index),fold)
        value=read(path)
        for output in value.get("outputs",[]):
            if output["fold"]!=fold:continue
            current=verified_path(output,path.parent)
            if current not in paths or current in observed:raise ValueError("Corrected graph source partition differs from consumed graph")
            observed.add(current)
            original=verified_path({"path":output["source_path"],"sha256":output["source_sha256"]},path.parent)
            expected=e1_coexpression_frame(pd.read_parquet(original),expected_fold=fold)
            try:pd.testing.assert_frame_equal(pd.read_parquet(current),expected,check_dtype=False,check_exact=True)
            except AssertionError as error:raise ValueError("Corrected graph differs from native E1 transformation") from error
            binding=output.get("upstream_transformation")
            if binding is None:
                inventory.gap(model,role,"LOCAL_CNV_GRAPH_UPSTREAM_SOURCES_ABSENT",fold,output["source_sha256"])
            else:
                if binding["output"]!={"path":str(original),"sha256":output["source_sha256"]}:
                    raise ValueError("Corrected graph upstream partition binding differs")
                inspect(inventory,binding,path.parent,model,role+".transformation."+str(index),fold)
    if observed!=paths:raise ValueError("Corrected graph source partition coverage is incomplete")
