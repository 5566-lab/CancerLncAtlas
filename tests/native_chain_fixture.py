"""Exercise native handoffs with existing explicitly frozen completion fixtures."""
from pathlib import Path
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record


def exercise(kind,manifest,root,complete):
    names=("risk","entity") if kind=="clinical" else ("completed",)
    inputs={"preparation":{"stage_ref":"prepared"},**{n:{"external_ref":n} for n in names}}
    source=root/"chain.json";write(source,{"format":chain.INPUT,"chain_id":kind+"-chain",
        "stages":[{"id":"prepared","operation":kind+".prepare","inputs":read(manifest)},
            {"id":"collected","operation":kind+".collect","inputs":inputs}]})
    output=root/"chain";chain.initialize(source,output);chain.advance(output);state=read(output/"CHAIN.json")
    assert state["status"]=="WAITING_EXTERNAL" and set(state["stages"])=={"prepared"},state
    prepared=Path(state["results"]["prepared"]["path"])
    completed=complete(prepared)
    paths=list(completed) if kind=="clinical" else [completed]
    before={str(p):record(p)["sha256"] for path in paths for p in path.parent.rglob("*") if p.is_file()}
    registry=root/"external.json";write(registry,{"format":chain.EXTERNAL,"inputs":{n:record(p) for n,p in zip(names,paths)}})
    chain.advance(output,registry);state=read(output/"CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    assert before=={str(p):record(p)["sha256"] for path in paths for p in path.parent.rglob("*") if p.is_file()}
    return read(state["results"]["collected"]["path"])
