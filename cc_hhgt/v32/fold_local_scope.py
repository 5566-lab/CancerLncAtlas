"""Annotation-defined candidates and training-only cross-cancer detection scope.

Candidate membership never uses measured detection. Each fold's scope uses
only that fold's selected training tumors, including when defining 'shared'.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .cnv_input_sources import require_host
from .formal_expression_sources import read, write, record, read_matrix
from .current_core_export import verified_path
from .fold_local_expression import native as normalization

INPUT = "FOLD_LOCAL_SCOPE_INPUT_V1"
FORMAT = "FOLD_LOCAL_SCOPE_RELEASE_V1"
SCOPE = "TRAIN_PATIENT_DETECTION_SCOPE_V1"
CANDIDATES = "ANNOTATION_DEFINED_CANDIDATES_V1"
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


def annotation_ids(item, root):
    from .native_annotation_workflow import annotation_table
    frame = annotation_table(item, root)
    if not {"ensembl_gene_id", "gene_type"} <= set(frame): raise ValueError("Incomplete lncRNA annotation")
    selected = frame.loc[frame.gene_type.eq("lncRNA")].copy()
    selected = selected.loc[~selected.ensembl_gene_id.str.endswith("_PAR_Y", na=False)]
    ids = selected.ensembl_gene_id.str.replace(r"\..*$", "", regex=True)
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any() or not len(ids):
        raise ValueError("Empty/ambiguous annotated lncRNA universe")
    return sorted("LNC:"+ids)


def pathway_ids(item, root):
    from .fold_local_activity import gene_sets
    path = verified_path(item, root); manifest = read(path)
    sources = {entry["collection"]:entry["file"] for entry in manifest.get("collections", [])}
    sets, _ = gene_sets({"inputs":{"gmt_manifest":item}, "gmt_sources":sources}, root)
    # Apply no measured expression, correlation, pathway activity or coverage
    # filter here. Unusable pathways retain unavailable rows downstream.
    return sorted(sets)


def candidate_frame(value, root):
    from scripts.prepare_v32_formal import make_candidate_pairs
    cancers = value["cancers"]
    if not cancers or cancers != sorted(set(cancers)): raise ValueError("Invalid declared cancer universe")
    ids = annotation_ids(value["annotation"], root); pathways = pathway_ids(value["gmt_manifest"], root)
    budget = value["budget_per_cancer"]
    if type(budget) is not int or not 0 < budget <= 250000: raise ValueError("Candidate budget exceeds frozen evaluation cap")
    grid = pd.MultiIndex.from_product([cancers,ids], names=["cancer_id","lncrna_id"]).to_frame(index=False)
    # Eligibility here means annotation membership only. Detection is absent.
    grid["within_cancer_eligible"] = True
    return make_candidate_pairs(grid,pathways,budget=budget,seed=value["seed"])


def validate_candidates(item, root, output=None, *, replay=False):
    path = verified_path(item, root); value = read(path)
    if (value.get("format") != CANDIDATES or value.get("target_host") != "149"
            or value.get("training_started") is not False or value.get("measured_detection_used") is not False
            or len(value.get("implementations", [])) != 3): raise ValueError("Invalid annotation-defined candidate receipt")
    target = verified_path(value["output"], path.parent)
    if output is not None and target != Path(output).resolve(): raise ValueError("Candidate receipt binds a different output")
    for source in [value["annotation"],value["gmt_manifest"],*value["implementations"]]: verified_path(source,path.parent)
    pathway_ids(value["gmt_manifest"],path.parent)
    if replay:
        expected = candidate_frame(value,path.parent)
        try: pd.testing.assert_frame_equal(pd.read_parquet(target),expected,check_exact=True,check_dtype=False)
        except AssertionError as error: raise ValueError("Candidates differ from annotation-only sampling") from error
    return path,value


def candidate_binding(output):
    path = Path(output).parent/"CANDIDATE_SOURCES.json"
    if Path(output).name != "candidates.parquet" or not path.is_file(): return None
    item=record(path); validate_candidates(item,path.parent,output); return item


def aggregate_scope(rates, *, fold, cancers, ids):
    required={"cancer_id","lncrna_id","detection_rate","training_tumor_n"}
    if not required <= set(rates): raise ValueError("Incomplete training detection table")
    frame=rates.copy().sort_values(["cancer_id","lncrna_id"],kind="stable").reset_index(drop=True)
    expected=pd.MultiIndex.from_product([cancers,ids],names=["cancer_id","lncrna_id"])
    observed=pd.MultiIndex.from_frame(frame[["cancer_id","lncrna_id"]])
    if (not observed.equals(expected) or not frame.detection_rate.between(0,1).all()
            or not frame.training_tumor_n.gt(0).all()): raise ValueError("Training scope must cover every declared cancer/lncRNA")
    frame["within_cancer_eligible"]=frame.detection_rate.ge(.10)
    counts=frame.groupby("lncrna_id",observed=True).within_cancer_eligible.sum()
    frame["detected_cancers"]=frame.lncrna_id.map(counts).astype(int)
    frame["assessed_cancers"]=len(cancers)
    frame["shared_or_local_scope"]=np.select(
        [frame.within_cancer_eligible & frame.detected_cancers.ge(3),frame.within_cancer_eligible],
        ["shared","cancer_local"],default="not_expression_eligible")
    frame["fold_id"]=fold;frame["fit_split"]="train"
    return frame


def scope_frame(value, root):
    cpath, candidate=validate_candidates(value["candidates"],root)
    ids=annotation_ids(candidate["annotation"],cpath.parent);cancers=candidate["cancers"]
    if set(value["normalizations"]) != set(cancers): raise ValueError("Scope normalization cancer coverage differs")
    pieces=[];patient_maps=set()
    for cancer in cancers:
        path,norm,samples=normalization(value["normalizations"][cancer],root)
        if norm["cancer_id"]!=cancer or norm["fold"]!=value["fold"]: raise ValueError("Scope normalization cancer/fold differs")
        patient_maps.add(str(verified_path(norm["inputs"]["patient_map"],path.parent)))
        matrix=read_matrix(verified_path(norm["outputs"]["lncrna"],path.parent))
        if sorted(matrix.index)!=ids: raise ValueError("Annotated lncRNA universe differs from unfiltered count rows")
        train=samples.loc[samples.exported & samples.split_role.eq("train"),"sample_id"].tolist()
        expected=samples.loc[samples.exported,"sample_id"].tolist()
        if list(matrix.columns)!=expected or not train or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Invalid scope training expression identity/values")
        rates=matrix.loc[:,train].gt(0).mean(axis=1)
        pieces.append(pd.DataFrame({"cancer_id":cancer,"lncrna_id":matrix.index,
                                   "detection_rate":rates.to_numpy(),"training_tumor_n":len(train)}))
    if len(patient_maps)!=1: raise ValueError("Cross-cancer scope must use one patient map")
    return aggregate_scope(pd.concat(pieces,ignore_index=True),fold=value["fold"],cancers=cancers,ids=ids)


def materialize(manifest, output, *, runner):
    require_host();manifest=Path(manifest).resolve(strict=True);source=record(manifest);spec=read(manifest)
    if spec.get("format")!=INPUT or not spec.get("normalizations"): raise ValueError("Invalid fold-local scope input")
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Use a new fold-local scope output")
    cancers=sorted(spec["normalizations"])
    successes={}
    for cancer,item in spec["normalizations"].items():
        path=verified_path(item,manifest.parent);value=read(path)
        if (value.get("status")!="PASS" or value.get("target_host")!="149" or value.get("training_started") is not False
                or value.get("cancer_id")!=cancer or set(value.get("folds",{}))!=set(map(str,range(5)))):
            raise ValueError("Incomplete cancer normalization source")
        successes[cancer]=value
    candidate={"format":CANDIDATES,"target_host":"149","training_started":False,"measured_detection_used":False,
        "annotation":spec["annotation"],"gmt_manifest":spec["gmt_manifest"],"cancers":cancers,
        "budget_per_cancer":spec.get("budget_per_cancer",100000),"seed":spec.get("seed",20260726),
        "implementations":[record(__file__),record(runner),record(Path(__file__).resolve().parents[2]/"scripts/prepare_v32_formal.py")]}
    # Resolve source references before relocating their receipt to the output.
    for key in ("annotation","gmt_manifest"):candidate[key]=record(verified_path(candidate[key],manifest.parent))
    candidates=candidate_frame(candidate,manifest.parent)
    output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"operation":"FOLD_LOCAL_SCOPE"})
    candidates.to_parquet(output/"candidates.parquet",index=False)
    candidate["output"]=record(output/"candidates.parquet")
    write(output/"CANDIDATE_SOURCES.json",candidate);candidate_item=record(output/"CANDIDATE_SOURCES.json")
    folds={}
    for fold in range(5):
        value={"format":SCOPE,"target_host":"149","training_started":False,"fold":fold,"candidates":candidate_item,
               "normalizations":{c:s["folds"][str(fold)] for c,s in successes.items()},
               "normalization_releases":{c:record(verified_path(item,manifest.parent)) for c,item in spec["normalizations"].items()},
               "implementations":[record(__file__),record(runner)]}
        frame=scope_frame(value,manifest.parent)
        directory=output/f"fold={fold}";directory.mkdir()
        target=directory/"scope.parquet";frame.to_parquet(target,index=False)
        value["output"]=record(target);path=directory/"SCOPE_SOURCES.json";write(path,value)
        folds[str(fold)]=record(path)
    if record(manifest)!=source: raise ValueError("Scope input manifest changed during processing")
    result={"format":FORMAT,"target_host":"149","training_started":False,"status":"PASS",
            "input":source,"candidates":candidate_item,"folds":folds}
    path=output/"SCOPE_RELEASE.json";write(path,result);return path


def validate_scope(item, root, output=None, *, replay=False):
    path=verified_path(item,root);value=read(path)
    if (value.get("format")!=SCOPE or value.get("target_host")!="149" or value.get("training_started") is not False
            or value.get("fold") not in range(5) or len(value.get("implementations",[]))!=2):
        raise ValueError("Invalid fold-local detection scope")
    target=verified_path(value["output"],path.parent)
    if output is not None and target!=Path(output).resolve(): raise ValueError("Scope binds a different output")
    for source in value["implementations"]:verified_path(source,path.parent)
    _,candidate=validate_candidates(value["candidates"],path.parent)
    if set(value.get("normalizations",{}))!=set(candidate["cancers"]): raise ValueError("Scope cancer coverage differs")
    for cancer,source in value["normalizations"].items():
        _,norm,_=normalization(source,path.parent)
        if norm["cancer_id"]!=cancer or norm["fold"]!=value["fold"]: raise ValueError("Scope normalization fold/cancer differs")
        if "normalization_releases" in value:
            from .normalization_source_replay import declared_release
            if set(value["normalization_releases"])!=set(value["normalizations"]):
                raise ValueError("Scope normalization release cancer coverage differs")
            _,release=declared_release(value["normalization_releases"][cancer],path.parent)
            if release.get("cancer_id")!=cancer or release["folds"][str(value["fold"])]["sha256"]!=source["sha256"]:
                raise ValueError("Scope normalization differs from its completed release")
    if replay:
        try:pd.testing.assert_frame_equal(pd.read_parquet(target),scope_frame(value,path.parent),check_exact=True,check_dtype=False)
        except AssertionError as error: raise ValueError("Scope differs from training-tumor detection") from error
    return path,value


def scope_binding(output):
    path=Path(output).parent/"SCOPE_SOURCES.json"
    if Path(output).name!="scope.parquet" or not path.is_file():return None
    item=record(path);validate_scope(item,path.parent,output);return item


def release(item, root, *, candidates, patient_map):
    path=verified_path(item,root);value=read(path)
    if (value.get("format")!=FORMAT or value.get("status")!="PASS" or value.get("target_host")!="149"
            or value.get("training_started") is not False or set(value.get("folds",{}))!=set(map(str,range(5)))):
        raise ValueError("Incomplete fold-local scope release")
    validate_candidates(value["candidates"],path.parent,candidates,replay=True)
    for fold,source in value["folds"].items():
        spath,scope=validate_scope(source,path.parent)
        if scope["fold"]!=int(fold) or scope["candidates"]!=value["candidates"]: raise ValueError("Scope release fold/candidate differs")
        for norm_source in scope["normalizations"].values():
            npath,norm,_=normalization(norm_source,spath.parent)
            if verified_path(norm["inputs"]["patient_map"],npath.parent)!=Path(patient_map).resolve():
                raise ValueError("Scope patient map differs from analysis")
    return path,value


def scope_input(release_value, root, fold, *, capture=None):
    item=release_value["folds"][str(fold)]
    path,value=validate_scope(item,root,replay=True);target=verified_path(value["output"],path.parent)
    frame=(pd.read_parquet(target) if capture is None else
           capture.read(target,pd.read_parquet,role="detection",fold=fold))
    return frame,{**record(target),"upstream_scope":item}


def verify_association_normalizations(capture, scope_item, root, fold):
    """Prevent a new scope from being combined with old/cohort associations."""
    path,value=validate_scope(scope_item,root)
    from .local_cnv_sources import validate as validate_local_cnv
    from .formal_expression_sources import validate as validate_expression
    observed=set()
    for source in capture.reads:
        if source["role"] not in {"train_associations","validation_associations"} or source["fold"]!=fold:continue
        transformation=source.get("upstream_transformation")
        if not transformation:raise ValueError("Fold-local scope requires native association source bindings")
        tpath,info=validate_local_cnv(transformation["sources"],path.parent)
        matches=[r for r in info["records"] if r["role"]=="lncrna_expression" and r.get("fold")==fold]
        if len(matches)!=1 or not matches[0].get("upstream_expression"):
            raise ValueError("Scope associations must consume fold-local expression")
        actual=verified_path(matches[0],tpath.parent)
        _,expression=validate_expression(matches[0]["upstream_expression"],tpath.parent,actual)
        if expression.get("normalization")!=value["normalizations"][source["cancer_id"]]:
            raise ValueError("Association and scope normalization sources differ")
        observed.add((source["role"],source["cancer_id"]))
    expected={(r,c) for r in ("train_associations","validation_associations") for c in value["normalizations"]}
    if observed!=expected:raise ValueError("Scope association source coverage is incomplete")


def inspect_candidates(inventory,item,root,output,model,role,fold):
    path,value=validate_candidates(item,root,output,replay=("candidate_scope",item["sha256"]) not in inventory.visited)
    inventory.visited.add(("candidate_scope",item["sha256"]))
    inventory.bind(item,root,model,role+".candidate_definition",fold)
    from .current_auxiliary_model_use import input_source
    input_source(inventory,value["annotation"],path.parent,model,role+".annotation",fold)
    inventory.bind(value["gmt_manifest"],path.parent,model,role+".gmt_manifest",fold)
    for i,entry in enumerate(read(verified_path(value["gmt_manifest"],path.parent))["collections"]):
        source=entry["file"];label=role+".gmt."+str(i)
        inventory.bind(source,path.parent,model,label,fold)
        from .preparation_sources import metadata_tokens,canonical_sha
        from .evidence_rule_summary import stable_id
        import json
        tokens=sorted(metadata_tokens(entry.get("study_metadata",{})))
        if not tokens:inventory.gap(model,label,"SOURCE_STUDY_METADATA_ABSENT",fold,source["sha256"])
        identity=stable_id("",[model,fold,label,source["sha256"],inventory.context_id])
        inventory.usage.append({"cancer_id":"","lncrna_id":"","pathway_id":"","source_id":"CANDIDATE-SOURCE:"+identity,
            "source_event_id":identity,"source_file_sha256":source["sha256"],"phase":"FEATURE_CONSTRUCTION","endpoint":"MODEL_USE",
            "label":np.nan,"is_experimental":False,"is_predicted":False,"mapped":True,
            "assay":"REGISTERED_GMT_METADATA","biosample":"","study_tokens_json":json.dumps(tokens),
            "raw_record_id":identity,"raw_row_sha256":canonical_sha(entry),"raw_identifiers_json":"{}",
            "model_id":model,"patient_fold_id":fold,"use_view":"AUXILIARY_SOURCE_METADATA","candidate_rows":0,
            "entity_scope":"WHOLE_REGISTERED_FILE_NO_ENTITY_ASSERTION","dependency_context":inventory.context_id})
    for i,source in enumerate(value["implementations"]):inventory.bind(source,path.parent,model,role+".implementation."+str(i),fold)


def inspect_scope(inventory,item,root,output,model,role,fold):
    key=("detection_scope",item["sha256"])
    path,value=validate_scope(item,root,output,replay=key not in inventory.visited)
    if value["fold"]!=fold:raise ValueError("Consumed detection scope belongs to a different model fold")
    inventory.visited.add(key);inventory.bind(item,root,model,role+".scope",fold)
    cpath,candidate=validate_candidates(value["candidates"],path.parent)
    inspect_candidates(inventory,value["candidates"],path.parent,verified_path(candidate["output"],cpath.parent),model,role,fold)
    from .fold_local_expression import inspect_normalization
    for i,(cancer,source) in enumerate(sorted(value["normalizations"].items())):
        inspect_normalization(inventory,source,path.parent,model,role+".normalization."+str(i),fold,
                              release=value.get("normalization_releases",{}).get(cancer))
    for i,source in enumerate(value["implementations"]):inventory.bind(source,path.parent,model,role+".implementation."+str(i),fold)
