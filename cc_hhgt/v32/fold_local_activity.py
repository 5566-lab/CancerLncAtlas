"""Native training-fitted ssGSEA, its actual consumers and source ancestry.

Conversion inspection checks coverage, training scaling and serialization.
Model-use inspection also recomputes GSVA raw scores when a completed native
release is bound; legacy unbound conversions retain an explicit replay gap.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from .cnv_input_sources import require_host
from .formal_expression_sources import read, write, record, read_matrix
from .current_core_export import verified_path
from .fold_local_expression import native as expression_native

FORMAT = "FOLD_LOCAL_ACTIVITY_CONVERSION_V1"
METHOD = {"name":"ssGSEA", "alpha":.25, "min_size":10, "max_size":500, "original_set_size_max":500,
          "cpm_threshold":1, "detection_fraction":.20, "min_training_detection_n":3,
          "symbol_collapse":"MEAN_LOGCPM", "scaling":"RAW_SCORE_DIVIDED_BY_TRAIN_MAX_MINUS_MIN",
          "emitted_roles":["train","validation"]}


def gene_sets(value, root):
    manifest = read(verified_path(value["inputs"]["gmt_manifest"], root))
    if manifest.get("format") != "SSGSEA_GMT_INPUT_V1" or not manifest.get("collections"):
        raise ValueError("Invalid ssGSEA GMT manifest")
    sets, sources = {}, {}
    for entry in manifest["collections"]:
        name = entry["collection"]
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or name in sources:
            raise ValueError("Duplicate/invalid GMT collection")
        source = verified_path(entry["file"], root); sources[name] = entry["file"]
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line: continue
            fields = line.split("\t")
            if len(fields) < 3 or not fields[0]: raise ValueError("Invalid GMT row")
            key = "MSIGDB:" + name + ":" + fields[0]
            genes = list(dict.fromkeys(g for g in fields[2:] if g))
            if not genes or key in sets: raise ValueError("Empty/duplicate GMT gene set")
            sets[key] = genes
    if sources != value["gmt_sources"] or not sets: raise ValueError("GMT sources differ from native reads")
    return sets, manifest


def native(item, root):
    path = verified_path(item, root); value = read(path)
    if (value.get("format") != "FOLD_LOCAL_SSGSEA_SOURCES_V1" or value.get("target_host") != "149"
            or value.get("training_started") is not False or value.get("method") != METHOD
            or value.get("fold") not in range(5) or len(value.get("implementations", [])) != 2
            or set(value.get("inputs", {})) != {"normalization_success", "annotation", "gmt_manifest"}
            or set(value.get("outputs", {})) != {"raw", "activity", "state", "coverage", "gene_detection"}):
        raise ValueError("Invalid native fold-local ssGSEA receipt")
    for source in [*value["inputs"].values(), *value["outputs"].values(), *value["implementations"]]:
        verified_path(source, path.parent)
    npath, norm, samples = expression_native(value["normalization"], path.parent)
    if (norm["fold"] != value["fold"] or norm["cancer_id"] != value["cancer_id"]
            or read(verified_path(value["inputs"]["normalization_success"], path.parent))["folds"][str(value["fold"])] != value["normalization"]):
        raise ValueError("ssGSEA normalization cancer/fold differs")
    gene_sets(value, path.parent)
    return path, value, npath, norm, samples


def frame_from_native(value, root, npath, norm, samples):
    sets, _ = gene_sets(value, root)
    raw = read_matrix(verified_path(value["outputs"]["raw"], root))
    activity = read_matrix(verified_path(value["outputs"]["activity"], root))
    emitted = samples.loc[samples.exported, "sample_id"].tolist()
    for matrix in (raw, activity):
        if list(matrix.index) != list(sets) or list(matrix.columns) != emitted:
            raise ValueError("ssGSEA pathway/sample closure differs")
    coverage = pd.read_csv(verified_path(value["outputs"]["coverage"], root), sep="\t", keep_default_na=False)
    detection = pd.read_csv(verified_path(value["outputs"]["gene_detection"], root), sep="\t", keep_default_na=False)
    x = read_matrix(verified_path(norm["outputs"]["gene"], npath.parent))
    annotation = pd.read_csv(verified_path(value["inputs"]["annotation"], root), sep="\t", keep_default_na=False)
    annotation["ensembl_gene_id"] = annotation.ensembl_gene_id.str.replace(r"\.[0-9]+$", "", regex=True)
    if annotation.ensembl_gene_id.duplicated().any(): raise ValueError("Ambiguous ssGSEA gene annotation")
    ann = annotation.set_index("ensembl_gene_id").reindex(x.index.str.removeprefix("GENE:"))
    symbols = ann.gene_symbol.fillna("").to_numpy()
    train = samples.loc[samples.exported & samples.split_role.eq("train"), "sample_id"].tolist()
    m = norm["mean_train_effective_library_size"]
    cpm = np.exp2(x.loc[:, train].to_numpy()) * (1+.5/m) - .25*1e6/m
    required = max(3, int(np.ceil(.2*len(train))))
    counts = (cpm >= 1).sum(axis=1)
    keep = ann.gene_type.eq("protein_coding").to_numpy() & (symbols != "") & (counts >= required)
    if (value["training_detection_required_n"] != required or detection.gene_id.tolist() != x.index.tolist()
            or not np.array_equal(detection.gene_symbol, symbols)
            or not np.array_equal(detection.training_cpm_ge1_n, counts)
            or not np.array_equal(detection.analysis_eligible, keep)):
        raise ValueError("ssGSEA gene filter differs from training tumors")
    detected = set(symbols[keep]); n_genes = np.array([len(s) for s in sets.values()])
    n_detected = np.array([len(set(s) & detected) for s in sets.values()])
    reason = np.where(n_genes>500, "PATHWAY_EXCEEDS_MAX_SIZE",
             np.where(n_detected<10, "INSUFFICIENT_TRAIN_DETECTED_GENES",
             np.where(n_detected>=len(detected), "NO_BACKGROUND_GENES", "")))
    usable = reason == ""
    expected = raw.copy(); expected.iloc[:, :] = np.nan
    if usable.any():
        values = raw.iloc[usable].loc[:, train].to_numpy()
        if not np.isfinite(raw.iloc[usable].to_numpy()).all(): raise ValueError("Nonfinite eligible raw ssGSEA")
        scale = float(values.max()-values.min())
        if value["training_score_range"] is None or not np.isclose(scale, value["training_score_range"], rtol=1e-13, atol=0):
            raise ValueError("ssGSEA scaling differs from training scores")
        if scale > 0: expected.iloc[usable] = raw.iloc[usable]/scale
        else: reason = np.where(usable, "DEGENERATE_TRAIN_SCORE_RANGE", reason)
    elif value["training_score_range"] is not None:
        raise ValueError("No eligible pathways may declare a training scale")
    if not raw.iloc[~usable].isna().all().all(): raise ValueError("Unavailable pathway contains raw score")
    if (coverage.pathway_id.tolist() != list(sets) or not np.array_equal(coverage.gene_count, n_genes)
            or not np.array_equal(coverage.detected_gene_count, n_detected)
            or not np.allclose(coverage.coverage, n_detected/n_genes, rtol=0, atol=1e-12)
            or not np.array_equal(coverage.analysis_eligible, reason=="")
            or not np.array_equal(coverage.unavailable_reason, reason)):
        raise ValueError("ssGSEA coverage/availability differs from frozen genes")
    if not np.allclose(activity, expected, rtol=1e-13, atol=0, equal_nan=True):
        raise ValueError("ssGSEA scores differ from training-range normalization")
    activity.index.name = raw.index.name = "pathway_id"
    activity.columns.name = raw.columns.name = "sample_id"
    frame = activity.stack(future_stack=True).rename("activity_score").reset_index()
    frame["raw_score"] = raw.stack(future_stack=True).to_numpy()
    frame = frame.merge(coverage, on="pathway_id", validate="many_to_one", sort=False)
    frame["cancer_id"] = value["cancer_id"]; frame["patient_id"] = frame.sample_id.str[:12]
    frame["fold_id"] = value["fold"]
    frame["split_role"] = frame.sample_id.map(samples.set_index("sample_id").split_role)
    return frame


def convert(source_roots, output_root, *, runner):
    require_host(); output_root = Path(output_root).resolve()
    if output_root.exists(): raise FileExistsError("Use a new fold-local activity conversion output")
    declarations = [(Path(p).resolve(strict=True), read(Path(p)/"SUCCESS.json")) for p in source_roots]
    if not declarations or len({v["cancer_id"] for _,v in declarations}) != len(declarations):
        raise ValueError("Empty or duplicate ssGSEA cancer scope")
    output_root.mkdir(parents=True)
    write(output_root/"LAUNCH_PREFLIGHT.json", {"target_host":"149", "training_started":False, "operation":"FOLD_LOCAL_SSGSEA_CONVERSION"})
    outputs = []
    for root, success in declarations:
        if (success.get("status") != "PASS" or success.get("target_host") != "149"
                or success.get("training_started") is not False or set(success.get("folds", {})) != set(map(str, range(5)))):
            raise ValueError("Incomplete five-fold ssGSEA source")
        for fold in range(5):
            item = success["folds"][str(fold)]
            path, value, npath, norm, samples = native(item, root)
            if value["fold"] != fold or value["cancer_id"] != success["cancer_id"]: raise ValueError("ssGSEA partition mismatch")
            frame = frame_from_native(value, path.parent, npath, norm, samples)
            target = output_root/f"fold={fold}"/("cancer_id="+value["cancer_id"])/"part-0.parquet"
            target.parent.mkdir(parents=True); frame.to_parquet(target, index=False, compression="zstd")
            receipt = {"format":FORMAT, "target_host":"149", "training_started":False, "fold":fold,
                       "cancer_id":value["cancer_id"], "native":item, "output":record(target),
                       "activity_release":record(root/"SUCCESS.json"),
                       "implementations":[record(__file__), record(runner)]}
            rpath = target.parent/"ACTIVITY_SOURCES.json"; write(rpath, receipt)
            validate(record(rpath), target.parent, target); outputs.append(record(rpath))
    write(output_root/"SOURCE_CONFIG_FRAGMENT.json", {"sources":{"pathway_activity_root":{"path":str(output_root),"layout":"fold_local"}}})
    write(output_root/"SUCCESS.json", {"status":"PASS", "target_host":"149", "training_started":False, "outputs":outputs})
    return outputs


def validate(item, root, output, *, replay=False):
    path = verified_path(item, root); value = read(path)
    if (value.get("format") != FORMAT or value.get("target_host") != "149" or value.get("training_started") is not False
            or len(value.get("implementations", [])) != 2): raise ValueError("Invalid fold-local activity conversion")
    if verified_path(value["output"], path.parent) != Path(output).resolve(): raise ValueError("Activity binds a different output")
    for implementation in value["implementations"]: verified_path(implementation, path.parent)
    native_path, info, npath, norm, samples = native(value["native"], path.parent)
    if value["fold"] != info["fold"] or value["cancer_id"] != info["cancer_id"]: raise ValueError("Activity fold/cancer mismatch")
    if value.get("activity_release") is not None:
        from .activity_source_replay import declared_release
        rpath,release = declared_release(value["activity_release"],path.parent)
        source = release["folds"][str(value["fold"])]
        if release["cancer_id"] != value["cancer_id"] or verified_path(source,rpath.parent) != native_path:
            raise ValueError("Activity conversion differs from its completed release")
    if replay:
        expected = frame_from_native(info, native_path.parent, npath, norm, samples)
        try: pd.testing.assert_frame_equal(pd.read_parquet(output), expected, check_dtype=False, check_exact=True)
        except AssertionError as error: raise ValueError("Activity differs from native conversion") from error
    return path, value


def binding(output):
    path = Path(output).parent/"ACTIVITY_SOURCES.json"
    if Path(output).name != "part-0.parquet" or not path.is_file(): return None
    item = record(path); validate(item, path.parent, output); return item


def activity_path(sources, cancer, fold, patient_map):
    spec = sources["pathway_activity_root"]; root = Path(spec["path"]); layout = spec.get("layout", "cohort")
    if layout == "cohort": return root/f"cancer_id={cancer}"/"part-0.parquet"
    if layout != "fold_local": raise ValueError("Unknown pathway activity layout")
    target = root/f"fold={fold}"/f"cancer_id={cancer}"/"part-0.parquet"
    _, value = validate(record(target.parent/"ACTIVITY_SOURCES.json"), target.parent, target)
    _, _, npath, norm, _ = native(value["native"], target.parent)
    if value["fold"] != fold or value["cancer_id"] != cancer: raise ValueError("Requested activity fold/cancer differs")
    if verified_path(norm["inputs"]["patient_map"], npath.parent) != Path(patient_map).resolve():
        raise ValueError("Activity patient map differs from analysis")
    from .fold_local_expression import expression_path
    from .formal_expression_sources import binding as expression_binding
    for key in ("lnc_expression_root", "gene_expression_root"):
        if key in sources and sources[key].get("layout") == "fold_local":
            expression = expression_path(sources, key, cancer, fold, patient_map)
            expression_info = read(expression_binding(expression)["path"])
            if expression_info["normalization"] != read(verified_path(value["native"], target.parent))["normalization"]:
                raise ValueError("Activity and expression use different normalization sources")
    return target


def inspect(inventory, item, root, output, model, role, fold, *, phase="FEATURE_CONSTRUCTION"):
    key = ("fold_local_activity", item["sha256"])
    path, value = validate(item, root, output, replay=key not in inventory.visited)
    if value["fold"] != fold: raise ValueError("Consumed activity belongs to a different model fold")
    inventory.visited.add(key); inventory.bind(item, root, model, role+".conversion", fold)
    apath, info, npath, norm, _ = native(value["native"], path.parent)
    inventory.bind(value["native"], path.parent, model, role+".ssgsea", fold)
    for i, source in enumerate([*value["implementations"], *info["implementations"]]):
        inventory.bind(source, path.parent, model, role+".implementation."+str(i), fold)
    for name, source in {**info["inputs"], **info["outputs"]}.items(): inventory.bind(source, apath.parent, model, role+".ssgsea."+name, fold)
    from .fold_local_expression import inspect_normalization
    inspect_normalization(inventory,info["normalization"],apath.parent,model,role,fold,
                          release=info["inputs"]["normalization_success"],phase=phase)
    from .activity_source_replay import inspect as inspect_replay
    inspect_replay(inventory,value.get("activity_release"),value["native"],info["normalization"],
                   path.parent,model,role,fold)
    from .current_auxiliary_model_use import input_source
    input_source(inventory, info["inputs"]["annotation"], apath.parent, model, role+".annotation", fold, phase=phase)
    _, manifest = gene_sets(info, apath.parent)
    from .preparation_sources import metadata_tokens, canonical_sha
    from .evidence_rule_summary import stable_id
    for i, entry in enumerate(manifest["collections"]):
        source = entry["file"]; label = role+".gmt."+str(i)
        inventory.bind(source, apath.parent, model, label, fold)
        tokens = sorted(metadata_tokens(entry.get("study_metadata", {})))
        if not tokens: inventory.gap(model, label, "SOURCE_STUDY_METADATA_ABSENT", fold, source["sha256"])
        identity = stable_id("", [model, fold, label, source["sha256"], inventory.context_id, phase])
        inventory.usage.append({"cancer_id":"", "lncrna_id":"", "pathway_id":"", "source_id":"SSGSEA-SOURCE:"+identity,
            "source_event_id":identity, "source_file_sha256":source["sha256"], "phase":phase, "endpoint":"MODEL_USE",
            "label":np.nan, "is_experimental":False, "is_predicted":False, "mapped":True,
            "assay":"REGISTERED_GMT_METADATA", "biosample":"", "study_tokens_json":json.dumps(tokens),
            "raw_record_id":identity, "raw_row_sha256":canonical_sha(entry), "raw_identifiers_json":"{}",
            "model_id":model, "patient_fold_id":fold, "use_view":"AUXILIARY_SOURCE_METADATA", "candidate_rows":0,
            "entity_scope":"WHOLE_REGISTERED_FILE_NO_ENTITY_ASSERTION", "dependency_context":inventory.context_id})
