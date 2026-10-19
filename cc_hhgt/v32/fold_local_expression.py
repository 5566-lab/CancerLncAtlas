"""Training-fitted expression inputs consumed by the native five-fold analyses.

Raw RDS exposure is conservatively retained. This does not attest independence
of candidate selection, pathway scores, or GDC acquisition before the RDS.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from .cnv_input_sources import require_host
from .formal_expression_sources import read, write, record, read_matrix
from .current_core_export import verified_path

FORMAT = "FOLD_LOCAL_EXPRESSION_CONVERSION_V1"
NORMALIZATION = {
    "method": "edgeR_TMM", "prior_count": 0.25, "counts_assay": "unstranded", "fold_local": True,
    "normalization_scope": "TRAIN_PATIENT_SELECTED_TUMOR_AND_NORMAL",
    "prior_scale": "MEAN_TRAIN_EFFECTIVE_LIBRARY_SIZE", "detection_scope": "TRAIN_SELECTED_TUMOR_ONLY",
    "detection_threshold": 0.10, "emitted_roles": ["train", "validation"],
}


def native(item, root):
    path = verified_path(item, root); value = read(path)
    if (value.get("format") != "FOLD_LOCAL_TMM_SOURCES_V1" or value.get("target_host") != "149"
            or value.get("training_started") is not False or value.get("fold") not in range(5)
            or value.get("normalization") != NORMALIZATION or value.get("raw_source_ancestry_complete") is not False
            or value.get("raw_file_exposure") != "WHOLE_COUNT_RDS_INCLUDING_HELDOUT_ROWS"
            or set(value.get("inputs", {})) != {"counts", "patient_map"}
            or set(value.get("outputs", {})) != {"lncrna", "gene", "state", "detection", "normalization_samples"}
            or len(value.get("implementations", [])) != 2):
        raise ValueError("Invalid fold-local expression normalization receipt")
    for source in [*value["inputs"].values(), *value["outputs"].values(), *value["implementations"]]:
        verified_path(source, path.parent)
    samples = pd.read_csv(verified_path(value["outputs"]["normalization_samples"], path.parent), sep="\t")
    folds = pd.read_csv(verified_path(value["inputs"]["patient_map"], path.parent), sep="\t")
    folds = folds.loc[folds.cancer_id.eq(value["cancer_id"])]
    if (folds.patient_id.duplicated().any() or set(folds.patient_fold_id) != set(range(5))
            or not folds.patient_id.str.fullmatch(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}").all()):
        raise ValueError("Invalid frozen expression patient folds")
    declared = samples.patient_id.map(folds.set_index("patient_id").patient_fold_id)
    fold = value["fold"]
    roles = np.where(declared.isna(), "unassigned", np.where(declared.eq(fold), "test",
                     np.where(declared.eq((fold + 1) % 5), "validation", "train")))
    expected_fit = samples.selected_within_patient_type & samples.sample_type.isin(["tumor", "normal"]) & (roles == "train")
    expected_emit = samples.selected_within_patient_type & samples.sample_type.eq("tumor") & np.isin(roles, ["train", "validation"])
    if (samples.sample_id.duplicated().any() or not samples.cancer_id.eq(value["cancer_id"]).all()
            or not samples.patient_id.eq(samples.sample_id.str[:12]).all()
            or not np.array_equal(samples.split_role, roles) or not np.array_equal(samples.used_for_fit, expected_fit)
            or not np.array_equal(samples.exported, expected_emit)
            or not samples.loc[samples.used_for_fit, "sample_id"].eq(value["reference_sample"]).any()
            or not expected_fit.any() or not expected_emit.any()):
        raise ValueError("Fold-local expression fit/emission patient roles differ")
    if not np.isfinite(samples.loc[expected_fit | expected_emit, "normalization_factor"]).all():
        raise ValueError("Missing fitted/applied normalization factors")
    return path, value, samples


def frame_from_native(value, root, kind, samples):
    entity = "lncrna_id" if kind == "lncrna" else "gene_id"
    matrix = read_matrix(verified_path(value["outputs"][kind], root))
    matrix.index = matrix.index.astype(str); matrix.columns = matrix.columns.astype(str)
    expected_samples = samples.loc[samples.exported, "sample_id"].tolist()
    if (matrix.empty or matrix.index.duplicated().any() or list(matrix.columns) != expected_samples
            or not matrix.index.str.fullmatch(("LNC:" if kind == "lncrna" else "GENE:") + r"[^\s]+").all()
            or not np.isfinite(matrix.to_numpy(dtype=float)).all()):
        raise ValueError("Invalid fold-local matrix identity/values or emitted patients")
    matrix.index.name = entity; matrix.columns.name = "sample_id"
    frame = matrix.stack(future_stack=True).rename("logcpm").reset_index()
    frame["cancer_id"] = value["cancer_id"]; frame["patient_id"] = frame.sample_id.str[:12]
    frame["fold_id"] = value["fold"]
    frame["split_role"] = frame.sample_id.map(samples.set_index("sample_id").split_role)
    frame["analysis_eligible"] = True; frame["unavailable_reason"] = ""
    if kind == "lncrna":
        detected = pd.read_csv(verified_path(value["outputs"]["detection"], root), sep="\t", keep_default_na=False)
        if (detected.lncrna_id.duplicated().any() or set(detected.lncrna_id) != set(matrix.index)
                or not detected.detection_rate_logcpm_gt0.between(0, 1).all()
                or not detected.analysis_eligible.eq(detected.detection_rate_logcpm_gt0.ge(.10)).all()):
            raise ValueError("Invalid training-only expression detection table")
        train_samples = samples.loc[samples.exported & samples.split_role.eq("train"), "sample_id"]
        actual = (matrix.loc[:, train_samples] > 0).mean(axis=1)
        rates = detected.set_index("lncrna_id").detection_rate_logcpm_gt0.reindex(matrix.index)
        if not np.allclose(actual, rates, rtol=0, atol=1e-12):
            raise ValueError("Expression detection does not match training tumors")
        eligible = frame[entity].map(detected.set_index("lncrna_id").analysis_eligible)
        frame["analysis_eligible"] = eligible
        frame["unavailable_reason"] = np.where(eligible, "", "BELOW_TRAIN_DETECTION_THRESHOLD")
    return frame[["cancer_id", "sample_id", "patient_id", entity, "logcpm", "fold_id", "split_role",
                  "analysis_eligible", "unavailable_reason"]]


def convert(source_roots, output_root, *, runner):
    require_host(); output_root = Path(output_root).resolve()
    if output_root.exists(): raise FileExistsError("Use a new fold-local expression conversion output")
    declared = []
    for root in source_roots:
        root = Path(root).resolve(strict=True); success = read(root / "SUCCESS.json")
        if (success.get("status") != "PASS" or success.get("target_host") != "149"
                or success.get("training_started") is not False or set(success.get("folds", {})) != set(map(str, range(5)))):
            raise ValueError("Incomplete five-fold normalization source")
        declared.append((root, success))
    if not declared or len({v["cancer_id"] for _, v in declared}) != len(declared):
        raise ValueError("Empty or duplicate expression cancer scope")
    output_root.mkdir(parents=True)
    write(output_root / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False,
                                                 "operation": "FOLD_LOCAL_EXPRESSION_CONVERSION"})
    outputs = []
    for root, success in declared:
        for fold in range(5):
            item = success["folds"][str(fold)]
            path, value, samples = native(item, root)
            if value["fold"] != fold or value["cancer_id"] != success["cancer_id"]:
                raise ValueError("Fold-local normalization partition mismatch")
            for kind in ("lncrna", "gene"):
                frame = frame_from_native(value, path.parent, kind, samples)
                target = output_root / kind / f"fold={fold}" / ("cancer_id=" + value["cancer_id"]) / "part-0.parquet"
                target.parent.mkdir(parents=True)
                frame.to_parquet(target, index=False, compression="zstd")
                receipt = {"format": FORMAT, "target_host": "149", "training_started": False,
                           "cancer_id": value["cancer_id"], "fold": fold, "kind": kind, "normalization": item,
                           "output": record(target), "implementations": [record(__file__), record(runner)]}
                receipt_path = target.parent / "EXPRESSION_SOURCES.json"; write(receipt_path, receipt)
                validate(record(receipt_path), target.parent, target)
                outputs.append(record(receipt_path))
    roots = {"lnc_expression_root": {"path": str(output_root / "lncrna"), "layout": "fold_local"},
             "gene_expression_root": {"path": str(output_root / "gene"), "layout": "fold_local"}}
    write(output_root / "SOURCE_CONFIG_FRAGMENT.json", {"sources": roots})
    write(output_root / "SUCCESS.json", {"status": "PASS", "target_host": "149", "training_started": False, "outputs": outputs})
    return outputs


def validate(item, root, output, *, replay=False):
    path = verified_path(item, root); value = read(path)
    if (value.get("format") != FORMAT or value.get("target_host") != "149" or value.get("training_started") is not False
            or value.get("kind") not in {"lncrna", "gene"} or len(value.get("implementations", [])) != 2):
        raise ValueError("Invalid fold-local expression conversion")
    actual = verified_path(value["output"], path.parent)
    if actual != Path(output).resolve(): raise ValueError("Fold-local expression binds a different output")
    for implementation in value["implementations"]: verified_path(implementation, path.parent)
    native_path, native_value, samples = native(value["normalization"], path.parent)
    if value["fold"] != native_value["fold"] or value["cancer_id"] != native_value["cancer_id"]:
        raise ValueError("Fold-local expression scope differs from normalization")
    if replay:
        expected = frame_from_native(native_value, native_path.parent, value["kind"], samples)
        try: pd.testing.assert_frame_equal(pd.read_parquet(actual), expected, check_dtype=False, check_exact=True)
        except AssertionError as error: raise ValueError("Fold-local expression differs from native conversion") from error
    return path, value


def expression_path(sources, key, cancer, fold, patient_map):
    """Explicit layout, never a fallback from missing fold data to cohort data."""
    spec = sources[key]; root = Path(spec["path"])
    layout = spec.get("layout", "cohort")
    if layout not in {"cohort", "fold_local"}: raise ValueError("Unknown expression layout")
    if layout == "cohort": return root / f"cancer_id={cancer}" / "part-0.parquet"
    target = root / f"fold={fold}" / f"cancer_id={cancer}" / "part-0.parquet"
    _, value = validate(record(target.parent / "EXPRESSION_SOURCES.json"), target.parent, target)
    native_path, info, _ = native(value["normalization"], target.parent)
    kind = "lncrna" if key == "lnc_expression_root" else "gene"
    if value["fold"] != fold or value["cancer_id"] != cancer or value["kind"] != kind:
        raise ValueError("Requested expression fold/cancer/entity differs")
    if verified_path(info["inputs"]["patient_map"], native_path.parent) != Path(patient_map).resolve():
        raise ValueError("Expression normalization patient map differs from analysis")
    return target


def analysis_columns(path, columns):
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    extra = [v for v in ("analysis_eligible", "unavailable_reason") if v in names]
    return list(dict.fromkeys([*columns, *extra]))


def read_analysis_frame(path, columns):
    """Preserve excluded entities in attrs while presenting eligible measurements."""
    columns = analysis_columns(path, columns)
    extra = [v for v in ("analysis_eligible", "unavailable_reason") if v in columns]
    frame = pd.read_parquet(path, columns=columns)
    if extra:
        if len(extra) != 2: raise ValueError("Incomplete expression eligibility columns")
        entities = [name for name in ("lncrna_id", "gene_id", "pathway_id") if name in frame]
        if len(entities) != 1: raise ValueError("Analysis input needs one feature identity")
        entity = entities[0]
        status = frame[[entity, *extra]].drop_duplicates()
        if status[entity].duplicated().any() or frame.analysis_eligible.isna().any():
            raise ValueError("Inconsistent expression feature eligibility")
        excluded = status.loc[~status.analysis_eligible].set_index(entity).unavailable_reason.to_dict()
        frame = frame.loc[frame.analysis_eligible].copy()
        frame.attrs["expression_exclusions"] = excluded
    return frame


def mark_excluded(frame, expression, *, entity="lncrna_id"):
    reasons = frame[entity].map(expression.attrs.get("expression_exclusions", {})).fillna("")
    excluded = reasons.ne("")
    frame.loc[excluded, "availability"] = "typed_unavailable"
    frame.loc[excluded, "unavailable_reason"] = reasons[excluded]
    # Candidate identity and row count remain fixed. Missing features are not
    # measured negative associations, including in the historical A0 columns.
    for name in frame:
        if name.startswith(("rho_A", "p_A", "fdr_")): frame.loc[excluded, name] = np.nan
        elif name.startswith("n_A"): frame.loc[excluded, name] = 0
        elif name.startswith("label_A"): frame.loc[excluded, name] = "unavailable"
        elif name.startswith("direction_A"): frame.loc[excluded, name] = "unavailable"
    return frame


def inspect(inventory, item, root, output, model, role, fold, *, phase="FEATURE_CONSTRUCTION"):
    key = ("fold_local_expression", item["sha256"])
    path, value = validate(item, root, output, replay=key not in inventory.visited)
    if fold != value["fold"]: raise ValueError("Consumed expression belongs to a different model fold")
    inventory.visited.add(key)
    inventory.bind(item, root, model, role + ".conversion", fold)
    native_path, info, _ = native(value["normalization"], path.parent)
    inventory.bind(value["normalization"], path.parent, model, role + ".normalization", fold)
    for i, source in enumerate([*value["implementations"], *info["implementations"]]):
        inventory.bind(source, path.parent, model, role + ".implementation." + str(i), fold)
    for name, source in {**info["inputs"], **info["outputs"]}.items():
        inventory.bind(source, native_path.parent, model, role + ".normalization." + name, fold)
    from .current_auxiliary_model_use import input_source
    input_source(inventory, info["outputs"]["normalization_samples"], native_path.parent,
                 model, role + ".raw_sample_exposure", fold, phase=phase)
    inventory.gap(model, role, "EXPRESSION_GDC_ACQUISITION_BEFORE_COUNT_RDS_UNVERIFIED", fold, info["inputs"]["counts"]["sha256"])
