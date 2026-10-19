"""Native TMM RDS conversion and actually consumed normalization sources.

Preserves historical cohort normalization; does not certify fold-local TMM or
reconstruct GDC acquisition before the input count RDS.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from .cnv_input_sources import require_host
from .current_core_export import sha256, verified_path

FORMAT = "FORMAL_EXPRESSION_CONVERSION_SOURCES_V1"
NORMALIZATION = {"method": "edgeR_TMM", "prior_count": 0.25, "counts_assay": "unstranded",
                 "normalization_scope": "SELECTED_TUMOR_AND_NORMAL_SAMPLES_WITHIN_CANCER", "fold_local": False}


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def record(path): return {"path": str(Path(path).resolve()), "sha256": sha256(path)}
def write(path, value): Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def matrix_to_frame(matrix, cancer):
    """Preserve exact matrix values and explicit TCGA sample/patient identity."""
    if not re.fullmatch(r"[A-Z0-9]+", cancer) or matrix.empty:
        raise ValueError("Invalid expression cancer or empty matrix")
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str); matrix.columns = matrix.columns.astype(str)
    if (matrix.index.duplicated().any() or matrix.columns.duplicated().any()
            or not all(re.fullmatch(r"LNC:[^\s]+", value) for value in matrix.index)
            or not all(re.match(r"^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-[0-9]{2}", value) for value in matrix.columns)):
        raise ValueError("Invalid or duplicate expression entity/sample identity")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Expression matrix contains nonfinite values")
    matrix.index.name = "lncrna_id"; matrix.columns.name = "sample_id"
    frame = matrix.stack(future_stack=True).rename("logcpm").reset_index()
    frame["cancer_id"] = cancer; frame["patient_id"] = frame.sample_id.str[:12]
    return frame[["cancer_id", "sample_id", "patient_id", "lncrna_id", "logcpm"]]


def read_matrix(path):
    import pyreadr
    values = pyreadr.read_r(str(path))
    if list(values) != [None]: raise ValueError("Expression RDS must contain one unnamed matrix")
    return values[None]


def normalization(item, root, matrix, cancer):
    path = verified_path(item, root); value = read(path)
    if (value.get("format") != "FORMAL_TMM_LOGCPM_SOURCES_V1" or value.get("target_host") != "149"
            or value.get("training_started") is not False or value.get("cancer_id") != cancer
            or value.get("normalization") != NORMALIZATION or value.get("raw_source_ancestry_complete") is not False
            or set(value.get("inputs", {})) != {"counts", "detection"}
            or set(value.get("outputs", {})) != {"matrix", "tumor_samples", "normalization_samples"}):
        raise ValueError("Invalid native expression normalization receipt")
    for source in [*value["inputs"].values(), *value["outputs"].values(), value["implementation"]]:
        verified_path(source, path.parent)
    if value["outputs"]["matrix"] != record(matrix):
        raise ValueError("Expression normalization does not bind the consumed matrix")
    return path, value


def convert(source_root, output_root, *, cancers=None, runner):
    require_host()
    source_root = Path(source_root).resolve(strict=True); output_root = Path(output_root).resolve()
    paths = sorted(source_root.glob("TCGA-*_formal_logcpm.RDS"))
    available = {p.name.split("-")[1].split("_")[0]: p for p in paths}
    if cancers is None:
        if len(available) != 33: raise ValueError("Expected 33 formal cancer matrices")
        cancers = sorted(available)
    if not cancers or len(set(cancers)) != len(cancers) or not set(cancers) <= set(available):
        raise ValueError("Requested expression cancer scope is unavailable or duplicated")
    if output_root.exists(): raise FileExistsError("Use a new expression conversion output directory")
    output_root.mkdir(parents=True)
    write(output_root / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False,
                                               "operation": "FORMAL_EXPRESSION_RDS_CONVERSION"})
    results = []
    for cancer in cancers:
        source = available[cancer]; before = record(source)
        upstream_path = source_root / ("TCGA-" + cancer + "_SOURCE_RECEIPT.json")
        upstream = record(upstream_path) if upstream_path.exists() else None
        if upstream: normalization(upstream, source_root, source, cancer)
        frame = matrix_to_frame(read_matrix(source), cancer)
        if record(source) != before: raise ValueError("Expression matrix changed during conversion")
        target = output_root / ("cancer_id=" + cancer) / "part-0.parquet"
        target.parent.mkdir(); frame.to_parquet(target, index=False, compression="zstd")
        qc = {"cancer_id": cancer, "rows": len(frame), "lncrna_n": int(frame.lncrna_id.nunique()),
              "sample_n": int(frame.sample_id.nunique())}
        write(target.parent / "QC.json", qc)
        value = {"format": FORMAT, "target_host": "149", "training_started": False, "cancer_id": cancer,
                 "matrix": before, "normalization": upstream, "output": record(target),
                 "implementations": [record(__file__), record(runner)], "raw_source_ancestry_complete": False}
        write(target.parent / "EXPRESSION_SOURCES.json", value)
        validate(record(target.parent / "EXPRESSION_SOURCES.json"), target.parent, target)
        results.append({**qc, "source_receipt": record(target.parent / "EXPRESSION_SOURCES.json")})
    write(output_root / "SUCCESS.json", {"status": "PASS", "target_host": "149", "training_started": False,
                                        "cancers": results, "raw_source_ancestry_complete": False})
    return results


def validate(item, root, output, *, replay=False):
    path = verified_path(item, root); value = read(path)
    if value.get("format") == "FOLD_LOCAL_EXPRESSION_CONVERSION_V1":
        from .fold_local_expression import validate as validate_fold_local
        return validate_fold_local(item, root, output, replay=replay)
    if (value.get("format") != FORMAT or value.get("target_host") != "149" or value.get("training_started") is not False
            or value.get("raw_source_ancestry_complete") is not False or len(value.get("implementations", [])) != 2):
        raise ValueError("Invalid expression conversion source receipt")
    actual = verified_path(value["output"], path.parent)
    if actual != Path(output).resolve(): raise ValueError("Expression receipt binds a different consumed output")
    source = verified_path(value["matrix"], path.parent)
    for implementation in value["implementations"]: verified_path(implementation, path.parent)
    if value.get("normalization"):
        normalization(value["normalization"], path.parent, source, value["cancer_id"])
    if replay:
        expected = matrix_to_frame(read_matrix(source), value["cancer_id"])
        try: pd.testing.assert_frame_equal(pd.read_parquet(actual), expected, check_dtype=False, check_exact=True)
        except AssertionError as error: raise ValueError("Expression output differs from native RDS conversion") from error
    return path, value


def binding(output):
    output = Path(output); path = output.parent / "EXPRESSION_SOURCES.json"
    if output.name != "part-0.parquet" or not path.is_file(): return None
    item = record(path); validate(item, path.parent, output)
    return item


def inspect(inventory, item, root, output, model, role, fold, *, phase="FEATURE_CONSTRUCTION"):
    if read(verified_path(item, root)).get("format") == "FOLD_LOCAL_EXPRESSION_CONVERSION_V1":
        from .fold_local_expression import inspect as inspect_fold_local
        return inspect_fold_local(inventory, item, root, output, model, role, fold, phase=phase)
    key = ("expression_conversion", item["sha256"], sha256(output))
    path, value = validate(item, root, output, replay=key not in inventory.visited)
    inventory.visited.add(key)
    inventory.bind(item, root, model, role + ".conversion", fold)
    for i, source in enumerate(value["implementations"]):
        inventory.bind(source, path.parent, model, role + ".implementation." + str(i), fold)
    matrix = inventory.bind(value["matrix"], path.parent, model, role + ".matrix", fold)
    if value.get("normalization"):
        native_path, native = normalization(value["normalization"], path.parent, matrix, value["cancer_id"])
        inventory.bind(value["normalization"], path.parent, model, role + ".normalization", fold)
        for name, source in {**native["inputs"], **native["outputs"], "implementation": native["implementation"]}.items():
            inventory.bind(source, native_path.parent, model, role + ".normalization." + name, fold)
        from .current_auxiliary_model_use import input_source
        input_source(inventory, native["inputs"]["detection"], native_path.parent,
                     model, role + ".detection", fold, phase=phase)
        # Normal samples participate in TMM even though only tumors are exported.
        input_source(inventory, native["outputs"]["normalization_samples"], native_path.parent,
                     model, role + ".normalization_samples", fold, phase=phase)
        inventory.gap(model, role, "COHORT_LEVEL_TMM_NORMALIZATION_NOT_FOLD_LOCAL", fold, value["normalization"]["sha256"])
        inventory.gap(model, role, "EXPRESSION_GDC_ACQUISITION_BEFORE_COUNT_RDS_UNVERIFIED", fold, native["inputs"]["counts"]["sha256"])
    else:
        inventory.gap(model, role, "EXPRESSION_NORMALIZATION_SOURCES_ABSENT", fold, value["matrix"]["sha256"])
