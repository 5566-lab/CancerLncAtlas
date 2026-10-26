"""Prepare canonical donor partitions for the original nested SC materializer.

No geometry, PCA, LASSO or neural fitting is dispatched by this workflow.
"""
from pathlib import Path
import math
import sys

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .current_analysis_workflow import write
from . import native_single_cell_mapping_workflow as mapping
from .single_cell_metacell_screen import donor_folds
from .single_cell_nested_episodes import nested_fold_plans

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_SINGLE_CELL_DONORS_INPUT_V1"
FORMAT = "NATIVE_SINGLE_CELL_DONORS_RESULTS_V1"
CODE = ("cc_hhgt/v32/native_single_cell_donors_workflow.py",
        "scripts/run_native_single_cell_donors_workflow.py",
        "scripts/materialize_single_cell_nested_hhgt_episodes_149.py",
        "cc_hhgt/v32/single_cell_nested_episodes.py",
        "cc_hhgt/v32/single_cell_metacell_screen.py")
FILES = {"CANONICAL_CELL_METADATA.parquet", "CANONICAL_DONOR_CROSSWALK.parquet",
         "DONOR_FOLDS.parquet", "NESTED_PLANS.json", "EXECUTION_COMMAND.json"}


def canonical_crosswalk(metadata):
    frame = metadata[["dataset_id", "donor_id"]].drop_duplicates().rename(columns={"donor_id": "query_donor_id"})
    frame["canonical_donor_id"] = frame.query_donor_id
    return frame.sort_values(["dataset_id", "query_donor_id"]).reset_index(drop=True)


def derived(value):
    """Keep matrix row order while collapsing declared aliases before splitting."""
    mapped = Path(value["mapped_success"]["path"]).parent
    metadata = pd.read_parquet(mapped / "CELL_METADATA.parquet")
    crosswalk = pd.read_parquet(value["donor_crosswalk"]["path"])
    keys = ["dataset_id", "query_donor_id", "canonical_donor_id"]
    if not set(keys) <= set(crosswalk) or crosswalk.empty:
        raise ValueError("SC donor crosswalk requires dataset, native and canonical donor IDs")
    for key in keys:
        if crosswalk[key].isna().any() or not crosswalk[key].map(lambda x: isinstance(x, str) and x == x.strip() and bool(x)).all():
            raise ValueError("SC donor crosswalk identifiers must be non-empty exact strings")
    if crosswalk.duplicated(keys[:2]).any():
        raise ValueError("SC donor crosswalk has duplicate native identities")
    lookup = crosswalk.set_index(keys[:2]).canonical_donor_id.to_dict()
    identities = list(zip(metadata.dataset_id.astype(str), metadata.donor_id.astype(str)))
    if any(key not in lookup for key in identities):
        raise ValueError("SC mapped donors lack complete dataset-qualified crosswalk")
    metadata = metadata.copy()
    metadata["native_donor_id"] = metadata.donor_id.astype(str)
    metadata["donor_id"] = [lookup[key] for key in identities]
    if metadata.donor_id.nunique() < 3:
        raise ValueError("Nested SC preparation requires at least three canonical donors")
    folds = donor_folds(value["cancer_id"], metadata.donor_id, folds=5, seed=value["seed"])
    plans = [p.as_dict() for outer in sorted(folds.outer_fold_id.unique())
             for p in nested_fold_plans(folds, cancer_id=value["cancer_id"], outer_fold_id=int(outer))]
    return metadata, folds, plans


def build_settings(value, output, deadline=1.0):
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
    output = Path(output)
    argv = ["--cancer-id", value["cancer_id"], "--source-result-root", str(output),
            "--graph-asset-root", value["graph_root"], "--output-root", str(output / "nested"),
            "--scratch-root", str(Path(value["mapped_success"]["path"]).parent.parent),
            "--candidates", value["candidates"]["path"], "--deadline", str(deadline),
            "--seed", str(value["seed"])]
    return native.parser().parse_args(argv), argv


def plan(manifest, output, *, new=True):
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve()
    spec = read(manifest)
    required = {"format", "mapping", "donor_crosswalk", "graph_preflight", "candidates"}
    if not required <= set(spec) or set(spec) - required - {"seed"} or spec["format"] != INPUT:
        raise ValueError("Invalid native SC donor preparation manifest")
    if new and output.exists():
        raise FileExistsError("Use a new native SC donor preparation output")
    seed = spec.get("seed", 1729)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("SC donor seed must be an integer in [0, 2**31)")
    path, bound = mapping.validate(spec["mapping"], manifest.parent)
    paths = {key: verified_path(spec[key], manifest.parent)
             for key in ("donor_crosswalk", "graph_preflight", "candidates")}
    if paths["graph_preflight"].name != "LAUNCH_PREFLIGHT.json" or read(paths["graph_preflight"]).get("target_host") != "149":
        raise ValueError("SC static graph preflight must target 149")
    graph = paths["graph_preflight"].parent
    for source in (path.parent, graph):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("SC donor output must be separate from source directories")
    if any(output == p or output in p.parents for p in (manifest, *paths.values())):
        raise ValueError("SC donor output cannot contain its source files")
    mapped = read(bound["mapped_success"]["path"])
    value = {"input": record(manifest), "mapping": record(path), "seed": seed,
             "cancer_id": mapped["cancer_id"], "dataset_id": mapped["dataset_id"],
             "mapped_success": bound["mapped_success"], "graph_root": str(graph),
             **{key: record(path) for key, path in paths.items()},
             "target_host": "149", "training_started": False,
             "implementations": [record(ROOT / name) for name in CODE]}
    metadata, folds, plans = derived(value)
    current, _ = build_settings(value, output)
    ids = read(Path(value["mapped_success"]["path"]).parent / "IDS.json")
    static, receipt, _, _ = native._validate_static(current, value["cancer_id"], ids)
    value["static_sources"] = [record(static / "SUCCESS.json"),
                               *[record(verified_path(pin, static)) for pin in receipt["files"].values()]]
    candidates = pd.read_parquet(paths["candidates"])
    columns = ["cancer_id", "lncrna_id", "pathway_id"]
    if not set(columns) <= set(candidates) or candidates[columns].isna().any().any() or candidates.duplicated(columns).any():
        raise ValueError("Invalid SC candidate identities")
    selected = candidates.loc[candidates.cancer_id.eq(value["cancer_id"])]
    if selected.empty or not set(selected.lncrna_id) <= set(ids["lncrna_ids"]) or not set(selected.pathway_id) <= set(ids["pathway_ids"]):
        raise ValueError("SC candidates do not match mapped identities")
    value.update(canonical_donors=int(metadata.donor_id.nunique()), native_donors=int(metadata.native_donor_id.nunique()),
                 active_outer_folds=int(folds.outer_fold_id.nunique()), nested_plans=len(plans),
                 source_ancestry_complete=False, cross_module_patient_closure_verified=False)
    return value


def nested_preflight(value, output, binding):
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
    return {"format": native.NESTED_PREFLIGHT_FORMAT, "target_host": "149", "cpu_only": True,
            "paid_compute": False, "formal_pointer_update_authorized": False,
            "source_metacell_result_root": str(output), "graph_asset_root": value["graph_root"],
            "candidate_path": value["candidates"]["path"], "native_donor_preparation": record(binding),
            "materializer": native.file_record(ROOT / "scripts/materialize_single_cell_nested_hhgt_episodes_149.py"),
            "nested_module": native.file_record(ROOT / "cc_hhgt/v32/single_cell_nested_episodes.py"),
            "bindings": {"graph_asset_preflight": native.file_record(Path(value["graph_preflight"]["path"])),
                         "candidate": native.file_record(Path(value["candidates"]["path"]))}}


def execution_command(value, output):
    _, argv = build_settings(value, output)
    argv[argv.index("--deadline") + 1] = "<UNIX_DEADLINE>"
    return {"argv": [sys.executable, "-m", "scripts.materialize_single_cell_nested_hhgt_episodes_149", *argv],
            "automatically_executed": False, "fits_pca": True, "fits_lasso": False, "fits_neural_model": False,
            "requires_separate_explicit_execution": True}


def run(manifest, output):
    value = plan(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    write(output / "LAUNCH_PREFLIGHT.json", value)
    metadata, folds, plans = derived(value)
    metadata.to_parquet(output / "CANONICAL_CELL_METADATA.parquet", index=False)
    canonical_crosswalk(metadata).to_parquet(output / "CANONICAL_DONOR_CROSSWALK.parquet", index=False)
    folds.to_parquet(output / "DONOR_FOLDS.parquet", index=False)
    write(output / "NESTED_PLANS.json", plans)
    write(output / "EXECUTION_COMMAND.json", execution_command(value, output))
    if value != plan(manifest, output, new=False):
        raise ValueError("SC donor preparation sources changed during execution")
    binding = output / "NATIVE_SINGLE_CELL_DONORS_BINDING.json"
    write(binding, {"format": FORMAT, "status": "PASS", "target_host": "149", "training_started": False,
                   "preflight": record(output / "LAUNCH_PREFLIGHT.json"),
                   "files": {name: record(output / name) for name in sorted(FILES)},
                   "source_ancestry_complete": False, "nested_geometry_executed": False})
    (output / "nested").mkdir()
    write(output / "nested/LAUNCH_PREFLIGHT.json", nested_preflight(value, output, binding))
    validate(record(binding), output)
    return binding


def validate(item, root):
    require_host(); path = verified_path(item, root); bound = read(path); output = path.parent
    if (bound.get("format") != FORMAT or bound.get("status") != "PASS" or bound.get("target_host") != "149"
            or any(bound.get(key) is not False for key in ("training_started", "source_ancestry_complete", "nested_geometry_executed"))):
        raise ValueError("Invalid SC donor preparation binding")
    preflight = verified_path(bound["preflight"], output); value = read(preflight)
    if preflight != output / "LAUNCH_PREFLIGHT.json" or value != plan(verified_path(value["input"], output), output, new=False):
        raise ValueError("SC donor preparation input or implementation differs")
    if set(bound["files"]) != FILES:
        raise ValueError("SC donor preparation inventory differs")
    for name, pin in bound["files"].items():
        if verified_path(pin, output) != output / name:
            raise ValueError("SC donor preparation output location differs")
    metadata, folds, plans = derived(value)
    try:
        pd.testing.assert_frame_equal(metadata, pd.read_parquet(output / "CANONICAL_CELL_METADATA.parquet"), check_exact=True)
        pd.testing.assert_frame_equal(folds, pd.read_parquet(output / "DONOR_FOLDS.parquet"), check_exact=True)
        pd.testing.assert_frame_equal(canonical_crosswalk(metadata), pd.read_parquet(output / "CANONICAL_DONOR_CROSSWALK.parquet"), check_exact=True)
    except AssertionError as error:
        raise ValueError("SC canonical donor preparation differs from mapped source replay") from error
    if plans != read(output / "NESTED_PLANS.json"):
        raise ValueError("SC nested donor plans differ from source replay")
    if read(output / "EXECUTION_COMMAND.json") != execution_command(value, output):
        raise ValueError("SC nested execution command differs")
    if read(output / "nested/LAUNCH_PREFLIGHT.json") != nested_preflight(value, output, path):
        raise ValueError("SC nested launch preflight differs")
    return path, bound


def consumer(settings, receipt):
    """Verify the exact original CLI configuration before it can fit geometry."""
    path, bound = validate(receipt["native_donor_preparation"], settings.output_root)
    value = read(bound["preflight"]["path"])
    if not math.isfinite(settings.deadline) or settings.deadline <= 0:
        raise ValueError("Native nested execution requires a finite positive deadline")
    expected, _ = build_settings(value, path.parent, deadline=settings.deadline)
    if vars(settings) != vars(expected) or receipt != nested_preflight(value, path.parent, path):
        raise ValueError("Native nested materializer configuration differs from preparation")
    metadata, folds, _ = derived(value)
    return metadata, folds, {"dataset_id": value["dataset_id"], "source_receipt_path": str(path)}
