"""Bind corrected E1 edges to the normalization actually used by detection.

Patient identity agreement alone does not establish expression-version agreement.
Both lncRNA and coding-gene conversions must replay from the same fold's native
normalization, including cancers whose corrected edge partition is empty.
"""
from pathlib import Path
import pandas as pd

from .current_core_export import verified_path
from .input_lineage import artifact_sha256
from .formal_expression_sources import read, record

FORMAT = "FOLD_LOCAL_GRAPH_NORMALIZATION_MATCH_V1"


def artifact(item, root):
    path = Path(item["path"])
    path = (path if path.is_absolute() else Path(root)/path).resolve()
    if not path.exists() or artifact_sha256(path) != item["sha256"]:
        raise ValueError("Fold-local graph artifact SHA256 differs")
    return path


def verify(receipt_item, coexpression_item, scope_item, root, fold):
    from .fold_local_scope import validate_scope
    from .fold_local_expression import native, validate as validate_expression
    from .local_cnv_sources import validate as validate_local, partition_binding
    from .local_cnv_core_preparation import e1_coexpression_frame
    receipt_path = verified_path(receipt_item, root)
    consumed = artifact(coexpression_item, root)
    scope_path, scope = validate_scope(scope_item, root)
    if scope["fold"] != fold: raise ValueError("Graph normalization scope belongs to a different fold")
    document = read(receipt_path)
    declared = document.get("fold_artifacts", {}).get(str(fold), {}).get("train_coexpression")
    if (set(document.get("cancer_scope", [])) != set(scope["normalizations"])
            or not declared or artifact(declared, receipt_path.parent) != consumed):
        raise ValueError("Graph receipt and consumed coexpression scope differ")
    adjustment = document.get("local_cnv_adjustment", {})
    declarations = adjustment.get("source_receipts", [])
    if (adjustment.get("target") != "E1_LOCAL" or adjustment.get("e0_fallback_allowed") is not False
            or not declarations):
        raise ValueError("Fold-local graph requires native corrected E1 source receipts")
    paths = {p.resolve() for p in consumed.rglob("*.parquet")} if consumed.is_dir() else {consumed}
    if not paths: raise ValueError("Corrected graph requires explicit partitions, including empty cancers")
    observed = set(); cancers = {}
    for declaration in declarations:
        path = verified_path(declaration, receipt_path.parent); value = read(path)
        cancer = value.get("cancer_id")
        if (value.get("format") != "CANCERLNCATLAS_V32_LOCAL_CNV_E1_GRAPH_CANCER_V1"
                or value.get("status") != "PASS" or cancer not in scope["normalizations"] or cancer in cancers
                or value.get("gates", {}).get("e0_fallback_allowed") is not False):
            raise ValueError("Corrected graph cancer declaration differs from detection scope")
        outputs = value.get("outputs", [])
        if len(outputs) != 5 or {r.get("fold") for r in outputs} != set(range(5)):
            raise ValueError("Corrected graph producer must bind exactly five folds")
        output = next(r for r in outputs if r["fold"] == fold)
        current = verified_path(output, path.parent)
        if current not in paths or current in observed:
            raise ValueError("Corrected graph consumed partition differs from source declaration")
        observed.add(current)
        original = verified_path({"path": output["source_path"], "sha256": output["source_sha256"]}, path.parent)
        native_rows = pd.read_parquet(original)
        if not native_rows.cancer_id.eq(cancer).all(): raise ValueError("Native E1 source cancer differs")
        expected = e1_coexpression_frame(native_rows, expected_fold=fold)
        actual = pd.read_parquet(current)
        if len(actual) != output["rows"]: raise ValueError("Corrected graph source row count differs")
        try: pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
        except AssertionError as error: raise ValueError("Corrected graph differs from actual native E1 selection") from error
        binding = output.get("upstream_transformation")
        if not binding: raise ValueError("Corrected graph lacks native transformation ancestry")
        producer_path = verified_path(binding["receipt"], path.parent)
        if partition_binding(original, producer_path) != binding:
            raise ValueError("Corrected graph upstream partition binding differs")
        source_path, source = validate_local(binding["sources"], path.parent)
        matches = [r for r in source["outputs"] if Path(r["path"]).resolve() == original]
        if (source["producer"] != "association" or source["cancer_id"] != cancer or len(matches) != 1
                or matches[0]["output_kind"] != "graph" or matches[0]["fold"] != fold):
            raise ValueError("Corrected graph source is not the matching native graph transformation")
        norm_item = scope["normalizations"][cancer]
        norm_path, norm, _ = native(norm_item, scope_path.parent)
        patients = [r for r in source["records"] if r["role"] == "patient_map"]
        expected_patients = verified_path(norm["inputs"]["patient_map"], norm_path.parent)
        if len(patients) != 1 or verified_path(patients[0], source_path.parent) != expected_patients:
            raise ValueError("Graph and detection normalization patient maps differ")
        inputs = {}
        for role, kind in (("lncrna_expression", "lncrna"), ("gene_expression", "gene")):
            selected = [r for r in source["records"] if r["role"] == role and r.get("fold", -1) in {-1, fold}]
            if len(selected) != 1 or selected[0].get("fold") != fold or not selected[0].get("upstream_expression"):
                raise ValueError("Corrected graph requires native fold-local "+role)
            expression = selected[0]; actual_input = verified_path(expression, source_path.parent)
            conversion_path, conversion = validate_expression(expression["upstream_expression"], source_path.parent, actual_input, replay=True)
            if (conversion["kind"] != kind or conversion["fold"] != fold or conversion["cancer_id"] != cancer
                    or verified_path(conversion["normalization"], conversion_path.parent) != norm_path
                    or conversion["normalization"]["sha256"] != norm_item["sha256"]):
                raise ValueError("Graph and detection normalization sources differ: "+role)
            inputs[role] = {"table": record(actual_input), "conversion": record(conversion_path)}
        cancers[cancer] = {"normalization": record(norm_path), "patient_map": record(expected_patients),
            "expression_inputs": inputs, "e1_producer": record(path), "transformation": binding,
            "corrected_partition": record(current), "edge_rows": len(actual)}
    if observed != paths or set(cancers) != set(scope["normalizations"]):
        raise ValueError("Corrected graph cancer/partition coverage differs from detection scope")
    return {"format": FORMAT, "fold": fold, "scope": record(scope_path), "graph_receipt": record(receipt_path),
            "coexpression": {"path": str(consumed), "sha256": artifact_sha256(consumed)},
            "cancers": dict(sorted(cancers.items()))}


def validate_binding(binding, fold):
    """Portable structural validation; actual source replay is separate."""
    detection = binding.get("static_artifacts", {}).get("detection", {})
    if not detection.get("upstream_scope"): return
    proof = binding.get("fold_normalization_match", {})
    if (proof.get("format") != FORMAT or proof.get("fold") != fold
            or proof.get("scope") != detection["upstream_scope"]
            or proof.get("graph_receipt") != binding.get("receipt")
            or proof.get("coexpression") != binding.get("fold_artifacts", {}).get("train_coexpression")
            or not proof.get("cancers") or binding.get("gates", {}).get("matching_fold_expression_normalization") is not True):
        raise ValueError("Native graph lacks matching fold-normalization evidence")
    for cancer, item in proof["cancers"].items():
        if (not cancer or set(item) != {"normalization", "patient_map", "expression_inputs", "e1_producer",
                "transformation", "corrected_partition", "edge_rows"}
                or set(item["expression_inputs"]) != {"lncrna_expression", "gene_expression"}
                or type(item["edge_rows"]) is not int or item["edge_rows"] < 0):
            raise ValueError("Native graph fold-normalization evidence is incomplete")


def replay_binding(binding, root, fold, *, scope_item=None):
    """Use the actual graph base when an experimental/predicted arm wraps it."""
    if binding.get("base_authority") is not None:
        return replay_binding(binding["base_authority"], root, fold, scope_item=scope_item)
    detection = binding.get("static_artifacts", {}).get("detection", {})
    if not detection.get("upstream_scope"):
        if scope_item is not None: raise ValueError("Native prediction scope lacks its native graph detection")
        return None
    validate_binding(binding, fold)
    if scope_item is not None and detection["upstream_scope"] != scope_item:
        raise ValueError("Prepared graph and prediction detection scopes differ")
    result = verify(binding["receipt"], binding["fold_artifacts"]["train_coexpression"],
                    detection["upstream_scope"], root, fold)
    if result != binding["fold_normalization_match"]:
        raise ValueError("Prepared graph normalization evidence differs from actual inputs")
    return result
