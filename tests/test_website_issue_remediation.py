from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative: str, module_name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app_tree() -> ast.Module:
    return ast.parse(
        (ROOT / "website/backend/app.py").read_text(encoding="utf-8")
    )


def _assigned_literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment not found: {name}")


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_predicted_candidates_are_not_swallowed_by_supported_class() -> None:
    module = _load_script(
        "scripts/52_materialize_v29_web_tables.py", "materialize_v29_for_test"
    )
    frame = pd.DataFrame(
        {
            "direct_evidence_probability": [0.80, np.nan, 0.50, 0.10],
            "evidence_integrated_probability": [np.nan, np.nan, 0.50, 0.10],
            "discovery_ranking_probability": [0.95, 0.92, 0.70, 0.50],
            "fused_confidence_probability": [0.95, 0.93, 0.85, 0.30],
        }
    )
    classes = module._classify_relationships(frame, {})
    assert classes.tolist() == [
        "observed_core",
        "predicted_candidate",
        "model_supported",
        "exploratory",
    ]
    assert int(classes.eq("predicted_candidate").sum()) == 1
    assert int(classes.ne("exploratory").sum()) == 3


def test_relationship_primary_keys_fail_closed_on_duplicates() -> None:
    module = _load_script(
        "scripts/52_materialize_v29_web_tables.py",
        "materialize_v29_duplicate_key_test",
    )
    frame = pd.DataFrame(
        {
            "cancer_id": ["LUAD", "LUAD"],
            "lncrna_id": ["LNC:1", "LNC:1"],
            "pathway_family_id": ["PF:1", "PF:1"],
        }
    )
    try:
        module._assert_unique_keys(
            frame,
            ["cancer_id", "lncrna_id", "pathway_family_id"],
            "formal relationships",
        )
    except ValueError as exc:
        assert "duplicate relationship keys" in str(exc)
        assert "LUAD" in str(exc)
    else:
        raise AssertionError("duplicate formal relationship keys were accepted")


def test_clinical_events_are_counted_once_per_patient_endpoint() -> None:
    module = _load_script(
        "scripts/67_materialize_v30_clinical_web_tables.py",
        "materialize_v30_clinical_for_test",
    )
    patient = pd.DataFrame(
        {
            "cancer_id": ["LUAD"] * 6,
            "endpoint": ["OS"] * 5 + ["PFI"],
            "patient_id": ["P1", "P1", "P1", "P2", "P2", "P1"],
            "endpoint_available": [1, 1, 1, 1, 1, 0],
            "event": [1, 1, 1, 0, 0, 1],
            "risk_score": [0.8, 0.7, 0.9, 0.2, 0.3, 0.5],
            "fold_c_index": [0.6, 0.6, 0.6, 0.6, 0.6, 0.5],
        }
    )
    summary = module._patient_endpoint_summary(patient)
    assert summary[["cancer_id", "endpoint"]].values.tolist() == [["LUAD", "OS"]]
    row = summary.iloc[0]
    assert row.n_patients == 2
    assert row.n_events == 1
    assert row.n_events <= row.n_patients


def test_chinese_cancer_aliases_are_complete_and_searchable() -> None:
    tree = _app_tree()
    aliases = _assigned_literal(tree, "CANCER_SEARCH_ALIASES")
    assert len(aliases) == 33
    assert "肺癌" in aliases["LUAD"]
    assert "乳腺癌" in aliases["BRCA"]

    method = _class_method(tree, "SiteStore", "search")
    isolated = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {}
    exec(compile(isolated, "app.py:SiteStore.search", "exec"), namespace)
    store = SimpleNamespace(
        _search_rows=[
            {
                "type": "Cancer",
                "id": "LUAD",
                "label": "Lung adenocarcinoma",
                "subtitle": "LUAD",
                "search_aliases": aliases["LUAD"],
            }
        ]
    )
    results = namespace["search"](store, "肺癌", 12)
    assert [row["id"] for row in results] == ["LUAD"]


def test_download_catalog_is_recomputed_from_filesystem_truth(tmp_path: Path) -> None:
    tree = _app_tree()
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "available_downloads"
    )
    module_tree = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module_tree)
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module_tree, "app.py:available_downloads", "exec"), namespace)

    live = tmp_path / "live.tsv"
    missing = tmp_path / "deleted.parquet"
    live.write_text("a\tb\n", encoding="utf-8")
    namespace["DOWNLOADS"] = {"live": live, "stale": missing}
    first = namespace["available_downloads"]()
    assert [row["key"] for row in first] == ["live"]

    live.unlink()
    missing.write_bytes(b"PAR1")
    second = namespace["available_downloads"]()
    assert [row["key"] for row in second] == ["stale"]


def test_issue_remediation_routes_are_registered_in_source() -> None:
    tree = _app_tree()
    paths = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            if isinstance(decorator.args[0], ast.Constant):
                paths.add(decorator.args[0].value)
    assert {
        "/api/site/lncrna/{lncrna}/visuals",
        "/api/site/genesets/{geneset_id}",
        "/api/site/sc-summary/{cancer}",
        "/api/site/sc-umap/{cancer}",
        "/api/site/clinical/visuals",
        "/api/site/downloads",
        "/api/search",
        "/api/cancers",
        "/api/cancers/{cancer_id}",
        "/api/downloads",
    }.issubset(paths)
