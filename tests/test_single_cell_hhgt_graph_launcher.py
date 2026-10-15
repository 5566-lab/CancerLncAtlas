from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import launch_single_cell_hhgt_graph_assets_batch_149 as launcher


def test_cancer_success_binds_static_and_exact_source_fold_inventory(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    cancer = "C1"
    (source / "cancers" / cancer / "outer_fold=0").mkdir(parents=True)
    (source / "cancers" / cancer / "outer_fold=1").mkdir(parents=True)
    (source / "cancers" / cancer / "SUCCESS.json").write_text("source")
    cancer_root = output / "cancers" / cancer
    (cancer_root / "STATIC").mkdir(parents=True)
    (cancer_root / "STATIC" / "SUCCESS.json").write_text(
        json.dumps(
            {
                "state": "CANCER_STATIC_GRAPH_ASSETS_READY",
                "cancer_id": cancer,
                "pathways": 2,
                "pathway_family_edges": 2,
                "pathways_without_family_edges": 0,
            }
        )
    )
    for fold in (0, 1):
        root = cancer_root / f"outer_fold={fold}"
        root.mkdir(parents=True)
        (root / "SUCCESS.json").write_text(
            json.dumps(
                {
                    "state": "FOLD_GRAPH_ASSETS_READY",
                    "cancer_id": cancer,
                    "outer_fold_id": fold,
                }
            )
        )
    output.mkdir(exist_ok=True)
    (output / "LAUNCH_PREFLIGHT.json").write_text("preflight")
    supervisor = launcher.Supervisor(
        SimpleNamespace(
            output_root=output,
            source_result_root=source,
            workers=1,
            threads_per_job=1,
        ),
        [cancer],
    )
    record = supervisor.write_cancer_success(cancer, [0, 1])
    receipt = json.loads((cancer_root / "SUCCESS.json").read_text())
    assert record == launcher.file_record(cancer_root / "SUCCESS.json")
    assert receipt["state"] == "CANCER_GRAPH_ASSETS_COMPLETE"
    assert receipt["outer_fold_ids"] == [0, 1]
    assert len(receipt["outer_fold_successes"]) == 2
    assert supervisor.write_cancer_success(cancer, [0, 1]) == record
