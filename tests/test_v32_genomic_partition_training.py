from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.v32.genomic_partition_training import collect_examples_partitioned
from cc_hhgt.v32.genomic_resource_staging import artifact_sha256
from cc_hhgt.v32.genomic_training import (
    EmbeddingLookup,
    FoldCoreEmbeddings,
    _candidate_core,
    _collect_examples,
    normalise_candidates,
)


class _DeterministicCompactCalls:
    def candidate_statistics_arrays(self, candidates, patients, min_pair_callable):
        rows = np.arange(len(candidates), dtype=np.float32)
        domain = np.column_stack(
            [rows + offset / 10.0 for offset in range(6)]
        ).astype(np.float32)
        labels = (np.arange(len(candidates)) % 2).astype(np.float32)
        available = np.ones(len(candidates), dtype=bool)
        reasons = np.full(len(candidates), "", dtype=object)
        return domain, labels, available, reasons


def _core(candidates: pd.DataFrame) -> FoldCoreEmbeddings:
    lnc_ids = sorted(set(candidates.lncrna_id) - {"L_BAD_BRCA", "L_BAD_LUAD"})
    pathway_ids = sorted(set(candidates.pathway_id) - {"P_BAD"})
    lnc_values = np.arange(len(lnc_ids) * 3, dtype=np.float32).reshape(len(lnc_ids), 3)
    pathway_values = np.arange(len(pathway_ids) * 3, dtype=np.float32).reshape(
        len(pathway_ids), 3
    )
    return FoldCoreEmbeddings(
        fold=0,
        lncrna=EmbeddingLookup(
            tuple(lnc_ids), lnc_values, {value: index for index, value in enumerate(lnc_ids)}
        ),
        pathway=EmbeddingLookup(
            tuple(pathway_ids),
            pathway_values,
            {value: index for index, value in enumerate(pathway_ids)},
        ),
        checkpoint_sha256="1" * 64,
        core_parameter_sha256="2" * 64,
        input_hashes={},
    )


def test_partition_collector_exact_rng_offset_concat_and_core_filter(tmp_path: Path) -> None:
    counts = {"BRCA": 7, "COAD": 4, "LUAD": 9}
    rows = []
    for cancer, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "cancer_id": cancer,
                    "lncrna_id": (
                        f"L_BAD_{cancer}" if cancer in {"BRCA", "LUAD"} and index == 0 else f"L_{cancer}_{index}"
                    ),
                    "pathway_id": "P_BAD" if cancer == "LUAD" and index == 1 else f"P_{index % 3}",
                }
            )
    candidates = normalise_candidates(pd.DataFrame(rows))
    records = []
    partition_root = tmp_path / "partitions"
    for cancer, local in candidates.groupby("cancer_id", observed=True, sort=False):
        path = partition_root / f"cancer_id={cancer}" / "part-0.parquet"
        path.parent.mkdir(parents=True)
        local.to_parquet(path, index=False)
        records.append(
            {
                "cancer_id": str(cancer),
                "rows": len(local),
                "path": str(path),
                "sha256": artifact_sha256(path),
            }
        )
    expected_order = tuple(candidates.cancer_id.unique())
    assert tuple(record["cancer_id"] for record in records) == expected_order
    assert [record["rows"] for record in records] == [7, 4, 9]

    calls = _DeterministicCompactCalls()
    # The middle cancer intentionally has zero eligible rows.  It must still
    # consume RNG offset=1 so LUAD uses seed+2 exactly as the original loop.
    lnc_by_cancer = {"BRCA": calls, "LUAD": calls}
    path_by_cancer = {"BRCA": calls, "LUAD": calls}
    patients = {cancer: [f"{cancer}_S0", f"{cancer}_S1"] for cancer in counts}
    core = _core(candidates)
    core_available = []
    for _, local in candidates.groupby("cancer_id", observed=True, sort=False):
        core_available.extend(_candidate_core(local, core)[1].tolist())
    assert any(core_available) and not all(core_available)

    expected = _collect_examples(
        candidates,
        lnc_by_cancer,
        path_by_cancer,
        patients,
        core,
        modality="cnv",
        min_pair_callable=2,
        maximum=30,
        seed=101,
    )
    observed = collect_examples_partitioned(
        records,
        lnc_by_cancer,
        path_by_cancer,
        patients,
        core,
        modality="cnv",
        min_pair_callable=2,
        maximum=30,
        seed=101,
    )
    for original, partitioned in zip(expected, observed, strict=True):
        assert original.dtype == partitioned.dtype
        assert original.shape == partitioned.shape
        assert original.tobytes(order="C") == partitioned.tobytes(order="C")
