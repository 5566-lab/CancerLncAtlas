from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket

import pandas as pd
import pyarrow.parquet as pq

from cc_hhgt.v32.single_cell_graph_assets import normalize_pathway_families


SOURCE_FOLD = Path(
    "/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/"
    "c_graph_global_binding_20260928_r1/results/"
    "single_cell_metacell_v32-single-cell-metacell-donor-oof-20261005-r1/"
    "cancers/BLCA/outer_fold=0"
)
FAMILY = Path(
    "/public8/DSC/CancerLncAtlas/results/model/cc_hhgt_v2_9_state_graph/"
    "tables/pathway_family_member.parquet"
)
MEMBERSHIP = Path(
    "/dell_2/DSC/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r7_portable_supersession_20260829/authority/"
    "exact_pathway_membership_2135.parquet"
)
EXTERNAL = Path("/public8/DSC/CancerLncAtlas/parquet/interaction_relation.parquet")
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_single_cell_hhgt_assets_149.py"
)


def test_real_blca_fold_static_contracts_on_149():
    assert socket.gethostname() == "149"
    receipt = json.loads((SOURCE_FOLD / "SUCCESS.json").read_text())
    membership = pd.read_parquet(SOURCE_FOLD / "METACELL_MEMBERSHIP.parquet")
    metadata = pd.read_parquet(SOURCE_FOLD / "METACELL_METADATA.parquet")
    assigned = membership.loc[membership.metacell_id.ge(0)].copy()
    grouped = assigned.groupby("metacell_id", observed=True).agg(
        n_cells=("cell_id", "size"),
        donors=("donor_id", "nunique"),
        contexts=("cell_type", "nunique"),
        states=("state_id", "nunique"),
    )
    observed = metadata.set_index("metacell_id").loc[grouped.index]
    assert grouped[["donors", "contexts", "states"]].eq(1).all().all()
    assert grouped.n_cells.equals(observed.n_cells.rename("n_cells"))
    assert set(receipt["source_donors"]).isdisjoint(receipt["query_donors"])

    universe = pd.read_parquet(SOURCE_FOLD / "UNIVERSE.parquet")
    pathways = sorted(universe.pathway_id.astype(str).unique())
    family_membership, full, families = normalize_pathway_families(
        pd.read_parquet(FAMILY), pathways
    )
    assert family_membership.shape == (len(pathways), len(families))
    assert family_membership.nnz == len(full)
    assert (family_membership.getnnz(axis=1) > 0).all()
    assert family_membership.nnz > len(pathways) // 2
    assert len(families) > 1

    exact = pd.read_parquet(MEMBERSHIP, columns=["gene_id"])
    lncs = sorted(universe.lncrna_id.astype(str).unique())
    genes = sorted(exact.gene_id.astype(str).unique())
    spec = importlib.util.spec_from_file_location("graph_asset_materializer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parquet = pq.ParquetFile(EXTERNAL)
    assert parquet.metadata.num_rows == 6_160_707
    sample = next(parquet.iter_batches(columns=module.EXTERNAL_COLUMNS, batch_size=262144))
    sample = sample.to_pandas()
    eligible = sample.loc[
        sample.mapping_status.astype(str).eq("lncrna_and_partner_mapped")
        & sample.is_experimental.fillna(False).astype(bool)
        & ~sample.is_predicted.fillna(True).astype(bool)
    ]
    assert not eligible.empty
    primary = module._primary_physical_binding(eligible)
    assert not primary.empty
    assert primary.relation.eq("external_physical_binding").all()
    assert primary[["source_id", "target_id"]].duplicated().sum() == 0
    first = primary.iloc[0]
    mapped = module._map_external_to_nodes(
        primary.iloc[[0]],
        lncrna_ids=[first.source_id],
        gene_ids=[first.target_id],
    )
    assert mapped[["source_index", "target_index"]].iloc[0].tolist() == [0, 0]
    edges, schema = module.normalize_external_evidence(
        eligible, lncrna_ids=lncs, gene_ids=genes
    )
    assert len(schema) > 1
    assert edges.relation.isin(set(schema.relation)).all()
    assert edges[["source_id", "target_id", "relation"]].duplicated().sum() == 0
