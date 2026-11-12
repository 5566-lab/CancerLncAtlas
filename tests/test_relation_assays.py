from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from cla_engine.relation_assays import SourceAssay, align_cptac_assay, append_assay_nodes


def test_alignment_keeps_unmeasured_patients_missing_and_uses_exact_order():
    frame = pd.DataFrame({'feature_id':['site1','site2'], 'P2::T':[4.,np.nan], 'P1::T':[2.,3.]})
    x = align_cptac_assay(frame, ('P1','P3','P2'))
    np.testing.assert_allclose(x, [[2.,3.],[np.nan,np.nan],[4.,np.nan]], equal_nan=True)
    with pytest.raises(ValueError, match='NONTUMOR'):
        align_cptac_assay(frame.rename(columns={'P1::T':'P1::N'}), ('P1',))


def test_antibody_group_is_one_measurement_with_two_component_edges():
    assay = SourceAssay('TCGA','phospho',('P1','P2'),('AKT-pS473','unmapped'),
        (('G1','G2'),('G3',)), np.array([[2.,np.nan],[4.,np.nan]]), {})
    ids = {'gene':['G1','G2']}
    graph = {'node_omics::gene':np.zeros((2,4),np.float32)}
    audit = append_assay_nodes(ids,graph,{'tcga_phospho_assay':assay})
    assert ids['tcga_phospho_assay'] == ['AKT-pS473','unmapped']
    np.testing.assert_allclose(graph['x::tcga_phospho_assay'], [[3.,1.,1.],[0.,0.,0.]])
    np.testing.assert_array_equal(graph['edge::tcga_phospho_assay|assay_group_component|gene'], [[0,0],[0,1]])
    assert audit['tcga_phospho_assay']['multi_gene_assays'] == 1
    assert audit['tcga_phospho_assay']['measurement_duplicated_per_gene'] is False
    assert not graph['node_omics::tcga_phospho_assay'].any()


def test_duplicate_measurements_are_rejected():
    assay = SourceAssay('CPTAC','protein',('P1',),('f','f'),(('G',),('G',)),np.ones((1,2)),{})
    with pytest.raises(ValueError, match='METADATA_AXIS'):
        append_assay_nodes({'gene':['G']}, {}, {'cptac_protein_assay':assay})
