import unittest
import tempfile
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.single_cell_manifest_reconciliation import adapt_verified_head_tables, sync_verified_manifest, verify_file
from cc_hhgt.v32.single_cell_input_builder import EXPECTED_CANCERS, validate_dataset_manifest
from cc_hhgt.v32.single_cell_partition_builder import _manifest_row
from cc_hhgt.v32.single_cell_training import normalise_dataset_manifest, normalise_single_cell_associations, exact_candidate_join, _normalise_lnc_celltype, build_domain_features


def manifest(cancer, count=2909, donor=False):
    return pd.DataFrame([dict(cancer_id=cancer, dataset_id="SC_"+cancer,
        formal_eligible=False, source_tier="approved", quality_status="LIMITED",
        feature_universe_status="LIMITED", donor_metadata_available=donor,
        lncrna_feature_universe_count=count, quality_flags="KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION")])


class ManifestTests(unittest.TestCase):
    def test_real_source_schema_reaches_candidates_and_domain_features(self):
        candidates=pd.DataFrame([dict(cancer_id="CESC",lncrna_id="LNC:ENSG1",pathway_id="P1")])
        raw=manifest("CESC")
        fixed,_=sync_verified_manifest(raw,{"CESC":dict(dataset_id="SC_CESC",lncrnas=3098,donor_metadata_available=True)})
        qualified=normalise_dataset_manifest(fixed)
        assoc=pd.DataFrame([dict(cancer_id="CESC",dataset_id="SC_CESC",lncrna_id="ENSG1",
            pathway_id="P1",compartment="immune",n_donors=10,spearman_rho=.5,bh_q_global_tests=.02)])
        cell=pd.DataFrame([dict(cancer_id="CESC",dataset_id="SC_CESC",lncrna_id="ENSG1",
            patient_id="donor1",compartment="immune",cell_count=n,detected_cell_count=d,
            mean_expression=e,expression_summary_scale="MEAN_LOG1P_CPM10000") for n,d,e in [(10,5,2.),(30,3,4.)]])
        adapted,pooled=adapt_verified_head_tables(assoc,cell,candidates,qualified)
        self.assertEqual(pooled.iloc[0].n_cells,40)
        self.assertAlmostEqual(pooled.iloc[0].detection_rate,.2)
        self.assertAlmostEqual(pooled.iloc[0].mean_log_expression,3.5)
        norm=normalise_single_cell_associations(adapted,qualified)
        self.assertTrue(norm.formal_row.all())
        self.assertEqual(norm.iloc[0].n_observations,10)
        joined=exact_candidate_join(norm,candidates)
        self.assertEqual(len(joined),1)
        features=build_domain_features(joined,_normalise_lnc_celltype(pooled),pd.DataFrame())
        self.assertEqual(features[0,7],1.)
        self.assertAlmostEqual(float(features[0,0]),.2)
        self.assertEqual(float(features[0,1]),3.5)
        cell["expression_summary_scale"]="MEAN_SOURCE_LOG_NORMALIZED"
        _,pooled=adapt_verified_head_tables(assoc,cell,candidates,qualified)
        self.assertEqual(pooled.iloc[0].mean_log_expression,3.5)
        cell.loc[0,"expression_summary_scale"]="MEAN_LOG1P_CPM10000"
        with self.assertRaisesRegex(ValueError,"Mixed"):
            adapt_verified_head_tables(assoc,cell,candidates,qualified)

    def test_all_cancer_names_can_pass_with_sufficient_features(self):
        rows=[dict(dataset_id="SC_"+c,cancer_id=c,expression_source_tier="raw_counts",
            measurement_scale="counts",source_generation="SOURCE_RAW_EXPRESSION_COUNTS",
            formal_eligible=True,quality_status="PASS",model_derived=False,outcome_derived=False,
            lncrna_feature_universe_count=2500,donor_metadata_available=True) for c in EXPECTED_CANCERS]
        result=validate_dataset_manifest(pd.DataFrame(rows))
        self.assertEqual(len(result),33)
        self.assertTrue(result.feature_universe_status.eq("PASS").all())
        self.assertTrue(normalise_dataset_manifest(result).qualified.all())

    def test_tampered_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"receipt.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                verify_file(path,"0"*64)

    def test_verified_rescues_enter_real_consumer(self):
        for cancer, count in (("CESC",3098),("UVM",11826),("BRCA",2901),("COAD",2901),("OV",2901),("TGCT",15565)):
            with self.subTest(cancer=cancer):
                fixed, changes = sync_verified_manifest(manifest(cancer), {
                    cancer: dict(dataset_id="SC_"+cancer, lncrnas=count, donor_metadata_available=True)})
                self.assertTrue(normalise_dataset_manifest(fixed).qualified.all())
                self.assertEqual(int(fixed.iloc[0].lncrna_feature_universe_count),count)
                self.assertTrue(changes)

    def test_no_evidence_does_not_promote_other_cancers(self):
        raw=manifest("BLCA")
        fixed, changes=sync_verified_manifest(raw,{})
        pd.testing.assert_frame_equal(raw,fixed)
        self.assertFalse(changes)

    def test_wrong_identity_or_bad_evidence_rejected(self):
        for dataset,count,donor in [("SC_OTHER",3098,True),("SC_CESC",47,True),("SC_CESC",3098,False)]:
            with self.subTest(dataset=dataset,count=count,donor=donor), self.assertRaises(ValueError):
                sync_verified_manifest(manifest("CESC"),{"CESC":dict(dataset_id=dataset,lncrnas=count,donor_metadata_available=donor)})

    def test_feature_gate_depends_on_measurement_not_cancer_name(self):
        for cancer in ("CESC","UVM","UCS","BRCA"):
            for count,expected in ((47,False),(999,False),(1000,True),(11826,True)):
                with self.subTest(cancer=cancer,count=count):
                    row=_manifest_row(dataset_id="SC_"+cancer,cancer=cancer,source_tier="raw_counts",
                        measurement_scale="raw_counts",lnc_feature_count=count,
                        donor_metadata_available=True,formal_eligible=True,quality_flags=[])
                    self.assertEqual(bool(row.iloc[0].formal_eligible),expected)

    def test_sync_is_idempotent(self):
        ev={"CESC":dict(dataset_id="SC_CESC",lncrnas=3098,donor_metadata_available=True)}
        first,_=sync_verified_manifest(manifest("CESC"),ev)
        second,changes=sync_verified_manifest(first,ev)
        pd.testing.assert_frame_equal(first,second)
        self.assertFalse(changes)


if __name__ == "__main__":
    unittest.main()
