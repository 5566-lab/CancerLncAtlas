"""Small static bookkeeping checks, with JSON fixtures only and no assay data.

Run directly with unittest; do not import the scientific pytest environment.
The actual receipt CLI remains host149-only. These checks make no claim about
the current remote job, patient matrices, prepared tensors or model behaviour.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gene_receipts", ROOT / "scripts/inspect_v32_gene_node_run.py")
inspection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspection)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(root):
    folds = root / "folds"; folds.mkdir()
    channels = ["cnv_signed", "methylation_promoter_beta"]
    product = root / "product.json"
    product_pin = write(product, {"format": "GENE_PATIENT_ASSAYS_VERIFIED_V1", "status": "PASS",
        "target_host": "149", "training_started": False, "node_type": "gene", "gene_type": "protein_coding",
        "genome_build": "GRCh38", "matrix_orientation": "patient_by_gene",
        "source_artifacts": {"patient_folds": {"path": "unread-folds.tsv", "sha256": "c" * 64}},
        "units": {"cnv_signed": "log2_copy_ratio", "methylation_promoter_beta": "beta"}, "raw_file_count": 4,
        "partitions": [{"cancer_id": c, "channels": {k: {} for k in channels}} for c in ["C1", "C2"]]})
    binding = root / "binding.json"
    write(binding, {"target_host": "149", "training_started": False, "cnv_files": 2, "methylation_files": 2,
                    "current_patient_fold_source": {"path": "unread-folds.tsv", "sha256": "c" * 64}})
    rows, checks = [], []
    for fold in range(5):
        evidence = {"producer_format": "CANCERLNCATLAS_NODE_INPUT_PREPARATION_V1", "scope": "cohort",
                    "outer_fold": fold, "channels": channels, "training_started": False,
                    "cancer_id": "PANCAN_TRAINING_POOL", "cohort_eligibility": "TRAINING_TUMORS_ONLY",
                    "cancer_ids": ["C1", "C2"], "table_sha256": "b" * 64}
        node_pin = write(folds / f"evidence_{fold}.json", evidence)
        rows.append({"fold": fold, "path": f"unread-parent-{fold}.pt", "bytes": 123,
                     "sha256": "a" * 64, "node_omics": node_pin})
        check = {"fold": fold, "prepared_validator": "PASS", "gene_nodes": 3, "node_input_dim": 6,
                 "observed_gene_nodes_by_channel": {"cnv_signed": 3, "methylation_promoter_beta": 2},
                 "input_authority_hashes": {"node_omics_manifest_sha256": node_pin["sha256"],
                                            "node_omics_table_sha256": "b" * 64}, "training_started": False}
        write(folds / f"FOLD_{fold}_PREPARED_VALIDATION.json", check)
        checks.append(check)
    manifest = {"format": "GENE_NODE_PREPARED_INPUT_MANIFEST_V1", "target_host": "149", "training_started": False,
                "gene_product": product_pin, "fold_inputs": rows}
    manifest_pin = write(folds / "INPUT_MANIFEST.json", manifest)
    config_pin = write(folds / "MODEL_CONFIG_PATCH.json", {"task_contract": {"node_omics_channels": channels}})
    write(folds / "COMPLETE.json", {"target_host": "149", "status": "PASS", "training_started": False,
          "learned_embeddings_generated": False, "input_manifest": manifest_pin, "model_config_patch": config_pin,
          "folds": checks})
    return product, folds, binding


class ReceiptChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.product, self.folds, self.binding = fixture(self.root)

    def inspect(self):
        with patch.object(inspection.socket, "gethostname", return_value="149"):
            return inspection.inspect_run(self.product, self.folds, self.binding, expected_cancers=2)

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def change(self, path, change, *, repin=None):
        value = self.read(path); change(value); pin = write(path, value)
        if repin:
            complete = self.read(self.folds / "COMPLETE.json")
            complete[repin] = pin; write(self.folds / "COMPLETE.json", complete)

    def test_complete_records_pass_without_loading_large_artifacts(self):
        result = self.inspect()
        self.assertEqual(result["status"], "RECORDED_PASS")
        self.assertEqual(result["raw_file_count"], 4)
        self.assertEqual(len(result["folds"]), 5)
        self.assertFalse(result["large_artifact_bytes_reverified"])
        self.assertFalse(result["scientific_validation_repeated"])
        self.assertEqual(len(result["receipt_hashes"]), 15)

    def test_partial_directory_is_never_pass_or_running(self):
        (self.folds / "COMPLETE.json").unlink()
        result = self.inspect()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["process_state"], "NOT_INFERRED_FROM_PARTIAL_FILES")

    def test_wrong_host_stops_before_any_receipt_access(self):
        with patch.object(inspection.socket, "gethostname", return_value="Windows"), \
                patch.object(inspection.JsonReader, "read", side_effect=AssertionError("read attempted")):
            with self.assertRaisesRegex(ValueError, "requires host 149"):
                inspection.inspect_run(self.product, self.folds, self.binding, expected_cancers=2)

    def test_four_folds_are_not_complete(self):
        self.change(self.folds / "COMPLETE.json", lambda d: d["folds"].pop())
        self.assertIn("five rows required", self.inspect()["error"])

    def test_duplicate_fold_cannot_replace_a_missing_fold(self):
        self.change(self.folds / "COMPLETE.json", lambda d: d["folds"][4].update(fold=3))
        self.assertIn("exactly 0..4", self.inspect()["error"])

    def test_missing_individual_validation_is_invalid(self):
        (self.folds / "FOLD_4_PREPARED_VALIDATION.json").unlink()
        self.assertEqual(self.inspect()["status"], "INVALID_RECEIPTS")

    def test_failed_validation_cannot_be_hidden_by_complete_pass(self):
        complete = self.read(self.folds / "COMPLETE.json")
        complete["folds"][2]["prepared_validator"] = "FAILED"
        write(self.folds / "FOLD_2_PREPARED_VALIDATION.json", complete["folds"][2])
        write(self.folds / "COMPLETE.json", complete)
        self.assertIn("validation did not pass", self.inspect()["error"])

    def test_individual_validation_must_equal_completion(self):
        self.change(self.folds / "FOLD_2_PREPARED_VALIDATION.json", lambda d: d.update(node_input_dim=2))
        self.assertIn("receipt differs", self.inspect()["error"])

    def test_signed_cnv_and_beta_channel_order_is_fixed(self):
        self.change(self.folds / "MODEL_CONFIG_PATCH.json", lambda d: d["task_contract"]["node_omics_channels"].reverse(),
                    repin="model_config_patch")
        self.assertIn("channel order differs", self.inspect()["error"])

    def test_stale_manifest_hash_is_rejected(self):
        self.change(self.folds / "INPUT_MANIFEST.json", lambda d: d.update(training_started=True))
        self.assertIn("SHA256 mismatch", self.inspect()["error"])

    def test_pending_manifest_is_not_a_final_release(self):
        pin = write(self.folds / "INPUT_MANIFEST.pending.json", self.read(self.folds / "INPUT_MANIFEST.json"))
        self.change(self.folds / "COMPLETE.json", lambda d: d.update(input_manifest=pin))
        self.assertIn("pending manifest", self.inspect()["error"])

    def test_formal_folds_cannot_refer_to_another_product(self):
        pin = write(self.root / "other-product.json", self.read(self.product))
        self.change(self.folds / "INPUT_MANIFEST.json", lambda d: d.update(gene_product=pin), repin="input_manifest")
        self.assertIn("different matrix product", self.inspect()["error"])

    def test_raw_file_total_must_match_source_binding(self):
        self.change(self.binding, lambda d: d.update(methylation_files=3))
        self.assertIn("raw file total", self.inspect()["error"])

    def test_binding_cannot_substitute_another_patient_fold_authority(self):
        self.change(self.binding, lambda d: d["current_patient_fold_source"].update(sha256="d" * 64))
        self.assertIn("patient-fold authority differs", self.inspect()["error"])

    def test_rehashed_node_manifest_cannot_replace_loaded_node_authority(self):
        node = self.read(self.folds / "evidence_0.json")
        node["table_sha256"] = "e" * 64
        pin = write(self.folds / "evidence_0.json", node)
        self.change(self.folds / "INPUT_MANIFEST.json", lambda d: d["fold_inputs"][0].update(node_omics=pin), repin="input_manifest")
        self.assertIn("node manifest authority differs", self.inspect()["error"])

    def test_recorded_table_authority_must_match_evidence(self):
        complete = self.read(self.folds / "COMPLETE.json")
        complete["folds"][1]["input_authority_hashes"]["node_omics_table_sha256"] = "e" * 64
        write(self.folds / "FOLD_1_PREPARED_VALIDATION.json", complete["folds"][1])
        write(self.folds / "COMPLETE.json", complete)
        self.assertIn("node table authority differs", self.inspect()["error"])

    def test_six_columns_required_even_when_receipts_agree(self):
        complete = self.read(self.folds / "COMPLETE.json")
        complete["folds"][0]["node_input_dim"] = 2
        write(self.folds / "FOLD_0_PREPARED_VALIDATION.json", complete["folds"][0])
        write(self.folds / "COMPLETE.json", complete)
        self.assertIn("input width must be six", self.inspect()["error"])

    def test_completed_flag_cannot_claim_trained_embeddings(self):
        self.change(self.folds / "COMPLETE.json", lambda d: d.update(learned_embeddings_generated=True))
        self.assertIn("embedding declaration", self.inspect()["error"])

    def test_mutation_of_previously_read_receipt_is_detected(self):
        reader = inspection.JsonReader(); reader.read(self.binding)
        self.change(self.binding, lambda d: d.update(cnv_files=7))
        with self.assertRaisesRegex(ValueError, "changed during inspection"):
            reader.unchanged()

    def test_reader_bounds_file_size_and_rejects_non_object_json(self):
        with patch.object(inspection, "MAX_JSON_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "size limit"):
                inspection.JsonReader().read(self.binding)
        write(self.binding, [])
        with self.assertRaisesRegex(ValueError, "JSON object"):
            inspection.JsonReader().read(self.binding)

    def test_real_producer_format_constant_matches_fixture(self):
        tree = ast.parse((ROOT / "cc_hhgt/v32/omics_node_inputs.py").read_text(encoding="utf-8"))
        value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "FORMAT" for t in n.targets))
        self.assertEqual(value, self.read(self.folds / "evidence_0.json")["producer_format"])


if __name__ == "__main__":
    unittest.main()
