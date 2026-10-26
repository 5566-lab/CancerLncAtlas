"""Private paired-patient scopes observed in the native State estimator.

Rows refer to the loaded fold authority, not individual contributing aliquots.
Equal patient masks share a record across lncRNAs. This describes the statistical
inputs and head eligibility, never optimizer execution or core training ancestry.
"""
import json

import numpy as np
import pandas as pd

from .contracts import canonical_json_sha256
from .state_training import validate_fold_manifest

FORMAT = "NATIVE_STATE_STATISTICAL_PATIENT_SCOPES_V1"
TABLE = "state_patient_scopes_private"
COLUMNS = ["format", "patient_fold_id", "split_role", "statistic_role", "cancer_id", "state_id",
    "patient_map_sha256", "cohort_id", "patient_map_row_indices_json", "native_patient_count",
    "lncrna_ids_json", "candidate_count", "statistic_available_json", "core_embedding_available_json",
    "private_head_input_eligible_json", "core_embedding_sha256", "core_parameter_sha256",
    "v32_core_checkpoint_sha256"]
ROLES = {"train": "TRAINING_FEATURE_AND_LABEL", "validation": "VALIDATION_FEATURE_AND_LABEL",
         "test": "HELD_OUT_EFFECT"}


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class Trace:
    def __init__(self, raw_patients, patient_pin, cores, parents, validation_offset):
        validated = validate_fold_manifest(raw_patients).set_index("sample_id")
        # Restore the original loaded authority order after native validation.
        self.patients = validated.loc[raw_patients.sample_id.astype(str)].reset_index()
        self.patient_pin = patient_pin
        self.cores = {fold: set(frame.node_id.astype(str)) for fold, frame in cores.items()}
        self.parents = parents
        self.offset = validation_offset
        self.events = []

    def observer(self, fold):
        def observe(cancer, state, split, lncrnas, patients, mask, counts, available):
            if split not in ROLES or fold not in range(5):
                raise ValueError("Unknown State patient scope split")
            patients = np.asarray(patients, dtype=str); lncrnas = np.asarray(lncrnas, dtype=str)
            if len(set(patients)) != len(patients) or len(set(lncrnas)) != len(lncrnas):
                raise ValueError("State patient scope has duplicate identities")
            if (mask.dtype != bool or mask.shape != (len(patients), len(lncrnas))
                    or np.shape(available) != (len(lncrnas),)
                    or not np.array_equal(mask.sum(axis=0), counts)):
                raise ValueError("State patient scope mask or counts differ")
            validation = (fold + self.offset) % 5
            allowed = {"train": set(range(5)) - {fold, validation}, "validation": {validation}, "test": {fold}}[split]
            authority = self.patients.cancer_id.eq(cancer)
            # Check every aligned row, including rows excluded by all masks.
            relevant = authority & self.patients.patient_id.isin(patients)
            if (set(self.patients.loc[relevant, "patient_id"]) != set(patients)
                    or not set(self.patients.loc[relevant, "patient_fold_id"]) <= allowed):
                raise ValueError("State patient scope is unknown or crosses its declared split")
            _, inverse = np.unique(np.packbits(mask, axis=0).T, axis=0, return_inverse=True)
            for group in range(int(inverse.max()) + 1 if len(inverse) else 0):
                columns = np.flatnonzero(inverse == group)
                selected = patients[mask[:, columns[0]]]
                rows = np.flatnonzero((authority & self.patients.patient_id.isin(selected)).to_numpy()).tolist()
                cohort = canonical_json_sha256({"patient_map_sha256": self.patient_pin["sha256"], "rows": rows})
                core = np.asarray([name in self.cores[fold] for name in lncrnas[columns]], dtype=bool)
                finite = np.asarray(available, dtype=bool)[columns]
                eligible = finite & core if split != "test" else np.zeros(len(columns), dtype=bool)
                parent = self.parents[fold]
                self.events.append({"format": FORMAT, "patient_fold_id": fold, "split_role": split,
                    "statistic_role": ROLES[split], "cancer_id": cancer, "state_id": state,
                    "patient_map_sha256": self.patient_pin["sha256"], "cohort_id": cohort,
                    "patient_map_row_indices_json": encoded(rows), "native_patient_count": len(selected),
                    "lncrna_ids_json": encoded(lncrnas[columns].tolist()), "candidate_count": len(columns),
                    "statistic_available_json": encoded(finite.tolist()),
                    "core_embedding_available_json": encoded(core.tolist()),
                    "private_head_input_eligible_json": encoded(eligible.tolist()),
                    **{key: parent[key] for key in ("core_embedding_sha256", "core_parameter_sha256",
                                                   "v32_core_checkpoint_sha256")}})
        return observe

    def table(self):
        return pd.DataFrame(self.events, columns=COLUMNS)
