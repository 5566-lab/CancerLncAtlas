"""Private raw-row contributions observed inside the original State alignment.

Positions address loaded input tables, before validation sorting/filtering. This
traces the supplied measurements through within-patient means and paired masks;
it does not establish how those measurements were acquired or normalized.
"""
import json

import numpy as np
import pandas as pd

from .contracts import canonical_json_sha256
from .state_patient_use import encoded
from .state_training import validate_fold_manifest

TABLE = "state_measurement_contributions_private"
LINKS = "state_measurement_scope_links_private"
PRIVATE_NAMES = {TABLE, LINKS}
COLUMNS = ["role", "source_sha256", "contribution_id", "cancer_id", "patient_id", "feature_id",
    "value_column", "source_row_indices_json", "finite_source_row_indices_json", "missing_source_row_indices_json",
    "native_patient_mean", "finite_value_count", "patient_map_sha256", "patient_map_row_indices_json"]
LINK_COLUMNS = ["patient_fold_id", "split_role", "cohort_id", "cancer_id", "state_id", "lncrna_ids_json",
    "role", "contribution_ids_json", "native_patient_count"]


class MeasurementTrace:
    def __init__(self, raw_patients, pins):
        validated = validate_fold_manifest(raw_patients).set_index("sample_id")
        self.patients = validated.loc[raw_patients.sample_id.astype(str)].reset_index()
        self.authority_rows = {key: positions.tolist() for key, positions in
            self.patients.groupby(["cancer_id", "patient_id"], observed=True).indices.items()}
        self.pins = pins
        self.positions = {}
        self.events = []
        self.observed = set()

    def register(self, role, raw, feature):
        keys = ["cancer_id", "sample_id", feature]
        # Non-program metadata rows are deliberately outside State validation.
        # Store all positions; only rows retained by the original join are read.
        positions = {}
        for position, key in enumerate(raw[keys].astype(str).itertuples(index=False, name=None)):
            positions.setdefault(key, []).append(position)
        self.positions[role] = positions

    def observe(self, role, joined, means, feature, value):
        if role in self.observed or role not in self.positions:
            raise ValueError("State measurement trace has an unregistered or repeated source")
        self.observed.add(role)
        mean_index = means.set_index(["cancer_id", "sample_id", feature])[value]
        for (cancer, patient, identity), group in joined.groupby(["cancer_id", "patient_id", feature], observed=True):
            rows = []; finite = []; missing = []
            for sample, number in group[["sample_id", value]].itertuples(index=False, name=None):
                positions = self.positions[role].get((str(cancer), str(sample), str(identity)), [])
                if len(positions) != 1:
                    raise ValueError("State contribution does not identify one loaded source row")
                position = positions[0]; rows.append(position)
                (finite if pd.notna(number) else missing).append(position)
            authority = self.authority_rows.get((cancer, patient), [])
            if not authority:
                raise ValueError("State contribution lacks its actual patient authority")
            digest = self.pins[role]["sha256"]
            mean = mean_index.loc[(cancer, patient, identity)]
            self.events.append({"role": role, "source_sha256": digest,
                "contribution_id": canonical_json_sha256({"role": role, "source_sha256": digest, "rows": sorted(rows)}),
                "cancer_id": cancer, "patient_id": patient, "feature_id": identity, "value_column": value,
                "source_row_indices_json": encoded(sorted(rows)), "finite_source_row_indices_json": encoded(sorted(finite)),
                "missing_source_row_indices_json": encoded(sorted(missing)),
                "native_patient_mean": np.nan if pd.isna(mean) else float(mean), "finite_value_count": len(finite),
                "patient_map_sha256": self.pins["v32_patient_fold_manifest"]["sha256"],
                "patient_map_row_indices_json": encoded(authority)})

    def tables(self, scopes):
        if self.observed != {"tumor_state_long_measurement", "lncrna_expression"}:
            raise ValueError("State measurement trace did not observe both original means")
        groups = pd.DataFrame(self.events, columns=COLUMNS)
        # Build once: a cohort must not scan every patient/feature in the study.
        lookup = {(r["role"], r["cancer_id"], r["patient_id"], r["feature_id"]): r for r in self.events}
        if len(lookup) != len(self.events):
            raise ValueError("State measurement trace has duplicate contribution groups")
        links = []
        for event in scopes.to_dict("records"):
            if event["patient_map_sha256"] != self.pins["v32_patient_fold_manifest"]["sha256"]:
                raise ValueError("State measurement scope uses another patient authority")
            positions = json.loads(event["patient_map_row_indices_json"])
            if any(type(i) is not int or i < 0 or i >= len(self.patients) for i in positions):
                raise ValueError("State measurement scope has an invalid patient position")
            patients = set(self.patients.iloc[positions].patient_id)
            if len(patients) != event["native_patient_count"]:
                raise ValueError("State measurement scope patient count differs")
            for role, features in (("tumor_state_long_measurement", [event["state_id"]]),
                                   ("lncrna_expression", json.loads(event["lncrna_ids_json"]))):
                selected = [lookup.get((role, event["cancer_id"], patient, feature))
                    for patient in patients for feature in features]
                if any(r is None or r["finite_value_count"] == 0 or not np.isfinite(r["native_patient_mean"]) for r in selected):
                    raise ValueError("State paired scope lacks its actual finite measurement contributions")
                links.append({**{key: event[key] for key in ("patient_fold_id", "split_role", "cohort_id", "cancer_id",
                    "state_id", "lncrna_ids_json", "native_patient_count")}, "role": role,
                    "contribution_ids_json": encoded(sorted(r["contribution_id"] for r in selected))})
        return {TABLE: groups, LINKS: pd.DataFrame(links, columns=LINK_COLUMNS)}
