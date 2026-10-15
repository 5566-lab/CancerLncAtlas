"""Source-bound pharmacology covariates for the existing drug association head.

Static target annotations and molecular structure are separate from assay design.
Dose features describe the observed protocol mixture; they are not dose-specific
response predictions. No drug-response measurements enter this feature encoder.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import sha256, verified_path

FORMAT = "DRUG_PHARMACOLOGY_INPUT_V1"
FEATURE_FORMAT = "DRUG_PHARMACOLOGY_FEATURES_V1"
ACTIONS = ("INHIBITOR", "ANTAGONIST", "AGONIST", "ACTIVATOR", "PARTIAL_AGONIST", "INVERSE_AGONIST", "MODULATOR", "BINDER", "UNKNOWN")
ENDPOINTS = ("KI", "KD", "IC50", "EC50")
FP_BITS = 256
STRUCTURE_FEATURES = ("molecular_weight", "logp", "tpsa", "h_bond_donors", "h_bond_acceptors", "rotatable_bonds")
STATIC_FEATURES = (
    "action_annotation_available", *["action_"+x.lower()+"_target_fraction" for x in ACTIONS],
    "multiple_action_target_fraction",
    *[f"{e.lower()}_{stat}" for e in ENDPOINTS for stat in ("exact_p_available", "exact_p_median", "exact_target_fraction", "censored_target_fraction")],
    "structure_available", *STRUCTURE_FEATURES, *[f"morgan_r2_chiral_bit_{i:03}" for i in range(FP_BITS)],
)
DESIGN_FEATURES = ("dose_available", "positive_dose_available", "log10_molar_dose_min", "log10_molar_dose_median", "log10_molar_dose_max",
                   "vehicle_fraction", "exposure_available", "log1p_exposure_hours_mean", "log1p_protocol_count")
TARGET_POOLS = tuple(["action_"+a.lower() for a in ACTIONS] + ["exact_"+e.lower()+"_p" for e in ENDPOINTS])
FEATURES = tuple("pharm_"+x for x in (*STATIC_FEATURES, *DESIGN_FEATURES,
                                     *[p+"_embedding_available" for p in TARGET_POOLS]))
_UNITS = {"M": 1., "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "μM": 1e-6, "nM": 1e-9, "pM": 1e-12}
_DESIGN_KEYS = ["dataset_id", "cell_line_id", "drug_id"]


def _read(path):
    from .drug_training import _read_table
    return _read_table(path)


def _identifiers(frame, columns):
    for name in columns:
        if name not in frame or frame[name].isna().any():
            raise ValueError("Missing pharmacology source identity: "+name)
        frame[name] = frame[name].astype(str).str.strip()
        if frame[name].eq("").any(): raise ValueError("Empty pharmacology source identity: "+name)


def _required(frame, columns):
    missing = set(columns)-set(frame)
    if missing: raise ValueError("Pharmacology fields were not supplied: "+str(sorted(missing)))


def concentration(value, unit, *, allow_zero=False):
    if pd.isna(value) and pd.isna(unit): return np.nan
    if pd.isna(value) or unit not in _UNITS:
        raise ValueError("Concentration requires an explicit supported molar unit")
    result = float(value)*_UNITS[unit]
    if not np.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError("Invalid molar concentration")
    return result


def normalize_targets(frame, authority):
    required = ["drug_id", "gene_id", "action_type", "affinity_type", "affinity_value", "affinity_unit", "affinity_relation", "source_id"]
    _required(frame, required); frame = frame[required].copy()
    _identifiers(frame, ["drug_id", "gene_id", "source_id"])
    allowed = set(authority[["drug_id", "gene_id"]].itertuples(index=False, name=None))
    if not set(frame[["drug_id", "gene_id"]].itertuples(index=False, name=None)).issubset(allowed):
        raise ValueError("Pharmacology targets differ from the actual drug target authority")
    frame["action_raw"] = frame.action_type
    action = frame.action_type.fillna("").astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    frame["action_type"] = action.where(action.isin(ACTIONS), "UNKNOWN")
    frame["action_observed"] = action.ne("") & action.ne("UNKNOWN")
    frame["p_value_molar"] = np.nan
    frame["p_relation"] = None
    for index, row in frame.iterrows():
        fields = [row.affinity_type, row.affinity_value, row.affinity_unit, row.affinity_relation]
        if all(pd.isna(x) for x in fields): continue
        if any(pd.isna(x) for x in fields): raise ValueError("Incomplete affinity/potency measurement")
        endpoint = str(row.affinity_type).upper()
        if endpoint not in ENDPOINTS: raise ValueError("Unsupported affinity/potency endpoint: "+endpoint)
        relation = str(row.affinity_relation).strip()
        if relation not in {"=", "<", ">", "<=", ">="}: raise ValueError("Unknown affinity censoring relation")
        if row.affinity_unit == "p_molar":
            value = float(row.affinity_value)
            if not np.isfinite(value): raise ValueError("Non-finite p-molar measurement")
        else:
            value = -np.log10(concentration(row.affinity_value, row.affinity_unit))
            relation = {"=":"=", "<":">", ">":"<", "<=":">=", ">=":"<="}[relation]
        frame.loc[index, ["affinity_type", "p_value_molar", "p_relation"]] = [endpoint, value, relation]
    return frame.drop_duplicates().reset_index(drop=True)


def structure_table(frame, drugs):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors, rdFingerprintGenerator
    _required(frame, ["drug_id", "smiles", "source_id"])
    frame = frame[["drug_id", "smiles", "source_id"]].copy(); _identifiers(frame, ["drug_id", "source_id"])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS, includeChirality=True)
    rows = []
    for drug in sorted(set(map(str, drugs))):
        local = frame.loc[frame.drug_id.eq(drug)]
        molecules = {}
        invalid = False
        for raw in local.smiles:
            if pd.isna(raw) or not str(raw).strip(): continue
            mol = Chem.MolFromSmiles(str(raw))
            if mol is None or mol.GetNumAtoms() == 0: invalid = True; continue
            molecules[Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)] = mol
        available = len(molecules) == 1 and not invalid
        row = {"drug_id": drug, "structure_available": available, "canonical_smiles": None,
            "structure_unavailable_reason": None if available else
                "INVALID_SMILES" if invalid else "CONFLICTING_STRUCTURES" if len(molecules)>1 else "NO_STRUCTURE_SOURCE",
            **{k:np.nan for k in (*STRUCTURE_FEATURES, *STATIC_FEATURES[-FP_BITS:])}}
        if available:
            canonical, mol = next(iter(molecules.items()))
            row["canonical_smiles"] = canonical
            values = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
                Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol)]
            row.update(zip(STRUCTURE_FEATURES, values))
            row.update(zip(STATIC_FEATURES[-FP_BITS:], generator.GetFingerprintAsNumPy(mol).astype(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_design(frame, response, mapping):
    required = [*_DESIGN_KEYS, "assay_id", "dose_value", "dose_unit", "exposure_hours", "source_id"]
    _required(frame, required)
    forbidden = {"response_value", "sensitivity_value", "auc", "ln_ic50", "label", "prediction", "probability"} & set(frame)
    if forbidden: raise ValueError("Assay design contains response-derived fields: "+str(sorted(forbidden)))
    frame = frame[required].copy(); _identifiers(frame, [*_DESIGN_KEYS, "assay_id", "source_id"])
    valid = response[_DESIGN_KEYS].drop_duplicates()
    checked = frame.merge(valid.assign(_observed=True), on=_DESIGN_KEYS, how="left", validate="many_to_one")
    if checked._observed.isna().any(): raise ValueError("Assay conditions do not match an observed response identity")
    frame["dose_molar"] = [concentration(v, u, allow_zero=True) for v, u in zip(frame.dose_value, frame.dose_unit)]
    exposure = pd.to_numeric(frame.exposure_hours, errors="raise")
    if (exposure.notna() & (~np.isfinite(exposure) | exposure.lt(0))).any(): raise ValueError("Invalid exposure duration")
    frame["exposure_hours"] = exposure
    condition = [*_DESIGN_KEYS, "assay_id"]
    if frame.groupby(condition, dropna=False)[["dose_molar", "exposure_hours"]].nunique(dropna=False).gt(1).any().any():
        raise ValueError("One native assay ID has conflicting dose/exposure conditions")
    frame = frame.drop_duplicates(condition)
    mapping = mapping[["dataset_id", "cell_line_id", "canonical_model_id", "cancer_id"]].drop_duplicates()
    frame = frame.merge(mapping, on=["dataset_id", "cell_line_id"], how="left", validate="many_to_one")
    if frame.canonical_model_id.isna().any(): raise ValueError("Assay design lacks canonical model/cancer identity")
    return frame


class PharmacologyInputs:
    def __init__(self, targets, structures, design, authority, *, sources=None):
        self.authority = authority[["drug_id", "gene_id"]].drop_duplicates().copy()
        self.targets = targets; self.structures = structures; self.design = design
        self.sources = sources or {}; self._context_cache = {}; self._core_cache = {}
        self._static = self.static_table()

    def static_table(self, excluded_targets=None):
        excluded_targets = excluded_targets or {}
        rows = []
        for drug in self.structures.drug_id:
            genes = set(self.authority.loc[self.authority.drug_id.eq(drug), "gene_id"])-set(excluded_targets.get(drug, ()))
            local = self.targets.loc[self.targets.drug_id.eq(drug) & self.targets.gene_id.isin(genes)]
            n = max(len(genes), 1)
            actions = local[["gene_id", "action_type"]].drop_duplicates()
            row = {"drug_id":drug, "action_annotation_available":bool(local.action_observed.any())}
            for action in ACTIONS:
                row["action_"+action.lower()+"_target_fraction"] = actions.loc[actions.action_type.eq(action), "gene_id"].nunique()/n
            row["multiple_action_target_fraction"] = float(actions.groupby("gene_id").action_type.nunique().gt(1).sum()/n)
            for endpoint in ENDPOINTS:
                values = local.loc[local.affinity_type.eq(endpoint) & local.p_value_molar.notna()]
                exact = values.loc[values.p_relation.eq("=")].drop_duplicates(["gene_id", "p_value_molar"])
                # Each target contributes once, irrespective of database replication.
                per_target = exact.groupby("gene_id").p_value_molar.median()
                row.update({endpoint.lower()+"_exact_p_available":not per_target.empty,
                    endpoint.lower()+"_exact_p_median":float(per_target.median()) if len(per_target) else np.nan,
                    endpoint.lower()+"_exact_target_fraction":len(per_target)/n,
                    endpoint.lower()+"_censored_target_fraction":values.loc[values.p_relation.ne("="), "gene_id"].nunique()/n})
            rows.append(row)
        return pd.DataFrame(rows).merge(self.structures, on="drug_id", validate="one_to_one")

    def design_table(self, models):
        key = tuple(sorted(set(map(str, models))))
        if key in self._context_cache: return self._context_cache[key]
        frame = self.design.loc[self.design.canonical_model_id.isin(key)]
        rows = []
        for (cancer, drug), local in frame.groupby(["cancer_id", "drug_id"], sort=False):
            dose = local.dose_molar.dropna(); positive = dose.loc[dose.gt(0)]
            exposure = local.exposure_hours.dropna()
            rows.append({"cancer_id":cancer, "drug_id":drug, "dose_available":not dose.empty, "positive_dose_available":not positive.empty,
                "log10_molar_dose_min":np.log10(positive.min()) if len(positive) else np.nan,
                "log10_molar_dose_median":np.log10(positive.median()) if len(positive) else np.nan,
                "log10_molar_dose_max":np.log10(positive.max()) if len(positive) else np.nan,
                "vehicle_fraction":float(dose.eq(0).mean()) if len(dose) else np.nan,
                "exposure_available":not exposure.empty,
                "log1p_exposure_hours_mean":np.log1p(exposure.mean()) if len(exposure) else np.nan,
                "log1p_protocol_count":np.log1p(len(local))})
        result = pd.DataFrame(rows, columns=["cancer_id", "drug_id", *DESIGN_FEATURES])
        self._context_cache[key] = result
        return result

    def table(self, candidates, models, *, excluded_targets=None):
        static = self._static if excluded_targets is None else self.static_table(excluded_targets)
        frame = candidates[["cancer_id", "drug_id"]].reset_index(drop=True).merge(static, on="drug_id", how="left", validate="many_to_one")
        frame = frame.merge(self.design_table(models), on=["cancer_id", "drug_id"], how="left", validate="many_to_one")
        for name in ["action_annotation_available", "structure_available", "dose_available", "positive_dose_available", "exposure_available",
                     *[e.lower()+"_exact_p_available" for e in ENDPOINTS]]:
            frame[name] = frame[name].eq(True)
        return frame

    def target_core(self, candidates, core, *, excluded_targets=None):
        """Action-specific means and endpoint-specific mean(p * gene embedding).

        Potency products are interaction covariates, not target occupancy or a
        signed biological effect. Censored measurements never become points.
        """
        width = core.gene.values.shape[1]; excluded_targets = excluded_targets or {}
        rows = {}; flags = {}
        for drug in candidates.drug_id.astype(str).unique():
            excluded = set(excluded_targets.get(drug, ()))
            key = (core.fold, core.core_parameter_sha256, drug)
            if not excluded and key in self._core_cache:
                rows[drug], flags[drug] = self._core_cache[key]; continue
            genes = set(self.authority.loc[self.authority.drug_id.eq(drug), "gene_id"])-excluded
            local = self.targets.loc[self.targets.drug_id.eq(drug) & self.targets.gene_id.isin(genes)]
            values = []; available = []
            for action in ACTIONS:
                selected = set(local.loc[local.action_type.eq(action), "gene_id"])
                if action == "UNKNOWN": selected |= genes-set(local.gene_id)
                positions = [core.gene.index[g] for g in sorted(selected) if g in core.gene.index]
                values.append(core.gene.values[positions].mean(0) if positions else np.zeros(width))
                available.append(bool(positions))
            for endpoint in ENDPOINTS:
                exact = local.loc[local.affinity_type.eq(endpoint) & local.p_relation.eq("=") & local.p_value_molar.notna()]
                per_target = exact.drop_duplicates(["gene_id", "p_value_molar"]).groupby("gene_id").p_value_molar.median()
                products = [core.gene.values[core.gene.index[g]]*v for g,v in per_target.items() if g in core.gene.index]
                values.append(np.mean(products,axis=0) if products else np.zeros(width)); available.append(bool(products))
            rows[drug] = np.concatenate(values).astype(np.float32); flags[drug] = np.asarray(available, np.float32)
            if not excluded: self._core_cache[key] = rows[drug], flags[drug]
        if candidates.empty: return np.empty((0,len(TARGET_POOLS)*width),np.float32), np.empty((0,len(TARGET_POOLS)),np.float32)
        return np.stack([rows[d] for d in candidates.drug_id]), np.stack([flags[d] for d in candidates.drug_id])

    def encode(self, candidates, models, *, excluded_targets=None, core=None):
        table = self.table(candidates, models, excluded_targets=excluded_targets)
        values = table[[*STATIC_FEATURES, *DESIGN_FEATURES]].astype(float).fillna(0).to_numpy(np.float32)
        flags = self.target_core(candidates, core, excluded_targets=excluded_targets)[1] if core is not None else np.zeros((len(candidates),len(TARGET_POOLS)),np.float32)
        values = np.concatenate([values,flags],axis=1)
        if not np.isfinite(values).all(): raise ValueError("Non-finite pharmacology model feature")
        return values

    def augment(self, stats, candidates, models, *, core):
        values = self.encode(candidates, models, core=core)
        return replace(stats, domain=np.concatenate([stats.domain, values], axis=1))

    def receipt(self):
        import rdkit
        return {"format":FEATURE_FORMAT, "features":list(FEATURES), "sources":self.sources,
            "implementation_sha256":sha256(Path(__file__)),
            "rdkit_version":rdkit.__version__, "morgan_radius":2, "morgan_bits":FP_BITS, "chirality":True,
            "model_missing_value":0, "missingness_indicators_included":True,
            "dose_estimand":"SPLIT_LOCAL_OBSERVED_PROTOCOL_DISTRIBUTION_NOT_DOSE_RESPONSE",
            "affinity_estimand":"ENDPOINT_SEPARATE_EXACT_P_MOLAR_MEDIANS_WITH_CENSORING_FLAGS",
            "additional_core_pools":list(TARGET_POOLS),
            "target_pool_estimand":"ACTION_SPECIFIC_MEAN_EMBEDDINGS_AND_ENDPOINT_SPECIFIC_MEAN_P_TIMES_EMBEDDING_NOT_OCCUPANCY",
            "causal_drug_action_claimed":False}

    def verify_sources(self):
        for record in self.sources.values(): verified_path(record, Path.cwd())


def source_paths(manifest_path):
    path = Path(manifest_path).resolve(); payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT: raise ValueError("Unknown pharmacology input format")
    return {"manifest":path, **{role:verified_path(payload[role], path.parent)
        for role in ("target_annotations", "structures", "assay_conditions")}}


def load_pharmacology(manifest_path, *, targets, response, mapping):
    paths = source_paths(manifest_path); path = paths["manifest"]
    sources = {"manifest":{"path":str(path), "sha256":sha256(path)}}
    tables = {}
    for role in ("target_annotations", "structures", "assay_conditions"):
        source = paths[role]
        sources[role] = {"path":str(source), "sha256":sha256(source)}
        tables[role] = _read(source)
    drugs = set(response.drug_id.astype(str)) | set(targets.drug_id.astype(str))
    if not set(tables["structures"].drug_id.astype(str)).issubset(drugs):
        raise ValueError("Structure registry references drugs outside the native assay/target universe")
    result = PharmacologyInputs(normalize_targets(tables["target_annotations"], targets),
        structure_table(tables["structures"], drugs), normalize_design(tables["assay_conditions"], response, mapping), targets,
        sources=sources)
    result.verify_sources()
    return result
