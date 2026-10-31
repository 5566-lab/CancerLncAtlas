"""Measured KO/KD/OE and rescue response graphs for the existing GNN encoder.

Each graph is one experiment in its own biological context and native scale.
The observed response direction is not a basal activation/repression sign.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_regulatory_inputs import (
    FORMAT, MOLECULAR_CLASSES, SourceFiles, sha256, stable_gene, text, truth,
)


class CausalRegulatoryNetwork:
    def __init__(self, manifest, *, source_root=None):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (self.manifest.get("format") != FORMAT or self.manifest.get("status") != "PASS"
                or self.manifest.get("target_host") != "149"):
            raise ValueError("Not a completed causal network input binding")
        self.manifest_sha256 = sha256(self.manifest_path)
        loaded = {}
        for name in ("index", "entities"):
            item = self.manifest[name]
            path = (self.manifest_path.parent / item["path"]).resolve()
            if not path.is_relative_to(self.manifest_path.parent) or sha256(path) != item["sha256"]:
                raise ValueError("Causal network index hash mismatch")
            loaded[name] = pd.read_parquet(path)
        self.index = loaded["index"]
        if self.index.duplicated(["source", "experiment_id"]).any():
            raise ValueError("Ambiguous experiment identities")
        self.files = SourceFiles(source_root or self.manifest["source_root"],
                                 self.manifest["source_manifest_sha256"])
        self.entities = loaded["entities"]
        self.by_id = self.entities.set_index("gene_id").to_dict("index")
        unique = self.entities.groupby("symbol").gene_id.nunique()
        symbols = self.entities.loc[self.entities.symbol.isin(unique[unique == 1].index)]
        self.by_symbol = symbols.set_index("symbol").gene_id.to_dict()
        self._tables = {}
        self._arrays = {}

    def experiments(self, *, intervention=None, source=None):
        frame = self.index
        if intervention is not None:
            frame = frame.loc[frame.intervention_type == intervention]
        if source is not None:
            frame = frame.loc[frame.source == source]
        return frame.copy()

    def experiment(self, source, identifier):
        rows = self.index.loc[(self.index.source == source) & (self.index.experiment_id == str(identifier))]
        if len(rows) != 1:
            raise ValueError("Unknown or ambiguous experiment: " + source + ":" + str(identifier))
        return rows.iloc[0].to_dict()

    def _table(self, name):
        if name not in self._tables:
            self._tables[name] = self.files.table(name)
        return self._tables[name]

    def _npz(self, name, *, cache=False):
        if name in self._arrays:
            return self._arrays[name]
        with np.load(self.files.path(name), allow_pickle=False) as handle:
            result = {key: handle[key] for key in handle.files}
        if cache:
            self._arrays[name] = result
        return result

    def _gene_axis(self, gene_ids, symbols=None):
        identifiers = [stable_gene(x) for x in gene_ids]
        symbols = [""] * len(identifiers) if symbols is None else list(map(str, symbols))
        result = []
        for gene, symbol in zip(identifiers, symbols, strict=True):
            if not gene.startswith("ENSG"):
                gene = self.by_symbol.get(symbol, "")
            known = self.by_id.get(gene, {})
            result.append((gene, known.get("symbol", symbol), known.get("gene_class", "unresolved")))
        return pd.DataFrame(result, columns=["response_id", "response_symbol", "response_class"])

    def _finish(self, row, axis, effect, scale, *, pvalue=None, fdr=None,
                direction=None, eligible=None, direct=None, record_ids=None, lfc_se=None):
        frame = axis.reset_index(drop=True).copy()
        effect = np.asarray(effect, dtype=float)
        if effect.shape != (len(frame),) or np.isinf(effect).any():
            raise ValueError("Invalid response vector shape or infinite effect")
        frame["source"] = row["source"]
        frame["experiment_id"] = row["experiment_id"]
        frame["intervention_type"] = row["intervention_type"]
        frame["perturbed_id"] = row["target_id"]
        frame["perturbed_symbol"] = row["target_symbol"]
        frame["perturbed_class"] = row["target_class"]
        frame["cell_context"] = row["cell_context"]
        frame["background"] = row["background"]
        frame["known_study_group"] = row["known_study_group"]
        frame["study_ids"] = row["study_ids"]
        frame["split"] = row["split"]
        frame["mechanism_holdout"] = bool(row["mechanism_holdout"])
        frame["effect"] = effect
        frame["effect_observed"] = np.isfinite(effect)
        frame["effect_scale"] = scale
        frame["response_direction"] = np.sign(effect) if direction is None else np.asarray(direction, dtype=float)
        frame["direction_available"] = np.isfinite(frame.response_direction)
        if not frame.loc[frame.direction_available, "response_direction"].isin([-1, 0, 1]).all():
            raise ValueError("Invalid ordinal response direction")
        frame["p_value"] = np.nan if pvalue is None else np.asarray(pvalue, dtype=float)
        frame["adjusted_p_value"] = np.nan if fdr is None else np.asarray(fdr, dtype=float)
        for name in ("p_value", "adjusted_p_value"):
            frame[name + "_available"] = np.isfinite(frame[name]) & frame[name].between(0, 1)
        frame["lfc_se"] = np.nan if lfc_se is None else lfc_se
        frame["directly_measured"] = True if direct is None else np.asarray(direct, dtype=bool)
        frame["network_eligible"] = bool(row["network_eligible"])
        if eligible is not None:
            frame["network_eligible"] &= np.asarray(eligible, dtype=bool)
        frame["network_eligible"] &= (
            frame.response_id.str.startswith("ENSG") & frame.response_class.isin(MOLECULAR_CLASSES)
            & frame.directly_measured & frame.direction_available)
        frame["readout"] = {"LINCS_iLINCS_OE":"RNA_signature",
                            "L2S2_KO":"RNA_response_gene_set",
                            "LncRNA2Target3":"RNA_response_direction"}.get(row["source"], "RNA_abundance")
        frame["directness"] = "total_response_direct_or_indirect"
        frame["target_efficiency_status"] = "not_independently_verified"
        frame["review_reason"] = row["review_reason"]
        frame["source_method"] = row["source_method"]
        frame["time"] = row["time"]
        frame["dose"] = row["dose"]
        frame["n_control"] = row["n_control"]
        frame["n_treatment"] = row["n_treatment"]
        frame["physical_binding_label"] = "not_determined"
        frame["regulatory_activation_sign"] = np.nan
        frame["rescue_success_label"] = pd.NA
        frame["record_id"] = (list(record_ids) if record_ids is not None else
                              [row["source"] + ":" + row["experiment_id"] + ":" + str(i) for i in range(len(frame))])
        duplicate = frame.response_id.duplicated(keep=False) & frame.response_id.ne("")
        frame.loc[duplicate, "network_eligible"] = False
        frame.loc[duplicate, "review_reason"] += ";DUPLICATE_RESPONSE_ID_REQUIRES_REVIEW"
        return frame

    def responses(self, source, identifier):
        """Return every readout row, with source missingness and eligibility."""
        row = self.experiment(source, identifier)
        if source == "PerturbAtlas":
            import h5py
            i = int(row["row_index"])
            genes = self._table("PerturbAtlas/genes.tsv")
            with h5py.File(self.files.path("PerturbAtlas/responses.h5"), "r") as handle:
                if handle["contrast_id"].asstr()[i] != identifier:
                    raise ValueError("PerturbAtlas contrast axis drift")
                gene_ids = handle["gene_id"].asstr()[:]
                if list(gene_ids) != list(genes.gene_id):
                    raise ValueError("PerturbAtlas gene axis drift")
                axis = self._gene_axis(gene_ids, genes.gene_name)
                return self._finish(row, axis, handle["log2FoldChange"][i], "DESeq2_log2FoldChange",
                                    fdr=handle["padj"][i], lfc_se=handle["lfcSE"][i])
        if source == "GPSAdb2":
            if not str(identifier).isalnum():
                raise ValueError("Unsafe GPSAdb dataset identity")
            values = self._npz("GPSAdb2/datasets/" + identifier + ".npz")
            axis = self._gene_axis(values["gene_id"])
            counts = values["raw_counts"]
            control, treatment = values["control"], values["treatment"]
            if (counts.shape != (len(axis), len(values["sample_id"])) or counts.dtype.kind not in "iu"
                    or (counts < 0).any() or not control.any() or not treatment.any()
                    or np.any(control & treatment) or not np.all(control | treatment)):
                raise ValueError("GPSAdb count/group contract mismatch")
            if control.sum() != int(row["n_control"]) or treatment.sum() != int(row["n_treatment"]):
                raise ValueError("GPSAdb replicate metadata drift")
            samples = self._table("GPSAdb2/samples.tsv")
            samples = samples.loc[samples.dataset_id == identifier]
            if samples.sample_id.duplicated().any() or set(samples.sample_id) != set(values["sample_id"]):
                raise ValueError("GPSAdb sample identity drift")
            groups = samples.set_index("sample_id").loc[values["sample_id"], "group"]
            if not np.array_equal(groups.eq("control").to_numpy(), control) or not np.array_equal(
                    groups.isin(["treat", "treatment"]).to_numpy(), treatment):
                raise ValueError("GPSAdb sample group drift")
            # The source producer stores logCPM as float32; compare to that native transform.
            logged = values["log2cpm_plus1"]
            effect = values["effect_mean_log2cpm_treat_minus_control"]
            expected = logged[:, treatment].mean(axis=1) - logged[:, control].mean(axis=1)
            if not np.allclose(expected, effect, atol=2e-5, rtol=1e-5):
                raise ValueError("GPSAdb effect/group replay mismatch")
            return self._finish(row, axis, effect, "difference_of_mean_log2_CPM_plus_1_descriptive")
        if source == "LINCS_iLINCS_OE":
            values = self._npz(source + "/landmark_signatures.npz", cache=True)
            i = int(row["row_index"])
            genes = self._table(source + "/landmark_genes.tsv")
            annotation = self._table("lncRNA_edge_audit/" + source + "_readout_annotation.tsv")
            if str(values["signature_id"][i]) != identifier or list(values["entrez_gene_id"]) != list(genes.ID_geneid):
                raise ValueError("iLINCS signature/gene axis drift")
            if list(annotation.ID_geneid) != list(genes.ID_geneid):
                raise ValueError("iLINCS readout annotation axis drift")
            axis = self._gene_axis(values["entrez_gene_id"], annotation.hgnc_readout_symbol)
            return self._finish(row, axis, values["effect"][:, i], "iLINCS_source_MODZ_signature")
        if source == "L2S2_KO":
            from scipy import sparse
            genes = self._table(source + "/genes.tsv")
            annotation = self._table("lncRNA_edge_audit/" + source + "_readout_annotation.tsv")
            if list(annotation.gene_symbol) != list(genes.gene_symbol):
                raise ValueError("L2S2 readout annotation axis drift")
            i = int(row["row_index"])
            for key in ("up_gene_sets", "down_gene_sets"):
                if key not in self._arrays:
                    self._arrays[key] = sparse.load_npz(self.files.path(source + "/" + key + ".npz"))
            up, down = (self._arrays[key] for key in ("up_gene_sets", "down_gene_sets"))
            if up.shape != down.shape or up.shape != (len(self.experiments(source=source)), len(genes)):
                raise ValueError("L2S2 sparse matrix axis drift")
            a, b = up.getrow(i).toarray().ravel(), down.getrow(i).toarray().ravel()
            if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
                raise ValueError("L2S2 membership must be binary")
            listed, conflict = (a != 0) | (b != 0), (a != 0) & (b != 0)
            direction = np.where(listed & ~conflict, a.astype(float) - b, np.nan)
            axis = self._gene_axis([""] * len(genes), annotation.hgnc_readout_symbol)
            # These are ordinal memberships; there is no measured effect size.
            result = self._finish(row, axis, np.full(len(genes), np.nan), "thresholded_set_direction_only",
                direction=direction, direct=genes.is_direct_landmark.map(truth).to_numpy(),
                eligible=listed & ~conflict)
            result["set_membership_conflict"] = conflict
            result["source_up_listed"] = a.astype(bool)
            result["source_down_listed"] = b.astype(bool)
            return result
        if source == "LncRNA2Target3":
            values = self._table(source + "/high_throughput_human_response.tsv.gz")
            values = values.loc[values.experiment_id == str(identifier)].copy()
            axis = self._gene_axis(values.response_entity_id, values.response_hgnc_symbol)
            numeric = lambda name: pd.to_numeric(values[name], errors="coerce").to_numpy()
            direction = np.array(numeric("response_direction_for_training"), dtype=float, copy=True)
            eligible = values.direction_training_eligible.map(truth).to_numpy()
            direction[~eligible] = np.nan
            result = self._finish(row, axis, np.full(len(values), np.nan),
                "source_direction_only_magnitude_unverified", direction=direction, eligible=eligible,
                pvalue=numeric("p_value"), fdr=numeric("adjusted_p_value"), record_ids=values.record_id)
            result["source_reported_effect_for_audit"] = numeric("effect_log2FC")
            result["source_numeric_issue"] = values.source_numeric_issue.to_numpy()
            return result
        if source == "GEO_rescue_GSE154812":
            return self._rescue_response(row)
        raise ValueError("Source has a separate evidence/fitness readout; use auxiliary()")

    def _rescue_response(self, row):
        source = row["source"]
        table = self._table(source + "/rescue_effects.tsv.gz")
        axis = self._gene_axis(table.ensembl_gene_id, table.gene_name)
        columns = {"NORAD_CRISPRi":"NORAD_CRISPRi_minus_control",
                   "circPRE2":"circPRE2_construct_minus_CRISPRi",
                   "circPRE8":"circPRE8_construct_minus_CRISPRi"}
        values = self._npz(source + "/expression.npz", cache=True)
        groups = self._table(source + "/samples.tsv")
        # Source effects must remain tied to the supplied experiment, not inferred
        # from opposite signs in unrelated OE and KO studies.
        if list(values["gene_id"]) != list(table.ensembl_gene_id):
            raise ValueError("Rescue gene axis drift")
        effect = pd.to_numeric(table[columns[row["experiment_id"]]], errors="coerce").to_numpy()
        if groups.sample_id.duplicated().any() or set(groups.sample_id) != set(values["sample_id"]):
            raise ValueError("Rescue sample identity drift")
        groups = groups.set_index("sample_id").loc[values["sample_id"]]
        target_group = {"NORAD_CRISPRi":"NORAD_CRISPRi", "circPRE2":"NORAD_CRISPRi_circPRE2",
                        "circPRE8":"NORAD_CRISPRi_circPRE8"}[row["experiment_id"]]
        baseline = "control" if row["experiment_id"] == "NORAD_CRISPRi" else "NORAD_CRISPRi"
        control = (groups.group == baseline).to_numpy()
        treated = (groups.group == target_group).to_numpy()
        raw = values["source_normalized_values"]
        if (raw.shape != (len(table), len(groups)) or not np.isfinite(raw).all()
                or (raw < 0).any() or control.sum() != 2 or treated.sum() != 2):
            raise ValueError("Rescue source expression/group mismatch")
        logged = np.log2(raw + 1.0)
        expected = logged[:, treated].mean(axis=1) - logged[:, control].mean(axis=1)
        if not np.allclose(effect, expected, atol=1e-10, rtol=1e-10):
            raise ValueError("Rescue effect does not replay from its actual groups")
        result = self._finish(row, axis, effect,
            "difference_of_mean_log2_source_normalized_plus_1_descriptive")
        for column in ("NORAD_CRISPRi_minus_control", "circPRE8_residual_vs_control", "circPRE8_minus_circPRE2"):
            result[column] = pd.to_numeric(table[column], errors="coerce").to_numpy()
        result["rescue_role"] = row["usage"]
        result["unperturbed_construct_control_available"] = False
        result["descriptive_reversal_only"] = (np.sign(result.effect)
            * np.sign(result.NORAD_CRISPRi_minus_control) < 0) if row["intervention_type"] == "RESCUE" else False
        return result

    def auxiliary(self, source):
        """Curated/fitness sources remain usable without becoming RNA-edge labels."""
        if source == "LncTarD2":
            return self._table(source + "/curated_relations_all.tsv.gz").copy()
        if source == "LncRNA2Target3_low":
            return self._table("LncRNA2Target3/low_throughput_human_relations.tsv.gz").copy()
        if source == "CasRx2024":
            values = self._npz(source + "/lncRNA_fitness.npz")
            rows = values["library"] == "ALBAROSSA_library"
            columns = values["casrx_present"].astype(bool)
            result = {key: values[key][rows][:, columns] for key in
                      ("fitness_lfc", "RRA_score", "p_value", "FDR", "observed_mask")}
            result.update(record_id=values["record_id"][rows], family_id=values["family_id"][rows],
                          condition_id=values["condition_id"][columns], readout="cell_fitness")
            # The source mask requires both LFC and FDR. A finite LFC alone
            # must not promote a record whose screening statistics are absent.
            result["fitness_lfc_available"] = np.isfinite(result["fitness_lfc"])
            result["FDR_available"] = np.isfinite(result["FDR"])
            expected = result["fitness_lfc_available"] & result["FDR_available"]
            if np.any(result["observed_mask"] != expected):
                raise ValueError("CasRx missing-value mask drift")
            result["observed_mask_semantics"] = "source_finite_fitness_lfc_and_FDR"
            return result
        raise ValueError("Unknown auxiliary source")

    def edges(self, source, identifier, *, max_fdr=None, cross_type_only=False):
        """Build one context's directed response edges; zero/missing are not edges."""
        row = self.experiment(source, identifier)
        frame = self.responses(source, identifier)
        frame = frame.loc[frame.network_eligible & frame.direction_available & (frame.response_direction != 0)].copy()
        if max_fdr is not None:
            if not 0 < max_fdr <= 1:
                raise ValueError("max_fdr must be in (0,1]")
            frame = frame.loc[frame.adjusted_p_value_available & (frame.adjusted_p_value <= max_fdr)].copy()
        if cross_type_only:
            frame = frame.loc[((frame.perturbed_class == "lncRNA") & (frame.response_class == "protein_coding"))
                              | ((frame.perturbed_class == "protein_coding") & (frame.response_class == "lncRNA"))].copy()
        node_types = {"protein_coding":"gene", "lncRNA":"lncRNA", "synthetic_RNA":"RNA_construct"}
        frame["source_type"] = frame.perturbed_class.map(node_types)
        frame["target_type"] = frame.response_class.map(node_types)
        target = row["target_id"]
        if target.startswith("SYMBOL:"):
            target = self.by_symbol.get(target.split(":", 1)[1], target)
        source_prefix = {"gene":"GENE:", "lncRNA":"LNC:", "RNA_construct":""}
        frame["source_canonical_id"] = frame.source_type.map(source_prefix) + target
        frame["target_canonical_id"] = frame.target_type.map(source_prefix) + frame.response_id
        frame["source_node_id"] = frame.source_type + ":" + frame.source_canonical_id
        frame["target_node_id"] = frame.target_type + ":" + frame.target_canonical_id
        frame["relation_type"] = ("experimental_" + row["intervention_type"] + "_RNA_response_"
                                  + pd.Series(np.where(frame.response_direction > 0, "up", "down"), index=frame.index))
        frame["relation_polarity"] = frame.response_direction.astype(int)
        # Magnitudes are used only within this experiment's declared native scale.
        frame["weight"] = np.where(frame.effect_observed, np.abs(frame.effect), 1.0)
        frame["weight_semantics"] = np.where(frame.effect_observed, "absolute_native_response",
                                             "unit_ordinal_membership_not_effect_size")
        frame["edge_id"] = [hashlib.sha256(str(x).encode()).hexdigest() for x in frame.record_id]
        if frame.edge_id.duplicated().any() or frame.source_type.isna().any() or frame.target_type.isna().any():
            raise ValueError("Duplicate edge identity or unresolved node types")
        return frame.reset_index(drop=True)

    def graph(self, source, identifier, *, max_fdr=None, cross_type_only=False, purpose="exploration"):
        """Real GraphBundle consumer; research validation only, no split invention."""
        if purpose not in {"exploration", "mechanism_validation"}:
            raise ValueError("Training requires a separately reviewed cross-source study split")
        row = self.experiment(source, identifier)
        if purpose == "mechanism_validation" and not row["mechanism_holdout"]:
            raise ValueError("Experiment is not in the reserved mechanism-validation group")
        edges = self.edges(source, identifier, max_fdr=max_fdr, cross_type_only=cross_type_only)
        if edges.empty:
            raise ValueError("No eligible measured response edges for this experiment and policy")
        columns = ["node_id", "node_type", "canonical_id"]
        nodes = pd.concat([
            edges[["source_node_id", "source_type", "source_canonical_id"]].set_axis(columns, axis=1),
            edges[["target_node_id", "target_type", "target_canonical_id"]].set_axis(columns, axis=1),
        ]).drop_duplicates().sort_values(["node_type", "canonical_id"]).reset_index(drop=True)
        nodes["node_index_within_type"] = nodes.groupby("node_type", sort=False).cumcount()
        from cc_hhgt.gnn import _materialize_graph_bundle
        bundle = _materialize_graph_bundle(nodes, edges, graph_contract="CONTRACT-T")
        bundle.experimental_regulatory_authority = dict(
            source=source, experiment_id=identifier, input_manifest_sha256=self.manifest_sha256,
            known_study_group=row["known_study_group"], intervention_type=row["intervention_type"],
            cell_context=row["cell_context"], purpose=purpose, learned_parameters=False,
            response_edges=len(edges), physical_binding=False, automatic_reverse_edges_are_computational=True)
        return bundle
