"""Gene Sets, ranked subtypes and coverage derived from one selected-score snapshot."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import verified_path, sha256
from .evidence_rule_summary import KEYS, stable_id
from .selected_score import materialize_selected_scores, load_selected_scores
from .genesets import genesets_to_gmt_lines
from .subtypes import RankedSubtypeConfig, classify_pathway_context_subtypes, classify_cancer_program_subtypes


@dataclass(frozen=True)
class SelectedGeneSetConfig:
    min_members: int = 10
    max_members: int = 200
    score_min: float = .5

    def validate(self):
        if (type(self.min_members) is not int or type(self.max_members) is not int
                or self.min_members < 1 or self.max_members < self.min_members
                or not np.isfinite(self.score_min) or not 0 <= self.score_min <= 1):
            raise ValueError("Invalid selected Gene Set thresholds")


def add_fold_support(scores, binding, config):
    result = scores.sort_values(KEYS).reset_index(drop=True).copy()
    source = binding["sources"]; available = source["selected_fold_estimates_available"]
    records = source["reference_fold_sources"]
    if len(records) != 5 or {r["fold"] for r in records} != set(range(5)): raise ValueError("Incomplete primary fold annotation sources")
    n = len(result); total = np.zeros(n); squared = np.zeros(n); selected = np.zeros(n); percentile = np.zeros(n); agreement = np.zeros(n)
    for record in sorted(records, key=lambda r: r["fold"]):
        path = verified_path(record, Path.cwd())
        columns = [*KEYS, "association_direction_probability"]
        if available: columns.append(source["selected_fold_score_column"])
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns))).sort_values(KEYS).reset_index(drop=True)
        if not result[KEYS].equals(frame[KEYS]): raise ValueError("Fold support candidate universe changed")
        p = pd.to_numeric(frame.association_direction_probability, errors="coerce").to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any(): raise ValueError("Invalid per-fold direction probabilities")
        direction = np.select([p>.5, p<.5], ["positive", "negative"], default="unknown")
        agreement += (direction == result.association_direction.to_numpy()) & result.direction_available.to_numpy()
        if available:
            values = frame[source["selected_fold_score_column"]].to_numpy(float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any(): raise ValueError("Invalid selected per-fold scores")
            total += values; squared += values**2
            # Selection means membership after the same directional threshold,
            # size and deterministic tie policy as the final Gene Sets.
            fold = frame.assign(_value=values, _direction=direction, _eligible=values >= config.score_min)
            fold_groups = fold.groupby(["cancer_id", "pathway_id", "_direction"])
            eligible_count = fold_groups["_eligible"].transform("sum").to_numpy()
            rank = fold_groups["_value"].rank(method="first", ascending=False).to_numpy()
            selected += (values >= config.score_min) & (rank <= config.max_members) & (eligible_count >= config.min_members)
            groups = frame.assign(_value=values).groupby(["cancer_id", "pathway_id"])
            ranks = groups["_value"].rank(method="average", ascending=False)
            counts = groups["_value"].transform("size")
            percentile += 1-(ranks.to_numpy()-1)/np.maximum(counts.to_numpy()-1, 1)
    result["primary_direction_fold_agreement"] = np.where(result.direction_available, agreement/5, np.nan)
    result["selected_fold_estimates_available"] = available
    result["selected_score_fold_count"] = 5 if available else 0
    result["selected_score_fold_sd"] = np.sqrt(np.maximum((squared-total**2/5)/4, 0)) if available else np.nan
    result["selected_fold_selection_frequency"] = selected/5 if available else np.nan
    result["selected_fold_selection_definition"] = "PER_FOLD_DIRECTIONAL_GENESET_MEMBERSHIP_WITH_SELECTED_THRESHOLDS"
    result["selected_fold_rank_percentile_mean"] = percentile/5 if available else np.nan
    result["selected_fold_unavailable_reason"] = None if available else "SELECTED_PRODUCER_HAS_NO_MATCHED_FOLD_ESTIMATES"
    return result


def selected_gene_sets(scores, config):
    config.validate()
    if not scores.selected_score.between(0, 1).all(): raise ValueError("Invalid selected scores")
    master_rows = []; member_frames = []; coverage_rows = []
    grouping = ["cancer_id", "pathway_id", "association_direction"]
    version = scores.selected_score_run_id.unique()
    if len(version) != 1: raise ValueError("Gene Sets cannot mix selected-score runs")
    for keys, group in scores.groupby(grouping, sort=True):
        cancer, pathway, direction = keys
        eligible = group.loc[group.selected_score.ge(config.score_min)].sort_values(["selected_score", "lncrna_id"], ascending=[False, True], kind="stable")
        chosen = eligible.head(config.max_members)
        passing = len(chosen) >= config.min_members
        count = len(chosen) if passing else 0
        boundary_ties = int(eligible.selected_score.eq(chosen.selected_score.iloc[-1]).sum()) if len(eligible)>config.max_members else 0
        coverage = {"cancer_id": cancer, "pathway_id": pathway, "association_direction": direction, "candidate_rows": len(group),
            "rows_passing_score": len(eligible), "exported_members": count, "geneset_available": passing,
            "unavailable_reason": None if passing else "FEWER_THAN_MIN_MEMBERS_AFTER_SELECTED_SCORE_THRESHOLD",
            "boundary_tie_count": boundary_ties, "selected_score_min": config.score_min,
            "min_members": config.min_members, "max_members": config.max_members}
        for column in group:
            if column.startswith("selected_aux_") or column == "selected_primary_fallback":
                coverage[column+"_rows"] = int(group[column].sum())
        coverage_rows.append(coverage)
        if not passing: continue
        identifier = stable_id("LGS-SELECTED:", [*keys, version[0], asdict(config)])
        name = f"{cancer}__{pathway}__{direction.upper()}__{version[0].split(':')[-1][:12]}"
        master_rows.append({"geneset_id": identifier, "geneset_name": name, "geneset_type": "selected_score_exact_pathway_ranked",
            "cancer_id": cancer, "pathway_id": pathway, "pathway_family_id": chosen.pathway_family_id.iloc[0], "direction": direction,
            "member_count": count, "shared_member_count": int(chosen.shared_or_local_scope.eq("shared").sum()),
            "local_member_count": int(chosen.shared_or_local_scope.eq("cancer_local").sum()), "pathway_target_level": "exact_pathway",
            "ranking_score_column": "selected_score", "rank_tie_policy": "SCORE_DESC_THEN_STABLE_LNCRNA_ID",
            "rule_evidence_used_for_ranking": False})
        ranked = chosen.copy(); ranked.insert(0, "geneset_id", identifier); ranked.insert(1, "geneset_rank", np.arange(1, count+1))
        ranked["pathway_target_level"] = "exact_pathway"; member_frames.append(ranked)
    master_columns = ["geneset_id", "geneset_name", "geneset_type", "cancer_id", "pathway_id", "pathway_family_id", "direction",
        "member_count", "shared_member_count", "local_member_count", "pathway_target_level", "ranking_score_column", "rank_tie_policy", "rule_evidence_used_for_ranking"]
    master = pd.DataFrame(master_rows, columns=master_columns)
    members = pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame(columns=["geneset_id", "geneset_rank", *scores.columns, "pathway_target_level"])
    return master, members, pd.DataFrame(coverage_rows)


def selected_subtypes(scores, members, config):
    config.validate()
    if config.score_column != "selected_score": raise ValueError("Current subtypes must weight the selected score")
    base = scores[["cancer_id", "pathway_id", "pathway_family_id"]].drop_duplicates().sort_values(["pathway_id", "cancer_id"])
    directed = members.loc[members.association_direction.isin(["positive", "negative"])].copy()
    context_columns = [*base.columns, "n_shared_members", "pathway_context_subtype_id", "pathway_context_subtype",
        "classification_status", "selected_k", "proposed_k", "silhouette", "bootstrap_ari", "mean_pairwise_similarity"]
    conservation_columns = ["pathway_id", "pathway_family_id", "n_evaluable_cancers", "mean_pairwise_similarity",
        "bootstrap_ci_lower", "bootstrap_ci_upper", "selected_k", "proposed_k", "silhouette", "bootstrap_ari", "classification_status"]
    pair_columns = ["pathway_id", "pathway_family_id", "cancer_a", "cancer_b", "rbo_similarity", "weighted_jaccard_similarity", "combined_similarity"]
    if directed.empty:
        context = pd.DataFrame(columns=context_columns); conservation = pd.DataFrame(columns=conservation_columns); pairwise = pd.DataFrame(columns=pair_columns)
    else: context, conservation, pairwise = classify_pathway_context_subtypes(directed, config)
    context = base.merge(context, on=list(base), how="left", validate="one_to_one")
    context["classification_status"] = context.classification_status.fillna("UNAVAILABLE")
    for column in ("pathway_context_subtype_id", "pathway_context_subtype"): context[column] = context[column].fillna("UNAVAILABLE")
    for column in ("n_shared_members", "selected_k", "proposed_k"): context[column] = context[column].fillna(0).astype(int)
    paths = base[["pathway_id", "pathway_family_id"]].drop_duplicates()
    conservation = paths.merge(conservation, on=list(paths), how="left", validate="one_to_one")
    conservation["classification_status"] = conservation.classification_status.fillna("UNAVAILABLE")
    for column in ("n_evaluable_cancers", "selected_k", "proposed_k"): conservation[column] = conservation[column].fillna(0).astype(int)
    program, program_similarity = classify_cancer_program_subtypes(context, conservation, pairwise, config)
    conservation["unavailable_reason"] = np.where(conservation.classification_status.eq("UNAVAILABLE"),
        "INSUFFICIENT_EVALUABLE_CANCERS", None)
    program["unavailable_reason"] = [None if r.classification_status != "UNAVAILABLE" else
        "INSUFFICIENT_EVALUABLE_PATHWAYS" if r.n_evaluable_pathways < config.min_program_pathways
        else "INSUFFICIENT_EVALUABLE_CANCERS" for r in program.itertuples()]
    member_counts = members.groupby(["cancer_id", "pathway_id"]).size()
    known_counts = directed.groupby(["cancer_id", "pathway_id"]).size()
    context["unavailable_reason"] = [None if r.classification_status != "UNAVAILABLE" else
        "NO_GENESET_PASSING_THRESHOLDS" if not member_counts.get((r.cancer_id, r.pathway_id), 0) else
        "NO_MODEL_DIRECTION_FOR_RANKED_SUBTYPE" if not known_counts.get((r.cancer_id, r.pathway_id), 0) else
        "INSUFFICIENT_SHARED_MEMBERS" if r.n_shared_members < config.min_members else "INSUFFICIENT_CANCER_COVERAGE"
        for r in context.itertuples()]
    tables = {"pathway_context_subtype": context, "pathway_conservation": conservation, "pathway_pairwise_similarity": pairwise,
        "cancer_program_subtype": program, "cancer_program_similarity": program_similarity}
    for frame in tables.values():
        frame["subtype_score_column"] = "selected_score"; frame["subtype_feedback_to_model"] = False
        if "classification_status" in frame:
            frame["available"] = frame.classification_status.ne("UNAVAILABLE")
            for column in ("selected_k", "proposed_k"):
                if column in frame: frame[column] = frame[column].where(frame.available).astype("Int64")
            for column in ("pathway_context_subtype_id", "pathway_context_subtype", "cancer_program_subtype_id", "cancer_program_subtype"):
                if column in frame: frame[column] = frame[column].where(frame.available).astype("string")
        for column in ("silhouette", "bootstrap_ari", "mean_pairwise_similarity", "bootstrap_ci_lower", "bootstrap_ci_upper",
                       "rbo_similarity", "weighted_jaccard_similarity", "combined_similarity", "program_similarity"):
            if column in frame: frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return tables


def analysis_tables(scores, binding, gene_config, subtype_config):
    scored = add_fold_support(scores, binding, gene_config)
    master, members, coverage = selected_gene_sets(scored, gene_config)
    return {"geneset_master": master, "geneset_members": members, "geneset_coverage": coverage,
            **selected_subtypes(scored, members, subtype_config)}


def gmt_text(master, members, run_id, score_sha):
    description = "CancerLncAtlas selected_score="+run_id+";binding_sha256="+score_sha
    for frame, columns in ((master, ["geneset_name"]), (members, ["lncrna_id"])):
        if any(frame[col].astype(str).str.contains(r"[\t\r\n]", regex=True).any() for col in columns):
            raise ValueError("GMT identifiers must not contain field separators")
    lines = genesets_to_gmt_lines(master, members, description=description)
    return "\n".join(lines)+("\n" if lines else "")


def implementation_hashes():
    from .selected_score import implementation_hashes as score_hashes
    root = Path(__file__).parent
    return {**score_hashes(), **{name: sha256(root/name) for name in ("selected_genesets.py", "genesets.py", "subtypes.py")}}


def materialize_selected_analysis(manifest_path, output):
    import socket
    if socket.gethostname() != "149": raise RuntimeError("Selected analysis must execute on host 149")
    manifest_path = Path(manifest_path).resolve(); source_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False, "input_manifest_sha256": source_sha}))
    score_path = materialize_selected_scores(manifest_path, output/"selected_score")
    scores, binding = load_selected_scores(score_path, expected_sha=sha256(score_path))
    gene_config = SelectedGeneSetConfig(**manifest["geneset"]); gene_config.validate()
    subtype_config = RankedSubtypeConfig(**manifest["subtype"])
    tables = analysis_tables(scores, binding, gene_config, subtype_config)
    outputs = {}
    score_sha = sha256(score_path)
    for name, frame in tables.items():
        frame["selected_score_run_id"] = binding["run_id"]; frame["selected_score_binding_sha256"] = score_sha
        frame["analysis_version"] = "CancerLncAtlas_V3.2_SELECTED_ANALYSIS"
        path = output/(name.upper()+".parquet"); frame.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {"path": str(path), "sha256": sha256(path), "rows": len(frame)}
    gmt = output/"SELECTED_GENESETS.gmt"
    gmt.write_text(gmt_text(tables["geneset_master"], tables["geneset_members"], binding["run_id"], score_sha), encoding="utf-8")
    outputs["gmt"] = {"path": str(gmt), "sha256": sha256(gmt), "genesets": len(tables["geneset_master"])}
    if sha256(manifest_path) != source_sha: raise ValueError("Selected analysis manifest changed during execution")
    release = {"format": "CURRENT_SELECTED_ANALYSIS_RELEASE_V1", "status": "PASS", "target_host": "149",
        "selected_score_binding": {"path": str(score_path), "sha256": score_sha}, "selected_score_run_id": binding["run_id"],
        "input_manifest": {"path": str(manifest_path), "sha256": source_sha}, "geneset_config": asdict(gene_config),
        "subtype_config": asdict(subtype_config), "outputs": outputs, "training_started": False,
        "subtype_feedback_to_model": False, "rule_evidence_used_for_ranking": False, "production_deployed": False,
        "implementation_sha256": implementation_hashes()}
    path = output/"SELECTED_ANALYSIS_BINDING.json"; path.write_text(json.dumps(release, indent=2)+"\n")
    audit = audit_selected_analysis(path, expected_sha=sha256(path))
    audit_path = output/"RECOMPUTATION_AUDIT.json"; audit_path.write_text(json.dumps(audit, indent=2)+"\n")
    (output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "binding_sha256": sha256(path),
        "audit_sha256": sha256(audit_path), "training_started": False}))
    return {"binding_path": str(path), "binding_sha256": sha256(path), "selected_score_run_id": binding["run_id"], "training_started": False}


def _load_analysis(binding_path, expected_sha):
    path = verified_path({"path": str(binding_path), "sha256": expected_sha}, Path.cwd())
    binding = json.loads(path.read_text(encoding="utf-8"))
    if (binding.get("format") != "CURRENT_SELECTED_ANALYSIS_RELEASE_V1" or binding.get("status") != "PASS"
            or binding.get("target_host") != "149" or binding.get("training_started") is not False
            or binding.get("rule_evidence_used_for_ranking") is not False
            or binding.get("subtype_feedback_to_model") is not False):
        raise ValueError("Invalid selected analysis binding")
    outputs = {name: verified_path(record, path.parent) for name, record in binding["outputs"].items()}
    score_path = verified_path(binding["selected_score_binding"], path.parent)
    return path, binding, outputs, score_path


def audit_selected_analysis(binding_path, *, expected_sha):
    path, binding, outputs, score_path = _load_analysis(binding_path, expected_sha)
    if binding["implementation_sha256"] != implementation_hashes():
        raise ValueError("Selected analysis replay requires its recorded implementation")
    manifest = json.loads(verified_path(binding["input_manifest"], path.parent).read_text(encoding="utf-8"))
    scores, score_binding = load_selected_scores(score_path, expected_sha=binding["selected_score_binding"]["sha256"])
    geneset = SelectedGeneSetConfig(**manifest["geneset"]); subtype = RankedSubtypeConfig(**manifest["subtype"])
    if (asdict(geneset) != binding["geneset_config"] or asdict(subtype) != binding["subtype_config"]
            or score_binding["run_id"] != binding["selected_score_run_id"]
            or score_binding["input_manifest"] != binding["input_manifest"]):
        raise ValueError("Selected analysis policy or run differs from its score snapshot")
    expected = analysis_tables(scores, score_binding, geneset, subtype)
    if set(outputs) != set(expected)|{"gmt"}: raise ValueError("Selected analysis output inventory drift")
    for name, frame in expected.items():
        frame["selected_score_run_id"] = score_binding["run_id"]
        frame["selected_score_binding_sha256"] = binding["selected_score_binding"]["sha256"]
        frame["analysis_version"] = "CancerLncAtlas_V3.2_SELECTED_ANALYSIS"
        observed = pd.read_parquet(outputs[name])
        if len(observed) != binding["outputs"][name]["rows"]: raise ValueError("Selected output row count drift")
        try: pd.testing.assert_frame_equal(observed, frame, check_dtype=False, rtol=0, atol=1e-12)
        except AssertionError as error: raise ValueError("Selected source recomputation disagrees: "+name) from error
    expected_gmt = gmt_text(expected["geneset_master"], expected["geneset_members"], score_binding["run_id"], binding["selected_score_binding"]["sha256"])
    if (outputs["gmt"].read_text(encoding="utf-8") != expected_gmt
            or binding["outputs"]["gmt"]["genesets"] != len(expected["geneset_master"])):
        raise ValueError("Selected GMT differs from selected ranked members")
    return {"status": "PASS", "binding_sha256": expected_sha, "source_scores_recomputed": True,
        "same_implementation_replay": True, "independent_statistical_implementation": False,
        "training_started": False, "selected_score_run_id": score_binding["run_id"]}


class SelectedAnalysisQuery:
    """Exact filters and version-bound downloads for the selected analysis."""
    def __init__(self, binding_path, *, expected_sha):
        path, self.binding, self.paths, score_path = _load_analysis(binding_path, expected_sha)
        success = json.loads((path.parent/"SUCCESS.json").read_text())
        audit_path = path.parent/"RECOMPUTATION_AUDIT.json"
        audit = json.loads(audit_path.read_text())
        if (success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha
                or success.get("audit_sha256") != sha256(audit_path)
                or audit.get("status") != "PASS" or audit.get("binding_sha256") != expected_sha):
            raise ValueError("Selected analysis lacks completed source replay")
        self.binding_sha256 = expected_sha
        self.score_binding_path = score_path

    def download(self, name):
        if name not in self.paths: raise ValueError("Unknown selected analysis download")
        record = self.binding["outputs"][name]
        verified_path(record, Path.cwd())
        return {**record, "selected_score_run_id": self.binding["selected_score_run_id"], "binding_sha256": self.binding_sha256}

    def query(self, *, table="geneset_members", filters=None, limit=100, offset=0):
        import duckdb
        if table not in self.paths or table == "gmt": raise ValueError("Unknown selected analysis table")
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 1000 or not 0 <= offset <= 1000000:
            raise ValueError("Invalid selected analysis pagination")
        filters = filters or {}
        allowed = {"cancer_id", "lncrna_id", "pathway_id", "pathway_family_id", "geneset_id", "association_direction", "classification_status"}
        if not set(filters).issubset(allowed) or not all(isinstance(v, str) for v in filters.values()):
            raise ValueError("Selected analysis filters require exact string identities")
        path = self.download(table)["path"]
        connection = duckdb.connect()
        try:
            columns = [r[0] for r in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path]).fetchall()]
            if not set(filters).issubset(columns): raise ValueError("Filter does not belong to this selected table")
            where = " AND ".join('"'+key+'"=?' for key in filters) or "true"
            order = ",".join('"'+key+'"' for key in ["cancer_id", "pathway_id", "geneset_id", "geneset_rank", "lncrna_id", "cancer_a", "cancer_b"] if key in columns) or "1"
            frame = connection.execute(f"SELECT * FROM read_parquet(?) WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [path, *filters.values(), limit, offset]).df()
        finally: connection.close()
        return {"selected_score_run_id": self.binding["selected_score_run_id"], "binding_sha256": self.binding_sha256,
                "rows": frame.astype(object).where(frame.notna(), None).to_dict(orient="records")}
