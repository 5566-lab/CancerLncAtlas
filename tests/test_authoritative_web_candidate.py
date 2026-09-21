from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_APP = (
    ROOT
    / "artifacts/web_authoritative_candidate_20260829_r1"
    / "authoritative/website/backend/app.py"
)


def test_authoritative_full_app_loads_and_optional_sources_fail_soft() -> None:
    program = r'''
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pandas as pd

# The black-box candidate only needs psutil in the diagnostics endpoint.  Keep
# this unit test independent of that optional runtime package while exercising
# the route and materialization semantics below.
sys.modules.setdefault("psutil", types.ModuleType("psutil"))

app_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(app_path.parent))
spec = importlib.util.spec_from_file_location("authoritative_candidate_app", app_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

assert any(
    getattr(route, "path", None) == "/api/site/genesets/{geneset_id}"
    for route in module.app.routes
)
assert len(module.CANCER_SEARCH_ALIASES) == 33
candidate = module.candidate_health()
assert candidate["candidate_id"] == "pytest-authoritative-candidate"
assert candidate["production_deployed"] is False
assert candidate["candidate_port"] == "8261"
assert candidate["app_sha256"] == "pytest-app-sha"
assert candidate["v32_staging_enabled"] is False

# A formal V3 table must be usable without the deleted V2.6 directory.  One
# missing optional component must not turn the whole cancer profile into 503,
# and the empty payload must carry an explicit typed-unavailable status.
from tempfile import TemporaryDirectory
with TemporaryDirectory() as temporary_root:
    temporary_root = Path(temporary_root)
    module.WEB_TABLES = temporary_root / "formal/web_tables"
    module.LEGACY_WEB_TABLES = temporary_root / "deleted_v26/web_tables"
    module.WEB_TABLES.mkdir(parents=True)
    for table_name in (
        "web_cancer_celltype_summary",
        "web_cancer_state_summary",
        "web_cancer_geneset_summary",
    ):
        pd.DataFrame(
            {"cancer_id": ["ACC"], "value": [table_name]}
        ).to_parquet(module.WEB_TABLES / f"{table_name}.parquet", index=False)

    ranking = pd.DataFrame({
        "cancer_id": ["ACC"],
        "lncrna_id": ["LNC:1"],
        "gene_symbol": ["LNC1"],
        "pathway_family_id": ["PF:1"],
        "functional_score_available": [True],
        "discovery_ranking_probability": [0.8],
        "fused_confidence_probability": [0.7],
        "clinical_relevance_score": [None],
        "translational_priority_score": [0.75],
        "strict_available": [1],
        "n_evidence_events": [0],
        "confidence_tier": ["medium"],
    })
    pathway = pd.DataFrame({
        "cancer_id": ["ACC"],
        "pathway_family_id": ["PF:1"],
        "rank_within_cancer": [1],
        "n_lncRNAs": [1],
        "mean_discovery_probability": [0.8],
        "mean_confidence_probability": [0.7],
    })

    class CancerProfileStub:
        cancer_component = module.SiteStore.cancer_component
        legacy_web_table = module.SiteStore.legacy_web_table
        _histogram = staticmethod(lambda values: [])
        def __init__(self):
            self._web_cache = {}
            self._legacy_web_cache = {}
        def web_table(self, name):
            if name == "web_cancer_pathway_ranking":
                return pathway
            return module.SiteStore.web_table(self, name)
        def cancer_overview(self):
            return pd.DataFrame({"cancer_id": ["ACC"]})
        def cancer_candidates(self, cancer_id):
            return ranking.copy()
        def family_names(self):
            return {"PF:1": "Pathway one"}
        def single_cell_cohort_summary(self, cancer_id):
            return []
        def clinical_patient_count(self, cancer_id):
            return None
        def bulk_sample_count(self, cancer_id):
            return None
        def cancer_evidence_event_count(self, cancer_id):
            return 0

    profile = module.SiteStore.cancer_profile(CancerProfileStub(), "ACC", 5)
    assert profile["celltype_summary"][0]["value"] == "web_cancer_celltype_summary"
    assert profile["component_status"]["celltype"]["source"] == "formal_release"
    assert profile["component_status"]["celltype"]["status"] == "available"
    assert profile["drug_summary"] == []
    assert profile["component_status"]["drug"]["status"] == "source_unavailable"
    assert profile["component_status"]["drug"]["reason"] == (
        "table_missing_in_formal_and_legacy_releases"
    )
    assert profile["module_status"]["drug"] == "source_unavailable"

module.scored_functional_rows = lambda frame: frame

class LncStub:
    def resolve_lnc(self, value):
        return "LNC:1"
    def v3_query(self, name, filters):
        return pd.DataFrame({
            "lncrna_id": ["LNC:1"],
            "cancer_id": ["LUAD"],
            "pathway_family_id": ["PF:1"],
            "fused_confidence_probability": [0.8],
            "discovery_ranking_probability": [0.9],
        })
    def bulk_adjusted_heatmap(self, value):
        raise module.HTTPException(status_code=503, detail="heatmap missing")
    def bulk_expression_pathway_scatter(self, *args):
        return {"status": "available", "points": []}
    def survival_curve(self, *args):
        raise module.HTTPException(status_code=503, detail="curve missing")
    def lnc_drug_evidence(self, *args):
        return {"status": "no_evidence", "rows": []}

lnc = module.SiteStore.lnc_visuals.__wrapped__(LncStub(), "MALAT1")
assert lnc["status"] == "available"
assert lnc["pathway_heatmap"]["status"] == "source_unavailable"
assert lnc["survival_curve"]["status"] == "source_unavailable"
assert lnc["bulk_scatter"]["status"] == "available"

module.require_clinical_release = lambda: None

class ClinicalStub:
    def clinical_query(self, name, filters, columns=None):
        if name == "web_clinical_endpoint_summary":
            return pd.DataFrame({"cancer_id": ["LUAD"], "endpoint": ["OS"], "c_index": [0.6]})
        if name == "web_lncRNA_survival_association":
            return pd.DataFrame(columns=["hazard_ratio", "ci_lower", "ci_upper", "subject_id"])
        if name == "web_translational_priority":
            return pd.DataFrame({
                "lncrna_id": ["LNC:1"],
                "pathway_family_id": ["PF:1"],
                "clinical_relevance_score": [0.5],
                "translational_priority_score": [0.7],
            })
        raise AssertionError(name)
    def clinical_endpoint_counts(self, cancer):
        return pd.DataFrame({"endpoint": ["OS"], "n_patients": [2], "n_events": [1]})
    def clinical_raw_query(self, *args, **kwargs):
        return pd.DataFrame(columns=["patient_id", "time_days", "event", "endpoint_available", "risk_score"])
    def survival_curve(self, *args):
        raise module.HTTPException(status_code=503, detail="curve missing")
    def resolve_lnc(self, value):
        return "LNC:1"

clinical = module.SiteStore.clinical_visuals(ClinicalStub(), "LUAD")
assert clinical["status"] == "available"
assert clinical["kaplan_meier"]["status"] == "source_unavailable"
assert clinical["endpoint_comparison"][0]["n_events"] == 1

class EmptyClinicalStub:
    def clinical_query(self, name, filters, columns=None):
        assert name == "web_clinical_endpoint_summary"
        return pd.DataFrame(columns=["cancer_id", "endpoint", "c_index"])

empty_clinical = module.SiteStore.clinical_visuals(EmptyClinicalStub(), "CHOL")
assert empty_clinical["status"] == "no_evidence"
assert empty_clinical["reason_code"] == "NO_ELIGIBLE_CLINICAL_ENDPOINT"
assert empty_clinical["endpoint_comparison"] == []
assert empty_clinical["kaplan_meier"]["status"] == "no_evidence"
assert empty_clinical["event_count_status"] == "no_evidence"

class Exact:
    def geneset_detail(self, value):
        return {"status": "available", "requested_id": value, "pathway_target_level": "exact_pathway"}

class GeneSetStub:
    exact_store = Exact()
    def formal_geneset_overview(self):
        raise module.HTTPException(status_code=503, detail="formal table missing")

detail = module.SiteStore.geneset_detail(GeneSetStub(), "PF:1")
assert detail["status"] == "available"
assert detail["pathway_target_level"] == "exact_pathway"

terms = module._search_tokens("LUAD", "Lung adenocarcinoma", module.CANCER_SEARCH_ALIASES["LUAD"])
search_stub = type("SearchStub", (), {"_search_rows": [{
    "type": "Cancer", "id": "LUAD", "label": "Lung adenocarcinoma",
    "subtitle": "LUAD", "href": "/cancer?id=LUAD", "_search_terms": "|".join(terms),
}]})()
search = module.SiteStore.search(search_stub, "肺癌", 10)
assert [row["id"] for row in search] == ["LUAD"]

print(json.dumps({"loaded": True, "routes": len(module.app.routes)}))
'''
    environment = dict(os.environ)
    environment["CANCERLNCATLAS_FRONTEND_ROOT"] = str(
        ROOT / "website/frontend"
    )
    environment["CANCERLNCATLAS_V31_RELEASE_ROOT"] = "__missing__"
    environment["CANCERLNCATLAS_CANDIDATE_ID"] = "pytest-authoritative-candidate"
    environment["CANCERLNCATLAS_PRODUCTION_DEPLOYED"] = "false"
    environment["CANCERLNCATLAS_CANDIDATE_PORT"] = "8261"
    environment["CANCERLNCATLAS_CANDIDATE_APP_SHA256"] = "pytest-app-sha"
    environment["CANCERLNCATLAS_ENABLE_V32_STAGING"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(CANDIDATE_APP)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["loaded"] is True
    assert payload["routes"] >= 60
