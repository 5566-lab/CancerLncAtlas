#!/usr/bin/env python3
"""Fail-closed HTTP smoke test for a candidate V3.1 exact-pathway website."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cc_hhgt.v30_integrity import atomic_write_json


def _get(base_url: str, path: str, *, binary: bool = False):
    request = Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "*/*"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - explicit smoke target
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {path}")
        body = response.read(128 if binary else None)
        if binary:
            if not body:
                raise RuntimeError(f"Empty download: {path}")
            return {"bytes_sampled": len(body), "content_type": response.headers.get("Content-Type")}
        return json.loads(body.decode("utf-8"))


def smoke(base_url: str, expected_lncRNAs: int) -> dict:
    stats = _get(base_url, "/api/site/stats")
    if (
        stats.get("version") != "3.1-exact-pathway"
        or int(stats.get("cancers", -1)) != 33
        or int(stats.get("scored_cancers", -1)) != 33
        or int(stats.get("reference_only_cancers", -1)) != 0
        or int(stats.get("lncrnas", -1)) != expected_lncRNAs
        or int(stats.get("exact_pathways", 0)) <= 0
        or stats.get("pathway_target_level") != "exact_pathway"
        or stats.get("pathway_family_role") != "auxiliary_hierarchy_only"
    ):
        raise RuntimeError(f"Invalid production V3.1 stats contract: {stats}")

    cancers_payload = _get(base_url, "/api/site/cancers")
    cancers = cancers_payload.get("cancers", [])
    cancer_ids = {str(row.get("cancer_id")) for row in cancers}
    if (
        len(cancers) != 33
        or len(cancer_ids) != 33
        or not {"HNSC", "LGG"}.issubset(cancer_ids)
        or any(bool(row.get("reference_only")) for row in cancers)
    ):
        raise RuntimeError("Production cancer overview is not 33-cancer/no-reference-only")

    checked_cancers = {}
    probe_pathway = None
    probe_lnc = None
    for cancer in ("HNSC", "LGG"):
        profile = _get(base_url, f"/api/site/cancer/{cancer}?limit=5")
        overview = profile.get("overview", {})
        if (
            overview.get("cancer_id") != cancer
            or bool(overview.get("reference_only"))
            or overview.get("model_version") != "V3.1-exact-pathway"
            or not profile.get("pathway_ranking")
        ):
            raise RuntimeError(f"Invalid formal cancer profile: {cancer}")
        genesets = _get(
            base_url,
            f"/api/site/genesets?{urlencode({'cancer': cancer, 'limit': 5})}",
        )
        if (
            genesets.get("pathway_target_level") != "exact_pathway"
            or int(genesets.get("total_genesets", 0)) <= 0
            or not genesets.get("members")
        ):
            raise RuntimeError(f"No exact-pathway gene set for formal cancer: {cancer}")
        checked_cancers[cancer] = int(genesets["total_genesets"])
        if probe_pathway is None:
            probe_pathway = str(genesets["genesets"][0]["pathway_id"])
            probe_lnc = str(genesets["members"][0]["lncrna_id"])

    search = _get(
        base_url,
        f"/api/site/search?{urlencode({'q': probe_pathway, 'limit': 20})}",
    )
    if not any(
        row.get("type") == "Exact pathway" and str(row.get("id")) == probe_pathway
        for row in search.get("results", [])
    ):
        raise RuntimeError("Exact-pathway search did not return the probed pathway")
    lnc = _get(
        base_url,
        f"/api/site/lncrna/{probe_lnc}?{urlencode({'cancer': 'HNSC'})}",
    )
    if (
        lnc.get("model_version") != "V3.1-exact-pathway"
        or lnc.get("pathway_target_level") != "exact_pathway"
        or not lnc.get("top_relationships")
    ):
        raise RuntimeError("V3.1 exact-pathway lncRNA profile smoke failed")

    downloads = {}
    for key in (
        "v31-genesets",
        "v31-genesets-ids",
        "v31-geneset-master",
        "v31-geneset-members",
    ):
        downloads[key] = _get(base_url, f"/api/site/download/{key}", binary=True)
    return {
        "status": "PASS",
        "production_switch_eligible": True,
        "base_url": base_url,
        "version": stats["version"],
        "cancers": 33,
        "required_formal_cancers": ["HNSC", "LGG"],
        "reference_only_cancers": 0,
        "pancancer_eligible_lncRNAs": expected_lncRNAs,
        "exact_pathways": int(stats["exact_pathways"]),
        "selected_model": stats["selected_model"],
        "checked_genesets": checked_cancers,
        "probe_pathway": probe_pathway,
        "probe_lncRNA": probe_lnc,
        "downloads": downloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-lncrnas", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Website smoke output already exists: {output}")
    payload = smoke(args.base_url, args.expected_lncrnas)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
