"""Validate, document, manifest, and package the completed v2.8 full-expert release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    ndcg_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results"
RELEASE = Path(os.getenv("CC_HHGT_RELEASE_DIR", str(RESULT / "v2_8_cancer_native_moe_release")))


def validate() -> None:
    required = [
        RESULT / "strict_global_run" / "SUCCESS.json",
        RESULT / "adapter_data" / "SUCCESS.json",
        RESULT / "cancer_adapter_run" / "SUCCESS.json",
        RELEASE / "PROBABILITY_RELEASE_SUCCESS.json",
        RELEASE / "CANCER_NATIVE_MOE_AUDIT.json",
        RESULT / "graph_expert_stacking" / "SUCCESS.json",
        RESULT / "evidence_event_model" / "EVIDENCE_TRANSFORMER_SUCCESS.json",
        RELEASE / "FULL_EXPERT_FUSION_SUCCESS.json",
        RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.json",
        RESULT / "lncrna_state_release" / "LNCRNA_STATE_MODEL_SUCCESS.json",
        RESULT / "lncrna_state_release" / "LNCRNA_STATE_AUDIT.json",
        RESULT / "lncrna_state_release" / "lncrna_state_final.parquet",
        RESULT / "custom_geneset_encoder" / "SUCCESS.json",
        RESULT / "protein_set_encoder" / "SUCCESS.json",
        ROOT / "query_assets" / "SUCCESS.json",
        RESULT / "query_benchmark.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"release is incomplete; missing: {missing}")
    moe_audit = json.loads(
        (RELEASE / "CANCER_NATIVE_MOE_AUDIT.json").read_text(encoding="utf-8")
    )
    if moe_audit.get("status") != "PASS":
        raise RuntimeError("cancer-native MoE audit did not pass")
    full_audit = json.loads((RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.json").read_text(encoding="utf-8"))
    if full_audit.get("status") != "PASS":
        raise RuntimeError("full model requirements audit did not pass")
    failures = list(ROOT.glob("results/**/FAILED.txt"))
    if failures:
        raise RuntimeError(f"release contains failure markers: {failures}")


def strict_metrics() -> pd.DataFrame:
    summary = pd.read_parquet(
        RESULT / "strict_release" / "strict_global_multiseed_summary.parquet"
    )
    test = summary[summary["split"].eq("test")]
    return (
        test.groupby("model_name", observed=True)
        .agg(
            n_folds=("fold_id", "nunique"),
            auprc_mean=("auprc_mean", "mean"),
            auroc_mean=("auroc_mean", "mean"),
            brier_mean=("brier_mean", "mean"),
            ece_mean=("ece_mean", "mean"),
        )
        .reset_index()
    )


def cold_start_metrics() -> pd.DataFrame:
    prediction = pd.read_parquet(
        RESULT / "strict_release" / "strict_cross_cancer_oof_prediction.parquet"
    )
    node = pd.read_parquet(
        RESULT / "strict_graph" / "graph_node.parquet",
        columns=["canonical_id", "node_type", "log_degree"],
    )
    node = node[node["node_type"].eq("lncRNA")].rename(
        columns={"canonical_id": "lncrna_id"}
    )
    prediction = prediction.merge(
        node[["lncrna_id", "log_degree"]], on="lncrna_id", how="left"
    )
    prediction["cold_start"] = prediction["log_degree"].fillna(0).eq(0)
    rows = []
    for cold_start, group in prediction.groupby("cold_start"):
        valid = group["proxy_label"].notna()
        labels = group.loc[valid, "proxy_label"].astype(int).to_numpy()
        probability = group.loc[
            valid, "cross_cancer_probability"
        ].to_numpy()
        rows.append(
            {
                "cold_start_status": (
                    "static_graph_cold_start"
                    if cold_start
                    else "static_graph_observed"
                ),
                "n_pairs": len(group),
                "auprc": (
                    average_precision_score(labels, probability)
                    if len(np.unique(labels)) > 1
                    else math.nan
                ),
                "auroc": (
                    roc_auc_score(labels, probability)
                    if len(np.unique(labels)) > 1
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def ranking_metrics() -> dict[str, float]:
    summary = pd.read_parquet(
        RELEASE / "cancer_specific_multiseed_summary.parquet"
    )
    values = []
    for _, group in summary.groupby("cancer_id", observed=True):
        if len(group) < 2:
            continue
        truth = group["heldout_patient_support_rate"].fillna(0).to_numpy()[None, :]
        score = group["cancer_specific_probability"].fillna(0).to_numpy()[None, :]
        values.append(float(ndcg_score(truth, score)))
    return {
        "cancer_specific_ndcg_macro": float(np.mean(values)),
        "n_cancers": len(values),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    line = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, line, *rows])


def write_reports() -> None:
    strict = strict_metrics()
    cold = cold_start_metrics()
    patient = pd.read_parquet(RELEASE / "patient_level_oof_metrics.parquet")
    custom = pd.read_parquet(
        RESULT / "custom_geneset_encoder" / "custom_geneset_metrics.parquet"
    ).iloc[0].to_dict()
    protein = pd.read_parquet(
        RESULT / "protein_set_encoder" / "protein_set_metrics.parquet"
    ).iloc[0].to_dict()
    benchmark = json.loads(
        (RESULT / "query_benchmark.json").read_text(encoding="utf-8")
    )
    ranking = ranking_metrics()
    strict.to_csv(RELEASE / "strict_loco_macro_metrics.tsv", sep="\t", index=False)
    cold.to_csv(RELEASE / "strict_cold_start_metrics.tsv", sep="\t", index=False)
    pd.DataFrame([ranking]).to_csv(
        RELEASE / "cancer_specific_ranking_metrics.tsv", sep="\t", index=False
    )

    model_card = f"""# CancerLncAtlas Full Expert Model Card

Generated: {datetime.now().isoformat()}

## Intended outputs

- `graph_ensemble_probability`: OOF stacking of already-trained R-GCN, HGT, and CC-HHGT-Strict experts.
- `cross_cancer_probability`: frozen CC-HHGT-Strict component retained for provenance.
- `cancer_native_probability`: patient-native expert that excludes strict scores and embeddings.
- `cancer_specific_probability`: internal patient native/joint combination with strict-availability masking.
- `indirect_mechanism_probability`: EventSetEncoder/Transformer score from indirect interaction, perturbation, drug, and cell-line routes; eligible for discovery.
- `direct_evidence_probability`: explicit lncRNA-pathway literature/experiment assertions; confidence only.
- `evidence_integrated_probability`: frozen historical evidence score retained as a separate confidence expert.
- `discovery_ranking_probability`: graph ensemble + patient-native + indirect event expert.
- `fused_confidence_probability`: discovery experts + direct event + historical evidence experts.

All graph and patient inputs to the final gates are OOF. The final score is not an arithmetic mean. Unavailable experts are masked before softmax. Direct target evidence is prohibited from the discovery branch.

## Strict LOCO performance

{markdown_table(strict)}

These metrics recover sampled proxy labels. They are not direct biological
accuracy estimates and must not be described as independent experimental
validation.

## Cold-start

{markdown_table(cold)}

## Patient-native candidate recovery

The candidate universe is the union of strict candidates, patient-native
train-fold candidates, and optional evidence candidates. Patient-native
candidates are selected separately in every fold from training patients only.
A strict-unavailable candidate remains scoreable by the patient-native branch.

Mean patient-level ranking NDCG is
{ranking['cancer_specific_ndcg_macro']:.4f}. Patient folds use train-only
scalers, residualizers, candidate selection, thresholds, and features.

## Direct lncRNA--tumor-state model

A separate task predicts `cancer x lncRNA x tumor_state` for telomerase and
stemness-related states. Tumor states are not treated as ordinary pathway
members. The state model combines a patient-native association-replication
head with frozen-embedding R-GCN/HGT/CC-HHGT state decoders through an
availability-masked MoE. Strict graph encoders are not retrained by this
module. Primary outputs are `lncrna_state_probability`, direction,
patient-fold replication class, and per-expert weights.

## Query encoders

- CustomGeneSet test Recall@5: {custom['test_recall_at_5']:.4f};
  MRR: {custom['test_mrr']:.4f}; OOD AUROC: {custom['ood_auroc']:.4f}.
- ProteinSet test AUPRC: {protein['test_auprc']:.4f};
  Hits@10: {protein['hits_at_10']:.4f}; MRR: {protein['mrr']:.4f}.
- CPU latency after load: custom {benchmark['custom_query_ms']:.1f} ms,
  protein {benchmark['protein_query_ms']:.1f} ms,
  mixed {benchmark['mixed_query_ms']:.1f} ms.

## Limitations

Patient-native candidate selection is association-based and does not establish
causality. Evidence-only candidates without patient or strict coverage do not
receive a discovery score. Physical interaction probability remains separate
from functional association probability.
"""
    (ROOT / "V2_8_FULL_EXPERT_MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    # Compatibility copy for existing server materializers.
    (ROOT / "V2_6_MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    api_doc = """# CC-HHGT v2.6 Query API

Start the CPU service:

```powershell
& '.\\.conda\\python.exe' -m uvicorn cc_hhgt_v26.api:app `
  --app-dir src --host 127.0.0.1 --port 8260
```

Endpoints:

- `GET /v2.6/health`
- `GET /v2.6/version`
- `POST /v2.6/predict/mixed-gene-set`
- `POST /v2.6/predict/custom-gene-set-lncrna`
- `POST /v2.6/predict/protein-set-lncrna`

POST body:

```json
{"members":["TP53","EGFR"],"cancer_id":"LUAD","top_k":20}
```

Mixed-set requests may additionally specify `network_nodes`. Limits are 100
input members, `top_k` 100, and 200 network nodes. Omitting `cancer_id`
returns a clearly labelled pan-cancer aggregate. Results use a normalized
request SHA256 disk cache. The batch CLI is `scripts/14_query_cli.py` and
supports JSONL `--resume`.

Protein responses report physical interaction and functional association
separately. Custom gene sets return
`prediction_scope=custom_virtual_pathway`; nearest pathway-family predictions
are never relabelled as exact pathways.
"""
    (ROOT / "V2_6_QUERY_API.md").write_text(api_doc, encoding="utf-8")

    leakage = """# CC-HHGT v2.6 Leakage Audit

## Strict global line

- Static pathway families use gene-set overlap, Reactome hierarchy, static PPI,
  and DrugCentral targets only.
- Context-specific test/validation cancer edges are absent from the strict
  graph.
- Fold degree is recomputed after context filtering.
- CC-HHGT-Strict decoder inputs exclude direct bulk/sc/UCell/drug/interaction
  pair evidence used to define proxy labels.
- Calibration uses the validation cancer only.
- All 297 tasks use their declared model; fallback is disabled.

## Cancer-specific line

- Strict OOF is no longer the sole candidate gateway.
- Patient-native candidates are selected from training patients only in each fold.
- The patient-native branch does not read strict probabilities or embeddings.
- Strict-unavailable rows receive zero strict-joint weight.
- Top-level MoE training uses OOF expert predictions only.
- Old `PF:*` adapter tables are not joined to new `SPF:*` identifiers.
- Static-family activity is rebuilt from exact pathways for every patient fold.
- Scalers, covariate design, residualization, thresholds, expression detection,
  and tumor-state summaries are fitted on training patients only.
- Every patient has one OOF test assignment per parent fold.
- Missing evidence is represented by availability and is not filled as
  contradictory zero evidence.

## Query encoders

- CustomGeneSet train/validation/test splits are pathway-family-disjoint.
- Physical positives require curated experimental physical binding.
- Interaction evidence events are deduplicated and pair-grouped by PMID split;
  pair overlap across splits is zero.
- LncACTdb-exclusive, cold-lncRNA, and cold-protein evaluations are reported.
- Coexpression/pathway concordance is never displayed as physical binding.

## Interpretation warning

Proxy-label AUPRC/AUROC quantify evidence or association recovery. Independent
experimental or temporal external validation remains necessary for a causal or
de-novo biological discovery claim.
"""
    (ROOT / "V2_8_FULL_EXPERT_LEAKAGE_AUDIT.md").write_text(leakage, encoding="utf-8")
    (ROOT / "V2_6_LEAKAGE_AUDIT.md").write_text(leakage, encoding="utf-8")


def release_files() -> list[Path]:
    files: set[Path] = set()
    for name in [
        "src",
        "scripts",
        "tests",
        "config",
        "audit",
        "models",
        "query_assets",
    ]:
        files.update(path for path in (ROOT / name).rglob("*") if path.is_file())
    selected_results = [
        RESULT / "strict_release",
        RELEASE,
        RESULT / "custom_geneset_encoder",
        RESULT / "protein_set_encoder",
        RESULT / "graph_expert_stacking",
        RESULT / "evidence_event_model",
        RESULT / "lncrna_state_release",
        RESULT / "lncrna_state_run",
    ]
    for directory in selected_results:
        files.update(path for path in directory.rglob("*") if path.is_file())
    for path in [
        RESULT / "strict_global_run" / "SUCCESS.json",
        RESULT / "strict_global_run" / "run_status.json",
        RESULT / "strict_global_run" / "task_manifest.tsv",
        RESULT / "cancer_adapter_run" / "SUCCESS.json",
        RESULT / "cancer_adapter_run" / "run_status.json",
        RESULT / "cancer_adapter_run" / "task_manifest.tsv",
        RESULT / "adapter_data" / "SUCCESS.json",
        RESULT / "adapter_data" / "data_contract.json",
        RESULT / "adapter_data" / "fold_provenance.parquet",
        RESULT / "query_benchmark.json",
        ROOT / "README.md",
        ROOT / "V2_8_FULL_EXPERT_MODEL_CARD.md",
        ROOT / "V2_8_FULL_EXPERT_LEAKAGE_AUDIT.md",
        ROOT / "V2_6_MODEL_CARD.md",
        ROOT / "V2_6_QUERY_API.md",
        ROOT / "V2_6_LEAKAGE_AUDIT.md",
        ROOT / "FULL_EXPERT_EXTENSION.md",
        ROOT / "CHAT_REQUIREMENTS_IMPLEMENTATION_AUDIT.md",
        ROOT / "PROTEIN_CODING_GENE_APPLICABILITY.md",
        RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.md",
        RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.tsv",
        RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.json",
    ]:
        if path.exists():
            files.add(path)
    return sorted(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package() -> Path:
    files = release_files()
    manifest_rows = []
    for index, path in enumerate(files, 1):
        manifest_rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        if index % 250 == 0:
            print(f"[manifest] {index}/{len(files)}", flush=True)
    manifest = ROOT / "V2_8_CANCER_NATIVE_MOE_RETURN_CONTENT_MANIFEST.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest, sep="\t", index=False)
    files.append(manifest)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = ROOT / f"CC_HHGT_V2_8_CANCER_NATIVE_MOE_RESULTS_RETURN_{stamp}.zip"
    with zipfile.ZipFile(
        archive, "w", allowZip64=True
    ) as output:
        for index, path in enumerate(files, 1):
            suffix = path.suffix.lower()
            compression = (
                zipfile.ZIP_STORED
                if suffix in {".pt", ".parquet", ".gz", ".zip"}
                else zipfile.ZIP_DEFLATED
            )
            output.write(
                path,
                arcname=path.relative_to(ROOT).as_posix(),
                compress_type=compression,
            )
            if index % 250 == 0:
                print(f"[zip] {index}/{len(files)}", flush=True)
    digest = sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    status = {
        "completed_at": datetime.now().isoformat(),
        "status": "COMPLETED",
        "archive": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": digest,
        "n_files": len(files),
    }
    (RESULT / "V2_8_CANCER_NATIVE_MOE_RELEASE_SUCCESS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    validate()
    write_reports()
    if args.package:
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
