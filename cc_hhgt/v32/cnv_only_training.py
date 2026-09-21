"""Independent CNV-only V3.2 private head runner.

This module deliberately does not import or read mutation call tables.  It uses
the already materialized, hash-bound full33 compact CNV store and a frozen core
embedding export, then writes fresh five-fold typed OOF predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .genomic_training import (
    ANALYSIS_VERSION,
    N_FOLDS,
    PRIVATE_CHECKPOINT_FORMAT,
    _DOMAIN_FEATURES,
    _candidate_core,
    _balanced_indices,
    _file_sha256,
    _fit_head,
    _predict_head,
    _read_table,
    _validate_core_manifest,
    GenomicTrainingConfig,
    normalise_candidates,
)


TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".partial")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class CompactCNV:
    """Memory-mapped compact CNV matrices; no mutation feature path exists."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.patient_ids = tuple(json.loads((root / "patient_ids.json").read_text(encoding="utf-8")))
        self.lncrna_ids = tuple(json.loads((root / "lncrna_ids.json").read_text(encoding="utf-8")))
        self.pathway_ids = tuple(json.loads((root / "pathway_ids.json").read_text(encoding="utf-8")))
        self.patient_index = {v: i for i, v in enumerate(self.patient_ids)}
        self.lncrna_index = {v: i for i, v in enumerate(self.lncrna_ids)}
        self.pathway_index = {v: i for i, v in enumerate(self.pathway_ids)}
        self.le = np.load(root / "lncrna_event.npy", mmap_mode="r", allow_pickle=False)
        self.lc = np.load(root / "lncrna_callable.npy", mmap_mode="r", allow_pickle=False)
        self.lb = np.load(root / "lncrna_burden.npy", mmap_mode="r", allow_pickle=False)
        self.pe = np.load(root / "pathway_event.npy", mmap_mode="r", allow_pickle=False)
        self.pc = np.load(root / "pathway_callable.npy", mmap_mode="r", allow_pickle=False)
        self.pb = np.load(root / "pathway_burden.npy", mmap_mode="r", allow_pickle=False)

    def candidate_statistics_arrays(self, candidates: pd.DataFrame, patients: Sequence[str], min_pair_callable: int):
        size = len(candidates)
        domain = np.zeros((size, 6), dtype=np.float32)
        labels = np.full(size, np.nan, dtype=np.float32)
        available = np.zeros(size, dtype=bool)
        reasons = np.full(size, "CNV_NO_EXPLICIT_PAIR_CALLABILITY", dtype=object)
        rows = np.asarray([self.patient_index[p] for p in patients if p in self.patient_index], dtype=int)
        if not len(rows):
            reasons[:] = "CNV_NOT_AVAILABLE_FOR_CANCER"
            return domain, labels, available, reasons
        li_all = np.asarray([self.lncrna_index.get(str(v), -1) for v in candidates.lncrna_id], dtype=int)
        pi_all = np.asarray([self.pathway_index.get(str(v), -1) for v in candidates.pathway_id], dtype=int)
        for start in range(0, size, 512):
            stop = min(size, start + 512)
            li, pi = li_all[start:stop], pi_all[start:stop]
            valid = (li >= 0) & (pi >= 0)
            local = np.flatnonzero(valid)
            if not len(local):
                reasons[start:stop][li < 0] = "CNV_LNCRNA_UNAVAILABLE"
                reasons[start:stop][pi < 0] = "CNV_PATHWAY_UNAVAILABLE"
                continue
            lx = self.le[np.ix_(rows, li[valid])].astype(float)
            py = self.pe[np.ix_(rows, pi[valid])].astype(float)
            lb = self.lb[np.ix_(rows, li[valid])].astype(float)
            pb = self.pb[np.ix_(rows, pi[valid])].astype(float)
            pair = self.lc[np.ix_(rows, li[valid])].astype(bool) & self.pc[np.ix_(rows, pi[valid])].astype(bool)
            count = pair.sum(axis=0).astype(float)
            safe = np.maximum(count, 1.0)
            x, y = np.where(pair, lx, 0.0), np.where(pair, py, 0.0)
            xs, ys, xys = x.sum(0), y.sum(0), (x * y).sum(0)
            n11, n10, n01 = xys, xs - xys, ys - xys
            n00 = count - n11 - n10 - n01
            den = np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
            phi = np.divide(n11*n00-n10*n01, den, out=np.full_like(count, np.nan), where=den > 0)
            idx = start + local
            domain[idx] = np.column_stack([np.log1p(count), count / len(rows), xs/safe, ys/safe,
                                            np.where(pair, np.abs(lb), 0).sum(0)/safe,
                                            np.where(pair, np.abs(pb), 0).sum(0)/safe]).astype(np.float32)
            eligible = (count >= int(min_pair_callable)) & (xs > 0) & (xs < count) & (ys > 0) & (ys < count) & np.isfinite(phi)
            labels[idx[eligible]] = (phi[eligible] > 0).astype(np.float32)
            available[idx[eligible]] = True
            reasons[idx[count < int(min_pair_callable)]] = "CNV_INSUFFICIENT_PAIR_CALLABILITY"
            reasons[idx[(count >= int(min_pair_callable)) & ~eligible]] = "CNV_NO_EXPLICIT_WT_EVENT_VARIATION"
            reasons[idx[eligible]] = ""
        return domain, labels, available, reasons


def _canonical_patient(value: Any) -> str:
    text = str(value).strip().replace(".", "-").upper()
    return text[:12] if text.startswith("TCGA-") else text


def run_cnv_only_oof(*, candidates_path: str | Path, folds_path: str | Path,
                     streaming_root: str | Path, core_manifest_path: str | Path,
                     output_root: str | Path, training_run_id: str,
                     config: GenomicTrainingConfig | None = None) -> dict[str, Any]:
    """Train fresh CNV heads only, or emit a typed BLOCKED receipt."""
    settings = config or GenomicTrainingConfig()
    settings.validate()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"CNV-only output reuse refused: {output}")
    output.mkdir(parents=True, exist_ok=True)
    folds_raw = _read_table(folds_path)
    patient_col = "patient_id" if "patient_id" in folds_raw.columns else "sample_id"
    folds = folds_raw[["cancer_id", patient_col, "patient_fold_id"]].copy()
    folds.columns = ["cancer_id", "patient_id", "patient_fold_id"]
    folds.cancer_id = folds.cancer_id.astype(str).str.upper()
    folds.patient_id = folds.patient_id.map(_canonical_patient)
    folds.patient_fold_id = pd.to_numeric(folds.patient_fold_id, errors="raise").astype(int)
    folds = folds.drop_duplicates(["cancer_id", "patient_id"])
    input_meta = {
        "format": "CC_HHGT_V3_2_CNV_ONLY_INPUT_PRECHECK_V1",
        "status": "PASS",
        "candidates_path": str(Path(candidates_path).resolve()),
        "candidates_sha256": _file_sha256(candidates_path),
        "folds_path": str(Path(folds_path).resolve()),
        "folds_sha256": _file_sha256(folds_path),
        "streaming_root": str(Path(streaming_root).resolve()),
        "core_manifest_path": str(Path(core_manifest_path).resolve()),
        "mutation_features_used": False,
        "mutation_tables_read_for_training": False,
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
    }
    _atomic_json(output / "INPUT_PRECHECK.json", input_meta)

    # Validate existence before attempting any fold.  Missing exported core
    # embeddings are a typed, non-success state; no fabricated OOF is emitted.
    missing_core: list[str] = []
    try:
        manifest, manifest_sha, core_composite = _validate_core_manifest(core_manifest_path)
        for fold in range(N_FOLDS):
            exports = manifest["folds"][str(fold)]["exports"]
            for node_type in ("lncRNA", "pathway"):
                declared = Path(str(exports[node_type]["path"]))
                resolved = None
                for anc in (Path(core_manifest_path).resolve().parent, *Path(core_manifest_path).resolve().parents):
                    p = (anc / declared)
                    if p.is_file(): resolved = p; break
                if resolved is None: missing_core.append(str(declared))
    except Exception as exc:
        missing_core.append(f"CORE_MANIFEST_VALIDATION:{type(exc).__name__}:{exc}")
    stream_success = Path(streaming_root).resolve() / "SUCCESS.json"
    if not stream_success.is_file():
        missing_core.append(f"STREAMING_SUCCESS_MISSING:{stream_success}")
    if missing_core:
        blocked = {
            "format": "CC_HHGT_V3_2_CNV_HEAD_OOF_V1",
            "status": "TYPED_BLOCKED_MISSING_FORMAL_CORE_INPUT",
            "training_run_id": str(training_run_id),
            "reason": "IRREPLACEABLE_FORMAL_CORE_EMBEDDING_EXPORT_MISSING",
            "missing": sorted(set(missing_core)),
            "five_fold_oof_emitted": False,
            "cnv_only": True,
            "mutation_features_used": False,
            "old_checkpoint_loaded": False,
            "old_predictions_used": False,
        }
        _atomic_json(output / "TYPED_BLOCKED.json", blocked)
        return blocked

    # Only after the irreplaceable-core gate passes do we materialize the
    # 3.3-million-row candidate authority.
    raw_candidates = pd.read_parquet(candidates_path, columns=["cancer_id", "lncrna_id", "pathway_id"])
    raw_candidates["cancer_id"] = raw_candidates["cancer_id"].astype(str).str.upper()
    # The FORMAL_CANDIDATE_UNIVERSE authority is already deduplicated and
    # canonically cancer-partitioned; avoid a second 3.3M-row global sort.
    if raw_candidates.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any():
        raise RuntimeError("Candidate authority contains duplicate exact keys")
    candidates = raw_candidates

    # This path is reached only when all frozen exports are present.  Load the
    # compact arrays directly; the runner has no mutation-table arguments.
    success_payload = json.loads(stream_success.read_text(encoding="utf-8"))
    compacts = {c: CompactCNV(Path(v["path"]).resolve()) for c, v in success_payload["cancers"].items()}
    core_manifest, core_manifest_sha, core_composite = _validate_core_manifest(core_manifest_path)
    checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir()
    pred_root = output / "predictions"; pred_root.mkdir()
    checkpoint_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    core_hashes_before: dict[str, str] = {}
    for fold in range(N_FOLDS):
        from .genomic_training import load_fold_core_embeddings, _candidate_core, _predict_head
        core = load_fold_core_embeddings(core_manifest_path, core_manifest, fold)
        core_hashes_before.update(core.input_hashes)
        split_patients = {s: {} for s in ("train", "validation", "test")}
        val_fold = (fold + 1) % N_FOLDS
        for cancer, group in folds.groupby("cancer_id", sort=False):
            split_patients["test"][cancer] = group.loc[group.patient_fold_id.eq(fold), "patient_id"].tolist()
            split_patients["validation"][cancer] = group.loc[group.patient_fold_id.eq(val_fold), "patient_id"].tolist()
            split_patients["train"][cancer] = group.loc[~group.patient_fold_id.isin([fold, val_fold]), "patient_id"].tolist()
        train_parts=[]; val_parts=[]
        for cancer in TCGA_CANCERS:
            local = candidates.loc[candidates.cancer_id.eq(cancer)]
            if local.empty: continue
            for split, parts, maximum, seed in (("train", train_parts, settings.max_train_rows, settings.seed + fold), ("validation", val_parts, settings.max_validation_rows, settings.seed + 50_000 + fold)):
                stats = compacts[cancer].candidate_statistics_arrays(local, split_patients[split].get(cancer, ()), settings.min_pair_callable)
                ok = np.flatnonzero(stats[2])
                if len(ok):
                    rng=np.random.default_rng(seed + TCGA_CANCERS.index(cancer)); take=rng.choice(ok,size=min(len(ok),max(2, int(np.ceil(maximum/33)))),replace=False)
                    cv, ca = _candidate_core(local.iloc[take], core); good=np.flatnonzero(ca)
                    if len(good): parts.append((cv[good], stats[0][take][good], stats[1][take][good]))
        if not train_parts or not val_parts: raise RuntimeError(f"CNV fold {fold} has no train/validation examples")
        trc=np.concatenate([x[0] for x in train_parts]); trd=np.concatenate([x[1] for x in train_parts]); try_=np.concatenate([x[2] for x in train_parts])
        vac=np.concatenate([x[0] for x in val_parts]); vad=np.concatenate([x[1] for x in val_parts]); vay=np.concatenate([x[2] for x in val_parts])
        train_choice = _balanced_indices(try_, len(try_), settings.seed + fold + 100_000)
        val_choice = _balanced_indices(vay, len(vay), settings.seed + 50_000 + fold + 100_000)
        if not len(train_choice) or not len(val_choice):
            raise RuntimeError(f"CNV fold {fold} lacks two explicit classes after balancing")
        trc,trd,try_ = trc[train_choice], trd[train_choice], try_[train_choice]
        vac,vad,vay = vac[val_choice], vad[val_choice], vay[val_choice]
        # Keep collector semantics and class balance exactly in the shared head.
        head, metadata, mean, scale, history = _fit_head(trc,trd,try_,vac,vad,vay,modality="cnv",fold=fold,config=settings)
        import torch
        ckpt = checkpoint_root / f"cnv_patient_fold_{fold}.pt"
        torch.save({"checkpoint_format": PRIVATE_CHECKPOINT_FORMAT, "analysis_version": ANALYSIS_VERSION, "modality":"cnv", "patient_fold":fold, "initialization":metadata, "model_state":head.state_dict(), "domain_features":list(_DOMAIN_FEATURES), "domain_mean":mean, "domain_scale":scale, "history":history, "core_checkpoint_sha256":core.checkpoint_sha256, "core_parameter_sha256":core.core_parameter_sha256, "old_checkpoint_loaded":False, "old_predictions_used":False}, ckpt)
        checkpoint_records.append({"fold":fold,"path":str(ckpt),"sha256":_file_sha256(ckpt),"train_rows":len(try_),"validation_rows":len(vay),"core_parameter_sha256":core.core_parameter_sha256})
        for cancer in TCGA_CANCERS:
            local=candidates.loc[candidates.cancer_id.eq(cancer)].reset_index(drop=True)
            stats=compacts[cancer].candidate_statistics_arrays(local, split_patients["test"].get(cancer, ()), settings.min_pair_callable)
            cv, ca=_candidate_core(local,core); mask=stats[2] & ca
            prob=np.full(len(local),np.nan,np.float32)
            if mask.any(): prob[mask]=_predict_head(head,cv[mask],stats[0][mask],mean,scale,settings.prediction_batch_size)
            observed_y = stats[1][mask]
            observed_p = prob[mask]
            if len(observed_y):
                clipped=np.clip(observed_p,1e-7,1-1e-7)
                brier=float(np.mean((observed_p-observed_y)**2)); logloss=float(-np.mean(observed_y*np.log(clipped)+(1-observed_y)*np.log(1-clipped)))
                ece=0.0
                for lo in np.linspace(0,1,11)[:-1]:
                    hi=lo+0.1; sel=(observed_p>=lo)&((observed_p<hi) if hi<1 else (observed_p<=hi))
                    if sel.any(): ece += float(sel.mean())*abs(float(observed_p[sel].mean())-float(observed_y[sel].mean()))
                metric_records.append({"patient_fold":fold,"cancer_id":cancer,"n":len(observed_y),"positive_rate":float(observed_y.mean()),"brier":brier,"logloss":logloss,"ece_10bin":ece,"available":int(mask.sum())})
            out=pd.DataFrame({"cancer_id":cancer,"lncrna_id":local.lncrna_id,"pathway_id":local.pathway_id,"cnv_probability":prob,"cnv_available":mask,"cnv_unavailable_reason":np.where(mask,"",stats[3].astype(str)),"patient_fold":fold,"model_version":"V3.2","training_run_id":training_run_id,"checkpoint_sha256":_file_sha256(ckpt)})
            path=pred_root/f"patient_fold={fold}"/f"cancer={cancer}.parquet"; path.parent.mkdir(parents=True,exist_ok=True); out.to_parquet(path,index=False,compression="zstd")
            prediction_records.append({"fold":fold,"cancer":cancer,"path":str(path),"rows":len(out),"available":int(mask.sum()),"unavailable":int((~mask).sum())})
        del head, trc,trd,try_,vac,vad,vay
    core_hashes_after = {p: _file_sha256(p) for p in core_hashes_before}
    if core_hashes_after != core_hashes_before or _file_sha256(core_manifest_path) != manifest_sha:
        raise RuntimeError("Frozen core embedding or manifest changed during CNV-only OOF")
    _atomic_json(output / "METRICS.json", {"format":"CC_HHGT_V3_2_CNV_HEAD_OOF_METRICS_V1","records":metric_records,"global":{"n":int(sum(x["n"] for x in metric_records)),"brier":float(np.average([x["brier"] for x in metric_records],weights=[x["n"] for x in metric_records])) if metric_records else None,"logloss":float(np.average([x["logloss"] for x in metric_records],weights=[x["n"] for x in metric_records])) if metric_records else None}})
    _atomic_json(output / "OOF_MANIFEST.json", {"format":"CC_HHGT_V3_2_CNV_HEAD_OOF_V1","status":"SUCCESS","cnv_only":True,"mutation_features_used":False,"checkpoint_count":5,"checkpoints":checkpoint_records,"predictions":prediction_records,"core_manifest_sha256":manifest_sha,"core_parameter_composite_sha256":core_composite,"core_parameters_before_sha256":core_hashes_before,"core_parameters_after_sha256":core_hashes_after,"old_checkpoint_loaded":False,"old_predictions_used":False,"metrics_path":str(output/"METRICS.json")})
    return {"status":"SUCCESS","checkpoint_count":5,"prediction_files":len(prediction_records)}
