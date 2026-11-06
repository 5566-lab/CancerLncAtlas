"""Validate saved real final models without changing any training artifact.

The audit covers bounded, outcome-independent target selections for the
base/module estimators supported by network_ridge_predict. It checks frozen
preprocessing, independent kernel predictions/refits, query permutation and
unsupported clinical rows. This is an artifact audit, not new validation data.
"""
from __future__ import annotations
import argparse, json, socket, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cla_engine.common import sha
from cc_hhgt.network_ridge import make_context, base_kernel, group_kernels
from cc_hhgt.network_ridge_predict import predict, coefficients
from cc_hhgt.graph_ridge import target_policy
from cc_hhgt.m0_covariates import complete_rows

VARIANTS = ("rna_only", "omics_only", "ridge_equal", "ridge_adaptive_omics",
            "module_ridge_ppi", "module_ridge_rpi", "module_ridge_union",
            "module_ridge_union_random_101")
ARMS = ("lncrna", "mrna", "combined")
BLOCKS = ("RNA", "CNV", "MUTATION", "PROTEIN", "PHOSPHO")

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")

def checked_receipt(folder, names=None):
    receipt = json.loads((folder/"COMPLETE.json").read_text())
    files = receipt["files"]
    selected = list(files) if names is None else names
    for name in selected:
        if name not in files or sha(folder/name) != files[name]:
            raise ValueError("ARTIFACT_CHECKSUM_DRIFT: "+str(folder/name))
    return sha(folder/"COMPLETE.json")

def select_targets(meta, audit):
    fitted = audit.loc[audit.status.eq("FITTED")]
    available = set(fitted.target_id)
    selected = []
    for layer, part in meta[meta.target_id.isin(available)].groupby("layer", sort=True):
        policies = part.target_id.map(lambda tid: target_policy(
            part.loc[part.target_id.eq(tid)].iloc[0].to_dict())["use_purity"])
        for purity in sorted(policies.unique()):
            candidates = part.loc[policies.eq(purity)].sort_values("target_id")
            for tid in candidates.target_id:
                a = fitted[fitted.target_id.eq(tid)]
                required = {(v, arm) for v in VARIANTS
                            for arm in (("shared",) if v == "omics_only" else ARMS)}
                if required.issubset(set(zip(a.variant, a.arm))):
                    selected.append((tid, str(layer), bool(purity)))
                    break
    if not selected:
        raise ValueError("NO_COMPLETE_TARGET_FOR_ARTIFACT_AUDIT")
    return selected

def audit_cancer(inputs, networks, result, output, cancer):
    if socket.gethostname() != "149":
        raise RuntimeError("HOST_149_REQUIRED")
    src, net, final = Path(inputs)/cancer, Path(networks)/cancer, Path(result)/cancer/"final"
    out = Path(output)/cancer
    out.mkdir(parents=True, exist_ok=False)
    dump(out/"PREFLIGHT.json", dict(target_host="149", cancer=cancer,
        input=str(src), networks=str(net), final=str(final), producer_sha256=sha(Path(__file__)),
        selection="First lexicographic fully fitted target per layer and purity policy; no performance selection",
        scope="Base and module estimators; legacy Laplacian prediction helper is excluded"))
    bound = dict(input=checked_receipt(src), networks=checked_receipt(net),
                 final=checked_receipt(final, ["MODELS.joblib", "AUDIT.parquet",
                       "TARGETS.tsv", "PREPROCESS_purity1.joblib", "PREPROCESS_purity0.joblib"]))
    dfs = {b: pd.read_parquet(src/(b+".parquet")) for b in BLOCKS}
    pids = dfs["RNA"].index
    assert all(f.index.equals(pids) for f in dfs.values())
    blocks = {b: f.to_numpy(float) for b, f in dfs.items()}
    del dfs
    cov = pd.read_parquet(src/"COVARIATES.parquet").reindex(pids)
    y = pd.read_parquet(src/"Y.parquet").reindex(pids)
    ft = pd.read_csv(src/"RNA_FEATURES.tsv", sep="\t")
    fmap = pd.read_parquet(net/"FEATURE_MAP.parquet")
    meta = pd.read_csv(final/"TARGETS.tsv", sep="\t")
    audit = pd.read_parquet(final/"AUDIT.parquet")
    selection = select_targets(meta, audit)
    dump(out/"SELECTION.json", selection)
    models = joblib.load(final/"MODELS.joblib")
    rows = []
    for purity in sorted({v[2] for v in selection}):
        frozen = joblib.load(final/("PREPROCESS_purity"+str(int(purity))+".joblib"))
        ctx = make_context(blocks, ft, fmap, cov, np.arange(len(pids)), purity)
        assert ctx["design_spec"] == frozen["design_spec"]
        np.testing.assert_array_equal(ctx["train"], frozen["train"])
        np.testing.assert_array_equal(ctx["supported"], frozen["supported"])
        np.testing.assert_allclose(ctx["design"], frozen["design"], equal_nan=True, rtol=0, atol=0)
        for b in BLOCKS:
            assert set(ctx["preprocess"][b]) == set(frozen["preprocess"][b])
            for key, value in ctx["preprocess"][b].items():
                np.testing.assert_equal(value, frozen["preprocess"][b][key])
        for key, value in ctx["scales"].items():
            np.testing.assert_allclose(value, frozen["scales"][key], rtol=0, atol=1e-10)
        for tid, layer, use_purity in selection:
            if use_purity != purity:
                continue
            policy = target_policy(meta.loc[meta.target_id.eq(tid)].iloc[0].to_dict())
            expected_train = np.flatnonzero(np.isfinite(y[tid].to_numpy(float)) & complete_rows(cov, purity))
            for variant in VARIANTS:
                for arm in (("shared",) if variant == "omics_only" else ARMS):
                    model = models[(tid, variant, arm)]
                    assert tuple(model["allowed_blocks"]) == policy["allowed_blocks"]
                    np.testing.assert_array_equal(model["train"], expected_train)
                    train = np.asarray(model["train"], int)
                    if model["kind"] == "module":
                        kg, mass, labels = group_kernels(ctx, arm, model["allowed_blocks"], model["mode"])
                        assert labels == model["group_labels"]
                        k = np.einsum("g,gij->ij", model["group_weights"], kg)
                    else:
                        k = base_kernel(ctx, arm, model["allowed_blocks"], variant)
                    expected = ctx["design"] @ model["nuisance"] + k[:, train] @ model["dual"]
                    got, supported = predict(model, frozen, blocks, blocks, cov, fmap, ft)
                    np.testing.assert_array_equal(supported, ctx["supported"])
                    assert supported.any() and np.isfinite(got[supported]).all() and np.isnan(got[~supported]).all()
                    np.testing.assert_allclose(got[supported], expected[supported], rtol=2e-7, atol=2e-7)
                    # Independent linear solve of residualized normal equations.
                    zz = ctx["design"][train]
                    zi = np.linalg.pinv(zz)
                    m = np.eye(len(train))-zz@zi
                    kt = k[np.ix_(train, train)]
                    yy = y[tid].to_numpy(float)[train]
                    residual = m@kt@m
                    residual = (residual+residual.T)/2
                    dual = np.linalg.solve(residual+len(train)*model["lam"]*np.eye(len(train)), m@yy)
                    nuisance = zi@(yy-kt@dual)
                    independent = ctx["design"]@nuisance + k[:, train]@dual
                    np.testing.assert_allclose(got[supported], independent[supported], rtol=2e-6, atol=2e-6)
                    order = np.flatnonzero(supported)[:3][::-1]
                    query = {b: raw[order] for b, raw in blocks.items()}
                    permuted, ok = predict(model, frozen, blocks, query, cov.iloc[order], fmap, ft)
                    assert ok.all()
                    np.testing.assert_allclose(permuted, got[order], rtol=2e-7, atol=2e-7)
                    bad = cov.iloc[order].copy()
                    bad.iloc[0, bad.columns.get_loc("age_years")] = np.nan
                    rejected, ok = predict(model, frozen, blocks, query, bad, fmap, ft)
                    assert not ok[0] and np.isnan(rejected[0])
                    coef = coefficients(model, frozen, blocks, fmap, ft)
                    used = set(coef)
                    expect_blocks = set(() if variant == "omics_only" else ("RNA",))
                    if variant != "rna_only":
                        expect_blocks.update(policy["allowed_blocks"])
                    assert used == expect_blocks
                    assert all(np.isfinite(v["standardized_coefficients"]).all() for v in coef.values())
                    rows.append(dict(cancer=cancer, target_id=tid, layer=layer, purity=purity,
                        variant=variant, arm=arm, n_train=len(train), n_supported=int(supported.sum()),
                        coefficients=sum(len(v["standardized_coefficients"]) for v in coef.values()),
                        max_abs_kernel_difference=float(np.max(np.abs(got[supported]-expected[supported]))),
                        max_abs_independent_refit_difference=float(np.max(np.abs(got[supported]-independent[supported]))),
                        reordered_query_pass=True, missing_clinical_rejected=True))
            print("TARGET_VERIFIED", cancer, layer, tid, flush=True)
        del ctx
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"CHECKS.tsv", sep="\t", index=False)
    dump(out/"COMPLETE.json", dict(target_host="149", status="PASS", bounded_artifact_check=True,
        new_external_validation=False, models_checked=len(frame), targets_checked=len(selection),
        bound_receipts=bound, files={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print("FINAL_MODEL_ARTIFACT_PASS", cancer, len(frame), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("inputs", "networks", "result", "output", "cancer"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args()
    audit_cancer(**vars(args))
