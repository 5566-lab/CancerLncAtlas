"""Observed point-dose associations and matched-model adjacent-dose contrasts.

These analyses use measured responses, not a fitted curve or a strength-head
prediction. Concentration, exposure, endpoint and response unit remain separate.
Adjacent-dose contrasts use the same canonical model within the same dataset.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from . import drug_training as base
from .drug_identity import text
from .drug_pharmacology import concentration, _identifiers
from .drug_pharmacology_native import verify_artifact, native_protocols, _strings
from .drug_staging import artifact_sha256
from .drug_response_analysis import iter_response_scan, annotated_contexts, DrugResponseScanConfig

FORMAT = "NATIVE_DRUG_POINT_RESPONSE_V1"
METRICS = {"POINT_VIABILITY", "POINT_GROWTH", "POINT_LOG_FOLD_CHANGE", "POINT_INHIBITION"}
RESPONSE_FIELDS = ("response_value", "higher_is_sensitive", "response_metric", "response_unit")
CONDITION = ["response_metric", "response_unit", "exposure_hours_key", "dose_molar_key"]
IDENTITY = ["dataset_id", "cell_line_id", "drug_id", "assay_id"]
FACTORS = {"M":"1", "mM":"0.001", "uM":"0.000001", "µM":"0.000001", "μM":"0.000001",
           "nM":"0.000000001", "pM":"0.000000000001"}


def _decimal_key(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0: raise ValueError("Dose/exposure must be finite and nonnegative")
    return format(result.normalize(), "f")


def dose_key(value, unit):
    if pd.isna(value) and (pd.isna(unit) or unit in FACTORS): return None
    measured = concentration(value, unit, allow_zero=True)
    if pd.isna(measured): return None
    exact = Decimal(str(value))*Decimal(FACTORS[unit])
    if exact > 0 and measured == 0: raise ValueError("Positive dose is outside the supported numerical range")
    return _decimal_key(exact)


def normalize_point_response(frame, mapping):
    required = [*IDENTITY, "dose_value", "dose_unit", "exposure_hours", "source_id", *RESPONSE_FIELDS]
    if missing := set(required)-set(frame): raise ValueError("Point-response fields missing: "+str(sorted(missing)))
    base._assert_no_result_columns(frame,"raw response")
    data = frame[required].copy().reset_index(drop=True)
    _identifiers(data,[*IDENTITY,"source_id","response_metric","response_unit"])
    if not set(data.response_metric).issubset(METRICS):
        raise ValueError("Only observed point responses are eligible; fitted AUC/IC50 cannot become dose measurements")
    data["higher_is_sensitive"] = base._bool_series(data.higher_is_sensitive,"point response orientation")
    data["response_value"] = pd.to_numeric(data.response_value, errors="raise")
    data["dose_molar_key"] = [dose_key(v,u) for v,u in zip(data.dose_value,data.dose_unit)]
    data["exposure_hours_key"] = [None if pd.isna(v) else _decimal_key(v) for v in data.exposure_hours]
    data["condition_available"] = data.dose_molar_key.notna() & data.exposure_hours_key.notna()
    data["measurement_available"] = data.condition_available & np.isfinite(data.response_value)
    data["measurement_unavailable_reason"] = np.select(
        [data.dose_molar_key.isna(),data.exposure_hours_key.isna(),~np.isfinite(data.response_value)],
        ["MISSING_POINT_DOSE","MISSING_EXPOSURE_DURATION","MISSING_OR_NONFINITE_POINT_RESPONSE"],default=None)
    # One assay ID denotes one observed well/measurement, not an entire plate.
    check = [*CONDITION,"response_value","higher_is_sensitive"]
    if data.groupby(IDENTITY,dropna=False)[check].nunique(dropna=False).gt(1).any().any():
        raise ValueError("One native point assay ID has conflicting measurements or conditions")
    orientation = data.groupby(["dataset_id","drug_id","response_metric","response_unit"]).higher_is_sensitive.nunique()
    if orientation.gt(1).any(): raise ValueError("Point response orientation changes within a dataset/endpoint")
    authority = base.normalise_cell_line_map(mapping)
    data = data.merge(authority,on=["dataset_id","cell_line_id"],how="left",validate="many_to_one")
    if data.canonical_model_id.isna().any(): raise ValueError("Point responses lack a canonical model/cancer identity")
    disposition = data[[*IDENTITY,"source_id","condition_available","measurement_available","measurement_unavailable_reason"]].copy()
    # Technical duplicates across source files do not reweight an assay.
    data = data.drop_duplicates(IDENTITY).reset_index(drop=True)
    data["sensitivity_value"] = data.response_value*np.where(data.higher_is_sensitive,1.,-1.)
    return data, disposition


def load_native_point_response(manifest_path, input_paths, tables):
    """Bind the actual response-scan arguments to one native staging producer."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or not manifest.get("measurements"):
        raise ValueError("Point response manifest requires native observed measurement sources")
    sources = {"point_manifest":{"path":str(path),"sha256":artifact_sha256(path)}}
    def bound(role,record,root=path.parent):
        target=verify_artifact(record,root); sources[role]={"path":str(target),"sha256":record["sha256"]}
        return target
    staged_path=bound("staging_manifest",manifest["staging_manifest"])
    staging=json.loads(staged_path.read_text(encoding="utf-8"))
    if (staging.get("status")!="SUCCESS" or staging.get("canonical_drug_id_policy")!="CURATED_STABLE_ID_OR_SOURCE_SCOPED_ASSAY_ID"
            or staging.get("input_artifacts_unchanged") is not True):
        raise ValueError("Point responses require the actual successful stable-ID native staging")
    roles={"response":"raw_cell_line_drug_response","expression":"raw_cell_line_lncrna_expression",
           "mapping":"cell_line_map","targets":"drug_gene_target","subtypes":"model_subtypes"}
    for argument,role in roles.items():
        record=staging["artifacts"][role]
        if artifact_sha256(input_paths[argument])!=record["sha256"]:
            raise ValueError("Point scan arguments differ from actual staging: "+argument)
        bound("staged_"+role,record,staged_path.parent)
    exact=staging["input_artifact_snapshot"]["start"]["v32_exact_candidates"]
    if artifact_sha256(input_paths["exact"])!=exact["sha256"]:
        raise ValueError("Point scan exact candidate authority differs from actual staging")
    bound("exact_candidates",exact,staged_path.parent)
    source_map=_strings(bound("source_drug_map",staging["artifacts"]["source_drug_map"],staged_path.parent))
    observed=base.normalise_raw_response(tables["response"])
    parts=[]; dispositions=[]
    for number,spec in enumerate(manifest["measurements"]):
        if spec.get("measurement_level")!="OBSERVED_POINT_RESPONSE" or spec.get("condition_kind")!="POINT_CONCENTRATION":
            raise ValueError("Fitted curves and concentration ranges are not observed point responses")
        source=bound("measurement_"+str(number),spec["source"])
        raw=_strings(source); columns=spec.get("columns",{}); constants=spec.get("constants",{})
        if set(columns)&set(constants) or not set(columns.values()).issubset(raw):
            raise ValueError("Native point columns/constants conflict or are absent")
        if not set(RESPONSE_FIELDS).issubset(set(columns)|set(constants)):
            raise ValueError("Native point measurements require response value, unit, metric and orientation")
        value_column=base._normalise_name(columns.get("response_value",""))
        if value_column in {"auc","aoc","ic50","ln_ic50","log_ic50","pic50","ec50","gr50","fitted_response","predicted_response"}:
            raise ValueError("A fitted summary metric column cannot be relabelled as an observed point response")
        if "response_value" in constants:
            raise ValueError("Observed response values must come from a native measurement column")
        protocol={**spec,"columns":{k:v for k,v in columns.items() if k not in RESPONSE_FIELDS},
                  "constants":{k:v for k,v in constants.items() if k not in RESPONSE_FIELDS}}
        design,identity_audit=native_protocols(raw,protocol,source_map,observed,source_sha=artifact_sha256(source))
        identity_audit["source_number"]=number; dispositions.append(identity_audit)
        indices=identity_audit.loc[identity_audit.retained,"source_row"].to_numpy(int)
        for field in RESPONSE_FIELDS:
            design[field] = raw.iloc[indices][columns[field]].to_numpy() if field in columns else constants[field]
        parts.append(design)
    joined=pd.concat(parts,ignore_index=True)
    normalized,measurement_audit=normalize_point_response(joined,tables["mapping"])
    for record in sources.values(): verify_artifact(record,path.parent)
    return normalized, pd.concat(dispositions,ignore_index=True), measurement_audit, {
        "format":FORMAT,"sources":sources,"staging_run_id":staging["staging_run_id"],
        "measurement_rows":len(normalized),"available_point_rows":int(normalized.measurement_available.sum()),
        "identified_condition_rows":int(normalized.condition_available.sum()),
        "native_rows_without_staged_identity":int(sum((~d.retained).sum() for d in dispositions)),
        "fitted_response_metrics_used_as_points":False,"training_started":False}


def condition_inputs(points, mapping):
    """Yield point and paired-dose response tables plus their complete universes."""
    measured=points.loc[points.measurement_available]
    declared=points.loc[points.condition_available]
    # Every canonical model has one mean per dataset/condition in both analyses.
    keys=["dataset_id","canonical_model_id","cancer_id","drug_id",*CONDITION]
    aggregated=measured.groupby(keys,sort=False).sensitivity_value.mean().reset_index()
    representatives=base.normalise_cell_line_map(mapping).sort_values(["dataset_id","canonical_model_id","cell_line_id"])
    representatives=representatives.drop_duplicates(["dataset_id","canonical_model_id"])[["dataset_id","canonical_model_id","cell_line_id"]]
    by_condition=aggregated.groupby(CONDITION,sort=False)
    # Retain conditions with no finite measurements for explicit candidate nulls.
    for values,registry in declared.groupby(CONDITION,sort=True):
        condition=dict(zip(CONDITION,values))
        local=by_condition.get_group(values) if values in by_condition.indices else aggregated.iloc[:0]
        local=local.merge(representatives,on=["dataset_id","canonical_model_id"],how="left",validate="many_to_one")
        response=local[["dataset_id","cell_line_id","drug_id","sensitivity_value"]].rename(columns={"sensitivity_value":"response_value"})
        response["higher_is_sensitive"]=True
        metadata={"response_metric":condition["response_metric"],"response_unit":condition["response_unit"],
            "exposure_hours":float(condition["exposure_hours_key"]),"dose_molar":float(condition["dose_molar_key"]),
            "exposure_hours_key":condition["exposure_hours_key"],"dose_molar_key":condition["dose_molar_key"],
            "sensitivity_estimand":"OBSERVED_POINT_SENSITIVITY"}
        yield "dose_associations",metadata,response,registry[["cancer_id","drug_id"]].drop_duplicates()
    # Average technical assays before pairing. Different datasets never supply
    # the two ends of one contrast, even when they share a canonical model.
    groups=["cancer_id","drug_id","response_metric","response_unit","exposure_hours_key"]
    by_group=aggregated.groupby(groups,sort=False)
    bundles={}
    for values,registry in declared.groupby(groups,sort=True):
        cancer,drug,metric,unit,exposure=values
        doses=sorted(registry.dose_molar_key.unique(),key=Decimal)
        local=by_group.get_group(values) if values in by_group.indices else aggregated.iloc[:0]
        for low,high in zip(doses,doses[1:]):
            pair_keys=["dataset_id","canonical_model_id"]
            left=local.loc[local.dose_molar_key.eq(low),[*pair_keys,"sensitivity_value"]]
            right=local.loc[local.dose_molar_key.eq(high),[*pair_keys,"sensitivity_value"]]
            paired=left.merge(right,on=pair_keys,how="inner",suffixes=("_low","_high"),validate="one_to_one")
            paired=paired.merge(representatives,on=pair_keys,how="left",validate="many_to_one")
            paired["drug_id"]=drug; paired["response_value"]=paired.sensitivity_value_high-paired.sensitivity_value_low
            paired["higher_is_sensitive"]=True
            key=(metric,unit,exposure,low,high)
            entry=bundles.setdefault(key,{"responses":[],"candidates":[]})
            entry["responses"].append(paired[["dataset_id","cell_line_id","drug_id","response_value","higher_is_sensitive"]])
            entry["candidates"].append({"cancer_id":cancer,"drug_id":drug})
    for (metric,unit,exposure,low,high),bundle in sorted(bundles.items()):
        metadata={"response_metric":metric,"response_unit":unit,"exposure_hours":float(exposure),
            "exposure_hours_key":exposure,
            "low_dose_molar":float(low),"high_dose_molar":float(high),"low_dose_molar_key":low,"high_dose_molar_key":high,
            "sensitivity_estimand":"WITHIN_DATASET_MODEL_HIGH_MINUS_LOW_SENSITIVITY",
            "dose_interval_selection":"ADJACENT_DECLARED_DOSES_WITHIN_CANCER_DRUG_ENDPOINT_EXPOSURE"}
        yield "dose_contrasts",metadata,pd.concat(bundle["responses"],ignore_index=True),pd.DataFrame(bundle["candidates"])


def iter_dose_response_scan(points, exact, expression, mapping, subtypes, targets, *, config=None):
    settings=config or DrugResponseScanConfig()
    mapping=base.normalise_cell_line_map(mapping)
    contexts={(c,s,t):models for c,s,t,models,_ in annotated_contexts(mapping,subtypes)}
    for kind,metadata,response,candidates in condition_inputs(points,mapping):
        linked=response.merge(mapping,on=["dataset_id","cell_line_id"],how="left",validate="many_to_one")
        # These descriptive summaries retain observed constant/zero changes even
        # when a rank correlation is undefined. Each cell model contributes once.
        summaries={}
        for receipt,frame in iter_response_scan(exact,response,expression,mapping,subtypes,targets,
                                                config=settings,candidate_drugs=candidates):
            identity={**metadata,"cancer_id":receipt["cancer_id"],"subtype_system":receipt["subtype_system"],
                      "subtype_id":receipt["subtype_id"]}
            context=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            receipt={**receipt,**metadata,"context_id":context,"analysis_kind":kind,
                     "response_rows_in_condition":len(response),"paired_model_contrast":kind=="dose_contrasts"}
            if not frame.empty:
                context_key=receipt["cancer_id"],receipt["subtype_system"],receipt["subtype_id"]
                if context_key not in summaries:
                    selected=linked.loc[linked.canonical_model_id.isin(contexts.get(context_key,set()))]
                    per_model=selected.groupby(["drug_id","canonical_model_id"]).response_value.mean().reset_index()
                    summary=per_model.groupby("drug_id").response_value.agg(
                        measured_unique_cell_models="size",mean_observed_oriented_response="mean",
                        median_observed_oriented_response="median",sd_observed_oriented_response="std").reset_index()
                    counts=selected.groupby("drug_id").dataset_id.nunique().rename("measured_contributing_datasets")
                    variable=selected.groupby(["drug_id","dataset_id"]).response_value.nunique().gt(1)
                    variable=variable.groupby("drug_id").any().rename("within_dataset_response_variation")
                    summaries[context_key]=summary.merge(counts,on="drug_id").merge(variable,on="drug_id")
                frame=frame.merge(summaries[context_key],on="drug_id",how="left",validate="many_to_one")
                for column in ("measured_unique_cell_models","measured_contributing_datasets"):
                    frame[column]=frame[column].fillna(0).astype(int)
                frame["measured_response_available"]=frame.measured_unique_cell_models.gt(0)
                frame["within_dataset_response_variation"]=frame.within_dataset_response_variation.eq(True)
                for key,value in metadata.items(): frame[key]=value
                frame["dose_context_id"]=context
                frame["fitted_dose_response_curve"]=False
                frame["dose_interaction_causal_claimed"]=False
                if kind=="dose_contrasts":
                    frame["observed_direction"]=frame.observed_direction.replace({
                        "HIGHER_EXPRESSION_HIGHER_SENSITIVITY":"HIGHER_EXPRESSION_LARGER_SENSITIVITY_INCREASE",
                        "HIGHER_EXPRESSION_LOWER_SENSITIVITY":"HIGHER_EXPRESSION_SMALLER_SENSITIVITY_INCREASE"})
            yield kind,receipt,frame


def materialize_dose_response(output, points, identity_audit, measurement_audit, tables, config):
    """Write bounded candidate chunks; retain raw identity dispositions privately."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    output=Path(output); coverage=[]; counts={"dose_associations":0,"dose_contrasts":0}
    for kind in counts: (output/kind).mkdir(exist_ok=False)
    active=None; writer=None
    try:
        for kind,receipt,frame in iter_dose_response_scan(points,**{k:tables[k] for k in
                ("exact","expression","mapping","subtypes","targets")},config=config):
            key=kind,receipt["context_id"]
            if active!=key:
                if writer is not None: writer.close(); writer=None
                coverage.append(receipt); active=key
            if frame.empty: continue
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(output/kind/(key[1]+".parquet"),table.schema,compression="zstd")
            writer.write_table(table.cast(writer.schema)); counts[kind]+=len(frame)
    finally:
        if writer is not None: writer.close()
    coverage_frame=pd.DataFrame(coverage) if coverage else pd.DataFrame(columns=["context_id","analysis_kind",
        "cancer_id","subtype_system","subtype_id","candidate_rows","context_available","response_metric","response_unit"])
    coverage_frame.to_parquet(output/"DOSE_CONTEXT_COVERAGE.parquet",index=False)
    identity_audit.to_parquet(output/"PRIVATE_DOSE_IDENTITY_DISPOSITION.parquet",index=False)
    measurement_audit.to_parquet(output/"PRIVATE_DOSE_MEASUREMENT_DISPOSITION.parquet",index=False)
    return {"rows":counts,"contexts":len(coverage),"point_response_prediction":False,
            "curve_fitting_performed":False,"paired_dose_contrast":True,"p_values_computed":False,
            "independent_validation_claimed":False,"causal_drug_effect_claimed":False,
            "pairing_unit":"DATASET_AND_CANONICAL_CELL_MODEL_AT_IDENTICAL_EXPOSURE_AND_ENDPOINT",
            "descriptive_response_unit":"ONE_CANONICAL_MODEL_MEAN_ACROSS_DATASETS_AFTER_TECHNICAL_ASSAY_AGGREGATION",
            "observed_constant_or_zero_response_preserved":True}
