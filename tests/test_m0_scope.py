"""M0 scope fails closed for unknown cancers and includes both named cohorts."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest
spec=importlib.util.spec_from_file_location("m0_runner",Path(__file__).resolve().parents[1]/"scripts/run_m0_linear.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture_config(tmp_path):
    pd.DataFrame({"cancer_id":["BRCA","LUAD","LUSC"]}).to_csv(tmp_path/"folds.tsv",sep="\t",index=False)
    pd.DataFrame({"source_cohort":["BRCA","LUAD","LSCC"],"cancer_id":["BRCA","LUAD","LUSC"]}).to_csv(tmp_path/"PATIENTS.tsv",sep="\t",index=False)
    return {"sources":{"patient_map":{"path":str(tmp_path/"folds.tsv")}},
            "cptac_input":{"complete":{"path":str(tmp_path/"COMPLETE.json")}}}

def test_scope_selects_matching_cancer_in_each_cohort(tmp_path):
    cfg=fixture_config(tmp_path)
    t=module.tasks(cfg,["TCGA","CPTAC"],"list",["LUSC"])
    assert {(r["dataset"],r["cohort"]) for r in t}=={("TCGA","LUSC"),("CPTAC","LSCC")}
    assert len(module.tasks(cfg,["TCGA","CPTAC"],"all",[]))==6

def test_unknown_and_empty_list_are_rejected(tmp_path):
    cfg=fixture_config(tmp_path)
    with pytest.raises(ValueError,match="Unknown"):module.tasks(cfg,["TCGA"],"list",["typo"])
    with pytest.raises(ValueError,match="requires"):module.tasks(cfg,["TCGA"],"list",[])
    with pytest.raises(ValueError,match="Do not supply"):module.tasks(cfg,["TCGA"],"all",["BRCA"])
