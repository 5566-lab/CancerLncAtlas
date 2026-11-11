"""Bind the fixed biological supervision scope to existing binary preparation.

This adapter changes admitted identities and routing only. It does not convert
positive-unlabelled functional evidence into binary labels.
"""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
from pathlib import Path
from .target_policy import load_target_policy
from .kegg381_backbone import load_backbone
from .common import atomic_json, sha, fail

FORMAT = "CLA_PREPARED_RELATION_TARGET_POLICY_V1"
FOLDER = "relation_target_policy"
RECEIPT = "RELATION_TARGET_POLICY.json"
BINDING_FIELDS = ("path","sha256","backbone_path","backbone_manifest_sha256",
                  "source_identity_map","source_identity_map_sha256")


def _verified(path, expected, role):
    path = Path(path)
    if not expected or not path.is_file() or sha(path) != expected:
        fail("RELATION_POLICY_CHECKSUM_MISMATCH",role,path=str(path))
    return path


def _map_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:
        reader = csv.DictReader(handle,delimiter="\t")
        if not {"source_target_id","canonical_id"} <= set(reader.fieldnames or []):
            fail("RELATION_POLICY_IDENTITY_FIELDS","source_target_id/canonical_id required")
        rows = list(reader)
    aliases = {}
    for row in rows:
        sid,cid = row["source_target_id"].strip(),row["canonical_id"].strip()
        if not sid or not cid:
            fail("RELATION_POLICY_IDENTITY_EMPTY","Identity mapping contains an empty ID")
        for alias in {sid,row.get("source_column","").strip()}-{""}:
            aliases.setdefault(alias,set()).add(cid)
    return aliases


def load_bound_policy(cfg, prepared_root=None):
    config = cfg.get("relation_target_policy")
    if config is None:
        return None
    if not isinstance(config,dict) or any(not config.get(k) for k in BINDING_FIELDS):
        fail("RELATION_POLICY_BINDING_INVALID","All policy/backbone/identity file paths and SHA256s are required")
    if prepared_root is None:
        policy_path = Path(config["path"])
        identity_path = Path(config["source_identity_map"])
        backbone_path = Path(config["backbone_path"])
    else:
        folder = Path(prepared_root)/FOLDER
        policy_path,identity_path,backbone_path = folder/"TARGET_POLICY.tsv",folder/"SOURCE_IDENTITY_MAP.tsv",folder/"backbone"
    _verified(policy_path,config["sha256"],"target_policy")
    _verified(identity_path,config["source_identity_map_sha256"],"source_identity_map")
    _verified(backbone_path/"MANIFEST.json",config["backbone_manifest_sha256"],"backbone_manifest")
    backbone = load_backbone(backbone_path,verify=True)
    if not backbone["rows"]:
        fail("RELATION_POLICY_BACKBONE_COUNT","The verified biological backbone must not be empty")
    policies = load_target_policy(policy_path)
    aliases = _map_rows(identity_path)
    binding = {"config":dict(config),"policies":policies,"aliases":aliases,
               "backbone":backbone,"policy_path":policy_path,
               "identity_path":identity_path,"backbone_path":backbone_path}
    if backbone["manifest"].get("format")=="CLA_KEGG_CANCER_BACKBONE_V2":
        files=config.get("context_graph",{})
        required={"MECHANISM_CONTEXT_LINKS.tsv","MECHANISM_CONTEXT_COMPONENTS.tsv"}
        if set(files)!=required:
            fail("RELATION_POLICY_CONTEXT_GRAPH_BINDING","Evidence-defined scope requires sealed mechanism context")
        binding["context_graph"]={}
        for name,item in files.items():
            path=(Path(prepared_root)/FOLDER/name) if prepared_root is not None else Path(item["path"])
            _verified(path,item["sha256"],name)
            with path.open(encoding="utf-8-sig",newline="") as h:
                binding["context_graph"][name]=list(csv.DictReader(h,delimiter="\t"))
    for row in backbone["rows"]:
        cid = resolve_identity(binding,row["source_target_id"])
        if cid != row["canonical_id"]:
            fail("RELATION_POLICY_BACKBONE_IDENTITY_DRIFT",row["source_target_id"])
        policy = policies.get(cid)
        if policy is None or not policy.supervised or policy.analysis_role != "primary_pathway" or policy.loss_group != "pathway":
            fail("RELATION_POLICY_BACKBONE_NOT_PRIMARY",cid)
    return binding


def resolve_identity(binding, source_id, source_column=None):
    matches = set()
    for value in (source_id,source_column):
        if value:
            if value in binding["policies"]:
                matches.add(value)
            matches.update(binding["aliases"].get(str(value),set()))
    if not matches:
        fail("RELATION_POLICY_IDENTITY_MISSING",str(source_id),source_column=source_column)
    if len(matches) != 1:
        fail("RELATION_POLICY_IDENTITY_AMBIGUOUS",str(source_id),canonical_ids=sorted(matches))
    return next(iter(matches))


def _unique_axis(binding, source_ids, columns=None):
    canonicals = [resolve_identity(binding,sid,None if columns is None else columns[i])
                  for i,sid in enumerate(source_ids)]
    duplicate = {cid:[sid for sid,mapped in zip(source_ids,canonicals) if mapped==cid]
                 for cid in set(canonicals) if canonicals.count(cid)>1}
    if duplicate:
        fail("RELATION_POLICY_MANY_TO_ONE_SUPERVISION",
             "Multiple original training targets share a canonical ID; explicit aggregation is required",
             duplicate=duplicate)
    return canonicals


def select_pathways(binding, membership_ids):
    if binding is None:
        return sorted(set(map(str,membership_ids)))
    expected = [row["source_target_id"] for row in binding["backbone"]["rows"]]
    missing = sorted(set(expected)-set(map(str,membership_ids)))
    if missing:
        fail("RELATION_POLICY_BACKBONE_MEMBERSHIP_MISSING",
             "Fixed backbone source definitions are absent from graph membership",missing=missing)
    _unique_axis(binding,expected)
    return sorted(expected)


def host_readout_qualified_axis(binding,pathways):
    """Generic host readouts qualify Reference mechanisms, not named perturbants."""
    if binding is None or binding['backbone']['manifest'].get('format') != 'CLA_KEGG_CANCER_BACKBONE_V2':
        return [True] * len(pathways)
    rows={r['source_target_id']:r for r in binding['backbone']['rows']}
    return [rows[sid]['network_type']=='Reference' for sid in pathways]


def select_state_targets(cfg,binding):
    primary = [row for row in cfg["targets"] if row["role"]=="primary"]
    if binding is None:
        return primary
    result = []
    for row in primary:
        # Routed PROGENy/DoRothEA readouts can share an identity with a genuine
        # State (e.g. Hypoxia). Their source role, not the merged display class,
        # keeps the regulator/program readout out of the State supervision axis.
        if row.get("source_kind") == "routed_program":
            continue
        cid = resolve_identity(binding,row["id"],row.get("column"))
        policy = binding["policies"].get(cid)
        if policy is None:
            fail("RELATION_POLICY_CANONICAL_MISSING",cid)
        if policy.supervised and policy.analysis_role=="state_auxiliary" and policy.loss_group=="state":
            result.append(row)
    _unique_axis(binding,[x["id"] for x in result],[x.get("column") for x in result])
    return result


def seal_prepared_policy(root,cfg,binding,pathways,state_targets,profile):
    if binding is None:
        return None
    root = Path(root)
    destination = root/FOLDER
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(binding["policy_path"],destination/"TARGET_POLICY.tsv")
    shutil.copyfile(binding["identity_path"],destination/"SOURCE_IDENTITY_MAP.tsv")
    for name,item in binding["config"].get("context_graph",{}).items():
        source=_verified(item["path"],item["sha256"],name)
        shutil.copyfile(source,destination/name)

    target_backbone = destination/"backbone"
    target_backbone.mkdir(exist_ok=True)
    for name in ["MANIFEST.json",*binding["backbone"]["manifest"]["files"]]:
        # All copied members must remain inside the pinned backbone directory.
        source = (binding["backbone_path"]/name).resolve()
        if not source.is_relative_to(binding["backbone_path"].resolve()) or Path(name).name != name:
            fail("RELATION_POLICY_BACKBONE_MEMBER_PATH",name)
        shutil.copyfile(source,target_backbone/name)
    state_ids = [x["id"] for x in state_targets]
    pathway_canonical = _unique_axis(binding,pathways)
    state_canonical = _unique_axis(binding,state_ids,[x.get("column") for x in state_targets])
    source_all = sorted(row["source_target_id"] for row in binding["backbone"]["rows"])
    if profile=="formal" and sorted(pathways)!=source_all:
        fail("RELATION_POLICY_FORMAL_BACKBONE_INCOMPLETE","Formal supervision cannot omit fixed backbone definitions")
    rows = {"format":FORMAT,"binding":binding["config"],"profile":profile,
            "pathway_source_ids":list(pathways),"pathway_canonical_ids":pathway_canonical,
            "state_source_ids":state_ids,"state_canonical_ids":state_canonical,
            "backbone_source_ids":source_all,
            "excluded_state_source_ids":[x["id"] for x in cfg["targets"]
                                        if x["role"]=="primary" and x["id"] not in state_ids],
            "display_category_used_for_routing":False,
            "label_definition_changed":False,"fused_y_converted_to_binary":False,
            "files":{str(p.relative_to(root)):{"sha256":sha(p),"bytes":p.stat().st_size}
                     for p in sorted(destination.rglob("*")) if p.is_file()}}
    atomic_json(root/RECEIPT,rows)
    return rows


def validate_prepared_policy(root,cfg,spec,profile=None):
    if cfg.get("relation_target_policy") is None:
        if "relation_target_policy_sha256" in spec or (Path(root)/RECEIPT).exists():
            fail("RELATION_POLICY_UNBOUND_CAPSULE","Prepared policy exists without configuration binding")
        return None
    root = Path(root)
    _verified(root/RECEIPT,spec.get("relation_target_policy_sha256"),"prepared_policy_receipt")
    receipt = json.loads((root/RECEIPT).read_text())
    if receipt.get("format") != FORMAT or receipt.get("binding") != cfg["relation_target_policy"]:
        fail("RELATION_POLICY_PREPARED_BINDING_DRIFT","Receipt and configured authority differ")
    for name,item in receipt["files"].items():
        path=(root/name).resolve()
        if not path.is_relative_to((root/FOLDER).resolve()):
            fail("RELATION_POLICY_CAPSULE_PATH",name)
        _verified(path,item["sha256"],name)
    binding=load_bound_policy(cfg,root)
    states=select_state_targets(cfg,binding)
    expected_states=[row["id"] for row in states]
    pathways=list(spec["nodes"]["pathway"])
    expected_pathways=select_pathways(binding,receipt["backbone_source_ids"])
    active_profile=profile or receipt["profile"]
    if active_profile != receipt["profile"]:
        fail("RELATION_POLICY_PROFILE_DRIFT",str(active_profile))
    if pathways != sorted(set(pathways)) or not set(pathways)<=set(expected_pathways):
        fail("RELATION_POLICY_PATHWAY_AXIS_DRIFT","Prepared targets are outside the fixed backbone")
    if active_profile=="formal" and pathways != expected_pathways:
        fail("RELATION_POLICY_FORMAL_BACKBONE_INCOMPLETE","Formal supervision omitted fixed backbone targets")
    if spec["states"] != expected_states:
        fail("RELATION_POLICY_STATE_AXIS_DRIFT","Prepared State axis differs from explicit policy")
    pathway_canonical=_unique_axis(binding,pathways)
    state_canonical=_unique_axis(binding,expected_states,[x.get("column") for x in states])
    expected={"pathway_source_ids":pathways,"pathway_canonical_ids":pathway_canonical,
              "state_source_ids":expected_states,"state_canonical_ids":state_canonical}
    for key,value in expected.items():
        if receipt.get(key)!=value:
            fail("RELATION_POLICY_RECEIPT_AXIS_DRIFT",key)
    for key,value in [("pathway_canonical_ids",pathway_canonical),("state_canonical_ids",state_canonical)]:
        if spec.get(key)!=value:
            fail("RELATION_POLICY_SPEC_AXIS_DRIFT",key)
    import numpy as np
    qualified=np.asarray(host_readout_qualified_axis(binding,pathways),bool)
    if not qualified.all():
        for archive in (root/'views').glob('*/pathway.npz'):
            with np.load(archive,allow_pickle=False) as data:
                if np.any(data['available'] & ~qualified[data['target']]):
                    fail("RELATION_POLICY_UNQUALIFIED_PERTURBANT_LABEL",str(archive))
    return {"binding":binding,"receipt":receipt,"state_targets":states}
