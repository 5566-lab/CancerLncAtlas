#!/usr/bin/env python3
"""Freeze KEGG NETWORK definitions/classifications on 149 without guessing edges."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CANCER_CATEGORIES = frozenset(("Cancer", "Cancer signaling"))
NETWORK_RE = re.compile(r"N[0-9]{5}")
MAP_RE = re.compile(r"nt[0-9]{5}")
RECORD_FIELDS = ["network_id","name","network_type","definition","expanded_definition",
                 "pathway_ids","gene_ids","class_map_ids","source_url","source_sha256"]
MAP_FIELDS = ["network_id","map_id","map_name","category","is_cancer_map",
              "category_path","network_class_confirmed","source_url","source_sha256"]

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)

def atomic_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode())

def write_tsv(path, rows, fields):
    tmp = Path(path).with_name(Path(path).name + ".part")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def parse_flat_records(text):
    """Read fixed-column fields, including EXPANDED subfields and continuations."""
    parsed = {}
    for block in text.split("///"):
        if not block.strip():
            continue
        fields, active = {}, None
        for line in block.splitlines():
            if not line.strip():
                continue
            key, value = line[:12].strip(), line[12:].strip()
            if key:
                active = key
                fields.setdefault(active, []).append(value)
            elif active:
                fields[active].append(value)
            else:
                raise ValueError("Malformed KEGG record before first field")
        entry = " ".join(fields.get("ENTRY", [])).split()
        if not entry:
            raise ValueError("KEGG record missing ENTRY")
        identifier = entry[0]
        if identifier in parsed:
            raise ValueError(f"Duplicate KEGG ENTRY: {identifier}")
        parsed[identifier] = fields
    return parsed

def joined(fields, key):
    return " ".join(fields.get(key, []))

def parse_network(identifier, fields):
    if not NETWORK_RE.fullmatch(identifier):
        raise ValueError(f"Not a NETWORK element: {identifier}")
    name = joined(fields, "NAME")
    if not name:
        raise ValueError(f"Network has no name: {identifier}")
    genes = set()
    for line in fields.get("GENE", []):
        match = re.match(r"(?:hsa:)?([0-9]+)(?:\s|$)", line)
        if not match:
            raise ValueError(f"Unexpected GENE field in {identifier}: {line}")
        genes.add(match.group(1))
    return {"network_id":identifier, "name":name, "network_type":joined(fields,"TYPE"),
        "definition":joined(fields,"DEFINITION"), "expanded_definition":joined(fields,"EXPANDED"),
        "pathway_ids":"|".join(sorted(set(re.findall(r"\bhsa[0-9]{5}\b",joined(fields,"PATHWAY"))))),
        "gene_ids":"|".join(sorted(genes,key=int)),
        "class_map_ids":"|".join(sorted(set(MAP_RE.findall(joined(fields,"CLASS")))))}

def parse_map(identifier, fields):
    if not MAP_RE.fullmatch(identifier):
        raise ValueError(f"Not a NETWORK map: {identifier}")
    name, path = joined(fields,"NAME"), joined(fields,"CATEGORY")
    categories = [x.strip() for x in path.split(";") if x.strip()]
    members = sorted(set(NETWORK_RE.findall(joined(fields,"MEMBER"))))
    if not name or not categories or not members:
        raise ValueError(f"Missing map name/category/members: {identifier}")
    return {"map_id":identifier,"map_name":name,"category":categories[-1],
        "category_path":path,"is_cancer_map":int(any(x in CANCER_CATEGORIES for x in categories)),
        "members":members}

def parse_brite_maps(tree):
    """Map explicit BRITE leaf identifiers to their parent category path."""
    output = {}
    def walk(node, parents=()):
        name = node.get("name","").strip()
        children = node.get("children",[])
        if children:
            for child in children:
                walk(child,parents+(name,))
        else:
            match = re.fullmatch(r"([0-9]{5})\s+(.+)",name)
            if not match:
                raise ValueError(f"Malformed BRITE network leaf: {name}")
            identifier = "nt"+match.group(1)
            if identifier in output:
                raise ValueError(f"Duplicate BRITE map: {identifier}")
            output[identifier] = {"name":match.group(2),"category":parents[-1],
                                  "category_path":"; ".join(parents[1:])}
    walk(tree)
    for category in CANCER_CATEGORIES:
        if not any(row["category"]==category for row in output.values()):
            raise ValueError(f"BRITE lacks exact category {category}")
    return output

def candidate_networks(definitions_source=None, curated_xrefs=None, mapping_tsv=None):
    ids, receipts = set(), []
    for path in (definitions_source,curated_xrefs,mapping_tsv):
        if path:
            receipts.append({"path":str(Path(path).resolve()),"sha256":sha(path)})
    if definitions_source:
        data = json.loads(Path(definitions_source).read_text(encoding="utf-8"))
        for row in data["rows"]:
            if row.get("source_name","").startswith("KEGG_MEDICUS") and row.get("status")=="VERIFIED_SOURCE_METADATA":
                value = row.get("exact_source","")
                if NETWORK_RE.fullmatch(value):
                    ids.add(value)
    if curated_xrefs:
        data = json.loads(Path(curated_xrefs).read_text(encoding="utf-8"))
        for row in data["records"]:
            value = row.get("network_id","")
            if not NETWORK_RE.fullmatch(value):
                raise ValueError(f"Invalid curated network ID: {value}")
            ids.add(value)
    if mapping_tsv:
        with Path(mapping_tsv).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                for value in row.get("network_id","").split("|"):
                    if value:
                        if not NETWORK_RE.fullmatch(value):
                            raise ValueError(f"Invalid mapped network ID: {value}")
                        ids.add(value)
    if not ids:
        raise ValueError("No source-verified candidate NETWORK IDs")
    return sorted(ids), receipts

class Fetcher:
    def __init__(self, root, proxy=None, timeout=40, interval=.45):
        if interval < 1/3:
            raise ValueError("KEGG request interval must be at least 1/3 second")
        self.root, self.timeout, self.interval = Path(root),timeout,interval
        self.last_request, self.sources, self.unresolved = 0., {}, {}
        handler = urllib.request.ProxyHandler({"http":proxy,"https":proxy} if proxy else {})
        self.opener = urllib.request.build_opener(handler)

    def fetch(self, relative, url):
        path = self.root/relative
        receipt_path = path.with_name(path.name+".receipt.json")
        if path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["url"] != url or receipt["sha256"] != sha(path):
                raise ValueError(f"Cached source checksum/URL mismatch: {path}")
            self.sources[relative] = receipt
            return path.read_text(encoding="utf-8")
        for attempt in range(3):
            time.sleep(max(0.,self.interval-(time.monotonic()-self.last_request)))
            self.last_request = time.monotonic()
            try:
                req = urllib.request.Request(url,headers={"User-Agent":"CancerLncAtlas-NETWORK-Snapshot/1.0"})
                with self.opener.open(req,timeout=self.timeout) as response:
                    payload = response.read()
                    status,content_type = response.status,response.headers.get("Content-Type","")
                if not payload:
                    raise ValueError(f"Empty official source: {url}")
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (400,404) or attempt==2:
                    raise
                time.sleep(2**attempt)
            except Exception:
                if attempt==2:
                    raise
                time.sleep(2**attempt)
        atomic_bytes(path,payload)
        receipt = {"path":relative,"url":url,"fetched_at_utc":now(),"sha256":sha(path),
                   "bytes":len(payload),"http_status":status,"content_type":content_type}
        atomic_json(receipt_path,receipt)
        self.sources[relative] = receipt
        return payload.decode("utf-8")

    def confirm_missing(self, identifier):
        """Require an individual REST 404 and official entry 'No such data'."""
        if not NETWORK_RE.fullmatch(identifier):
            raise ValueError(f"Unresolved map cannot be published: {identifier}")
        relative = "raw/unresolved/"+identifier+".json"
        path = self.root/relative
        receipt_path = path.with_name(path.name+".receipt.json")
        url = "https://rest.kegg.jp/get/"+identifier
        if path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["sha256"] != sha(path) or receipt["url"] != url:
                raise ValueError(f"Missing-entry receipt checksum failure: {identifier}")
        else:
            try:
                text = self.fetch("raw/entries/"+identifier+".txt",url)
                records = parse_flat_records(text)
                if set(records) != {identifier}:
                    raise ValueError(f"Individual response has wrong ID: {identifier}")
                receipt = self.sources["raw/entries/"+identifier+".txt"]
                return (records[identifier],url,receipt["sha256"])
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                response_body = error.read().decode("utf-8",errors="replace")
                atomic_json(path,{"network_id":identifier,"url":url,"http_status":404,
                    "checked_at_utc":now(),"response_body":response_body})
                receipt = {"path":relative,"url":url,"fetched_at_utc":now(),
                    "sha256":sha(path),"bytes":path.stat().st_size,"http_status":404,
                    "content_type":"application/json; receipt of HTTP response"}
                atomic_json(receipt_path,receipt)
        self.sources[relative] = receipt
        entry_relative = "raw/unresolved/"+identifier+".html"
        entry_url = "https://www.kegg.jp/entry/"+identifier
        entry_text = self.fetch(entry_relative,entry_url)
        if "No such data was found." not in entry_text:
            raise ValueError(f"REST 404 but official entry is not confirmed absent: {identifier}")
        self.unresolved[identifier] = {"network_id":identifier,
            "reason":"official_missing_confirmed","error":"REST HTTP 404; official entry: No such data was found.",
            "source_url":entry_url,"source_sha256":self.sources[entry_relative]["sha256"],
            "http_receipt_path":relative,"entry_response_path":entry_relative}
        return None

    def records(self, identifiers):
        output = {}
        requested = set(identifiers)
        # Reuse verified batches even when candidate additions change batch boundaries.
        for receipt_path in sorted((self.root/"raw/entries").glob("*.receipt.json")):
            receipt = json.loads(receipt_path.read_text())
            raw_path = self.root/receipt["path"]
            cached = parse_flat_records(raw_path.read_text(encoding="utf-8"))
            matched = requested.intersection(cached)
            if not matched:
                continue
            if sha(raw_path) != receipt["sha256"]:
                raise ValueError(f"Cached source checksum mismatch: {raw_path}")
            self.sources[receipt["path"]] = receipt
            for identifier in matched:
                value = (cached[identifier],receipt["url"],receipt["sha256"])
                if identifier in output and output[identifier][0] != value[0]:
                    raise ValueError(f"Conflicting cached definitions for {identifier}")
                output[identifier] = value
        missing = [x for x in identifiers if x not in output]
        if output:
            print(f"Reused {len(output)}/{len(identifiers)} cached entries",flush=True)
        for offset in range(0,len(missing),10):
            batch = missing[offset:offset+10]
            relative = "raw/entries/"+"_".join(batch)+".txt"
            url = "https://rest.kegg.jp/get/"+"+".join(("nt:"+x) if MAP_RE.fullmatch(x) else x for x in batch)
            try:
                records = parse_flat_records(self.fetch(relative,url))
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                records = {}
            if set(records) - set(batch):
                raise ValueError(f"Unexpected KEGG entries: {sorted(set(records)-set(batch))}")
            for identifier in set(batch)-set(records):
                result = self.confirm_missing(identifier)
                if result is not None:
                    output[identifier] = result
            for identifier,fields in records.items():
                output[identifier] = (fields,url,self.sources[relative]["sha256"])
            print(f"Fetched {offset+len(batch)}/{len(identifiers)} entries",flush=True)
        return output

def verify_snapshot(root, manifest):
    for receipt in manifest["sources"]+manifest["outputs"]:
        path = Path(root)/receipt["path"]
        if not path.is_file() or sha(path)!=receipt["sha256"]:
            raise ValueError(f"Frozen snapshot checksum failure: {path}")

def freeze(output_root, definitions_source=None, curated_xrefs=None, mapping_tsv=None, proxy=None, timeout=40):
    hostname = socket.gethostname()
    if hostname!="149":
        raise RuntimeError(f"CancerLncAtlas preparation must run on 149, got {hostname}")
    root = Path(output_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    out = root/"network_snapshot"
    out.mkdir(exist_ok=True)
    candidate_ids,inputs = candidate_networks(definitions_source,curated_xrefs,mapping_tsv)
    manifest_path = out/"NETWORK_SNAPSHOT_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["candidate_network_ids"]!=candidate_ids or manifest["input_sources"]!=inputs:
            raise ValueError("Existing immutable snapshot has different candidate inputs")
        verify_snapshot(root,manifest)
        print(json.dumps({"status":"verified_existing_snapshot",**manifest["counts"]}))
        return manifest
    atomic_json(out/"LAUNCH_PREFLIGHT.json",{"target_host":"149","verified_hostname":hostname,
        "timestamp_utc":now(),"candidate_count":len(candidate_ids),
        "cancer_categories":sorted(CANCER_CATEGORIES),
        "classification_rule":"exact CATEGORY tokens on official nt records"})
    fetcher = Fetcher(root,proxy,timeout)
    brite = parse_brite_maps(json.loads(fetcher.fetch("raw/br08907.json","https://rest.kegg.jp/get/br:br08907/json")))
    directory_maps = set(brite)
    if not directory_maps:
        raise ValueError("Official NETWORK directory contains no nt map IDs")
    raw_networks = fetcher.records(candidate_ids)
    network_rows,class_maps = [],set()
    for identifier in candidate_ids:
        if identifier not in raw_networks:
            continue
        fields,url,source_sha = raw_networks[identifier]
        row = parse_network(identifier,fields)
        row.update(source_url=url,source_sha256=source_sha)
        class_maps.update(filter(None,row["class_map_ids"].split("|")))
        network_rows.append(row)
    map_ids = sorted(directory_maps|class_maps)
    raw_maps = fetcher.records(map_ids)
    map_rows,map_registry,classified_cancer = [],[],set()
    element_to_maps = {r["network_id"]:set(filter(None,r["class_map_ids"].split("|"))) for r in network_rows}
    for identifier in map_ids:
        fields,url,source_sha = raw_maps[identifier]
        parsed = parse_map(identifier,fields)
        if identifier in brite and parsed["category"] != brite[identifier]["category"]:
            raise ValueError(f"Map category differs from BRITE for {identifier}")
        members = parsed.pop("members")
        map_registry.append({**parsed,"member_count":len(members),"source_url":url,"source_sha256":source_sha})
        for member in members:
            if member in element_to_maps:
                if parsed["is_cancer_map"]:
                    classified_cancer.add(member)
                map_rows.append({"network_id":member,**parsed,
                    "network_class_confirmed":int(identifier in element_to_maps[member]),
                    "source_url":url,"source_sha256":source_sha})
    covered_pairs = {(r["network_id"],r["map_id"]) for r in map_rows}
    missing_pairs = [(nid,mid) for nid,mids in element_to_maps.items() for mid in mids if (nid,mid) not in covered_pairs]
    if missing_pairs:
        raise ValueError(f"Network CLASS claims absent from map MEMBER: {missing_pairs}")
    for category in CANCER_CATEGORIES:
        if not any(r["category"]==category for r in map_registry):
            raise ValueError(f"Official map directory lacks category {category}")
    if set(candidate_ids) != set(raw_networks)|set(fetcher.unresolved):
        raise ValueError("Candidate resolution incomplete")
    write_tsv(out/"NETWORK_UNRESOLVED.tsv",[fetcher.unresolved[x] for x in sorted(fetcher.unresolved)],
        ["network_id","reason","error","source_url","source_sha256","http_receipt_path","entry_response_path"])
    write_tsv(out/"NETWORK_RECORDS.tsv",network_rows,RECORD_FIELDS)
    write_tsv(out/"NETWORK_MAP_MEMBERSHIP.tsv",sorted(map_rows,key=lambda r:(r["network_id"],r["map_id"])),MAP_FIELDS)
    write_tsv(out/"NETWORK_MAPS.tsv",map_registry,["map_id","map_name","category","is_cancer_map","category_path","member_count","source_url","source_sha256"])
    counts = {"candidate_networks":len(candidate_ids),"parsed_networks":len(network_rows),
        "unresolved_current_networks":len(fetcher.unresolved),
        "maps":len(map_registry),"cancer_maps":sum(r["is_cancer_map"] for r in map_registry),
        "cancer_category_maps":sum(r["category"]=="Cancer" for r in map_registry),
        "cancer_signaling_maps":sum(r["category"]=="Cancer signaling" for r in map_registry),
        "candidate_map_memberships":len(map_rows),"cancer_candidate_networks":len(classified_cancer),
        "networks_without_map":sum(not r["class_map_ids"] for r in network_rows)}
    outputs = []
    for name in ["NETWORK_RECORDS.tsv","NETWORK_MAP_MEMBERSHIP.tsv","NETWORK_MAPS.tsv","NETWORK_UNRESOLVED.tsv","LAUNCH_PREFLIGHT.json"]:
        path = out/name
        outputs.append({"path":str(path.relative_to(root)),"sha256":sha(path),"bytes":path.stat().st_size})
    manifest = {"schema_version":1,"target_host":hostname,"frozen_at_utc":now(),
        "candidate_network_ids":candidate_ids,"input_sources":inputs,
        "cancer_categories":sorted(CANCER_CATEGORIES),
        "classification_rule":"Exact semicolon-delimited CATEGORY tokens on official KEGG nt map records",
        "definition_policy":"Original DEFINITION/EXPANDED text retained; no guessed graph edges",
        "sources":list(fetcher.sources.values()),"outputs":outputs,"counts":counts}
    atomic_json(manifest_path,manifest)
    verify_snapshot(root,manifest)
    print(json.dumps({"status":"frozen_and_verified",**counts},ensure_ascii=False),flush=True)
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--definitions-source")
    parser.add_argument("--curated-xrefs")
    parser.add_argument("--mapping-tsv")
    parser.add_argument("--proxy",help="Explicit proxy; unset ignores inherited proxy settings")
    parser.add_argument("--timeout",type=float,default=40)
    freeze(**vars(parser.parse_args()))

if __name__=="__main__":
    main()
