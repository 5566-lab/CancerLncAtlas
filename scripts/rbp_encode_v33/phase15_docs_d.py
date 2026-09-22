"""Phase 15D: the ENCODE ingestion report, generated from the receipts.

Every number in the produced document is read from an artifact the pipeline
wrote; nothing is transcribed by hand.  A receipt that does not exist yet is
reported as PENDING rather than guessed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
MAN = WORK / "manifests"
ENC_DIR = WORK / "outputs" / "phase3_encode_eclip"
LAYER_DIR = WORK / "outputs" / "phase9_10_layers"

#: The release tree keeps its markdown under docs/rbp_encode_v33/.  The work root
#: writes the same file next to the other generated reports so the two can be
#: compared; the repo copy is what ships.
REPORT = WORK / "reports" / "RBP_ENCODE_INGESTION_REPORT.md"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def mark(value: object) -> str:
    return "PENDING" if value is None else str(value)


def table(rows: list[tuple[str, object]], header: tuple[str, str]) -> list[str]:
    lines = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
    lines += [f"| {a} | {mark(b)} |" for a, b in rows]
    return lines


manifest_receipt = load(MAN / "ENCODE_FILE_MANIFEST.json")
download = load(MAN / "ENCODE_DOWNLOAD_RECEIPT.json")
coords = load(MAN / "LNCRNA_COORDINATE_AUTHORITY.json")
chain = load(WORK / "inputs" / "liftover" / "CHAIN_RECEIPT.json")
lift = load(MAN / "PATHB_LIFTOVER_RECEIPT.json")
overlap = load(ENC_DIR / "PHASE3_ENCODE_ECLIP_AUDIT.json")
gate = load(ENC_DIR / "PHASE3D_ENCODE_LAYER_GATE.json")
sensitivity = load(ENC_DIR / "PHASE3C_THRESHOLD_SENSITIVITY.json")
lineage = load(ENC_DIR / "PATHB_STEP5_LINEAGE.json")
merge = load(ENC_DIR / "PHASE3B_MERGE_AUDIT.json")
layers = load(LAYER_DIR / "PHASE9_10_LAYER_DECLARATION.json")
ablation = load(MAN / "RBP_FOUR_MODE_ABLATION.json")
preflight = load(MAN / "PHASE12_CPU_PREFLIGHT.json")

out: list[str] = []
add = out.append

add("# ENCODE RBP ingestion report")
add("")
add(f"Generated {datetime.now(timezone.utc).isoformat()} by "
    "`scripts/rbp_encode_v33/phase15_docs_d.py` from the pipeline receipts.")
add("")
add("This report answers one question: **did ENCODE data actually reach the model, "
    "and can every edge be traced back to bytes on disk?**")
add("")

# ---------------------------------------------------------------- finding 1
add("## 1. The plan's accessions were not experiments")
add("")
audit = (manifest_receipt or {}).get("plan_accession_audit") or {}
probe = audit.get("probe") or {}
add("A `type=Experiment` search returns HTTP 404 / `total = 0` for all six accessions "
    "named by the plan, while the control experiment resolves. The six are "
    "`PublicationData` FileSets, so the plan's manifest could only ever report the "
    "FileSet's own dominant assembly.")
add("")
add("| accession | `type=Experiment` total |")
add("| --- | --- |")
for accession, row in probe.items():
    add(f"| {accession} | {row.get('total')} (http {row.get('http')}) |")
add("")
add("File-level enumeration of those FileSets:")
add("")
add("| accession | role | files | FileSet `assembly` field |")
add("| --- | --- | --- | --- |")
for accession, row in ((manifest_receipt or {}).get("plan_filesets") or {}).items():
    add(f"| {accession} | {row['role']} | {row['n_files']:,} | {row['assembly_field']} |")
add("")

# ---------------------------------------------------------------- finding 2
add("## 2. The resource is not hg19-only")
add("")
add("The plan parked ENCODE as `ASSEMBLY_MISMATCH` because the FileSet reports `hg19`. "
    "At file level the eCLIP narrowPeak resource releases both assemblies:")
add("")
surface = (manifest_receipt or {}).get("eclip_narrowpeak_surface") or {}
add("| assembly | biosample | files | MiB | RBPs |")
add("| --- | --- | --- | --- | --- |")
for key, row in sorted(surface.items()):
    assembly, biosample = key.split("|")
    add(f"| {assembly} | {biosample} | {row['files']:,} | {row['bytes'] / 1048576:.1f} | "
        f"{row['n_rbps']} |")
add("")
add(f"Total: **{mark((manifest_receipt or {}).get('eclip_narrowpeak_surface_files'))} files, "
    f"{((manifest_receipt or {}).get('eclip_narrowpeak_surface_bytes') or 0) / 1048576:.1f} MiB**.")
add("")
add("The GRCh38-native half needs no coordinate change at all; only the hg19 half is "
    "lifted, and it is lifted explicitly.")
add("")

# ---------------------------------------------------------------- finding 3
add("## 3. Coordinate authority")
add("")
add("| item | value |")
add("| --- | --- |")
for row in table([
    ("lncRNA nodes", (coords or {}).get("nodes")),
    ("nodes without coordinates", (coords or {}).get("nodes_without_coordinates")),
    ("coordinate source", (coords or {}).get("coordinate_source")),
    ("annotation release", (coords or {}).get("coordinate_source_annotation_release")),
    ("assembly", (coords or {}).get("assembly")),
    ("external annotation introduced", (coords or {}).get("external_annotation_introduced")),
], ("item", "value")):
    add(row)
add("")
add("Assembly verification on three probe loci (GRCh38 truth):")
add("")
add("| locus | observed | expected GRCh38 | delta (start, end) | verdict |")
add("| --- | --- | --- | --- | --- |")
for check in (coords or {}).get("assembly_verification") or []:
    add(f"| {check['symbol']} | {'-'.join(str(x) for x in check['observed'])} | "
        f"{'-'.join(str(x) for x in check['expected_GRCh38'])} | "
        f"({check['delta_start']:,}, {check['delta_end']:,}) | {check['verdict']} |")
add("")
add("The coordinates come from the project's own standardised `dim_lncRNA` asset, so the "
    "`LNC:ENSG…` identifier space cannot drift. No external annotation is introduced.")
add("")

# ---------------------------------------------------------------- download
add("## 4. Download verification")
add("")
if download:
    add("| item | value |")
    add("| --- | --- |")
    add(f"| files on the consumption surface | {download['surface_files']:,} |")
    add(f"| verified downloads | {download['verified_files']:,} |")
    add(f"| verified bytes | {download['verified_bytes'] / 1048576:.1f} MiB |")
    add(f"| failures | {len(download['failures'])} |")
    add("")
    add(f"Verification: {download['verification']}.")
else:
    add("PENDING (`phase6b_encode_download.py` has not produced a receipt).")
add("")

# ---------------------------------------------------------------- liftover
add("## 5. Explicit hg19 to GRCh38 liftOver")
add("")
add("Silent liftOver is forbidden, so the chain file is pinned and its direction is "
    "verified by chromosome **size** (`chr1` 249,250,621 hg19 -> 248,956,422 GRCh38), "
    "not by chromosome name, which is identical in both assemblies.")
add("")
add("| item | value |")
add("| --- | --- |")
add(f"| chain sha256 | {mark((chain or {}).get('sha256'))} |")
add(f"| chain records | {mark((chain or {}).get('chain_records'))} |")
add(f"| direction verified by | {mark((chain or {}).get('direction_verified_by'))} |")
add(f"| aligned blocks indexed | {mark((lift or {}).get('aligned_blocks'))} |")
add("")
if lift:
    add("| outcome | peaks |")
    add("| --- | --- |")
    add(f"| total | {lift['peaks_total']:,} |")
    add(f"| mapped | {lift['peaks_mapped']:,} ({lift['mapped_fraction'] * 100:.2f}%) |")
    add(f"| quarantined | {lift['peaks_quarantined']:,} |")
    add(f"| dropped | {lift['peaks_dropped']} |")
    add("")
    add("Quarantine reasons:")
    add("")
    add("| reason | peaks |")
    add("| --- | --- |")
    for reason, count in lift["quarantine_reasons"].items():
        add(f"| `{reason}` | {count:,} |")
    add("")
    add(f"Criterion: {lift['criterion']}")
else:
    add("PENDING (`pathb_step2_liftover.py` has not produced a receipt).")
add("")

# ---------------------------------------------------------------- edges
add("## 6. What the overlap produced - and why it is not in the graph")
add("")
if overlap:
    add("| item | value |")
    add("| --- | --- |")
    add(f"| eCLIP files used | {overlap['eclip_files']['total']:,} "
        f"(GRCh38-native {overlap['eclip_files']['native_GRCh38']:,}, "
        f"lifted {overlap['eclip_files']['lifted_from_hg19']:,}) |")
    add(f"| RBP symbols resolved to a protein node | "
        f"{overlap['rbp_symbols_resolved']:,} / {overlap['rbp_symbols']:,} |")
    add(f"| evidence rows (lncRNA x RBP x file) | {overlap['evidence_rows_lncrna_rbp_file']:,} |")
    add(f"| global `binds_protein_eclip` edges built | {overlap['global_edges']:,} |")
    add(f"| context-specific edges built | {overlap['context_edges']:,} |")
    add(f"| lncRNAs touched | {overlap['lncrnas_touched']:,} |")
    add(f"| protein nodes touched | {overlap['proteins_touched']:,} |")
    add("")
    add(f"Biosample to cancer mapping: `{overlap['biosample_to_cancer']}`. "
        f"Biosamples with **no** cancer mapping: `{overlap['biosamples_without_cancer_mapping']}`. "
        "K562 is CML, which is not one of the 33 cancers, so it is never forced onto LAML; "
        "HepG2 produces LIHC context edges only and is never broadcast to all 33.")
else:
    add("PENDING (`phase3_encode_eclip_overlap.py` has not produced a receipt).")
add("")

if gate:
    add("### The layer is gated off")
    add("")
    add(f"**Decision: {gate['decision']}.** Switch "
        f"`{gate['switch']['name']}` defaults to `{gate['switch']['default']}`.")
    add("")
    who = gate["why"]
    add("A peak overlapping a lncRNA gene body is not by itself evidence that the RBP "
        "bound that lncRNA. Two null models were applied and they disagree:")
    add("")
    add("| null | result |")
    add("| --- | --- |")
    add(f"| uniform placement, peak level | "
        + ", ".join(f"{k} {v:.2f}x" for k, v in who["peak_level_measurement"].items())
        + " - **depletion** |")
    pp = who["conflicting_per_pair_view"]
    add(f"| per-pair Poisson | {pp['pairs_at_two_fold_or_more_fraction']:.1%} of pairs at "
        f">=2x, median {pp['median_fold_over_uniform']:.1f}x |")
    add("")
    add(who["explanation"])
    add("")
    cf = gate["counterfactual_if_merged"]
    add(f"Merging would have taken the typed binding table from "
        f"{cf['legacy_typed_binding_rows']:,} to {cf['would_become']:,} rows. "
        "That is exactly why the measurement was made before merging.")
    add("")
    add("Criterion before this layer may be admitted:")
    add("")
    for item in gate["criterion_for_admission"]:
        add(f"1. {item}")
    add("")
elif sensitivity:
    add("The overlap exists but the gate decision has not been recorded yet "
        "(`phase3d_encode_layer_gate.py`).")
    add("")

if merge:
    add("### Novelty against the frozen databases")
    add("")
    add("| item | pairs |")
    add("| --- | --- |")
    add(f"| distinct ENCODE pairs | {merge['distinct_encode_pairs']:,} |")
    add(f"| already in the frozen eCLIP layer | {merge['encode_pairs_already_in_frozen_eclip']:,} |")
    add(f"| new to the graph | {merge['encode_pairs_new']:,} "
        f"({merge['novelty_fraction'] * 100:.1f}%) |")
    add("")
    add("The merged table is written to a new path; the pre-ENCODE artifact is left in place.")
    add("")
    add("| table | rows |")
    add("| --- | --- |")
    add(f"| legacy typed binding | {merge['legacy_binding']['rows']:,} |")
    add(f"| merged | {merge['merged']['rows']:,} |")
    add("")
else:
    add("### Merge")
    add("")
    add("Not performed. `phase3b_merge_encode_binding.py` is the step that would merge "
        "the layer, and the gate above is why it was not run.")
    add("")
if lineage:
    add("### Lineage")
    add("")
    add(f"{lineage['lineage_rows']:,} lineage rows cover "
        f"{lineage['distinct_files_cited']:,} source files "
        f"(native {lineage['native_GRCh38_files']:,}, lifted {lineage['lifted_hg19_files']:,}), "
        f"{lineage['distinct_lncrnas']:,} lncRNAs and {lineage['distinct_rbps']:,} RBPs. "
        "Checks: " + ", ".join(f"`{k}`" for k in lineage["checks"]) + ".")
    add("")

# ---------------------------------------------------------------- layers
add("## 7. Every ENCODE layer, and its gate")
add("")
add("| layer | built | admitted to the main graph | switch |")
add("| --- | --- | --- | --- |")
if overlap:
    add(f"| eCLIP gene-body overlap | {overlap['global_edges']:,} edges | "
        f"**no** - see section 6 | `include_encode_eclip_overlap = False` |")
add("| eCLIP coordinates and lineage | "
    f"{mark((lineage or {}).get('lineage_rows'))} rows | yes (provenance only) | - |")
if layers:
    for key, label in (("rbp_knockdown", "RBP knockdown RNA-seq"), ("rbns", "RBNS")):
        row = layers["layers"][key]
        add(f"| {label} | {row['files']:,} files inventoried | "
            f"**no** | `{row['default_switch']}` |")
add("")
add("The reason each is gated off:")
add("")
for key, label in (("rbp_knockdown", "RBP knockdown RNA-seq"), ("rbns", "RBNS")):
    if layers:
        row = layers["layers"][key]
        add(f"* **{label}** - {row['why_not_admitted']}")
add("* **eCLIP gene-body overlap** - the peak-level enrichment measurement came out "
    "below 1 (see section 6).")
add("")
if layers:
    rbn = layers["layers"]["rbns"]
    kd = layers["layers"]["rbp_knockdown"]
    add(f"Inventoried but not downloaded: the knockdown resource is "
        f"{kd['bytes'] / 1024 ** 3:.1f} GiB. The RBNS small products "
        f"({rbn['small_products']['files']} files, "
        f"{rbn['small_products']['bytes'] / 1048576:.1f} MiB) were fetched and verified, "
        "because at that size the layer can be present rather than merely described.")
add("")

# ---------------------------------------------------------------- modes
add("## 8. Four-mode ablation")
add("")
if ablation:
    add("| mode | binding layer | binding rows | active binding edges | status |")
    add("| --- | --- | --- | --- | --- |")
    for name, row in ablation["modes"].items():
        status = row.get("status") or ("materialised" if row.get("active_binding_edges") is not None
                                       else "not materialised")
        add(f"| {row['mode']} `{name}` | {row['binding_layer']} | "
            f"{mark(row.get('binding_rows'))} | {mark(row['active_binding_edges'])} | {status} |")
    add("")
    c = ablation["modes"].get("typed_binding_plus_encode_eclip") or {}
    if c.get("gate"):
        gate = c["gate"]
        add(f"Mode C is `{gate['decision']}`. It built "
            f"{(gate.get('edges_built') or {}).get('global')} global and "
            f"{(gate.get('edges_built') or {}).get('context')} context ENCODE edges, and did "
            "not materialise them into a graph generation. Until it does, modes B, C and D "
            "produce the same graph, which is stated here rather than hidden behind a "
            "table of identical numbers.")
        add("")
    add("Mode D is graph-identical to mode C by construction: knockdown evidence is "
        "declared `evidence_only` and `assert_fair_comparison` forbids it from the "
        "primary graph.")
    add("")
    add("Resolved on CPU: " + "; ".join(ablation["resolved_on_cpu"]) + ".")
    add("")
    add("**Not** resolved on CPU: " + "; ".join(ablation["not_resolved_on_cpu"]) + ".")
else:
    add("PENDING (`phase13_four_mode_ablation.py` has not produced a receipt).")
add("")

# ---------------------------------------------------------------- gates
add("## 9. CPU gate")
add("")
if preflight:
    add("| gate | status |")
    add("| --- | --- |")
    for key in ("encode_manifest", "genome_assembly", "encode_download",
                "liftover_quarantine", "encode_adds_no_nodes"):
        if key in preflight:
            add(f"| {key} | {preflight[key].get('status')} |")
    add("")
    if "encode_novelty" in preflight:
        nov = preflight["encode_novelty"]
        add(f"ENCODE novelty: {nov['new_to_graph']:,} new pairs of "
            f"{nov['distinct_encode_pairs']:,} ({nov['novelty_fraction'] * 100:.1f}%).")
else:
    add("PENDING (`phase12_preflight_compute.py` has not produced a receipt).")
add("")

# ---------------------------------------------------------------- prohibitions
add("## 10. Prohibitions, and how each is held")
add("")
add("| prohibition | mechanism |")
add("| --- | --- |")
add("| no new RBP node type | ENCODE RBPs map onto the existing `protein` nodes; "
    "`encode_adds_no_nodes` fails if any new node appears |")
add("| knockdown is not lncRNA functional truth | declared `evidence_only`; "
    "`rbp_kd_is_lncrna_function_truth` is a hard `False` in every mode |")
add("| no context-specific broadcast | only HepG2 -> LIHC is mapped; K562 is CML and is "
    "never forced onto LAML |")
add("| no silent liftOver | the chain is pinned by SHA256 and every mapped peak records "
    "its chain id, strand and original coordinates |")
add("| no overwriting of formal artifacts | every output goes to a new path; the typed "
    "authority refuses to write into a populated directory |")
add("| no GPU while the CPU gate is open | no training is launched by any script here |")
add("")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"written: {REPORT}")
print(f"  {len(out)} lines")
