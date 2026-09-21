#!/usr/bin/env python3
"""Download and freeze the official GSE85011 GEO sample metadata snapshot."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, load_config, utc_now, write_table


GEO_SOFT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
    "acc=GSE85011&targ=gsm&form=text&view=brief"
)
CONTROL_TARGETS = {"gal4", "gal4-3", "nc688", "nc736", "nc727", "nc-3150"}


def parse_soft(text: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            if current:
                records.append(current)
            current = {"gsm": line.split(" = ", 1)[1].strip()}
            continue
        if current is None or not line.startswith("!Sample_") or " = " not in line:
            continue
        key, value = line[1:].split(" = ", 1)
        value = value.strip()
        if key == "Sample_title":
            current["title"] = value
        elif key == "Sample_source_name_ch1":
            current["source_name"] = value
        elif key == "Sample_characteristics_ch1" and value.lower().startswith("cell type:"):
            current["cell_line"] = value.split(":", 1)[1].strip()
        elif key == "Sample_description" and value.lower().startswith("processed data file:"):
            current["processed_data_file"] = value.split(":", 1)[1].strip()
        elif key == "Sample_platform_id":
            current["platform_id"] = value
    if current:
        records.append(current)

    frame = pd.DataFrame(records)
    required = {"gsm", "title", "source_name", "cell_line", "processed_data_file"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"GSE85011 SOFT snapshot is missing fields: {sorted(missing)}")

    def target_from_source(value: object) -> str:
        text_value = str(value).strip()
        return text_value.split("_", 1)[1] if "_" in text_value else text_value

    frame["target_raw"] = frame.source_name.map(target_from_source)
    normalized_target = (
        frame.target_raw.astype(str).str.lower().str.replace("_", "-", regex=False)
    )
    normalized_source = frame.source_name.astype(str).str.lower()
    frame["is_control"] = (
        normalized_source.str.startswith(("input_", "k9me3_"))
        | normalized_target.isin(CONTROL_TARGETS)
        | normalized_target.str.startswith(("gal4", "nc"))
    )
    frame["dataset_accession"] = "GSE85011"
    frame["pmid"] = "27980086"
    frame["metadata_source_url"] = GEO_SOFT_URL
    frame["downloaded_at"] = utc_now()
    columns = [
        "dataset_accession",
        "gsm",
        "title",
        "source_name",
        "cell_line",
        "target_raw",
        "is_control",
        "processed_data_file",
        "platform_id",
        "pmid",
        "metadata_source_url",
        "downloaded_at",
    ]
    return frame[columns].sort_values("gsm").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_v2_cpu.yaml")
    parser.add_argument("--soft-file", help="Use an already-downloaded SOFT file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = cfg["_root"] / cfg["inputs"]["gse85011_sample_metadata"]
    if args.soft_file:
        soft_path = Path(args.soft_file)
        payload = soft_path.read_bytes()
    else:
        request = urllib.request.Request(
            GEO_SOFT_URL,
            headers={"User-Agent": "CancerLncAtlas/2.1 external-validation materializer"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    text = payload.decode("utf-8", errors="replace")
    frame = parse_soft(text)
    if len(frame) < 100 or frame.gsm.nunique() != len(frame):
        raise RuntimeError(
            f"Unexpected GSE85011 sample snapshot: rows={len(frame)}, "
            f"unique_gsm={frame.gsm.nunique()}"
        )
    write_table(frame, output)
    soft_cache = output.with_suffix(".soft.txt")
    soft_cache.parent.mkdir(parents=True, exist_ok=True)
    soft_cache.write_bytes(payload)
    print(
        f"Wrote {len(frame)} samples to {output}; "
        f"non-control RNA-seq targets="
        f"{frame.loc[frame.processed_data_file.str.contains('rnaseq_tpm', case=False, na=False) & ~frame.is_control, 'target_raw'].nunique()}; "
        f"SOFT sha256={file_sha256(soft_cache)}"
    )


if __name__ == "__main__":
    main()
