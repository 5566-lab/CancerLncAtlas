#!/usr/bin/env python3
"""Create a checksum-audited ZIP containing GPU results to return to the server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


MODELS = ("rgcn", "hgt", "cc_hhgt")
EXPECTED = (
    "best.pt",
    "training_history.tsv",
    "prediction_raw.parquet",
    "metrics.tsv",
    "metadata.json",
    "node_embeddings.pt",
    "embedding_metadata.json",
    "calibration.json",
    "prediction_calibrated.parquet",
    "metrics_calibrated.tsv",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_fold_ids(results: Path) -> list[str]:
    with (results / "tables" / "fold_manifest.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return [row["fold_id"] for row in csv.DictReader(handle, delimiter="\t")]


def configured_seeds(config_path: Path) -> tuple[int, list[int]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    multiseed = config.get("multiseed", {})
    primary = int(multiseed["primary_seed"])
    replicates = [int(seed) for seed in multiseed["replicate_seeds"]]
    return primary, [primary, *replicates]


def complete_model_dirs(
    results: Path,
    config_path: Path,
) -> tuple[list[Path], list[str], list[int]]:
    complete: list[Path] = []
    missing: list[str] = []
    primary, seeds = configured_seeds(config_path)
    folds = read_fold_ids(results)
    for seed in seeds:
        seed_root = (
            results / "models"
            if seed == primary
            else results / "models_multiseed" / f"seed_{seed}"
        )
        for model in MODELS:
            for fold in folds:
                fold_dir = seed_root / model / fold
                absent = [
                    name
                    for name in EXPECTED
                    if not (fold_dir / name).is_file()
                    or (fold_dir / name).stat().st_size == 0
                ]
                if absent:
                    missing.append(
                        f"seed={seed}/{model}/{fold}: {','.join(absent)}"
                    )
                else:
                    complete.append(fold_dir)
    return complete, missing, seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results = root / "results" / "model" / "cc_hhgt_v2_1_gpu"
    config_path = root / "config" / "model_v2_1_gpu_4070tis.yaml"
    complete, incomplete, seeds = complete_model_dirs(results, config_path)
    expected_count = len(read_fold_ids(results)) * len(MODELS) * len(seeds)
    if not args.allow_partial and (len(complete) != expected_count or incomplete):
        raise RuntimeError(
            f"Refusing incomplete return package: complete model/folds={len(complete)}/{expected_count}, "
            f"incomplete={len(incomplete)}. Use --allow-partial only for diagnostic return."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else root / f"CC_HHGT_GPU_RESULTS_RETURN_{stamp}.zip"
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    files: set[Path] = set()
    for directory in (
        results / "models",
        results / "models_multiseed",
        results / "status",
        results / "reports",
        results / "tables" / "external_validation",
        root / "logs",
    ):
        if directory.exists():
            files.update(path for path in directory.rglob("*") if path.is_file())
    files.update(
        {
            config_path,
            root / "README_GPU_训练与回传.md",
            root / "CODE_AND_DATA_PROVENANCE.md",
            root / "交接.md",
        }
    )
    files = {path for path in files if path.exists() and path.resolve() != output.resolve()}

    manifest_rows = ["sha256\tsize_bytes\tpath"]
    for path in sorted(files):
        digest = file_sha256(path)
        manifest_rows.append(f"{digest}\t{path.stat().st_size}\t{path.relative_to(root).as_posix()}")
    manifest_text = "\n".join(manifest_rows) + "\n"
    return_note = (
        "Copy this ZIP and its .sha256 file back to the server.\n"
        "Do not rename or modify files inside the ZIP.\n"
        f"Complete model/fold directories included: {len(complete)}/{expected_count}.\n"
        f"Experiment seeds: {','.join(map(str, seeds))}.\n"
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}\n"
    )

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(files):
            archive.write(path, arcname=path.relative_to(root).as_posix())
        archive.writestr("RETURN_CONTENT_MANIFEST.tsv", manifest_text)
        archive.writestr("RETURN_INSTRUCTIONS.txt", return_note)

    archive_digest = file_sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{archive_digest}  {output.name}\n", encoding="utf-8")
    print(f"Return package: {output}")
    print(f"SHA-256: {archive_digest}")
    print(f"Complete model/folds: {len(complete)}/{expected_count}")
    print(f"Seeds: {seeds}")


if __name__ == "__main__":
    main()
