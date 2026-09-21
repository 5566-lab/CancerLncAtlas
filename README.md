# CancerLncAtlas CC-HHGT v3.2

CancerLncAtlas is a reproducible pan-cancer lncRNA functional atlas. This repository contains the source code for the complete workflow: raw-source acquisition, identifier and modality harmonization, graph construction, covariate-adjusted baselines, CC-HHGT multitask training, validation, pathway/state materialization, evidence integration, and the website API/frontend.

The v3.2 model uses one shared CC-HHGT encoder with pathway and tumor-state heads. The two heads are trained jointly with the declared loss weights in `config/model_v3_2_full_multitask.yaml`; pathway and state artifacts are comparison/output tables, not two unrelated deep-learning models.

## Repository layout

- `cc_hhgt/` — reusable model, graph, data-contract, validation, and release modules.
- `cc_hhgt/v32/` — v3.2 multimodal extensions, training, evidence fusion, subtype and web-release code.
- `scripts/` — ordered acquisition, preprocessing, training, audit, materialization, and packaging entry points.
- `config/`, `configs/`, `schemas/`, `sql/` — versioned configuration and data contracts.
- `website/backend/` and `website/frontend/` — API and static website implementation.
- `tests/` — unit and contract tests.
- `data/` — public input contracts and instructions; large raw datasets are downloaded at run time.
- `examples/` — small manifests and reproducibility examples.
- `docs/` — protocols, data provenance, validation, and release notes.

## Reproducibility

The repository intentionally does not commit raw TCGA/GDC, single-cell, ATAC, clinical, drug, or materialized web tables. These files are large, may have redistribution restrictions, and can contain controlled-access references. Use the acquisition scripts and manifests to fetch permitted public inputs, record their source versions, and verify SHA-256 checksums before running downstream stages.

1. Create an isolated Python environment (Python 3.11 is the reference runtime) and install the pinned dependencies:

   ```bash
   bash install_environment.sh cc_hhgt
   ```

2. Set paths and public data locations in a local copy of the appropriate config under `config/` or `configs/`. Do not commit local paths, credentials, tokens, or private endpoints.

3. Build and audit inputs:

   ```bash
   python scripts/00_audit_model_inputs.py --help
   python scripts/01_standardize_model_inputs.py --help
   python scripts/02_build_id_crosswalks.py --help
   python scripts/04_build_pathway_state_edges.py --help
   ```

4. Run CPU preparation and baseline checks:

   ```bash
   bash run_model_workflow.sh config/model_v3_2_code_only.yaml prepare
   bash run_model_workflow.sh config/model_v3_2_code_only.yaml baseline
   ```

5. Run the formal GPU workflow only after the input manifest, code archive, environment, and GPU preflight are verified:

   ```bash
   bash 03_train_all_linux.sh
   bash 04_train_multiseed_linux.sh
   bash 05_package_results_linux.sh
   ```

6. Build website tables and start the local website using the instructions in `docs/` and `website/`. Results must be generated from a frozen manifest and published with a SHA-256 receipt.

The commands above are entry points. Each formal release should retain its own manifest, configuration, environment lock, validation report, and output checksum file.

## Data and privacy

Only use public data or data for which your project has permission. Do not commit credentials, SSH keys, cloud account files, controlled-access patient data, raw clinical records, unreviewed identifiers, or private server paths. See `SECURITY.md` and `data/README.md`.

## Citation

Until a project DOI is assigned, cite the repository commit and the versioned model protocol in `docs/`. A citation template is provided in `CITATION.cff`.

## License

Source code is released under the Apache License 2.0. Third-party datasets and dependencies retain their own licenses; see `NOTICE` and the acquisition manifests.
