# Data policy

Large raw inputs and generated result tables are intentionally excluded from Git. They are obtained by the versioned acquisition scripts in `scripts/` and `cc_hhgt/v32/download_catalog.py`.

For every run:

1. record source URL, release/version, access date and license;
2. compute SHA-256 before preprocessing;
3. store the resulting manifest outside the source tree or in a reviewed release artifact;
4. never commit controlled-access clinical data, credentials, private server paths or cloud configuration files.

The checked-in `model_input_contract.yaml` describes required tables, columns, types and missingness semantics. `examples/INPUT_MANIFEST_CODE_ONLY.json` is a code-only example and contains no raw data.
