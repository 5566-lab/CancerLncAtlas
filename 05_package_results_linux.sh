#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"$PWD/.venv/bin/python" scripts/24_package_gpu_results.py
