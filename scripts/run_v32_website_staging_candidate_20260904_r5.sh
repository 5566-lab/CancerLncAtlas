#!/usr/bin/env bash
set -Eeuo pipefail

# Server-only, non-production V3.2 website candidate.
#
# The script is intentionally bounded. It binds already-produced, hash-
# checked sidecars, starts one loopback uvicorn process on a non-production
# port, performs a finite HTTP smoke, and stops that process before writing a
# receipt. It never trains, downloads, rewrites authority data, or touches
# ports used by the existing services.

HOST_EXPECTED=149
HOST_OBSERVED="$(hostname -s 2>/dev/null || hostname)"
[[ "$HOST_OBSERVED" == "$HOST_EXPECTED" ]] || {
  echo "BLOCKED_WRONG_HOST expected=$HOST_EXPECTED observed=$HOST_OBSERVED" >&2
  exit 42
}

WORK_ROOT="${V32_STAGING_WORK_ROOT:-${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r3}"
CODE_ROOT="${V32_STAGING_CODE_ROOT:-$WORK_ROOT/code}"
OVERLAY_ROOT="${V32_STAGING_CODE_OVERLAY:-$WORK_ROOT/code_overlay_20260904_r5}"
MANIFEST="${V32_STAGING_MANIFEST:-$CODE_ROOT/config/v32_server_unified_staging_bindings_20260904_r2.json}"
CATALOG_DIR="${V32_STAGING_CATALOG_DIR:-$WORK_ROOT/catalog_candidate_20260904_r5}"
CATALOG="$CATALOG_DIR/v32-capability-catalog.json"
BOOTSTRAP_CATALOG="${V32_STAGING_BOOTSTRAP_CATALOG:-$WORK_ROOT/catalog_candidate_20260904_r4/v32-capability-catalog.json}"
BOOTSTRAP_CATALOG_SHA="${V32_STAGING_BOOTSTRAP_CATALOG_SHA256:-fa71a3c62abd0ae751ecf5017517b19388991b94799d3ec7c42db1c0020780f8}"
SCOPE="$WORK_ROOT/capability_scope_20260904_r5.json"
SMOKE_REPORT="$WORK_ROOT/receipts/WEBSITE_STAGING_SMOKE_20260904_r5.json"
RECEIPT="$WORK_ROOT/receipts/WEBSITE_STAGING_CANDIDATE_20260904_r5.json"
VENV="${V32_STAGING_VENV:-$WORK_ROOT/.venv-web-r5}"
# An already-created, isolated web interpreter may be supplied for a bounded
# staging retry.  When set, it is read-only from this launcher: no venv
# creation or package installation is attempted in that environment.
PYTHON_OVERRIDE="${V32_STAGING_PYTHON:-}"
# Optional read-only site-packages overlay for an already verified interpreter.
# This is intentionally separate from the code overlay so a staging retry can
# supply one missing pure/binary wheel without mutating a shared environment.
PYTHON_EXTRA_PATH="${V32_STAGING_EXTRA_PYTHONPATH:-}"
PORT="${V32_STAGING_PORT:-8295}"
BASE_URL="http://127.0.0.1:$PORT"
READY_TIMEOUT_SEC="${V32_STAGING_READY_TIMEOUT_SEC:-180}"

SC_RUNTIME="${V32_SC_R11_RUNTIME:-./data/CancerLncAtlas/runtime/single_cell_cell_level_r11_rescued_20260903_r1}"
R11_HANDOFF="${V32_SC_R11_HANDOFF:-$SC_RUNTIME/TRAINING_HANDOFF.json}"
R11_RUN_STATUS="${V32_SC_R11_RUN_STATUS:-$SC_RUNTIME/RUN_STATUS.json}"
R11_SUCCESS="${V32_SC_R11_SUCCESS:-$SC_RUNTIME/SUCCESS.json}"
GAP_BINDING="${V32_SC_GAP_BINDING:-$CODE_ROOT/artifacts/v32_single_cell_gap_audit_20260826_r2_codebound/SINGLE_CELL_GAP_AUDIT_BINDING.json}"
GAP_AUDIT_BINDING="${V32_SC_GAP_AUDIT_BINDING:-$CODE_ROOT/artifacts/v32_single_cell_gap_audit_20260826_r2_codebound_independent_audit/INDEPENDENT_AUDIT_BINDING.json}"

R11_HANDOFF_SHA="${V32_SC_R11_HANDOFF_SHA256:-ab2eccf729a34930ea3e3ec58068132fb1f39eba28dcce461a82b3f53b55336c}"
R11_RUN_STATUS_SHA="${V32_SC_R11_RUN_STATUS_SHA256:-3c01080cabe27c0031c9f508f6c96d67ae6c388e48d96c59e03aa16c8db45a59}"
R11_SUCCESS_SHA="${V32_SC_R11_SUCCESS_SHA256:-92826b85e1dd981e57ef3a68586798981692796d70199dd0a602b69b0fc5fb68}"
GAP_BINDING_SHA="${V32_SC_GAP_BINDING_SHA256:-26e94b0e0a9aa7c217d50494c2b92ac477f4c01fdcb2a2147d61dc89eab76ccf}"
GAP_AUDIT_SHA="${V32_SC_GAP_AUDIT_BINDING_SHA256:-1e909bc81a2180f1ab301233bf57d77911305120627f6443f7acafce7cbc1784}"

# R7/HNSC gap artifacts are optional compatibility history.  If an operator
# explicitly supplies any legacy override, require the complete pair and
# validate it; otherwise use the pair only when both default files happen to
# be present.  A formal23-only bundle therefore has no legacy-gap dependency.
LEGACY_GAP_ARGS=()
if [[ -n "${V32_SC_GAP_BINDING:-}" || -n "${V32_SC_GAP_BINDING_SHA256:-}" || \
      -n "${V32_SC_GAP_AUDIT_BINDING:-}" || -n "${V32_SC_GAP_AUDIT_BINDING_SHA256:-}" ]]; then
  [[ -n "$GAP_BINDING" && -n "$GAP_BINDING_SHA" && -n "$GAP_AUDIT_BINDING" && -n "$GAP_AUDIT_SHA" ]] || {
    echo "BLOCKED_INCOMPLETE_LEGACY_GAP_OVERRIDE" >&2
    exit 44
  }
  [[ -f "$GAP_BINDING" && ! -L "$GAP_BINDING" && -f "$GAP_AUDIT_BINDING" && ! -L "$GAP_AUDIT_BINDING" ]] || {
    echo "BLOCKED_MISSING_OR_UNSAFE_LEGACY_GAP_OVERRIDE" >&2
    exit 44
  }
  LEGACY_GAP_ARGS=(
    --gap-binding "$GAP_BINDING"
    --gap-binding-sha256 "$GAP_BINDING_SHA"
    --gap-audit-binding "$GAP_AUDIT_BINDING"
    --gap-audit-binding-sha256 "$GAP_AUDIT_SHA"
  )
elif [[ -f "$GAP_BINDING" || -f "$GAP_AUDIT_BINDING" ]]; then
  [[ -f "$GAP_BINDING" && ! -L "$GAP_BINDING" && -f "$GAP_AUDIT_BINDING" && ! -L "$GAP_AUDIT_BINDING" ]] || {
    echo "BLOCKED_INCOMPLETE_LEGACY_GAP_PAIR" >&2
    exit 44
  }
  LEGACY_GAP_ARGS=(
    --gap-binding "$GAP_BINDING"
    --gap-binding-sha256 "$GAP_BINDING_SHA"
    --gap-audit-binding "$GAP_AUDIT_BINDING"
    --gap-audit-binding-sha256 "$GAP_AUDIT_SHA"
  )
fi

[[ "$PORT" =~ ^[0-9]+$  && "$PORT" -ge 1024 && "$PORT" -le 65535 ]] || {
  echo "BLOCKED_INVALID_STAGING_PORT=$PORT" >&2
  exit 43
}
case "$PORT" in
  8260|8261|8262)
    echo "BLOCKED_PRODUCTION_OR_LEGACY_PORT=$PORT" >&2
    exit 43
    ;;
esac

mkdir -p "$WORK_ROOT/receipts" "$CATALOG_DIR"

# The first HTTP pass needs a catalog file so the loopback app can start. Copy
# only the previous isolated candidate into the new isolated directory; it is
# treated as bootstrap input and is replaced atomically after the route scope
# is probed. Never fall back to the canonical/production catalog.
if [[ ! -f "$CATALOG" ]]; then
  [[ -f "$BOOTSTRAP_CATALOG" && ! -L "$BOOTSTRAP_CATALOG" ]] || {
    echo "BLOCKED_MISSING_BOOTSTRAP_CATALOG=$BOOTSTRAP_CATALOG" >&2
    exit 44
  }
  [[ "$(sha256sum "$BOOTSTRAP_CATALOG" | awk '{print $1}')" == "$BOOTSTRAP_CATALOG_SHA" ]] || {
    echo "BLOCKED_BOOTSTRAP_CATALOG_SHA256_DRIFT=$BOOTSTRAP_CATALOG" >&2
    exit 44
  }
  cp --reflink=auto -- "$BOOTSTRAP_CATALOG" "$CATALOG"
fi
[[ -f "$CATALOG" && ! -L "$CATALOG" ]] || {
  echo "BLOCKED_MISSING_OR_UNSAFE_CATALOG=$CATALOG" >&2
  exit 44
}

for required in \
  "$CODE_ROOT/scripts/serve_v32_unified_staging_candidate.py" \
  "$CODE_ROOT/scripts/probe_v32_website_staging_candidate_20260904.py" \
  "$CODE_ROOT/scripts/build_v32_staging_catalog_from_manifest_20260904.py" \
  "$CODE_ROOT/cc_hhgt/v32/unified_staging_bindings.py" \
  "$CODE_ROOT/scripts/build_v32_staging_web_catalog.py" \
  "$MANIFEST" \
  "$R11_HANDOFF" "$R11_RUN_STATUS" "$R11_SUCCESS"; do
  [[ -f "$required" && ! -L "$required" ]] || {
    echo "BLOCKED_MISSING_OR_UNSAFE_INPUT=$required" >&2
    exit 44
  }
done
[[ -d "$OVERLAY_ROOT" && ! -L "$OVERLAY_ROOT" ]] || {
  echo "BLOCKED_MISSING_OR_UNSAFE_CODE_OVERLAY=$OVERLAY_ROOT" >&2
  exit 44
}

runtime_mode="new_or_candidate_venv"
if [[ -n "$PYTHON_OVERRIDE" ]]; then
  [[ -x "$PYTHON_OVERRIDE" && ! -L "$PYTHON_OVERRIDE" ]] || {
    echo "BLOCKED_UNSAFE_EXPLICIT_PYTHON=$PYTHON_OVERRIDE" >&2
    exit 44
  }
  PY="$PYTHON_OVERRIDE"
  runtime_mode="existing_hash_checked_venv"
else
  if [[ -x "$VENV/bin/python" ]]; then
    # Reuse an already-created isolated environment without modifying it.
    runtime_mode="existing_hash_checked_venv"
  else
    python3 -m venv "$VENV"
  fi
  PY="$VENV/bin/python"
fi
if [[ -n "$PYTHON_EXTRA_PATH" ]]; then
  export PYTHONPATH="$PYTHON_EXTRA_PATH:$OVERLAY_ROOT:$CODE_ROOT"
else
  export PYTHONPATH="$OVERLAY_ROOT:$CODE_ROOT"
fi
export PYTHONDONTWRITEBYTECODE=1

deps_installed=false
if ! "$PY" -c 'import fastapi, uvicorn, duckdb, pandas, pyarrow' >/dev/null 2>&1; then
  if [[ -n "$PYTHON_OVERRIDE" ]]; then
    echo "BLOCKED_EXPLICIT_PYTHON_MISSING_DEPENDENCY=$PY" >&2
    exit 44
  fi
  "$PY" -m pip install --disable-pip-version-check --no-cache-dir \
    --timeout 30 --retries 1 \
    'fastapi>=0.100,<1' 'uvicorn>=0.20,<1' 'duckdb>=1.1,<2' \
    'pandas>=2,<3' 'pyarrow>=14,<22'
  deps_installed=true
fi
"$PY" -c 'import fastapi, uvicorn, duckdb, pandas, pyarrow'

SERVER_LOG="$WORK_ROOT/receipts/website_staging_candidate_20260904_r5.server.log"
SERVER_PID_FILE="$WORK_ROOT/receipts/website_staging_candidate_20260904_r5.server.pid"
SERVER_PID=""

cleanup() {
  set +e
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  [[ -f "$SERVER_PID_FILE" ]] && rm -f -- "$SERVER_PID_FILE"
}
trap cleanup EXIT INT TERM

"$PY" "$CODE_ROOT/scripts/serve_v32_unified_staging_candidate.py" \
  --repo-root "$CODE_ROOT" \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --overlay-root "$OVERLAY_ROOT" \
  --host 127.0.0.1 \
  --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$SERVER_PID_FILE"

ready=false
for _ in $(seq 1 "$READY_TIMEOUT_SEC"); do
  if curl --silent --show-error --fail --max-time 2 "$BASE_URL/v3.2-staging/health" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done
[[ "$ready" == true ]] || {
  echo "BLOCKED_STAGING_SERVER_NOT_READY pid=$SERVER_PID" >&2
  exit 45
}

# First pass discovers the mounted route set. It deliberately allows an old
# catalog so the scope can be used to rebuild a corrected catalog.
"$PY" "$CODE_ROOT/scripts/probe_v32_website_staging_candidate_20260904.py" \
  --base-url "$BASE_URL" \
  --repo-root "$CODE_ROOT" \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --output "$SCOPE" \
  --allow-stale-catalog

SCOPE_SHA="$(sha256sum "$SCOPE" | awk '{print $1}')"

# Rebuild only the catalog in the isolated candidate root. All scientific
# inputs are read through their hash-bound loaders; no authority output path
# is passed as a write target.
"$PY" "$CODE_ROOT/scripts/build_v32_staging_catalog_from_manifest_20260904.py" \
  --repo-root "$CODE_ROOT" \
  --manifest "$MANIFEST" \
  --scope "$SCOPE" \
  --scope-sha256 "$SCOPE_SHA" \
  --output "$CATALOG" \
  --r11-handoff "$R11_HANDOFF" \
  --r11-handoff-sha256 "$R11_HANDOFF_SHA" \
  --r11-run-status "$R11_RUN_STATUS" \
  --r11-run-status-sha256 "$R11_RUN_STATUS_SHA" \
  --r11-success "$R11_SUCCESS" \
  --r11-success-sha256 "$R11_SUCCESS_SHA" \
  "${LEGACY_GAP_ARGS[@]}"

# Strict second pass verifies the newly rebuilt catalog and writes a separate
# smoke report, leaving the scope hash embedded in the catalog unchanged.
"$PY" "$CODE_ROOT/scripts/probe_v32_website_staging_candidate_20260904.py" \
  --base-url "$BASE_URL" \
  --repo-root "$CODE_ROOT" \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --scope "$SCOPE" \
  --scope-sha256 "$SCOPE_SHA" \
  --output "$SMOKE_REPORT"

CATALOG_SHA="$(sha256sum "$CATALOG" | awk '{print $1}')"
MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
OVERLAY_API_SHA="$(sha256sum "$OVERLAY_ROOT/website/backend/v32_staging_api.py" | awk '{print $1}')"
OVERLAY_BUILDER_SHA="$(sha256sum "$OVERLAY_ROOT/scripts/build_v32_staging_web_catalog.py" 2>/dev/null | awk '{print $1}' || true)"

# Stop before producing the receipt. The trap performs the bounded shutdown;
# this explicit wait makes the final process state auditable.
cleanup
SERVER_STOPPED=true
SERVER_PID=""

"$PY" - "$RECEIPT" "$SCOPE" "$SMOKE_REPORT" "$CATALOG" "$MANIFEST" \
  "$CATALOG_SHA" "$SCOPE_SHA" "$MANIFEST_SHA" "$OVERLAY_API_SHA" "$OVERLAY_BUILDER_SHA" \
  "$deps_installed" "$PORT" "$PY" "$runtime_mode" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(
    receipt_path,
    scope_path,
    smoke_path,
    catalog_path,
    manifest_path,
    catalog_sha,
    scope_sha,
    manifest_sha,
    overlay_api_sha,
    overlay_builder_sha,
    deps_installed,
    port,
    python_executable,
    dependency_runtime_mode,
) = sys.argv[1:]
smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
payload = {
    "format": "CANCERLNCATLAS_V32_WEBSITE_STAGING_CANDIDATE_RECEIPT_V1",
    "status": "PASS",
    "host": "149",
    "environment": "staging",
    "production_deployed": False,
    "release_ready": False,
    "server_port": int(port),
    "server_loopback_only": True,
    "server_stopped": True,
    "gpu_started": False,
    "training_started": False,
    "authority_data_modified": False,
    "sealed_test_read": False,
    "formal_single_cell_scope": "23_PLUS_10_TYPED_UNAVAILABLE",
    "full_33_single_cell_coverage_claimed": False,
    "catalog_path": str(Path(catalog_path).resolve()),
    "catalog_sha256": catalog_sha,
    "capability_scope_path": str(Path(scope_path).resolve()),
    "capability_scope_sha256": scope_sha,
    "manifest_path": str(Path(manifest_path).resolve()),
    "manifest_sha256": manifest_sha,
    "overlay_hashes": {
        "v32_staging_api": overlay_api_sha,
        "build_v32_staging_web_catalog": overlay_builder_sha or None,
    },
    "dependencies_installed_in_isolated_venv": deps_installed.lower() == "true",
    "python_executable": str(Path(python_executable).resolve()),
    "dependency_runtime_mode": dependency_runtime_mode,
    "smoke_report_path": str(Path(smoke_path).resolve()),
    "smoke_report_sha256": sha(Path(smoke_path)),
    "smoke_status": smoke.get("status"),
    "probe_count": smoke.get("probe_count"),
    "semantic_summary": smoke.get("semantic_summary"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
Path(receipt_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": payload["status"], "receipt": str(Path(receipt_path).resolve())}))
PY

sha256sum "$RECEIPT" | tee "$RECEIPT.sha256"
echo "STAGING_CANDIDATE_READY=$CATALOG"
echo "RELEASE_READY=false"
echo "PRODUCTION_DEPLOYED=false"
