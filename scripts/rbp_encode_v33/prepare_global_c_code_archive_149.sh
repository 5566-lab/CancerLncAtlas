#!/usr/bin/env bash
# Package only the final G2 trainer code on the required CPU host.
set -euo pipefail
test "$(hostname)" = 149
NEW=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1
CODE="$NEW/code"
ARCHIVE="$NEW/GLOBAL_G2_TRAIN_CODE.tar.gz"
test -s "$CODE/cc_hhgt/v32/training.py"
test -s "$CODE/cc_hhgt/v32/c_graph_overlay.py"
test -s "$CODE/scripts/v32_pipeline.py"
test ! -e "$ARCHIVE"
test ! -e "$NEW/GLOBAL_G2_TRAIN_CODE_READY.json"
python3 - "$NEW/CODE_ARCHIVE_PREFLIGHT.json" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
path = Path(sys.argv[1])
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps({"target_host":"149", "workload":"final_G2_code_archive_CPU",
                            "paid_gpu_allowed":False, "file_hashes_computed":False},indent=2)+"\n")
PY
temporary="$NEW/.GLOBAL_G2_TRAIN_CODE.tar.gz.$$.tmp"
trap 'rm -f -- "$temporary"' EXIT
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -czf "$temporary" -C "$CODE" cc_hhgt scripts/v32_pipeline.py
python3 - "$temporary" "$ARCHIVE" "$NEW/GLOBAL_G2_TRAIN_CODE_READY.json" <<'PY'
import json, os, socket, sys, tarfile
from pathlib import Path
assert socket.gethostname() == "149"
temporary, archive, receipt = map(Path, sys.argv[1:])
with tarfile.open(temporary, "r:gz") as handle:
    names = {member.name for member in handle.getmembers() if member.isfile()}
required = {"cc_hhgt/v32/training.py", "cc_hhgt/v32/c_graph_overlay.py",
            "cc_hhgt/v32/formal_graph.py", "scripts/v32_pipeline.py"}
if not required <= names: raise RuntimeError(f"Code archive missing: {sorted(required-names)}")
os.replace(temporary, archive)
result={"status":"PASS_GLOBAL_G2_TRAIN_CODE_READY", "target_host":"149",
        "path":str(archive), "bytes":archive.stat().st_size,
        "required_members":sorted(required), "file_hashes_computed":False}
receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,sort_keys=True))
PY
