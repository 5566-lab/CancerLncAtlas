#!/usr/bin/env bash
# Step E: the single materialisation of the v2 generation.
#
# Points phase7_typed_authority.py at the merged binding table and a new output
# directory, so the frozen pre-ENCODE generation is left untouched and the new one
# carries its own receipt.
set -u
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
V2_BINDING="$R/outputs/phase3_typed_binding_v2/lnc_protein_binding_typed_v2.parquet"
V2_OUT="$R/outputs/phase7_authority_v2"

if [ ! -f "$V2_BINDING" ]; then
  echo "FAIL-CLOSED: $V2_BINDING does not exist"
  exit 1
fi
if [ -d "$V2_OUT" ] && [ -n "$(ls -A "$V2_OUT" 2>/dev/null)" ]; then
  echo "FAIL-CLOSED: $V2_OUT already populated; refusing to overwrite a generation"
  exit 1
fi

echo "binding : $V2_BINDING"
echo "sha256  : $(sha256sum "$V2_BINDING" | cut -d' ' -f1)"
echo "output  : $V2_OUT"
echo "folds   : 0 1 2 3 4"
echo

export RBP_BINDING_PARQUET="$V2_BINDING"
export RBP_AUTHORITY_OUT="$V2_OUT"
bash /tmp/run_task.sh phase7_typed_authority.py 0 1 2 3 4
