#!/usr/bin/env bash
set -euo pipefail
umask 027

root="${1:-}"
expected_root=./data/CancerLncAtlas
if [[ "$(readlink -f "$root")" != "$expected_root" ]]; then
  echo "Refusing output root outside $expected_root: $root" >&2
  exit 2
fi

run_id=single_cell_rescue_20260828
source_tar=./data/CancerLncAtlas/raw/single_cell_23/GSE299623/GSE299623_RAW.tar
stage="$root/staging/$run_id/UCS_GSE299623_without_GSM9042132_CS6"
manifest="$root/manifests/$run_id"
mkdir -p "$stage" "$manifest"

# Validate all member names before extraction.  This rejects absolute paths,
# parent traversal, subdirectories, and unexpected characters.
tar -tf "$source_tar" > "$manifest/UCS_GSE299623_tar_members.txt"
if grep -Ev '^[A-Za-z0-9_.-]+$' "$manifest/UCS_GSE299623_tar_members.txt" > "$manifest/UCS_GSE299623_rejected_member_names.txt"; then
  echo "Unsafe tar member name detected; refusing extraction" >&2
  exit 3
fi

# The official GEO sample table maps GSM9042132 to CS6.  Tar members use the
# prefix GSM9042132_A2_1, so exclusion is accession-based and fail-closed.
if ! grep -q '^GSM9042132_' "$manifest/UCS_GSE299623_tar_members.txt"; then
  echo "Expected GSM9042132/CS6 members are absent; refusing ambiguous staging" >&2
  exit 4
fi
if [[ "$(grep -c '^GSM9042132_' "$manifest/UCS_GSE299623_tar_members.txt")" -ne 3 ]]; then
  echo "Expected exactly three GSM9042132/CS6 members" >&2
  exit 5
fi

tar -xf "$source_tar" -C "$stage" --exclude='GSM9042132_*'
find "$stage" -maxdepth 1 -type f -printf '%f\t%s\n' | sort > "$manifest/UCS_GSE299623_staged_files.tsv"
if find "$stage" -maxdepth 1 -type f -name 'GSM9042132_*' | grep -q .; then
  echo "CS6 exclusion failed" >&2
  exit 6
fi

source_members=$(wc -l < "$manifest/UCS_GSE299623_tar_members.txt")
staged_members=$(find "$stage" -maxdepth 1 -type f | wc -l)
excluded_members=$((source_members - staged_members))
if [[ "$excluded_members" -ne 3 ]]; then
  echo "Expected three excluded CS6 files; observed $excluded_members" >&2
  exit 7
fi

sha256sum "$source_tar" > "$manifest/UCS_GSE299623_source_tar.sha256"
find "$stage" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$manifest/UCS_GSE299623_staged_files.sha256"
cat > "$manifest/UCS_GSE299623_EXCLUSION_GATE.txt" <<EOF
status=PASS
source_accession=GSE299623
excluded_sample_accession=GSM9042132
excluded_source_label=CS6
excluded_tar_prefix=GSM9042132_A2_1
source_members=$source_members
staged_members=$staged_members
excluded_members=$excluded_members
raw_source_deleted=false
source_policy=PUBLIC8_READ_ONLY
EOF
sha256sum "$manifest/UCS_GSE299623_EXCLUSION_GATE.txt"
