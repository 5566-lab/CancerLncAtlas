#!/usr/bin/env bash
set -u
R=${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1
echo "now: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== jobs ==="
if pgrep -f phase7_typed_authority.py >/dev/null; then
  echo "  authority build : RUNNING"
  for pid in $(pgrep -f 'phase7_typed_authority.py'); do
    ps -o pid,etime,time,%cpu,rss,stat,comm -p "$pid" 2>/dev/null | tail -1 | sed 's/^/    /'
  done
else
  echo "  authority build : FINISHED"
fi
pgrep -f _waiter.sh >/dev/null && echo "  waiter          : ARMED" || echo "  waiter          : idle/finished"
pgrep -f phase14_bundle_check.py >/dev/null && echo "  bundle check    : RUNNING"
echo
echo "=== folds: $(ls -d "$R"/outputs/phase7_typed_authority/fold_* 2>/dev/null | wc -l)/5 ==="
for i in 0 1 2 3 4 ; do
  f="$R/outputs/phase7_typed_authority/fold_$i/AUTHORITY_SUMMARY.json"
  if [ -f "$f" ]; then printf "  fold_%s  %s bytes  %s\n" "$i" "$(stat -c %s "$f")" "$(stat -c %y "$f" | cut -d. -f1)"; else printf "  fold_%s  (pending)\n" "$i"; fi
done
echo
echo "=== combined summary ==="
python3 - "$R" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "outputs/phase7_typed_authority/PHASE7_TYPED_AUTHORITY_SUMMARY.json"
if p.exists():
    s = json.loads(p.read_text())
    folds = s.get("folds") or {}
    print("  generation:", s.get("relation_schema_generation"), " folds recorded:", len(folds))
    for k in sorted(folds):
        i = folds[k]; v = i["variant_edge_counts"]
        print(f"    fold {k}: edges={i['edges']:,}  G0={v['G0']:,} G1={v['G1']:,} G2={v['G2']:,}"
              f"  binding={sum(i['binding_relations'].values()):,}")
PY
echo
echo "=== bundle check log (tail) ==="
tail -24 "$R/reports/PHASE14_BUNDLE_CHECK.log" 2>/dev/null || echo "  (not started)"
