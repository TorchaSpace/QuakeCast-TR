#!/bin/bash
# Verilen yapılandırmaları sırayla, yakınsayana kadar uydurur (her çağrı ~165 s bütçe).
# Kullanım: bash src/etas/run_fits.sh configs/a.json configs/b.json ...
start=$(date +%s)
for cfg in "$@"; do
  out=$(python3 -c "import json;print(json.load(open('$cfg'))['out_dir'])")
  if [ -f "$out/durum.json" ] && grep -q '"yakinsadi": true' "$out/durum.json"; then continue; fi
  left=$((150 - $(date +%s) + start))
  [ $left -lt 40 ] && { echo "SURE DOLDU"; exit 0; }
  BUDGET=$((left - 25)) timeout $left python3 src/etas/fit_etas.py "$cfg" 2>&1 | grep "^iter\|YAKIN" | tail -n 1 | sed "s|^|$cfg: |"
done
echo "TUMU BITTI"
