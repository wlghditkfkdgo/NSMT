#!/usr/bin/env bash
# mem_all.sh <out> <stop-file>: every 2 s, per-process GPU memory of all compute processes with the run tag of each train_forecast.py
# process (2026-11-20, check-ettune); ends when <stop-file> exists. Peak per tag: awk '{if($4>m[$3])m[$3]=$4} END{for(t in m)print m[t],t}'
OUT=$1; STOP=$2; : > "$OUT"
while [ ! -e "$STOP" ]; do
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r p m; do
    t=$(ps -o args= -p "$p" 2>/dev/null | grep -oE -- '--tag [^ ]+ seed[0-9]+' | tr ' ' '_')
    [ -n "$t" ] && echo "$(date +%T) $p ${t#--tag_} $m"
  done >> "$OUT"
  sleep 2
done
