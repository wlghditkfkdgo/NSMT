#!/usr/bin/env bash
# Run this ON THE TARGET SERVER, from inside the copied NSMT directory.
# Runners derive their paths automatically; this links existing datasets.
set -eu
NEW_ROOT="$(cd "$(dirname "$0")/.." && pwd)"       # .../NSMT
REPO_ROOT="$(dirname "$NEW_ROOT")"                  # the Bio-inspired-... repo root
echo "[setup] NSMT root : $NEW_ROOT"
echo "[setup] repo root : $REPO_ROOT"

# Datasets: retain real directories already copied into NSMT.
mkdir -p "$NEW_ROOT/forecasting" "$NEW_ROOT/anomaly_detection"
for pair in "forecasting/dataset" "anomaly_detection/dataset"; do
  src="$REPO_ROOT/$pair"
  dst="$NEW_ROOT/$pair"
  if [ -d "$dst" ] && [ ! -L "$dst" ]; then
    echo "[setup] keeping existing dataset directory $dst"
  elif [ -e "$src" ]; then
    ln -sfnT "$src" "$dst" && echo "[setup] linked $dst -> $src"
  else
    echo "[setup] WARNING: $src not found -- set $dst manually"
  fi
done
for task in neorecall_v1/forecasting neorecall_v2/forecasting model_v1/forecasting neorecall_ad_v1/anomaly_detection; do
  task_kind="${task##*/}"
  dst="$NEW_ROOT/$task/dataset"
  if [ -d "$dst" ] && [ ! -L "$dst" ]; then
    echo "[setup] keeping existing dataset directory $dst"
  else
    ln -sfnT "../../$task_kind/dataset" "$dst"
  fi
done
echo "[setup] done. Run model scripts from <model>/<task>/scripts/."
