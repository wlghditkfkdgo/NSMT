#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_ett.sh" --suite ett-tcn-h96-20260914 --pred_len 96 "$@"
bash "$SCRIPT_DIR/run_ett.sh" --suite ett-tcn-h720-20260914 --pred_len 720 "$@"
