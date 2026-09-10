#!/bin/bash
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT" || exit 1

bash scripts/PSM.sh
bash scripts/SMAP.sh
bash scripts/MSL.sh 
bash scripts/SMD.sh # 다시 실험 해야함 -> original
bash scripts/SWaT.sh