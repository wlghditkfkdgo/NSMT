#!/bin/bash

# Run from the shared dataset directory, independent of caller location.
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT/dataset" || exit 1

# Unzip selected dataset archives
# unzip MSL.zip
# unzip PSM.zip
# unzip SMAP.zip
# unzip SMD.zip
unzip SWaT.zip
