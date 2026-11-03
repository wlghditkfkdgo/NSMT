#!/usr/bin/env bash
# make_queue_hm.sh (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG H1 pre-registration): queue files of the H1 head runs (head_mem.py:
# the head also reads the last neuron layer's charge; runners get HEADMEM=1). gpu_pool2.sh lines without the gpu argument, run from
# NSMT/. Each task uses its tuned settings (the selected_*.json of its tuning suite) for muxlin and the Spikformer-style model,
# seeds 42 / 7 / 2026; one suite per task (flif-v1-headmem-<task>, so the per-task energy recounts read one task).
#   hm_cinc    CinC fold 0                                              (6)
#   hm_ett     ETTh2-96 + ETTm2-96 (user choice 11-03)                  (12)
#   hm_mitbih  MIT-BIH, whole training file, last-epoch weights tested  (6)
#   hm_ad      MSL / SMAP / PSM, whole training file (TSLib)            (18)
# usage: bash make_queue_hm.sh hm_cinc|hm_ett|hm_mitbih|hm_ad
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="$W/scripts/queues"; mkdir -p "$Q"; D=$(date +%Y%m%d)
stage=${1:?hm_cinc|hm_ett|hm_mitbih|hm_ad}
f="$Q/$stage-$D.txt"; : > "$f"
python3 - "$W/results" "$stage" "$f" <<'PYEOF'
import json, os, sys
res, stage, out = sys.argv[1:4]
def sel(suite):
    return {s: json.load(open(os.path.join(res, suite, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
SEEDS = (42, 7, 2026)
lines = []
if stage == 'hm_cinc':
    s, run = sel('flif-v1-rhythm-spk-tune-cinc'), 'bash cinc_mixer_v1/scripts/run_rhythm_spk_mux.sh'
    m = 'spk_lif_muxlin_or'
    for seed in SEEDS:
        lines.append(f"HEADMEM=1 SUITE=flif-v1-headmem-cinc COND=spk_lif_muxlin_{s['res'][m]} LR={s['lr'][m]:g} MLAYERS={s['layers'][m]} {run} {seed}")
    for seed in SEEDS:
        lines.append(f"HEADMEM=1 SUITE=flif-v1-headmem-cinc COND=spk_ssa LR={s['lr']['spk_ssa']:g} {run} {seed}")
elif stage == 'hm_ett':
    s, run = sel('flif-v1-ett-spkmux-tune'), 'bash cinc_mixer_v1/scripts/run_ett_spk.sh'
    m = 'spkmx_lif_muxlin_or'
    for ds in ('ETTm2', 'ETTh2'):
        for seed in SEEDS:
            lines.append(f"HEADMEM=1 SUITE=flif-v1-headmem-ett COND=spkmx_lif_muxlin_{s['res'][m][ds]} LR={s['lr'][m][ds]:g} "
                         f"LAYERS={s['layers'][m][ds]} {run} {ds} 96 {seed}")
    for ds in ('ETTm2', 'ETTh2'):
        for seed in SEEDS:                                          # the runner's default LAYERS, as the tuned control rows
            lines.append(f"HEADMEM=1 SUITE=flif-v1-headmem-ett COND=spk_ssa LR={s['lr']['spk_ssa'][ds]:g} {run} {ds} 96 {seed}")
elif stage == 'hm_mitbih':
    s, run = sel('flif-v1-mitbih-spkmux-tune'), 'bash cinc_mixer_v1/scripts/run_mitbih_spk.sh'
    m = 'spkmx_lif_muxlin_or'
    for seed in SEEDS:
        lines.append(f"HEADMEM=1 SUITE=flif-v1-headmem-mitbih COND=spkmx_lif_muxlin_{s['res'][m]} LR={s['lr'][m]:g} MLAYERS={s['layers'][m]} {run} {seed}")
    for seed in SEEDS:
        lines.append(f"HEADMEM=1 SUITE=flif-v1-headmem-mitbih COND=spk_ssa LR={s['lr']['spk_ssa']:g} {run} {seed}")
elif stage == 'hm_ad':
    s, run = sel('flif-v1-ad-spkmux-tune'), 'bash cinc_mixer_v1/scripts/run_ad_spk.sh'
    m = 'spkmx_lif_muxlin_or'
    for ds in ('SMAP', 'PSM', 'MSL'):
        for seed in SEEDS:
            lines.append(f"HEADMEM=1 HOLDOUT=0 SUITE=flif-v1-headmem-ad COND=spkmx_lif_muxlin_{s['res'][m][ds]} LR={s['lr'][m][ds]:g} "
                         f"MLAYERS={s['layers'][m][ds]} {run} {ds} {seed}")
            lines.append(f"HEADMEM=1 HOLDOUT=0 SUITE=flif-v1-headmem-ad COND=spk_ssa LR={s['lr']['spk_ssa'][ds]:g} MLAYERS=2 {run} {ds} {seed}")
else:
    raise SystemExit(f'bad stage {stage}')
open(out, 'w').write(''.join(x + '\n' for x in lines))
PYEOF
echo "$f: $(grep -c . "$f") jobs"
