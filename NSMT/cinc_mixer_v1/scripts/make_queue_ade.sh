#!/usr/bin/env bash
# make_queue_ade.sh (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG AD energy pre-registration; user request 11-19 "AD energy를 낮추는 실험을
# 하자", decision "(가) (나) 둘 다"): queue files (gpu_pool2.sh lines, no gpu argument; the pool runs them from NSMT/). SpkMixer muxlin,
# LIF in every layer, AD datasets MSL / SMAP / PSM, seeds 42 / 7 / 2026, suite flif-v1-ad-energy.
#   (A) mean pooling: the 11-03 tuned settings (results/flif-v1-ad-spkmux-tune/selected_{lr,layers,res}.json, muxlin) + MIXPOOL=mean
#   (B) 1 mixer layer per direction for every dataset: lr (OR) then residual re-selected on validation as on 11-03 (summarize_ade.py)
#   smoke   1 epoch, seed 42: (A) x 3 datasets whole file; (B) OR holdout x 3 datasets; (B) IAND whole file MSL           (7)
#   lrmean  (A) 9 whole-file runs + (B) lr {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3} OR holdout x 3 datasets x 3 seeds          (9 + 45)
#   res     (B) residual {IAND, ADD} holdout at the selected lr (results/flif-v1-ad-energy/selected_lr.json)              (18)
#   final   (B) whole file: the selected residual (selected_res.json "res"), plus the best 0/1 residual ("spike") where
#           the selected one is ADD                                                                                       (9-18)
# usage: bash make_queue_ade.sh smoke|lrmean|res|final ; prints <queue file>:<jobs>
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="bash cinc_mixer_v1/scripts/run_ad_spk.sh"                 # args: <DS> <seed>, the pool adds the gpu
Q="$W/scripts/queues"; mkdir -p "$Q"; D=20261119
SUITE=flif-v1-ad-energy
stage=${1:?smoke|lrmean|res|final}
f="$Q/ade-$stage-$D.txt"; : > "$f"
python3 - "$W/results" "$stage" "$RUN" "$SUITE" "$f" "$D" <<'PYEOF'
import json, os, sys
res, stage, run, suite, out, day = sys.argv[1:7]
DS, SEEDS, LRS = ('SMAP', 'PSM', 'MSL'), (42, 7, 2026), ('0.0001', '0.00025', '0.0005', '0.001', '0.002')
MK = 'spkmx_lif_muxlin_or'
t = lambda s: json.load(open(os.path.join(res, 'flif-v1-ad-spkmux-tune', f'selected_{s}.json')))['selected'][MK]
mine = lambda s: json.load(open(os.path.join(res, suite, f'selected_{s}.json')))
lines = []
if stage in ('smoke', 'lrmean'):
    lr0, lay0, res0 = t('lr'), t('layers'), t('res')
    A = lambda ds, s, e='', sui=suite: f'{e}HOLDOUT=0 LR={lr0[ds]:g} MLAYERS={lay0[ds]} MIXPOOL=mean SUITE={sui} COND=spkmx_lif_muxlin_{res0[ds]} {run} {ds} {s}'
if stage == 'smoke':
    sm = f'smoke-ade-{day}'
    lines += [A(ds, 42, 'EPOCHS=1 ', sm) for ds in DS]
    lines += [f'EPOCHS=1 HOLDOUT=1 LR=0.0005 MLAYERS=1 SUITE={sm} COND=spkmx_lif_muxlin_or {run} {ds} 42' for ds in DS]
    lines.append(f'EPOCHS=1 HOLDOUT=0 LR=0.0005 MLAYERS=1 SUITE={sm} COND=spkmx_lif_muxlin_iand {run} MSL 42')
elif stage == 'lrmean':
    lines += [A(ds, s) for ds in DS for s in SEEDS]
    lines += [f'HOLDOUT=1 LR={lr} MLAYERS=1 SUITE={suite} COND=spkmx_lif_muxlin_or {run} {ds} {s}' for ds in DS for s in SEEDS for lr in LRS]
elif stage == 'res':
    lr = mine('lr')['selected']
    lines += [f'HOLDOUT=1 LR={lr[ds]} MLAYERS=1 SUITE={suite} COND=spkmx_lif_muxlin_{r} {run} {ds} {s}' for ds in DS for s in SEEDS for r in ('iand', 'add')]
elif stage == 'final':
    lr, sel = mine('lr')['selected'], mine('res')['selected']
    for ds in DS:
        rs = [sel['res'][ds]] + ([sel['spike'][ds]] if sel['spike'][ds] != sel['res'][ds] else [])
        lines += [f'HOLDOUT=0 LR={lr[ds]} MLAYERS=1 SUITE={suite} COND=spkmx_lif_muxlin_{r} {run} {ds} {s}' for r in rs for s in SEEDS]
else:
    sys.exit(f'bad stage {stage}')
open(out, 'w').write(''.join(l + '\n' for l in lines))
PYEOF
echo "$f:$(wc -l < "$f")"
