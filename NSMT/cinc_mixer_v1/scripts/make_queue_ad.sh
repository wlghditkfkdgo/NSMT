#!/usr/bin/env bash
# make_queue_ad.sh (2026-11-02, exp/f-lif-pop-v3; PROJECT_LOG AD pre-registration): queue files of the anomaly-detection tuning
# (gpu_pool2.sh lines, no gpu argument; the pool runs them from NSMT/). Datasets MSL / SMAP / PSM (user decision 11-02: the three
# small ones first), seeds 42 / 7 / 2026, suite flif-v1-ad-spkmux-tune. Later stages read results/<suite>/selected_<stage>.json
# ({condition: {dataset: value}}, written by summarize_ad_tune.py).
#   adsmoke      1 epoch, seed 42: {muxlin, sdlin, Spikformer-style} x 3 datasets x {holdout, whole file}        (18 jobs)
#   adsmoke2     1 epoch, seed 42, whole file: {muxlin, sdlin, Spikformer-style, muxlin ADD} x 3 datasets          (12 jobs)
#   adtune_lr    S1 holdout: lr {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3} x {muxlin, sdlin, Spikformer-style} (OR, 2 layers)  (135)
#   adtune_layers S2 holdout: mixer layers per direction {1, 3} (2* = S1 runs) x {muxlin, sdlin} at the selected lr   (36)
#   adtune_res   S3 holdout: residual {IAND, ADD} (OR* = earlier runs) at the selected lr and layers                  (36)
#   adfinal      whole training file (TSLib), tuned settings of the 3 conditions + muxlin at the defaults           (<= 36)
# usage: bash make_queue_ad.sh adsmoke|adtune_lr|adtune_layers|adtune_res|adfinal
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="bash cinc_mixer_v1/scripts/run_ad_spk.sh"                 # args: <DS> <seed>, the pool adds the gpu
Q="$W/scripts/queues"; mkdir -p "$Q"; D=$(date +%Y%m%d)
SUITE=flif-v1-ad-spkmux-tune
stage=${1:?adsmoke|adtune_lr|adtune_layers|adtune_res|adfinal}
f="$Q/$stage-$D.txt"; : > "$f"
case $stage in
  adsmoke)
    for ds in SMAP PSM MSL; do for ho in 1 0; do for c in spkmx_lif_muxlin_or spkmx_lif_sdlin_or spk_ssa; do
      echo "EPOCHS=1 HOLDOUT=$ho SUITE=smoke-ad-spkmux-$D COND=$c $RUN $ds 42" >> "$f"; done; done; done ;;
  adsmoke2)                                                    # whole-file smoke again after the head-count fix (+ ADD)
    for ds in SMAP PSM MSL; do for c in spkmx_lif_muxlin_or spkmx_lif_sdlin_or spk_ssa spkmx_lif_muxlin_add; do
      echo "EPOCHS=1 HOLDOUT=0 SUITE=smoke-ad-spkmux-${D}b COND=$c $RUN $ds 42" >> "$f"; done; done ;;
  adtune_lr)
    for ds in SMAP PSM MSL; do for s in 42 7 2026; do for lr in 0.0001 0.00025 0.0005 0.001 0.002; do
      for c in spkmx_lif_muxlin_or spkmx_lif_sdlin_or spk_ssa; do
        echo "HOLDOUT=1 LR=$lr SUITE=$SUITE COND=$c $RUN $ds $s" >> "$f"; done; done; done; done ;;
  adtune_layers|adtune_res|adfinal)
    python3 - "$W/results/$SUITE" "$stage" "$RUN" "$SUITE" "$f" <<'PYEOF'
import json, os, sys
res, stage, run, suite, out = sys.argv[1:6]
sel = lambda s: json.load(open(os.path.join(res, f'selected_{s}.json')))['selected']
DS, SEEDS, LR0, L0, R0 = ('SMAP', 'PSM', 'MSL'), (42, 7, 2026), 0.0005, 2, 'or'
MS = ('spkmx_lif_muxlin_or', 'spkmx_lif_sdlin_or')
lines = []
lr = sel('lr')
if stage == 'adtune_layers':
    for ds in DS:
        for s in SEEDS:
            for c in MS:
                for L in (1, 3):
                    lines.append(f'HOLDOUT=1 LR={lr[c][ds]:g} MLAYERS={L} SUITE={suite} COND={c} {run} {ds} {s}')
elif stage == 'adtune_res':
    lay = sel('layers')
    for ds in DS:
        for s in SEEDS:
            for c in MS:
                for r in ('iand', 'add'):
                    lines.append(f'HOLDOUT=1 LR={lr[c][ds]:g} MLAYERS={lay[c][ds]} SUITE={suite} COND={c[:-3]}_{r} {run} {ds} {s}')
else:                                                          # adfinal: whole training file
    lay, rr = sel('layers'), sel('res')
    for ds in DS:
        cfgs = [(f'{c[:-3]}_{rr[c][ds]}', lr[c][ds], lay[c][ds]) for c in MS] + [('spk_ssa', lr['spk_ssa'][ds], L0)]
        dflt = ('spkmx_lif_muxlin_or', LR0, L0)
        if dflt != cfgs[0]:
            cfgs.append(dflt)                                  # A4: muxlin at the defaults (skipped when the tuned setting is it)
        for s in SEEDS:
            for c, l, L in cfgs:
                lines.append(f'HOLDOUT=0 LR={l:g} MLAYERS={L} SUITE={suite} COND={c} {run} {ds} {s}')
open(out, 'w').write(''.join(x + '\n' for x in lines))
PYEOF
    ;;
  *) echo "bad stage $stage"; exit 1 ;;
esac
echo "$f: $(grep -c . "$f") jobs"
