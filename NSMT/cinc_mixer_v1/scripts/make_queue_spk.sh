#!/usr/bin/env bash
# make_queue_spk.sh (2026-10-29, exp/f-lif-pop-v3): queue files for the SpkMixer rerun (gpu_pool2.sh lines, no gpu argument).
#   smoke : 1 epoch, seed 42: 6 neurons x sdlin x iand + lif x {or, sdor} x {or, add}  (10 jobs)    -> rhythm-spk-smoke-<date>.txt
#   b1    : LIF x {or, sdor, sdlin} x {iand, or, add} x seeds 42/7/2026 (27 jobs)                    -> rhythm-spk-b1-<date>.txt
#   b2 CF : the other 5 neurons x {or, sdor, sdlin} x seeds with residual CF (45 jobs)               -> rhythm-spk-b2-<CF>-<date>.txt
#   c  CF : LIF linmx reference with residual CF, 3 seeds (3 jobs)                                    -> rhythm-spk-c-<CF>-<date>.txt
#   muxsmoke / mux : 10-30 fusion MUX, LIF x {muxor, muxlin} x residual OR (2 smoke jobs / 6 jobs)  -> rhythm-spk-{muxsmoke,mux}-<date>.txt
#   ettsmoke / ettrepro / ett : 10-31 ETT check of muxlin (4 smoke / 1 reproduction / 24 jobs)       -> ett-spkmux-*-<date>.txt
#   tunesmoke / mitrepro / mitbih / tune_lr / tune_layers / tune_res : 11-01 ETT tuning of muxlin / sdlin (+ Spikformer-style lr)
#         and MIT-BIH (5 smoke / 1 reproduction / 6 / 144 / 48 / 48 jobs; tune_layers / tune_res read selected_<stage>.json)
#   t2smoke / crepro / mtune_lr|layers|res / mfinal / ctune_lr|layers|res : 11-01 MIT-BIH and CinC tuning (later stages read
#         results/<suite>/selected_<stage>.json; mfinal retrains the tuned non-default settings on the whole training file)
# usage: bash make_queue_spk.sh smoke|b1|b2|c [cf] | muxsmoke | mux | ettsmoke | ettrepro | ett | tunesmoke | mitrepro | mitbih | tune_lr | tune_layers | tune_res | t2smoke | crepro | mtune_lr | mtune_layers | mtune_res | mfinal | ctune_lr | ctune_layers | ctune_res
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="bash cinc_mixer_v1/scripts/run_rhythm_spk.sh"           # gpu_pool2.sh runs jobs from NSMT/
RUN_MUX="bash cinc_mixer_v1/scripts/run_rhythm_spk_mux.sh"   # 10-30: muxor / muxlin (run_rhythm_spk.sh is in use by the B2 / C pool)
RUN_ETT="bash cinc_mixer_v1/scripts/run_ett_spk.sh"            # 10-31: ETT forecasting (args: <DS> <PL> <seed>, the pool adds the gpu)
RUN_MIT="bash cinc_mixer_v1/scripts/run_mitbih_spk.sh"         # 11-01: MIT-BIH (arg: <seed>, the pool adds the gpu)
Q="$W/scripts/queues"; mkdir -p "$Q"; D=$(date +%Y%m%d)
stage=${1:?smoke|b1|b2|c|muxsmoke|mux|ettsmoke|ettrepro|ett|tunesmoke|mitrepro|mitbih|tune_lr|tune_layers|tune_res|t2smoke|crepro|mtune_lr|mtune_layers|mtune_res|mfinal|ctune_lr|ctune_layers|ctune_res}; cf=${2:-}
case $stage in
  smoke) f="$Q/rhythm-spk-smoke-$D.txt"; : > "$f"
         for n in lif flif pop_lif pop_flif popg_lif popg_flif; do echo "EPOCHS=1 SUITE=smoke-rhythm-spk-$D COND=spk_${n}_sdlin_iand $RUN 42" >> "$f"; done
         for c in or sdor; do for r in or add; do echo "EPOCHS=1 SUITE=smoke-rhythm-spk-$D COND=spk_lif_${c}_${r} $RUN 42" >> "$f"; done; done ;;
  b1)    f="$Q/rhythm-spk-b1-$D.txt"; : > "$f"
         for s in 42 7 2026; do for c in or sdor sdlin; do for r in iand or add; do echo "SUITE=flif-v1-rhythm-spk-cinc COND=spk_lif_${c}_${r} $RUN $s" >> "$f"; done; done; done ;;
  b2)    [ -n "$cf" ] || { echo "b2 needs cf"; exit 1; }; f="$Q/rhythm-spk-b2-$cf-$D.txt"; : > "$f"
         for s in 42 7 2026; do for n in flif pop_lif pop_flif popg_lif popg_flif; do for c in or sdor sdlin; do echo "SUITE=flif-v1-rhythm-spk-cinc COND=spk_${n}_${c}_${cf} $RUN $s" >> "$f"; done; done; done ;;
  c)     [ -n "$cf" ] || { echo "c needs cf"; exit 1; }; f="$Q/rhythm-spk-c-$cf-$D.txt"; : > "$f"
         for s in 42 7 2026; do echo "SUITE=flif-v1-rhythm-spk-cinc COND=spk_lif_linmx_${cf} $RUN $s" >> "$f"; done ;;
  muxsmoke) f="$Q/rhythm-spk-muxsmoke-$D.txt"; : > "$f"                                     # 10-30 fusion MUX smoke (1 epoch)
         for c in muxor muxlin; do echo "EPOCHS=1 SUITE=smoke-rhythm-spk-mux-$D COND=spk_lif_${c}_or $RUN_MUX 42" >> "$f"; done ;;
  mux)   f="$Q/rhythm-spk-mux-$D.txt"; : > "$f"                                          # 10-30 fusion MUX: LIF x {muxor, muxlin} x OR residual
         for s in 42 7 2026; do for c in muxor muxlin; do echo "SUITE=flif-v1-rhythm-spk-mux-cinc COND=spk_lif_${c}_or $RUN_MUX $s" >> "$f"; done; done ;;
  ettsmoke) f="$Q/ett-spkmux-smoke-$D.txt"; : > "$f"                                    # 10-31 ETT check of muxlin: 1 epoch
         for c in muxlin sdlin; do for dp in "ETTh2 96" "ETTm2 720"; do echo "EPOCHS=1 SUITE=smoke-ett-spkmux-$D COND=spkmx_lif_${c}_or $RUN_ETT $dp 42" >> "$f"; done; done ;;
  ettrepro) f="$Q/ett-spkmux-repro-$D.txt"; : > "$f"                                    # 10-31 check t4: Spikformer-style reproduction
         echo "SUITE=check-ett-spkssa-$D COND=spk_ssa $RUN_ETT ETTh2 96 42" >> "$f" ;;
  ett)   f="$Q/ett-spkmux-$D.txt"; : > "$f"                                             # 10-31 ETT check: {muxlin, sdlin} x 4 cells x 3 seeds
         for dp in "ETTm2 720" "ETTh2 720" "ETTm2 96" "ETTh2 96"; do for s in 42 7 2026; do for c in muxlin sdlin; do
           echo "SUITE=flif-v1-ett-spkmux COND=spkmx_lif_${c}_or $RUN_ETT $dp $s" >> "$f"; done; done; done ;;
  tunesmoke) f="$Q/tune-mit-smoke-$D.txt"; : > "$f"                                       # 11-01: ETT tuning + MIT-BIH smoke (1 epoch)
         echo "EPOCHS=1 LR=0.0001 LAYERS=2 SUITE=smoke-ett-tune-$D COND=spkmx_lif_muxlin_add $RUN_ETT ETTh2 96 42" >> "$f"
         echo "EPOCHS=1 LR=0.0001 SUITE=smoke-ett-tune-$D COND=spk_ssa $RUN_ETT ETTh2 96 42" >> "$f"
         echo "EPOCHS=1 SUITE=smoke-ett-tune-$D COND=spkmx_lif_muxlin_or $RUN_ETT ETTh2 96 42" >> "$f"
         for c in muxlin sdlin; do echo "EPOCHS=1 SUITE=smoke-mitbih-spkmux-$D COND=spkmx_lif_${c}_or $RUN_MIT 42" >> "$f"; done ;;
  mitrepro) f="$Q/mitbih-repro-$D.txt"; : > "$f"                                          # 11-01 check k4: Spikformer-style MIT-BIH seed 42
         echo "SUITE=check-mitbih-spkssa-$D COND=spk_ssa $RUN_MIT 42" >> "$f" ;;
  mitbih) f="$Q/mitbih-spkmux-$D.txt"; : > "$f"                                           # 11-01 MIT-BIH: {muxlin, sdlin} x 3 seeds
         for s in 42 7 2026; do for c in muxlin sdlin; do echo "SUITE=flif-v1-mitbih-spkmux COND=spkmx_lif_${c}_or $RUN_MIT $s" >> "$f"; done; done ;;
  tune_lr) f="$Q/ett-tune-lr-$D.txt"; : > "$f"                                            # 11-01 ETT tuning S1: lr x {muxlin, sdlin, spk_ssa} x 4 cells
         for dp in "ETTm2 720" "ETTm2 96" "ETTh2 720" "ETTh2 96"; do for s in 42 7 2026; do for lr in 0.0001 0.00025 0.001 0.002; do
           for c in spkmx_lif_muxlin_or spkmx_lif_sdlin_or spk_ssa; do
             echo "LR=$lr SUITE=flif-v1-ett-spkmux-tune COND=$c $RUN_ETT $dp $s" >> "$f"; done; done; done; done ;;
  tune_layers|tune_res)                                                                   # 11-01 S2 / S3 on the earlier stages' selections
         SEL="$W/results/flif-v1-ett-spkmux-tune"; f="$Q/ett-$stage-$D.txt"; : > "$f"
         python3 - "$SEL" "$stage" "$RUN_ETT" "$f" <<'PYEOF'
import json, os, sys
sel, stage, run, out = sys.argv[1:5]
lr = json.load(open(os.path.join(sel, 'selected_lr.json')))['selected']
L0 = {'ETTh2': 1, 'ETTm2': 2}
lines = []
if stage == 'tune_layers':
    cand = {'ETTh2': (2, 3), 'ETTm2': (1, 3)}
    for ds, pl in (('ETTm2', 720), ('ETTm2', 96), ('ETTh2', 720), ('ETTh2', 96)):
        for s in (42, 7, 2026):
            for L in cand[ds]:
                for c in ('spkmx_lif_muxlin_or', 'spkmx_lif_sdlin_or'):
                    lines.append(f'LR={lr[c][ds]} LAYERS={L} SUITE=flif-v1-ett-spkmux-tune COND={c} {run} {ds} {pl} {s}')
else:
    lay = json.load(open(os.path.join(sel, 'selected_layers.json')))['selected']
    for ds, pl in (('ETTm2', 720), ('ETTm2', 96), ('ETTh2', 720), ('ETTh2', 96)):
        for s in (42, 7, 2026):
            for cf in ('iand', 'add'):
                for c in ('spkmx_lif_muxlin_or', 'spkmx_lif_sdlin_or'):
                    lines.append(f'LR={lr[c][ds]} LAYERS={lay[c][ds]} SUITE=flif-v1-ett-spkmux-tune COND={c[:-3]}_{cf} {run} {ds} {pl} {s}')
open(out, 'w').write('\n'.join(lines) + '\n')
PYEOF
         ;;
  t2smoke) f="$Q/tune2-smoke-$D.txt"; : > "$f"                                            # 11-01 MIT-BIH / CinC tuning smoke (1 epoch)
         echo "EPOCHS=1 VAL_FOLD=0 LR=0.005 MLAYERS=3 SUITE=smoke-mitbih-tune-$D COND=spkmx_lif_muxlin_or $RUN_MIT 42" >> "$f"
         echo "EPOCHS=1 SUITE=smoke-mitbih-tune-$D COND=spkmx_lif_muxlin_add $RUN_MIT 42" >> "$f"
         echo "EPOCHS=1 LR=0.0005 SUITE=smoke-cinc-tune-$D COND=spk_ssa $RUN_MUX 42" >> "$f"
         echo "EPOCHS=1 MLAYERS=1 SUITE=smoke-cinc-tune-$D COND=spk_lif_muxlin_or $RUN_MUX 42" >> "$f" ;;
  crepro) f="$Q/cinc-repro-$D.txt"; : > "$f"                                              # 11-01 check c4: Spikformer-style CinC seed 42
         echo "SUITE=check-cinc-spkssa-$D COND=spk_ssa $RUN_MUX 42" >> "$f" ;;
  mtune_lr) f="$Q/mitbih-tune-lr-$D.txt"; : > "$f"                                        # 11-01 MIT-BIH tuning S1 (validation fold 0)
         for s in 42 7 2026; do for lr in 0.0025 0.005 0.01 0.02; do for c in spkmx_lif_muxlin_or spkmx_lif_sdlin_or spk_ssa; do
           echo "VAL_FOLD=0 LR=$lr SUITE=flif-v1-mitbih-spkmux-tune COND=$c $RUN_MIT $s" >> "$f"; done; done; done ;;
  ctune_lr) f="$Q/cinc-tune-lr-$D.txt"; : > "$f"                                          # 11-01 CinC tuning S1 (lr 1e-3 runs reused)
         for s in 42 7 2026; do for lr in 0.00025 0.0005 0.002; do for c in spk_lif_muxlin_or spk_lif_sdlin_or spk_ssa; do
           echo "LR=$lr SUITE=flif-v1-rhythm-spk-tune-cinc COND=$c $RUN_MUX $s" >> "$f"; done; done; done ;;
  mtune_layers|mtune_res|mfinal|ctune_layers|ctune_res)                                    # 11-01 later stages: read the selections
         f="$Q/$stage-$D.txt"; : > "$f"
         python3 - "$W/results" "$stage" "$RUN_MIT" "$RUN_MUX" "$f" <<'PYEOF'
import json, os, sys
res, stage, run_mit, run_cinc, out = sys.argv[1:6]
mit = stage.startswith('m')
suite = 'flif-v1-mitbih-spkmux-tune' if mit else 'flif-v1-rhythm-spk-tune-cinc'
sel = lambda s: json.load(open(os.path.join(res, suite, f'selected_{s}.json')))['selected']
pref = 'spkmx_lif_' if mit else 'spk_lif_'
L0, LR0 = 2, (0.01 if mit else 0.001)
run = run_mit if mit else run_cinc
vf = 'VAL_FOLD=0 ' if mit else ''
lines = []
for s in (42, 7, 2026):
    for comb in ('muxlin', 'sdlin'):
        ck = f'{pref}{comb}_or'
        if stage.endswith('layers'):
            for L in (1, 3):
                lines.append(f'{vf}LR={sel("lr")[ck]} MLAYERS={L} SUITE={suite} COND={ck} {run} {s}')
        elif stage.endswith('res'):
            lr, L = sel('lr')[ck], sel('layers')[ck]
            for cf in ('iand', 'add'):
                if not mit and comb == 'sdlin' and float(lr) == LR0 and int(L) == L0:
                    continue                                   # B1 has sdlin IAND / ADD at the default lr and layers (reused)
                lines.append(f'{vf}LR={lr} MLAYERS={L} SUITE={suite} COND={pref}{comb}_{cf} {run} {s}')
    if stage == 'mfinal':                                      # retrain the tuned settings on the whole training file
        for comb in ('muxlin', 'sdlin', 'ssa'):
            ck = 'spk_ssa' if comb == 'ssa' else f'{pref}{comb}_or'
            lr = sel('lr')[ck]
            L = L0 if comb == 'ssa' else sel('layers')[ck]
            cf = '-' if comb == 'ssa' else sel('res')[ck]
            if float(lr) == LR0 and int(L) == L0 and cf in ('-', 'or'):
                continue                                       # the default: the 11-01 runs / the reused Spikformer-style rows
            cond = 'spk_ssa' if comb == 'ssa' else f'{pref}{comb}_{cf}'
            lines.append(f'LR={lr} MLAYERS={L} SUITE={suite} COND={cond} {run} {s}')
open(out, 'w').write('\n'.join(lines) + ('\n' if lines else ''))
PYEOF
         ;;
  *) echo "bad stage"; exit 1 ;;
esac
echo "$f: $(wc -l < "$f") jobs"
