#!/usr/bin/env bash
# make_queue_spk.sh (2026-10-29, exp/f-lif-pop-v3): queue files for the SpkMixer rerun (gpu_pool2.sh lines, no gpu argument).
#   smoke : 1 epoch, seed 42: 6 neurons x sdlin x iand + lif x {or, sdor} x {or, add}  (10 jobs)    -> rhythm-spk-smoke-<date>.txt
#   b1    : LIF x {or, sdor, sdlin} x {iand, or, add} x seeds 42/7/2026 (27 jobs)                    -> rhythm-spk-b1-<date>.txt
#   b2 CF : the other 5 neurons x {or, sdor, sdlin} x seeds with residual CF (45 jobs)               -> rhythm-spk-b2-<CF>-<date>.txt
#   c  CF : LIF linmx reference with residual CF, 3 seeds (3 jobs)                                    -> rhythm-spk-c-<CF>-<date>.txt
#   muxsmoke / mux : 10-30 fusion MUX, LIF x {muxor, muxlin} x residual OR (2 smoke jobs / 6 jobs)  -> rhythm-spk-{muxsmoke,mux}-<date>.txt
#   ettsmoke / ettrepro / ett : 10-31 ETT check of muxlin (4 smoke / 1 reproduction / 24 jobs)       -> ett-spkmux-*-<date>.txt
# usage: bash make_queue_spk.sh smoke|b1|b2|c [cf] | muxsmoke | mux | ettsmoke | ettrepro | ett
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="bash cinc_mixer_v1/scripts/run_rhythm_spk.sh"           # gpu_pool2.sh runs jobs from NSMT/
RUN_MUX="bash cinc_mixer_v1/scripts/run_rhythm_spk_mux.sh"   # 10-30: muxor / muxlin (run_rhythm_spk.sh is in use by the B2 / C pool)
RUN_ETT="bash cinc_mixer_v1/scripts/run_ett_spk.sh"            # 10-31: ETT forecasting (args: <DS> <PL> <seed>, the pool adds the gpu)
Q="$W/scripts/queues"; mkdir -p "$Q"; D=$(date +%Y%m%d)
stage=${1:?smoke|b1|b2|c|muxsmoke|mux|ettsmoke|ettrepro|ett}; cf=${2:-}
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
  *) echo "bad stage"; exit 1 ;;
esac
echo "$f: $(wc -l < "$f") jobs"
