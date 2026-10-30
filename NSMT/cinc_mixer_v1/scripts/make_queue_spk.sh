#!/usr/bin/env bash
# make_queue_spk.sh (2026-10-29, exp/f-lif-pop-v3): queue files for the SpkMixer rerun (gpu_pool2.sh lines, no gpu argument).
#   smoke : 1 epoch, seed 42: 6 neurons x sdlin x iand + lif x {or, sdor} x {or, add}  (10 jobs)    -> rhythm-spk-smoke-<date>.txt
#   b1    : LIF x {or, sdor, sdlin} x {iand, or, add} x seeds 42/7/2026 (27 jobs)                    -> rhythm-spk-b1-<date>.txt
#   b2 CF : the other 5 neurons x {or, sdor, sdlin} x seeds with residual CF (45 jobs)               -> rhythm-spk-b2-<CF>-<date>.txt
#   c  CF : LIF linmx reference with residual CF, 3 seeds (3 jobs)                                    -> rhythm-spk-c-<CF>-<date>.txt
#   muxsmoke / mux : 10-30 fusion MUX, LIF x {muxor, muxlin} x residual OR (2 smoke jobs / 6 jobs)  -> rhythm-spk-{muxsmoke,mux}-<date>.txt
# usage: bash make_queue_spk.sh smoke|b1|b2|c [cf] | muxsmoke | mux
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="bash cinc_mixer_v1/scripts/run_rhythm_spk.sh"           # gpu_pool2.sh runs jobs from NSMT/
RUN_MUX="bash cinc_mixer_v1/scripts/run_rhythm_spk_mux.sh"   # 10-30: muxor / muxlin (run_rhythm_spk.sh is in use by the B2 / C pool)
Q="$W/scripts/queues"; mkdir -p "$Q"; D=$(date +%Y%m%d)
stage=${1:?smoke|b1|b2|c|muxsmoke|mux}; cf=${2:-}
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
  *) echo "bad stage"; exit 1 ;;
esac
echo "$f: $(wc -l < "$f") jobs"
