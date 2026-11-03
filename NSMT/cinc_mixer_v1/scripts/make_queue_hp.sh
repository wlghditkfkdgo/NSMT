#!/usr/bin/env bash
# make_queue_hp.sh (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG head input form pre-registration): queue files of the head input form
# runs (head_pool.py: a zero-initialised head term on another input form of the last block; runners get HEADPOOL=<form>). gpu_pool2.sh
# lines without the gpu argument, run from NSMT/. Each task uses its tuned settings (the selected_*.json of its tuning suite) for
# muxlin and the Spikformer-style model, seeds 42 / 7 / 2026, forms max / seg4 / layers / attn (attn = reference only).
#   hp_smoke   1-epoch smoke: flag off (suite smoke-hp-off-<date>, 4; same lines as the H1 flag-off smoke) and every form
#              (smoke-hp-on-<date>, 16), CinC + MIT-BIH, seed 42
#   hp_cinc    CinC fold 0 (24; muxlin first: the long runs start first)
#   hp_mitbih  MIT-BIH, whole training file, last-epoch weights tested (24)
# usage: bash make_queue_hp.sh hp_smoke|hp_cinc|hp_mitbih
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="$W/scripts/queues"; mkdir -p "$Q"; D=$(date +%Y%m%d)
stage=${1:?hp_smoke|hp_cinc|hp_mitbih}
f="$Q/$stage-$D.txt"; : > "$f"
python3 - "$W/results" "$stage" "$f" "$D" <<'PYEOF'
import json, os, sys
res, stage, out, date = sys.argv[1:5]
def sel(suite):
    return {s: json.load(open(os.path.join(res, suite, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
SEEDS, FORMS = (42, 7, 2026), ('max', 'seg4', 'layers', 'attn')
RR, RM = 'bash cinc_mixer_v1/scripts/run_rhythm_spk_mux.sh', 'bash cinc_mixer_v1/scripts/run_mitbih_spk.sh'
c, m = sel('flif-v1-rhythm-spk-tune-cinc'), sel('flif-v1-mitbih-spkmux-tune')
cm, mm = 'spk_lif_muxlin_or', 'spkmx_lif_muxlin_or'
def cinc(env, seed):
    return [f"{env} COND=spk_lif_muxlin_{c['res'][cm]} LR={c['lr'][cm]:g} MLAYERS={c['layers'][cm]} {RR} {seed}",
            f"{env} COND=spk_ssa LR={c['lr']['spk_ssa']:g} {RR} {seed}"]
def mitbih(env, seed):
    return [f"{env} COND=spkmx_lif_muxlin_{m['res'][mm]} LR={m['lr'][mm]:g} MLAYERS={m['layers'][mm]} {RM} {seed}",
            f"{env} COND=spk_ssa LR={m['lr']['spk_ssa']:g} {RM} {seed}"]
lines = []
if stage == 'hp_smoke':
    lines += cinc(f'EPOCHS=1 SUITE=smoke-hp-off-{date}', 42) + mitbih(f'EPOCHS=1 SUITE=smoke-hp-off-{date}', 42)
    for form in FORMS:
        env = f'EPOCHS=1 HEADPOOL={form} SUITE=smoke-hp-on-{date}'
        lines += cinc(env, 42) + mitbih(env, 42)
elif stage == 'hp_cinc':
    rows = [cinc(f'HEADPOOL={form} SUITE=flif-v1-headpool-cinc', seed) for form in FORMS for seed in SEEDS]
    lines = [r[0] for r in rows] + [r[1] for r in rows]                # muxlin (long) runs first
elif stage == 'hp_mitbih':
    rows = [mitbih(f'HEADPOOL={form} SUITE=flif-v1-headpool-mitbih', seed) for form in FORMS for seed in SEEDS]
    lines = [r[0] for r in rows] + [r[1] for r in rows]
else:
    raise SystemExit(f'bad stage {stage}')
open(out, 'w').write(''.join(x + '\n' for x in lines))
PYEOF
echo "$f: $(grep -c . "$f") jobs"
