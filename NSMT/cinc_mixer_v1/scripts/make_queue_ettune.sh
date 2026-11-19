#!/usr/bin/env bash
# make_queue_ettune.sh (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning pre-registration; user decisions 11-19: plan C + patch length,
# LIF SpkMixer only (mixer-only population not adopted, results/flif-v1-ett-spkmux-mxpop/selected_structure.json), SF-de not tuned).
# Stepwise tuning of the LIF SpkMixer (muxlin + IAND, 1 mixer layer per direction, lr ETTh2 5e-4 / ETTm2 1e-4, input 96), one knob per
# stage, the winners of the earlier stages carried over (per dataset):
#   patch  PATCH  {4, 8*, 16}            dmodel EMB    {32, 64*, 128, 256}      mlp  MLPR {0.5, 1, 2, 4} (* ETTh2 0.5, ETTm2 1)
#   blocks BLOCKS {1*, 2, 3}             wd     WD     {0, 0.01, 0.06*, 0.1}    (* = current value; its rows are not re-run)
# Each stage queues the non-current values x 4 cells (ETTh2 / ETTm2 x H96 / H720) x seeds 42 / 7 / 2026 into the suite
# flif-v1-ett-spkmux-ettune (patch 24, dmodel 36, mlp 36, blocks 24, wd 36 = 156); slow cells first (ETTm2 H96, ETTm2 H720, ETTh2 H720,
# ETTh2 H96). A stage needs selected_<stage>.json of every earlier stage (summarize_ettune.py), else it stops (exit 1, no queue).
# Only non-current values reach the env (the runner treats a default value as unset anyway).
#   smoke  1 epoch, seed 42, H720 (suite smoke-ettune-<date>): per dataset the default, every non-default value alone and the largest
#          shape (PATCH=4 EMB=256 MLPR=4 BLOCKS=3) = 30 runs
# env ET_DRY=<folder>: read the selections from <folder>/flif-v1-ett-spkmux-ettune and write the queue to <folder>/queues (dry runs)
# usage (from anywhere): bash make_queue_ettune.sh smoke|patch|dmodel|mlp|blocks|wd ; prints <queue file>:<jobs>
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage=${1:?smoke|patch|dmodel|mlp|blocks|wd}
if [ -n "${ET_DRY:-}" ]; then SEL="$ET_DRY/flif-v1-ett-spkmux-ettune"; Q="$ET_DRY/queues"; else SEL="$W/results/flif-v1-ett-spkmux-ettune"; Q="$W/scripts/queues"; fi
mkdir -p "$Q"; D=$(date +%Y%m%d)
f="$Q/ettune-$stage-$D.txt"
python3 - "$SEL" "$stage" "$f" "$D" <<'PYEOF'
import json, os, sys
sel_dir, stage, out, date = sys.argv[1:5]
E = 'bash cinc_mixer_v1/scripts/run_ett_spk.sh'
SUITE = 'flif-v1-ett-spkmux-ettune'
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
STAGES = (('patch', 'PATCH', ('4', '8', '16')), ('dmodel', 'EMB', ('32', '64', '128', '256')), ('mlp', 'MLPR', ('0.5', '1', '2', '4')),
          ('blocks', 'BLOCKS', ('1', '2', '3')), ('wd', 'WD', ('0', '0.01', '0.06', '0.1')))
DEFAULT = {'ETTh2': {'PATCH': '8', 'EMB': '64', 'MLPR': '0.5', 'BLOCKS': '1', 'WD': '0.06'},
           'ETTm2': {'PATCH': '8', 'EMB': '64', 'MLPR': '1', 'BLOCKS': '1', 'WD': '0.06'}}
ORDER = ('PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD')
CELLS = (('ETTm2', 96), ('ETTm2', 720), ('ETTh2', 720), ('ETTh2', 96))           # slow cells first
SEEDS = (42, 7, 2026)


def env(ds, cfg):
    return ' '.join(f'{k}={cfg[k]}' for k in ORDER if cfg[k] != DEFAULT[ds][k])


def line(prefix, ds, pl, seed, cfg):
    kn = env(ds, cfg)
    return f"{prefix} COND=spkmx_lif_muxlin_iand LR={LR[ds]} LAYERS=1{' ' + kn if kn else ''} {E} {ds} {pl} {seed}"


lines = []
if stage == 'smoke':
    big = {'PATCH': '4', 'EMB': '256', 'MLPR': '4', 'BLOCKS': '3', 'WD': '0.06'}
    for ds in ('ETTm2', 'ETTh2'):
        p = f'EPOCHS=1 SUITE=smoke-ettune-{date}'
        lines.append(line(p, ds, 720, 42, big))
        for _, knob, vals in STAGES:
            for v in vals:
                if v != DEFAULT[ds][knob]:
                    lines.append(line(p, ds, 720, 42, dict(DEFAULT[ds], **{knob: v})))
        lines.append(line(p, ds, 720, 42, DEFAULT[ds]))
    assert len(lines) == 30, len(lines)
else:
    names = [s[0] for s in STAGES]
    if stage not in names:
        sys.exit(f'bad stage {stage}')
    k = names.index(stage)
    base = {ds: dict(DEFAULT[ds]) for ds in DEFAULT}
    for name, knob, vals in STAGES[:k]:                                   # winners of the earlier stages
        p = os.path.join(sel_dir, f'selected_{name}.json')
        if not os.path.exists(p):
            sys.exit(f'STOP: {p} missing (stage {name} not selected yet)')
        s = json.load(open(p))
        for ds in DEFAULT:
            v = str(s['selected'][ds])
            if v not in vals:
                sys.exit(f'STOP: {p} selects {v} for {ds}, not in {vals}')
            base[ds][knob] = v
    _, knob, vals = STAGES[k]
    for ds, pl in CELLS:
        for v in vals:
            if v == base[ds][knob]:
                continue
            for s in SEEDS:
                lines.append(line(f'SUITE={SUITE}', ds, pl, s, dict(base[ds], **{knob: v})))
    want = {'patch': 24, 'dmodel': 36, 'mlp': 36, 'blocks': 24, 'wd': 36}[stage]
    assert len(lines) == want, (len(lines), want)
open(out, 'w').write('\n'.join(lines) + '\n')
print(f'{out}:{len(lines)}')
PYEOF
