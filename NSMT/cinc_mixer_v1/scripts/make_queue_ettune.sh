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
# env ET_CHAIN=x (2026-11-20 extension, PROJECT_LOG; user decisions 11-20): the same stepwise chain with the wider candidate lists
#   patch {4, 8*, 16, 24, 32} and wd {0, 0.005, 0.01, 0.02, 0.05, 0.06*} (0.1 dropped: outside the related-work range); selections in
#   and new runs to the suite flif-v1-ett-spkmux-ettune2 (under ET_DRY when set); a (cell, seed, configuration) that already has a RESULT
#   row in flif-v1-ett-spkmux-tune (default configuration), flif-v1-ett-spkmux-ettune or the new suite is not queued again, so the number
#   of jobs depends on the earlier selections (0 is possible); queue file ettune2-<stage>-<date>.txt. Without ET_CHAIN nothing changes.
# usage (from anywhere): bash make_queue_ettune.sh smoke|patch|dmodel|mlp|blocks|wd ; prints <queue file>:<jobs>
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage=${1:?smoke|patch|dmodel|mlp|blocks|wd}
case ${ET_CHAIN:-} in '') SU=flif-v1-ett-spkmux-ettune; QP=ettune ;; x) SU=flif-v1-ett-spkmux-ettune2; QP=ettune2 ;; *) echo "bad ET_CHAIN"; exit 1 ;; esac
if [ -n "${ET_DRY:-}" ]; then SEL="$ET_DRY/$SU"; Q="$ET_DRY/queues"; else SEL="$W/results/$SU"; Q="$W/scripts/queues"; fi
mkdir -p "$Q"; D=$(date +%Y%m%d)
f="$Q/$QP-$stage-$D.txt"
python3 - "$SEL" "$stage" "$f" "$D" "$W/results" <<'PYEOF'
import json, os, sys
sel_dir, stage, out, date, res = sys.argv[1:6]
CHAIN = os.environ.get('ET_CHAIN', '')
E = 'bash cinc_mixer_v1/scripts/run_ett_spk.sh'
SUITE = 'flif-v1-ett-spkmux-ettune'
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
STAGES = (('patch', 'PATCH', ('4', '8', '16')), ('dmodel', 'EMB', ('32', '64', '128', '256')), ('mlp', 'MLPR', ('0.5', '1', '2', '4')),
          ('blocks', 'BLOCKS', ('1', '2', '3')), ('wd', 'WD', ('0', '0.01', '0.06', '0.1')))
if CHAIN == 'x':                                                          # 2026-11-20 extension (user decisions 11-20)
    SUITE = 'flif-v1-ett-spkmux-ettune2'
    STAGES = (('patch', 'PATCH', ('4', '8', '16', '24', '32')), ('dmodel', 'EMB', ('32', '64', '128', '256')),
              ('mlp', 'MLPR', ('0.5', '1', '2', '4')), ('blocks', 'BLOCKS', ('1', '2', '3')),
              ('wd', 'WD', ('0', '0.005', '0.01', '0.02', '0.05', '0.06')))
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


def have_rows():
    """ET_CHAIN=x: (ds, pl, seed, configuration) with a RESULT row in the 11-01 default rows, flif-v1-ett-spkmux-ettune or the new suite."""
    have = set()
    for l in open(os.path.join(res, 'flif-v1-ett-spkmux-tune', 'raw.txt')):
        if not l.startswith('RESULT'):
            continue
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        ds = r.get('ds')
        if r.get('cond') == 'spkmx_lif_muxlin_iand' and ds in LR and r.get('tag') == f'spkmx_lif_muxlin_iand_lr{LR[ds]}_L1':
            have.add((ds, int(r['pl']), int(r['seed']), tuple(DEFAULT[ds][k] for k in ORDER)))
    for p in (os.path.join(res, 'flif-v1-ett-spkmux-ettune', 'raw.txt'), os.path.join(sel_dir, 'raw.txt')):
        for l in (open(p) if os.path.exists(p) else []):
            if l.startswith('RESULT'):
                r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
                have.add((r['ds'], int(r['pl']), int(r['seed']), tuple(r[f] for f in ('patch', 'emb', 'mlpr', 'blocks', 'wd'))))
    return have


lines = []
if stage == 'smoke' and CHAIN == 'x':
    sys.exit('smoke is a chain-1 stage (ET_CHAIN unset)')
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
    have = have_rows() if CHAIN == 'x' else set()
    for ds, pl in CELLS:
        for v in vals:
            if v == base[ds][knob]:
                continue
            for s in SEEDS:
                c = dict(base[ds], **{knob: v})
                if (ds, pl, s, tuple(c[q] for q in ORDER)) in have:
                    continue
                lines.append(line(f'SUITE={SUITE}', ds, pl, s, c))
    if CHAIN != 'x':
        want = {'patch': 24, 'dmodel': 36, 'mlp': 36, 'blocks': 24, 'wd': 36}[stage]
        assert len(lines) == want, (len(lines), want)
open(out, 'w').write(''.join(l + '\n' for l in lines))
print(f'{out}:{len(lines)}')
PYEOF
