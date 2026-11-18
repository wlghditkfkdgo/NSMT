"""MODI 2026-11-18: learning-rate selection of the long-input lr test (docs/MODI_PROJECT_LOG.md, pre-registered rules; step 2 of user
decision 11-18 "1. 진행  2. 1번에서 개선이 없으면 진행").

Phase A trained seed 42 of the three compared models (spk_ssa, nsmt4q_pop_lif_msn, nsmt4q_pop_flif_msn) on ETTh2 / ETTm2 x H96 / H720 at
inputs 336 and 720 with lr in GRID (suites flif-v1-ett-lr<L>-<lr>). The default lr 0.0005 is the seed-42 run of the input-length
suites flif-v1-ett-len<L> (reused).
Selection (validation only, test is never read here): per model and input length, s(lr) = mean over the four cells of
val_loss_best(lr) / val_loss_best(0.0005) - 1 (seed 42); the selected lr is the one with the lowest s; the default stays unless
another lr is strictly lower. A missing, duplicated or non-finite row, a wrong seq_len or a run folder of another lr raises.
Writes the selection JSON and, with --queue_b, the phase-B queue: seeds 7 / 2026 of every (model, length) whose selected lr is not the
default (ETTm2 first, input 720 first), into the same suite as its seed-42 run.
usage (classification folder): python select_lrlong.py [--out_json F] [--queue_b F] [--fc DIR]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
LENGTHS = (720, 336)
MODELS = ('spk_ssa', 'nsmt4q_pop_lif_msn', 'nsmt4q_pop_flif_msn')
DEFAULT = '0.0005'
GRID = ('5e-05', '0.0001', '0.00025')
ET = 'nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh'
MX = 'nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh'


def suite(L, lr):
    return f'flif-v1-ett-len{L}' if lr == DEFAULT else f'flif-v1-ett-lr{L}-{lr}'


def rows(fc, s, **sel):
    out = []
    for ln in open(os.path.join(fc, s, 'raw.txt')):
        if not ln.startswith('RESULT'):
            continue
        d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
        if all(d.get(k) == str(v) for k, v in sel.items()):
            out.append(d)
    return out


def val42(fc, L, lr, cond, ds, pl):
    r = rows(fc, suite(L, lr), ds=ds, pl=pl, cond=cond, seed=42)
    if len(r) != 1:
        raise RuntimeError(f'{suite(L, lr)} {cond} {ds} H{pl} seed 42: {len(r)} rows')
    d = r[0]
    if int(d.get('seq_len', -1)) != L:
        raise RuntimeError(f'{suite(L, lr)} {cond} {ds} H{pl}: seq_len {d.get("seq_len")} != {L}')
    if lr != DEFAULT and d.get('lr') != lr:
        raise RuntimeError(f'{suite(L, lr)} {cond} {ds} H{pl}: RESULT lr {d.get("lr")} != {lr}')
    if f'+lr+{lr}+' not in d['dir']:
        raise RuntimeError(f'{suite(L, lr)} {cond} {ds} H{pl}: run folder is not lr {lr}: {d["dir"]}')
    v = float(d['val_loss_best'])
    if not math.isfinite(v) or v <= 0:
        raise RuntimeError(f'{suite(L, lr)} {cond} {ds} H{pl}: val_loss_best {v}')
    return v


def select(fc):
    out = {}
    for L in LENGTHS:
        for c in MODELS:
            val = {lr: {f'{ds} H{pl}': val42(fc, L, lr, c, ds, pl) for ds, pl in CELLS} for lr in (DEFAULT,) + GRID}
            s = {lr: 100 * sum(val[lr][k] / val[DEFAULT][k] - 1 for k in val[DEFAULT]) / len(CELLS) for lr in val}
            best = min(GRID, key=lambda x: s[x])
            sel = best if s[best] < s[DEFAULT] else DEFAULT
            out[f'{c} L{L}'] = dict(cond=c, seq_len=L, selected=sel, rel_val_pct=s, val=val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fc', default=os.path.join(HERE, '..', 'forecasting', 'results'))
    ap.add_argument('--out_json', default=os.path.join(HERE, 'results', 'flif-v1-lrlong-selected.json'))
    ap.add_argument('--queue_b', default='')
    a = ap.parse_args()
    out = select(a.fc)
    json.dump(out, open(a.out_json, 'w'), indent=1)
    for k, v in out.items():
        print(f"{k}: selected {v['selected']}  " + '  '.join(f"{lr} {v['rel_val_pct'][lr]:+.2f} %" for lr in (DEFAULT,) + GRID))
    if a.queue_b:
        lines = []
        for L in LENGTHS:
            for ds in ('ETTm2', 'ETTh2'):
                for pl in (720, 96):
                    for s in (7, 2026):
                        for c in MODELS:
                            sel = out[f'{c} L{L}']['selected']
                            if sel != DEFAULT:
                                lines.append(f"COND={c} SUITE={suite(L, sel)} SEQ={L} LR={sel} bash {MX if c == 'spk_ssa' else ET} {ds} {pl} {s}")
        open(a.queue_b, 'w').write(''.join(x + '\n' for x in lines))
        print(f'{a.queue_b}: {len(lines)} jobs')
    print(a.out_json)


if __name__ == '__main__':
    main()
