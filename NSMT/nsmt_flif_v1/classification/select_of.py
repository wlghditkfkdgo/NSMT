"""MODI 2026-11-19: selections of the overfitting test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user request 11-19 "과적합에
대한 여러 조치를 취해서 해결한 상태에서의 test 값을 Spikformer(DE)와 비교해줘"). Validation only; test is never read here.

Models (ETT, input 96, every neuron SpikeDE): spk = Spikformer(DE) spkall_lif (run_ett_mixer.sh); pop_lif / pop_flif = the SpikeDE msn base
nsmt4q_<n>_msn_sde + the aux form chosen in stage S (run_ett_dual.sh). Current setting: lr 0.0005, head dropout 0, weight decay 0.06.
Stages (each writes the queue of the next one; the state goes to the selection JSON):
  S   per neuron, the aux form among off (flif-v1-ett-sde), low-pass (flif-v1-ett-aux, _aux), full signal (flif-v1-ett-auxf, _auxf):
      s(form) = mean over the 4 cells x 3 seeds of val(form) / val(off) - 1; the lowest, if strictly below off  -> queue A1
  A1  learning rate in {0.0001, 5e-05} (+ the current 0.0005)                                                     -> queue A2
  A2  head dropout in {0.1, 0.3} (+ 0), at the model's A1 setting                                                  -> queue A3
  A3  weight decay in {0.2, 0.5} (+ 0.06), at the model's A2 setting                                               -> queue B
  A-stages: seed 42; s(v) = mean over the 4 cells of val(v) / val(ref) - 1, ref = the previous stage's setting; the lowest new value
  if strictly below ref, else ref. New runs of stage X, value v go to suite flif-v1-ett-of-X-v (the run folder name has no dropout /
  weight decay). B: seeds 7 / 2026 of every model whose final setting is not the current one, in the suite of its final seed-42 run.
Fail-closed: a missing / duplicated row, a SKIP line of a needed run, a non-finite validation, a run folder without the SpikeDE suffix
_nalif or with another lr, a RESULT lr / wd / drop field that does not match the setting, a stage out of order or already done.
usage (classification folder): python select_of.py --stage S|A1|A2|A3 --queue_next FILE [--state FILE] [--fc DIR]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SEEDS = (42, 7, 2026)
MODELS = ('spk', 'pop_lif', 'pop_flif')
DEF = dict(lr='0.0005', drop='0', wd='0.06')
GRID = {'A1': ('lr', ('0.0001', '5e-05')), 'A2': ('drop', ('0.1', '0.3')), 'A3': ('wd', ('0.2', '0.5'))}
EDGE = {'lr': '5e-05', 'drop': '0.3', 'wd': '0.5'}
ORDER = ('S', 'A1', 'A2', 'A3')
VARIANTS = {'off': ('flif-v1-ett-sde', ''), 'aux': ('flif-v1-ett-aux', '_aux'), 'auxf': ('flif-v1-ett-auxf', '_auxf')}
MX = 'nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh'
ET = 'nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh'


def cond(model, variant):
    return 'spkall_lif' if model == 'spk' else f'nsmt4q_{model}_msn_sde{VARIANTS[variant][1]}'


def ref_suite(model, variant):
    return 'flif-v1-ett-sde' if model == 'spk' else VARIANTS[variant][0]


def stage_suite(stage, value):
    return f'flif-v1-ett-of-{stage}-{value}'


def read(fc, suite):
    path = os.path.join(fc, suite, 'raw.txt')
    res, skip = [], []
    for ln in open(path):
        if ln.startswith('RESULT'):
            res.append(dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t))
        elif ln.startswith('SKIP'):
            skip.append(ln)
    return res, skip


def row(fc, suite, c, ds, pl, seed, setting):
    res, skip = read(fc, suite)
    if any(f' cond={c} ' in s and f' ds={ds} ' in s and f' pl={pl} ' in s and f' seed={seed} ' in s for s in skip):
        raise RuntimeError(f'{suite} {c} {ds} H{pl} seed {seed}: SKIP line')
    r = [d for d in res if d.get('cond') == c and d.get('ds') == ds and d.get('pl') == str(pl) and d.get('seed') == str(seed)]
    if len(r) != 1:
        raise RuntimeError(f'{suite} {c} {ds} H{pl} seed {seed}: {len(r)} rows')
    d = r[0]
    if int(d.get('seq_len', 96)) != 96 or '_nalif' not in d['dir'] or f"+lr+{setting['lr']}+" not in d['dir']:
        raise RuntimeError(f'{suite} {c} {ds} H{pl} seed {seed}: seq_len / SpikeDE suffix / lr folder ({d["dir"]})')
    for k in ('lr', 'wd', 'drop'):
        want = setting[k] if setting[k] != DEF[k] else None
        if d.get(k) != want:
            raise RuntimeError(f'{suite} {c} {ds} H{pl} seed {seed}: RESULT {k}={d.get(k)} but the setting is {setting[k]}')
    v = float(d['val_loss_best'])
    if not math.isfinite(v) or v <= 0:
        raise RuntimeError(f'{suite} {c} {ds} H{pl} seed {seed}: val_loss_best {v}')
    return d


def env(setting):
    e = []
    if setting['lr'] != DEF['lr']:
        e.append(f"LR={setting['lr']}")
    if setting['drop'] != DEF['drop']:
        e.append(f"DROP={setting['drop']}")
    if setting['wd'] != DEF['wd']:
        e.append(f"WD={setting['wd']}")
    return ' '.join(e)


def line(model, variant, suite, setting, ds, pl, seed):
    pre = ' '.join(x for x in (f'COND={cond(model, variant)}', f'SUITE={suite}', env(setting)) if x)
    return f"{pre} bash {MX if model == 'spk' else ET} {ds} {pl} {seed}"


def suite_of(model, variant, src):
    return ref_suite(model, variant) if src is None else stage_suite(*src)


def stage_S(fc):
    out = {}
    for n in ('pop_lif', 'pop_flif'):
        val = {f: {(ds, pl, s): float(row(fc, VARIANTS[f][0], cond(n, f), ds, pl, s, DEF)['val_loss_best']) for ds, pl in CELLS for s in SEEDS}
               for f in VARIANTS}
        rel = {f: 100 * sum(val[f][k] / val['off'][k] - 1 for k in val['off']) / len(val['off']) for f in VARIANTS}
        best = min(('aux', 'auxf'), key=lambda f: rel[f])
        out[n] = dict(selected=best if rel[best] < rel['off'] else 'off', rel_val_pct=rel)
    return out


def stage_A(fc, state, stage):
    key, values = GRID[stage]
    prev = ORDER[ORDER.index(stage) - 1]
    out = {}
    for m in MODELS:
        variant = 'off' if m == 'spk' else state['S'][m]['selected']
        ref = state[prev][m] if prev != 'S' else dict(setting=dict(DEF), src=None)
        val = {'ref': {(ds, pl): float(row(fc, suite_of(m, variant, ref['src']), cond(m, variant), ds, pl, 42, ref['setting'])['val_loss_best'])
                       for ds, pl in CELLS}}
        for v in values:
            st = dict(ref['setting'], **{key: v})
            val[v] = {(ds, pl): float(row(fc, stage_suite(stage, v), cond(m, variant), ds, pl, 42, st)['val_loss_best']) for ds, pl in CELLS}
        rel = {v: 100 * sum(val[v][c] / val['ref'][c] - 1 for c in CELLS) / len(CELLS) for v in val}
        best = min(values, key=lambda v: rel[v])
        if rel[best] < 0:
            res = dict(setting=dict(ref['setting'], **{key: best}), src=[stage, best])
        else:
            res = dict(setting=dict(ref['setting']), src=ref['src'])
        res.update(variant=variant, rel_val_pct={('ref ' + ref['setting'][key] if k == 'ref' else k): x for k, x in rel.items()},
                   edge=res['setting'][key] == EDGE[key])
        out[m] = res
    return out


def queue(state, nxt):
    lines = []
    for ds in ('ETTm2', 'ETTh2'):
        for pl in (720, 96):
            for m in MODELS:
                variant = 'off' if m == 'spk' else state['S'][m]['selected']
                if nxt in GRID:
                    key, values = GRID[nxt]
                    prev = ORDER[ORDER.index(nxt) - 1]
                    base = state[prev][m]['setting'] if prev != 'S' else dict(DEF)
                    for v in values:
                        lines.append(line(m, variant, stage_suite(nxt, v), dict(base, **{key: v}), ds, pl, 42))
                else:                                                   # B
                    fin = state['A3'][m]
                    if fin['src'] is not None:
                        for s in (7, 2026):
                            lines.append(line(m, variant, stage_suite(*fin['src']), fin['setting'], ds, pl, s))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=ORDER, required=True)
    ap.add_argument('--queue_next', required=True)
    ap.add_argument('--state', default=os.path.join(HERE, 'results', 'flif-v1-of-selected.json'))
    ap.add_argument('--fc', default=os.path.join(HERE, '..', 'forecasting', 'results'))
    a = ap.parse_args()
    state = json.load(open(a.state)) if os.path.exists(a.state) else {}
    i = ORDER.index(a.stage)
    if a.stage in state or any(s not in state for s in ORDER[:i]):
        raise RuntimeError(f'stage {a.stage}: out of order or already selected (state has {sorted(state)})')
    state[a.stage] = stage_S(a.fc) if a.stage == 'S' else stage_A(a.fc, state, a.stage)
    nxt = ORDER[i + 1] if i + 1 < len(ORDER) else 'B'
    lines = queue(state, nxt)
    json.dump(state, open(a.state, 'w'), indent=1)
    open(a.queue_next, 'w').write(''.join(x + '\n' for x in lines))
    for m, r in state[a.stage].items():
        print(f'{a.stage} {m}: ' + (f"form {r['selected']}" if a.stage == 'S' else f"setting {r['setting']} (src {r['src']}{', EDGE' if r['edge'] else ''})")
              + '  ' + '  '.join(f'{k} {v:+.2f} %' for k, v in r['rel_val_pct'].items()))
    print(f'{a.queue_next}: {len(lines)} jobs ({nxt})')


if __name__ == '__main__':
    main()
