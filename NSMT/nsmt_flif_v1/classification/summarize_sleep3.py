"""MODI 2026-11-27 (user decisions 11-27 "1. 진행" = Neocortex membrane readout, "2. 축적만 우선 진행" = past-reconstruction aux):
pre-registered summary of the Sleep-EDF runs of the two changes (docs/MODI_PROJECT_LOG.md 2026-11-27 pre-registration).

Suites (all lr 0.003, seed 42, folds 0-4): new flif-v1-sleep3 (60 runs: q_pop_<n>_{mem, acc, mem_acc, mem_cshuf, mem_nml, acc_cshuf});
references: q_pop_<n> (fold 0 flif-v1-sleep-lr, folds 1-4 flif-v1-sleep), spkall_lif lr 0.003 (fold 0 flif-v1-sleep-lr, folds 1-4
flif-v1-sleep2), q_pop_<n>_{cshuf, nml} (flif-v1-sleep2). Fail closed as summarize_sleep2.py (each needed row once, epochs 30, batch 64,
seed 42, finite, no SKIP, one sha within each suite; the default Sleep path of the new code reproduces the earlier smoke exactly).
Metric: test 5-class macro-F1 %, pairs = folds; verdict: mean >= +1.0 and >= 4/5 positive = better, mean <= -1.0 and >= 4/5 negative =
worse, otherwise no clear difference.
  E (effect of the change, per neuron): E1 mem - spike, E2 acc - spike, E3 mem_acc - spike (spike = q_pop_<n>).
  S (vs Spikformer(DE) at the same lr 0.003): mem, acc, mem_acc - spkall_lif.
  O (order): mem - mem_cshuf, acc - acc_cshuf (the spike model's O was no clear difference, 11-27 04:26).
  M (memory): mem - mem_nml (the spike model's M was no clear difference).
  Reading per variant and neuron: "candidate for the base" if S is better AND (O or M) is better for that neuron; otherwise the reason.
Report only: energy (syops) and its ratio to Spikformer(DE) at lr 0.003; accuracy; N1 F1.
usage (classification folder): python summarize_sleep3.py
"""
import json
import os

import numpy as np

import select_sleep as SS

HERE = os.path.dirname(os.path.abspath(__file__))
NEU = ('pop_lif', 'pop_flif')
NEW = ('mem', 'acc', 'mem_acc', 'mem_cshuf', 'mem_nml', 'acc_cshuf')


def verdict(d):
    if d.mean() >= 1.0 and (d > 0).sum() >= 4:
        return 'better'
    if d.mean() <= -1.0 and (d < 0).sum() >= 4:
        return 'worse'
    return 'no clear difference'


def main():
    T = {}

    def add(suite, allowed, strict=False):
        shas = set()
        for r in SS.rows(suite):
            key = (r['cond'], r['lr'], int(r['fold']))
            if key not in allowed:
                if strict:
                    raise SystemExit(f'STOP: unexpected row {key} in {suite}')
                continue
            if key in T:
                raise SystemExit(f'STOP: duplicate {key} ({suite})')
            SS.check(r, key); T[key] = r; shas.add(r['sha'])
        if len(shas) > 1:
            raise SystemExit(f'STOP: {suite} code versions {sorted(shas)}')
        return shas
    lr = '0.003'
    need1 = {(c, lr, 0) for c in ('spkall_lif', 'q_pop_lif', 'q_pop_flif')}
    needm = {(f'q_{n}', lr, f) for n in NEU for f in range(1, 5)}
    need2 = {('spkall_lif', lr, f) for f in range(1, 5)} | {(f'q_{n}_{s}', lr, f) for n in NEU for s in ('cshuf', 'nml') for f in range(5)}
    need3 = {(f'q_{n}_{v}', lr, f) for n in NEU for v in NEW for f in range(5)}
    sh = [add('flif-v1-sleep-lr', need1), add('flif-v1-sleep', needm), add('flif-v1-sleep2', need2), add('flif-v1-sleep3', need3, strict=True)]
    miss = sorted((need1 | needm | need2 | need3) - set(T))
    if miss:
        raise SystemExit(f'STOP: missing {miss[:4]} ({len(miss)})')
    g = lambda c, k='macro_f1': np.array([float(T[(c, lr, f)][k]) for f in range(5)]) * (100 if k in ('macro_f1', 'acc', 'f1_1') else 1)
    e0 = g('spkall_lif', 'syn_energy_uJ').mean()
    out = ['# Sleep-EDF: Neocortex membrane readout and past-reconstruction aux (lr 0.003)\n',
           f'Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Code versions: {sh}. Test macro-F1 %, folds 0-4, paired by fold.\n',
           '| model | macro-F1 | accuracy | N1 F1 | syops energy µJ | × Spikformer(DE) |', '|---|---|---|---|---|---|']
    rows = ['spkall_lif'] + [f'q_{n}{s}' for n in NEU for s in ('', '_cshuf', '_nml') + tuple('_' + v for v in NEW)]
    for c in rows:
        m = g(c)
        out.append(f'| {c} | {m.mean():.2f} ± {m.std():.2f} | {g(c, "acc").mean():.2f} | {g(c, "f1_1").mean():.1f} | {g(c, "syn_energy_uJ").mean():.2f} | '
                   f'{g(c, "syn_energy_uJ").mean() / e0:.2f} |')
    dec = {}
    tests = [('E1 mem − spike', '{n}_mem', '{n}'), ('E2 acc − spike', '{n}_acc', '{n}'), ('E3 mem+acc − spike', '{n}_mem_acc', '{n}'),
             ('S mem − Spikformer(DE)', '{n}_mem', 'SF'), ('S acc − Spikformer(DE)', '{n}_acc', 'SF'), ('S mem+acc − Spikformer(DE)', '{n}_mem_acc', 'SF'),
             ('O mem − mem_cshuf', '{n}_mem', '{n}_mem_cshuf'), ('O acc − acc_cshuf', '{n}_acc', '{n}_acc_cshuf'), ('M mem − mem_nml', '{n}_mem', '{n}_mem_nml')]
    out += ['', '| test | neuron | d per fold | mean | positive | verdict |', '|---|---|---|---|---|---|']
    for name, a_, b_ in tests:
        for n in NEU:
            A = 'q_' + a_.format(n=n); B = 'spkall_lif' if b_ == 'SF' else 'q_' + b_.format(n=n)
            d = g(A) - g(B); v = verdict(d)
            dec[f'{name.split()[0]}|{name}|{n}'] = dict(d=d.tolist(), mean=float(d.mean()), pos=int((d > 0).sum()), verdict=v)
            out.append(f'| {name} | {n} | {" / ".join(f"{x:+.2f}" for x in d)} | {d.mean():+.2f} | {int((d > 0).sum())}/5 | **{v}** |')
    read = {}
    for n in NEU:
        for var in ('mem', 'acc', 'mem_acc'):
            s_ok = dec[f'S|S {var.replace("mem_acc", "mem+acc")} − Spikformer(DE)|{n}']['verdict'] == 'better'
            o_key = {'mem': 'O|O mem − mem_cshuf', 'acc': 'O|O acc − acc_cshuf'}.get(var)
            m_key = 'M|M mem − mem_nml' if var == 'mem' else None
            o_ok = o_key is not None and dec[f'{o_key}|{n}']['verdict'] == 'better'
            m_ok = m_key is not None and dec[f'{m_key}|{n}']['verdict'] == 'better'
            if var == 'mem_acc':
                r = 'beats Spikformer(DE) (no own order / memory control)' if s_ok else 'does not beat Spikformer(DE)'
            elif s_ok and (o_ok or m_ok):
                r = 'candidate for the base (beats Spikformer(DE) and the Neocortex order / memory counts)'
            elif s_ok:
                r = 'beats Spikformer(DE), but the Neocortex order / memory share is not shown'
            elif o_ok or m_ok:
                r = 'the Neocortex order / memory now counts, but Spikformer(DE) is not beaten'
            else:
                r = 'neither'
            read[f'{var}|{n}'] = r
    dec['reading'] = read
    out += ['', '**Reading (pre-registered):** ' + '; '.join(f'{k.replace("|", " / ")}: {v}' for k, v in read.items())]
    txt = '\n'.join(out) + '\n'
    print(txt)
    os.makedirs(os.path.join(HERE, 'results', 'flif-v1-sleep3'), exist_ok=True)
    open(os.path.join(HERE, 'results', 'flif-v1-sleep3', 'summary.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(HERE, 'results', 'flif-v1-sleep3', 'decision.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
