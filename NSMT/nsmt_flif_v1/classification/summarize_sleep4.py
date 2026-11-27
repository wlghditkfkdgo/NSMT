"""MODI 2026-11-27 (user decisions 11-27: "(b) + 지금까지 시도해왔던 수정 사항들에 대해서 standard LIF neuron으로만 재실험 진행",
"기본으로 LIF, f-LIF, pop LIF 이렇게 3종만 넣어줘"): pre-registered summary of the plain LIF / f-LIF batch on Sleep-EDF
(docs/MODI_PROJECT_LOG.md 2026-11-27 pre-registration).

New suite flif-v1-sleep4 (70 runs, seed 42, folds 0-4): q_lif and kv_lif at lr 0.001 and 0.003; at lr 0.003 q_lif_{cshuf, nml, mem, acc,
mem_acc, mem_cshuf, mem_nml, acc_cshuf}, q_flif, q_flif_mem. References (earlier suites, same runner): spkall_lif lr 0.001 (fold 0
flif-v1-sleep-lr, folds 1-4 flif-v1-sleep) and lr 0.003 (fold 0 flif-v1-sleep-lr, folds 1-4 flif-v1-sleep2); q_pop_lif lr 0.003 (fold 0
flif-v1-sleep-lr, folds 1-4 flif-v1-sleep) and lr 0.001 (fold 0 flif-v1-sleep-lr, folds 1-4 flif-v1-sleep2); kv_pop_lif lr 0.003 (fold 0
flif-v1-sleep-lr, folds 1-4 flif-v1-sleep); q_pop_lif_{cshuf, nml} (flif-v1-sleep2) and q_pop_lif_{mem, acc, mem_acc, mem_cshuf, mem_nml,
acc_cshuf} (flif-v1-sleep3) — the pop LIF controls are for the side-by-side table; only q_pop_lif_mem enters a test (N). Fail closed as summarize_sleep3.py (each needed row once,
epochs 30, batch 64, seed 42, finite, no SKIP, one sha within each suite, no unexpected row in the new suite).
Metric: test 5-class macro-F1 %, pairs = folds; verdict: mean >= +1.0 and >= 4/5 positive = better, mean <= -1.0 and >= 4/5 negative =
worse, otherwise no clear difference.
  lr choice (as select_sleep.py): for q_lif and kv_lif the lr with the higher fold-0 best validation macro-F1; a tie keeps 0.001.
  Q1 (as 11-26): q_lif and kv_lif at their chosen lr - Spikformer(DE) at its chosen lr (0.001, selected 11-26).
  Q2 (as 11-26): q_lif - kv_lif, each at its chosen lr.
  L (as 11-27 04:26): q_lif(lr) - spkall_lif(lr) for lr 0.003 and 0.001; reading "beats Spikformer(DE) at the same lr" (better at both),
     "depends on lr" (one), "not supported" (none).
  O / M (spike readout, lr 0.003): q_lif - q_lif_cshuf, q_lif - q_lif_nml.
  E (lr 0.003): q_lif_{mem, acc, mem_acc} - q_lif; q_flif_mem - q_flif.
  S (lr 0.003): q_lif_{mem, acc, mem_acc} - spkall_lif; q_flif - spkall_lif, q_flif_mem - spkall_lif.
  O / M of the variants (lr 0.003): q_lif_mem - q_lif_mem_cshuf, q_lif_acc - q_lif_acc_cshuf, q_lif_mem - q_lif_mem_nml.
  Reading per LIF model (spike, mem, acc; as summarize_sleep3.py): S better (spike: L at 0.003) AND (O or M) better = "candidate for the
     base"; S only = "beats Spikformer(DE), but the Neocortex order / memory share is not shown"; O / M only = "the Neocortex order / memory
     now counts, but Spikformer(DE) is not beaten"; else "neither". mem_acc and the f-LIF models have no own controls: S only.
  N (neuron comparison, lr 0.003, words "LIF better" etc.): q_lif - q_pop_lif, q_flif - q_pop_lif, kv_lif - kv_pop_lif, q_lif_mem -
     q_pop_lif_mem, q_flif_mem - q_pop_lif_mem.
Report only: one table per readout with the three default neurons (LIF, f-LIF, pop LIF) side by side; accuracy, N1 F1, syops energy and
its ratio to Spikformer(DE) at lr 0.003, firing rate; Q2 and kv - Spikformer(DE) at the same lr; q_lif - q_pop_lif at lr 0.001.
usage (classification folder): python summarize_sleep4.py [--out <dir>]
"""
import argparse
import json
import os

import numpy as np

import select_sleep as SS

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = 'flif-v1-sleep4'
LIFV = ('cshuf', 'nml', 'mem', 'acc', 'mem_acc', 'mem_cshuf', 'mem_nml', 'acc_cshuf')


def verdict(d, pos='better', neg='worse'):
    if d.mean() >= 1.0 and (d > 0).sum() >= 4:
        return pos
    if d.mean() <= -1.0 and (d < 0).sum() >= 4:
        return neg
    return 'no clear difference'


def needed():
    a, b = '0.001', '0.003'
    new = {(c, lr, f) for c in ('q_lif', 'kv_lif') for lr in (a, b) for f in range(5)} | \
          {(f'q_lif_{v}', b, f) for v in LIFV for f in range(5)} | {(c, b, f) for c in ('q_flif', 'q_flif_mem') for f in range(5)}
    ref = {'flif-v1-sleep-lr': {(c, lr, 0) for c in ('spkall_lif', 'q_pop_lif') for lr in (a, b)} | {('kv_pop_lif', b, 0)},
           'flif-v1-sleep': {(c, lr, f) for c, lr in (('spkall_lif', a), ('q_pop_lif', b), ('kv_pop_lif', b)) for f in range(1, 5)},
           'flif-v1-sleep2': {(c, lr, f) for c, lr in (('spkall_lif', b), ('q_pop_lif', a)) for f in range(1, 5)} |
                             {(f'q_pop_lif_{v}', b, f) for v in ('cshuf', 'nml') for f in range(5)},
           'flif-v1-sleep3': {(f'q_pop_lif_{v}', b, f) for v in LIFV[2:] for f in range(5)}}
    return new, ref


def load(rows=SS.rows):
    T, sh = {}, {}
    new, ref = needed()

    def add(suite, allowed, strict):
        shas = set()
        for r in rows(suite):
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
        sh[suite] = sorted(shas)
    for s, allowed in ref.items():
        add(s, allowed, False)
    add(NEW, new, True)
    miss = sorted((new | set().union(*ref.values())) - set(T))
    if miss:
        raise SystemExit(f'STOP: missing {miss[:4]} ({len(miss)})')
    return T, sh


def main(rows=SS.rows, out_dir=None):
    T, sh = load(rows)
    pc = lambda k: 100 if k in ('macro_f1', 'acc', 'f1_1') else 1
    g = lambda c, lr, k='macro_f1': np.array([float(T[(c, lr, f)][k]) for f in range(5)]) * pc(k)
    val0 = lambda c, lr: float(T[(c, lr, 0)]['best_epoch_val'].split(':')[1])
    a, b = '0.001', '0.003'
    sel = {c: (b if val0(c, b) > val0(c, a) else a) for c in ('q_lif', 'kv_lif')}
    dec = {'lr': {c: dict(lr=sel[c], val={a: val0(c, a), b: val0(c, b)}) for c in sel}, 'sha': sh}
    e0 = g('spkall_lif', b, 'syn_energy_uJ').mean()
    out = ['# Sleep-EDF: plain LIF / f-LIF Neocortex (structure, order / memory, membrane readout, accumulation aux)\n',
           f'Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Code versions: {sh}. Test macro-F1 %, folds 0-4, paired by fold.\n',
           '## Three default neurons side by side (lr 0.003; — = not run)\n',
           '| model | LIF | f-LIF | pop LIF |', '|---|---|---|---|']
    cell = lambda c: f'{g(c, b).mean():.2f} ± {g(c, b).std():.2f}' if (c, b, 0) in T else '—'
    out.append(f'| Spikformer(DE) | {cell("spkall_lif")} (no Neocortex) | | |')
    for lab, suf in (('query, spike', ''), ('key/value, spike', 'kv'), ('query, membrane', '_mem'), ('query, accumulation aux', '_acc'),
                     ('query, both', '_mem_acc'), ('query, context shuffled', '_cshuf'), ('query, memoryless Neocortex', '_nml'),
                     ('membrane + context shuffled', '_mem_cshuf'), ('membrane + memoryless', '_mem_nml'), ('aux + context shuffled', '_acc_cshuf')):
        cs = [f'kv_{n}' if suf == 'kv' else f'q_{n}{suf}' for n in ('lif', 'flif', 'pop_lif')]
        out.append(f'| {lab} | ' + ' | '.join(cell(c) for c in cs) + ' |')
    out += ['', '## Every run set\n', '| model | lr | macro-F1 | accuracy | N1 F1 | syops energy µJ | × Spikformer(DE) @0.003 | firing rate |',
            '|---|---|---|---|---|---|---|---|']
    for (c, lr) in sorted({(k[0], k[1]) for k in T}):
        out.append(f'| {c} | {lr} | {g(c, lr).mean():.2f} ± {g(c, lr).std():.2f} | {g(c, lr, "acc").mean():.2f} | {g(c, lr, "f1_1").mean():.1f} | '
                   f'{g(c, lr, "syn_energy_uJ").mean():.2f} | {g(c, lr, "syn_energy_uJ").mean() / e0:.2f} | {g(c, lr, "syops_firing_rate").mean():.4f} |')
    out += ['', f'lr choice (fold-0 validation, tie -> 0.001): ' + '; '.join(f'{c} {v["val"][a]:.4f} / {v["val"][b]:.4f} -> {v["lr"]}' for c, v in dec['lr'].items()),
            '', '| test | d per fold | mean | positive | verdict |', '|---|---|---|---|---|']

    def test(key, A, B, pos='better', neg='worse', judged=True):
        d = g(*A) - g(*B); v = verdict(d, pos, neg)
        dec[key] = dict(A=list(A), B=list(B), d=d.tolist(), mean=float(d.mean()), pos=int((d > 0).sum()), verdict=v, judged=judged)
        out.append(f'| {key}{"" if judged else " (report)"} | {" / ".join(f"{x:+.2f}" for x in d)} | {d.mean():+.2f} | {int((d > 0).sum())}/5 | '
                   + (f'**{v}**' if judged else v) + ' |')
        return v
    SF = ('spkall_lif', a)
    test('Q1 q_lif − Spikformer(DE) (chosen lrs)', ('q_lif', sel['q_lif']), SF)
    test('Q1 kv_lif − Spikformer(DE) (chosen lrs)', ('kv_lif', sel['kv_lif']), SF)
    test('Q2 q_lif − kv_lif (chosen lrs)', ('q_lif', sel['q_lif']), ('kv_lif', sel['kv_lif']), 'query better', 'key/value better')
    Lv = {lr: test(f'L q_lif − Spikformer(DE) @{lr}', ('q_lif', lr), ('spkall_lif', lr)) for lr in (b, a)}
    nb = sum(v == 'better' for v in Lv.values())
    dec['L reading'] = 'beats Spikformer(DE) at the same lr' if nb == 2 else 'depends on lr' if nb == 1 else 'not supported'
    for lr in (b, a):
        test(f'Q2 q_lif − kv_lif @{lr}', ('q_lif', lr), ('kv_lif', lr), 'query better', 'key/value better', judged=False)
        test(f'kv_lif − Spikformer(DE) @{lr}', ('kv_lif', lr), ('spkall_lif', lr), judged=False)
    test('O q_lif − q_lif_cshuf', ('q_lif', b), ('q_lif_cshuf', b))
    test('M q_lif − q_lif_nml', ('q_lif', b), ('q_lif_nml', b))
    for v in ('mem', 'acc', 'mem_acc'):
        test(f'E q_lif_{v} − q_lif', (f'q_lif_{v}', b), ('q_lif', b))
    test('E q_flif_mem − q_flif', ('q_flif_mem', b), ('q_flif', b))
    for c in ('q_lif_mem', 'q_lif_acc', 'q_lif_mem_acc', 'q_flif', 'q_flif_mem'):
        test(f'S {c} − Spikformer(DE) @0.003', (c, b), ('spkall_lif', b))
    test('O q_lif_mem − q_lif_mem_cshuf', ('q_lif_mem', b), ('q_lif_mem_cshuf', b))
    test('O q_lif_acc − q_lif_acc_cshuf', ('q_lif_acc', b), ('q_lif_acc_cshuf', b))
    test('M q_lif_mem − q_lif_mem_nml', ('q_lif_mem', b), ('q_lif_mem_nml', b))
    for A, B in (('q_lif', 'q_pop_lif'), ('q_flif', 'q_pop_lif'), ('kv_lif', 'kv_pop_lif'), ('q_lif_mem', 'q_pop_lif_mem'), ('q_flif_mem', 'q_pop_lif_mem')):
        na = 'f-LIF' if 'flif' in A else 'LIF'
        test(f'N {A} − {B} @0.003', (A, b), (B, b), f'{na} better', 'pop LIF better')
    test('N q_lif − q_pop_lif @0.001', ('q_lif', a), ('q_pop_lif', a), 'LIF better', 'pop LIF better', judged=False)
    read = {}
    for c, s_key, o_key, m_key in (('q_lif', 'L q_lif − Spikformer(DE) @0.003', 'O q_lif − q_lif_cshuf', 'M q_lif − q_lif_nml'),
                                   ('q_lif_mem', 'S q_lif_mem − Spikformer(DE) @0.003', 'O q_lif_mem − q_lif_mem_cshuf', 'M q_lif_mem − q_lif_mem_nml'),
                                   ('q_lif_acc', 'S q_lif_acc − Spikformer(DE) @0.003', 'O q_lif_acc − q_lif_acc_cshuf', None),
                                   ('q_lif_mem_acc', 'S q_lif_mem_acc − Spikformer(DE) @0.003', None, None),
                                   ('q_flif', 'S q_flif − Spikformer(DE) @0.003', None, None),
                                   ('q_flif_mem', 'S q_flif_mem − Spikformer(DE) @0.003', None, None)):
        s_ok = dec[s_key]['verdict'] == 'better'
        om = (o_key is not None and dec[o_key]['verdict'] == 'better') or (m_key is not None and dec[m_key]['verdict'] == 'better')
        if o_key is None and m_key is None:
            r = 'beats Spikformer(DE) (no own order / memory control)' if s_ok else 'does not beat Spikformer(DE)'
        elif s_ok and om:
            r = 'candidate for the base (beats Spikformer(DE) and the Neocortex order / memory counts)'
        elif s_ok:
            r = 'beats Spikformer(DE), but the Neocortex order / memory share is not shown'
        elif om:
            r = 'the Neocortex order / memory now counts, but Spikformer(DE) is not beaten'
        else:
            r = 'neither'
        read[c] = r
    dec['reading'] = read
    out += ['', f'**L reading (q_lif):** {dec["L reading"]}.', '',
            '**Reading per model (pre-registered):** ' + '; '.join(f'{k}: {v}' for k, v in read.items())]
    txt = '\n'.join(out) + '\n'
    print(txt)
    od = out_dir or os.path.join(HERE, 'results', NEW)
    os.makedirs(od, exist_ok=True)
    open(os.path.join(od, 'summary.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(od, 'decision.json'), 'w'), indent=1)
    return dec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None)
    main(out_dir=ap.parse_args().out)
