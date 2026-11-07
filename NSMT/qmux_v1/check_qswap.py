"""Checks of summarize_qswap.py on synthetic runs with known answers (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG query-swap pre-registration). No model and no GPU; nothing under results/ is written.
The 60 expected runs are built from the real queue list and RESULT lines (clean = the logged metric), the replaced metrics from fixed factors, so the verdicts are known in advance:
  ETT      other +2 % on all pairs -> used 12/12; perm -2 % -> lower; zero +0.5 % -> no clear effect (R below 1); attn_off +3 % on 6 pairs, -1 % on 6 -> R +1, 6/12 -> no clear effect
  MIT-BIH  other -1 %p -> used; perm +1 %p -> higher; zero -0.4 %p -> no clear effect; attn_off -4, +0.5, +0.5 %p -> mean -1, 1/3 seeds -> no clear effect
Fail-closed cases that must stop with exit 2: a missing run, a NaN metric, a clean metric that differs from the RESULT line, a failed check field, a smoke run.
usage: python -B check_qswap.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'scripts'))
import make_queue_qswap as mq                                          # noqa: E402
import summarize_qswap as sq                                           # noqa: E402

CHECKS = dict(split_max_abs=0., neo_repeat_equal=True, other_min_distance=200, gap=192, perm_rows_changed=5, perm_rows=5, zero_all_zero=True, other_bits_changed=5)


def rates():
    return dict(rate_query=.1, rate_attn=.1, rate_hippo=.1)


def build(root):
    os.makedirs(root)
    jobs = mq.ett_jobs() + mq.mit_jobs()
    for task, raw, run, name, _ in jobs:
        ref = sq.result_line(raw, run)
        if task == 'ett':
            mse, mae = float(ref['mse']), float(ref['mae'])
            fac = lambda p: dict(mse=mse * (1 + p / 100), mae=mae * (1 + p / 100), n=1, **rates())
            c_i, s_i = sq.CELLS.index((ref['ds'], int(ref['pl']))), sq.SEEDS.index(int(ref['seed']))
            att = 3. if (c_i + s_i) % 2 == 0 else -1.                  # per neuron: 6 pairs +3 %, 6 pairs -1 %, R exactly +1
            m = dict(clean=fac(0), other=[fac(1.), fac(2.), fac(3.)], perm=[fac(-1.), fac(-2.), fac(-3.)], zero=fac(.5), attn_off=fac(att))
        else:
            f1 = float(ref['f1'])
            fac = lambda p: dict(f1=f1 + p / 100, f1_cls=[f1] * 5, acc=f1, n=1, **rates())
            s = name.rsplit('_s', 1)[1]
            m = dict(clean=fac(0), other=[fac(-1.)] * 3, perm=[fac(1.)] * 3, zero=fac(-.4), attn_off=fac(-4. if s == '42' else .5))
        rec = dict(name=name, task=task, raw=raw, run_dir=run, N=24 if task == 'ett' else 11, checks=dict(CHECKS), metrics=m,
                   neo=dict(rate=.05, rate_by_token=[.05] * (24 if task == 'ett' else 11), agree_other=.9, agree_chance=.9))
        with open(os.path.join(root, name + '.json'), 'w') as f:
            json.dump(rec, f)


def fails(root, mutate):
    tmp = root + '_m'
    shutil.copytree(root, tmp)
    try:
        mutate(tmp)
        try:
            sq.load(tmp)
        except SystemExit as e:
            return e.code == 2
        return False
    finally:
        shutil.rmtree(tmp)


def edit(root, name, fn):
    p = os.path.join(root, name + '.json')
    r = json.load(open(p))
    fn(r)
    with open(p, 'w') as f:
        json.dump(r, f)


def main():
    root = os.path.join(tempfile.mkdtemp(), 'runs')
    build(root)
    run = sq.load(root)
    E, M = sq.ett_table(run), sq.mit_table(run)
    bad = []
    for n in sq.NEURONS:
        e = E[n, 'other']
        if not (abs(e['R'] - 2) < 1e-6 and e['up'] == 12 and e['verdict'] == 'used'):
            bad.append(('ett other', n, e))
        e = E[n, 'perm']
        if not (abs(e['R'] + 2) < 1e-6 and e['down'] == 12 and e['verdict'] == 'lower'):
            bad.append(('ett perm', n, e))
        if E[n, 'zero']['verdict'] != 'no clear effect' or abs(E[n, 'zero']['R'] - .5) > 1e-6:
            bad.append(('ett zero', n, E[n, 'zero']))
        if E[n, 'attn_off']['verdict'] != 'no clear effect' or abs(E[n, 'attn_off']['R'] - 1) > 1e-6 or E[n, 'attn_off']['up'] != 6:
            bad.append(('ett attn_off', n, E[n, 'attn_off']))
        if M[n, 'other']['verdict'] != 'used' or abs(M[n, 'other']['d'] + 1) > 1e-6 or M[n, 'other']['lo'] != 3:
            bad.append(('mit other', n, M[n, 'other']))
        if M[n, 'perm']['verdict'] != 'higher':
            bad.append(('mit perm', n, M[n, 'perm']))
        if M[n, 'zero']['verdict'] != 'no clear effect':
            bad.append(('mit zero', n, M[n, 'zero']))
        if M[n, 'attn_off']['verdict'] != 'no clear effect' or abs(M[n, 'attn_off']['d'] + 1) > 1e-6 or M[n, 'attn_off']['lo'] != 1:
            bad.append(('mit attn_off', n, M[n, 'attn_off']))
    if bad:
        print('FAIL rules:', bad[:3])
        sys.exit(1)
    print('rules ok: known verdicts reproduced for 4 neurons x 4 conditions x 2 tasks')
    for task, cnt in (('ett', 12), ('mitbih', 3)):
        for n in sq.NEURONS:
            rr = sq.of(run, task, n)
            if len(rr) != cnt or any(('_pop_' in r['name']) != n.startswith('pop') for r in rr):
                print(f'FAIL row selection: {task} {n}', [r['name'] for r in rr])
                sys.exit(1)
    print('row selection ok: LIF / f-LIF rows hold no population run')
    one = 'ett_ETTh2_H96_pop_lif_s42'

    def nan_replaced(t):
        edit(t, one, lambda r: r['metrics']['zero'].update(mse=float('nan')))
    cases = dict(missing=lambda t: os.remove(os.path.join(t, one + '.json')),
                 wrong_clean=lambda t: edit(t, one, lambda r: r['metrics']['clean'].update(mse=r['metrics']['clean']['mse'] * 1.01)),
                 failed_check=lambda t: edit(t, one, lambda r: r['checks'].update(neo_repeat_equal=False)),
                 smoke=lambda t: edit(t, one, lambda r: r.update(smoke=True)))
    for k, fn in cases.items():
        if not fails(root, fn):
            print(f'FAIL fail-closed case {k} did not stop')
            sys.exit(1)
    tmp = root + '_nan'
    shutil.copytree(root, tmp)
    nan_replaced(tmp)
    try:
        sq.ett_table(sq.load(tmp))
        print('FAIL fail-closed case nan did not stop')
        sys.exit(1)
    except SystemExit as e:
        if e.code != 2:
            print('FAIL nan exit code', e.code)
            sys.exit(1)
    print('fail-closed ok: missing, wrong clean, failed check, smoke, NaN all stop with exit 2')
    shutil.rmtree(os.path.dirname(root))
    if not math.isfinite(sum(E[n, 'other']['R'] for n in sq.NEURONS)):
        sys.exit(1)


if __name__ == '__main__':
    main()
