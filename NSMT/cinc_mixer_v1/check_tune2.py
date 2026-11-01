"""Checks for the MIT-BIH and CinC tuning of the fusion MUX (2026-11-01, PROJECT_LOG, exp/f-lif-pop-v3). Fail-closed.

  c0  default paths unchanged: (a) the Spikformer-style MIT-BIH model built by the edited train.py module equals the snapshot taken
      before the edit (mit_k3_snapshot.py; --k3_before); (b) runner arguments: the MIT-BIH ADD smoke vs the 11-01 MIT-BIH muxlin run
      and the CinC 1-layer smoke vs the 10-31 CinC muxlin run differ only in date / paths / tag / epoch and the requested change
      (connect_f; time_num_layers)
  c1  validation split: create_loader(train_val_ratio=[4, 1], n_folds=5, fold_id=0) on the MIT-BIH training file gives disjoint
      train / validation index sets, |val| ~ 1/5, the class proportions of the file in both parts (within 0.5 %p), and the same
      split twice
  c2  knobs: the MIT-BIH validation smoke has lr 0.005 / time_num_layers 3 in logargs, val_fold 0 and val_f1_last == val_last.csv,
      n_val == |fold 0|; the CinC smokes have the requested lr / time_num_layers
  c3  CinC Spikformer-style model: this branch's code and the baseline commit 9094bfaf9 (git archive, --src9094, `python -B`) give the
      same initial state_dict, validation windows, eval / training outputs and CAL-weighted CE gradients (cinc_c3_snapshot.py)
  c4  CinC reproduction: this branch's Spikformer-style seed 42 run (lr 1e-3) is within 0.2 %p of the reused row (test 67.2548,
      best validation 71.1220)
  c5  MIT-BIH ADD energy: energy_recount_mitbih.py on the MIT-BIH smoke suite passed; the ADD row's corrected total equals
      total - syn + syn_int recomputed here; the 0/1 row keeps its logged total
usage: python check_tune2.py --scratch <dir> --src9094 <exported classification dir> --k3_before <k3_mine.pt> [--only c0 c1]
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

RESULTS = []
DATA_MIT = '/home/yschoi/CLS_spiking_transformer/data'
DATE = '20261101'


def report(name, ok, msg):
    RESULTS.append((name, bool(ok), msg)); print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
            except Exception as e:                                      # noqa: BLE001
                report(name, False, f'{type(e).__name__}: {e}')
        run.__name__ = name
        return run
    return deco


def child_env():
    """environment for the helper processes: without the MKL_THREADING_LAYER that the parent's torch / mkl import sets (it clashes with
    libgomp in a fresh interpreter: 'MKL_THREADING_LAYER=INTEL is incompatible with libgomp')."""
    env = dict(os.environ)
    env.pop('MKL_THREADING_LAYER', None)
    return env


def logargs(run_dir):
    out = {}
    for line in open(os.path.join(run_dir, 'logargs.txt')):
        m = re.match(r'^(\S+?)-{2,}(.*)$', line.rstrip('\n'))
        if m:
            out[m.group(1)] = m.group(2)
    return out


def args_diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


BASE_ALLOWED = {'date', 'save_result_path', 'save_log_path', 'save_model_state_path', 'tag', 'epoch', 'log_dir', 'txt', 'set_args', 'saved_epoch'}


def one(rows, **kw):
    rr = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    if len(rr) != 1:
        raise RuntimeError(f'{len(rr)} rows for {kw}')
    return rr[0]


@check('c0')
def c0():
    out = os.path.join(ARGS.scratch, 'k3_after.pt')
    with open(os.path.join(ARGS.scratch, 'k3_after.log'), 'w') as lf:
        subprocess.run([sys.executable, '-B', os.path.join(HERE, 'mit_k3_snapshot.py'), HERE, out, os.path.join(ARGS.scratch, 'k3log_after'), DATA_MIT],
                       check=True, cwd=ARGS.scratch, stdout=lf, stderr=subprocess.STDOUT, env=child_env())
    a, b = torch.load(ARGS.k3_before), torch.load(out)
    same = (a['sd'].keys() == b['sd'].keys() and all(torch.equal(a['sd'][k], b['sd'][k]) for k in a['sd'])
            and all(torch.equal(a[k], b[k]) for k in ('x', 'y', 'out_eval', 'out_train'))
            and all(torch.equal(a['grads'][k], b['grads'][k]) for k in a['grads']))
    ms = parse(os.path.join(HERE, 'results', f'smoke-mitbih-tune-{DATE}', 'raw.txt'))
    m_add = one(ms, cond='spkmx_lif_muxlin_add')
    m_ref = one(parse(os.path.join(HERE, 'results', 'flif-v1-mitbih-spkmux', 'raw.txt')), cond='spkmx_lif_muxlin_or', seed='42.0')
    d_mit = args_diff(logargs(m_add['dir']), logargs(m_ref['dir']))
    cs = parse(os.path.join(HERE, 'results', f'smoke-cinc-tune-{DATE}', 'raw.txt'))
    c_l1 = one(cs, cond='spk_lif_muxlin_or')
    c_ref = one(parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-spk-mux-cinc', 'raw.txt')), cond='spk_lif_muxlin_or', seed='42.0')
    d_cinc = args_diff(logargs(c_l1['dir']), logargs(c_ref['dir']))
    ok = same and set(d_mit) <= BASE_ALLOWED | {'connect_f'} and set(d_cinc) <= BASE_ALLOWED | {'time_num_layers', 'data_info'}
    report('c0', ok, f'MIT-BIH Spikformer-style model before / after the train.py edit bitwise equal: {same}; MIT-BIH ADD smoke vs 11-01 run differing '
                     f'keys {d_mit}; CinC 1-layer smoke vs 10-31 run differing keys {d_cinc}')


@check('c1')
def c1():
    from dataloader import create_loader
    sets = []
    for _ in range(2):
        (tr, va), cns = create_loader(train=True, batch_size=512, num_classes=5, data_root=DATA_MIT + '/5class_mitbih_train_7_187.csv', train_val_ratio=[4, 1],
                                      data_window_size=187, sampling='None', n_folds=5, fold_id=0, num_workers=0)
        sets.append((np.array(tr.dataset.indices), np.array(va.dataset.indices), np.asarray(tr.dataset.dataset.y).ravel()))
    (ti, vi, y), (ti2, vi2, _) = sets
    n = len(y)
    disjoint = len(np.intersect1d(ti, vi)) == 0 and len(ti) + len(vi) == n
    same = np.array_equal(ti, ti2) and np.array_equal(vi, vi2)
    pf, pt, pv = (np.bincount(y.astype(int), minlength=5) / n, np.bincount(y[ti].astype(int), minlength=5) / len(ti),
                  np.bincount(y[vi].astype(int), minlength=5) / len(vi))
    strat = float(np.abs(pv - pf).max()) < 0.005 and float(np.abs(pt - pf).max()) < 0.005
    size = abs(len(vi) / n - 0.2) < 0.001
    report('c1', disjoint and same and strat and size, f'train {len(ti)} / validation {len(vi)} of {n} (disjoint {disjoint}, repeatable {same}); '
                                                      f'class shares file {np.round(100 * pf, 2).tolist()} validation {np.round(100 * pv, 2).tolist()}')


@check('c2')
def c2():
    ms = parse(os.path.join(HERE, 'results', f'smoke-mitbih-tune-{DATE}', 'raw.txt'))
    v = one(ms, cond='spkmx_lif_muxlin_or')
    la = logargs(v['dir'])
    vl = list(csv.DictReader(open(os.path.join(v['dir'], 'log', 'val_last.csv'))))[-1]
    ok_m = (abs(float(la['lr']) - 0.005) < 1e-12 and int(la['time_num_layers']) == 3 and int(v['val_fold']) == 0
            and abs(float(vl['f1']) - float(v['val_f1_last'])) < 1e-6 and int(vl['n_val']) == int(v['n_val']) and abs(int(v['n_val']) - 15323) <= 2)
    cs = parse(os.path.join(HERE, 'results', f'smoke-cinc-tune-{DATE}', 'raw.txt'))
    s1 = one(cs, cond='spk_ssa'); s2 = one(cs, cond='spk_lif_muxlin_or')
    l1, l2 = logargs(s1['dir']), logargs(s2['dir'])
    ok_c = (abs(float(l1['lr']) - 0.0005) < 1e-12 and int(l1['time_num_layers']) == 2 and abs(float(l2['lr']) - 0.001) < 1e-12
            and int(l2['time_num_layers']) == 1 and int(s2['layers']) == 1 and s1['tag'] == 'spk_ssa_lr0.0005_L2')
    report('c2', ok_m and ok_c, f"MIT-BIH validation smoke: lr {la['lr']}, layers {la['time_num_layers']}, val_fold {int(v['val_fold'])}, n_val {int(v['n_val'])}, "
                                f"val_f1_last {float(v['val_f1_last']):.4f} == val_last.csv {float(vl['f1']):.4f} | CinC smokes: spk_ssa lr {l1['lr']} L {l1['time_num_layers']}, "
                                f"muxlin lr {l2['lr']} L {l2['time_num_layers']} (tags {s1['tag']}, {s2['tag']})")


@check('c3')
def c3():
    outs = {}
    for tag, root in (('mine', HERE), ('src', ARGS.src9094)):
        out = os.path.join(ARGS.scratch, f'c3_{tag}.pt')
        with open(os.path.join(ARGS.scratch, f'c3_{tag}.log'), 'w') as lf:   # helper output kept for diagnosis
            subprocess.run([sys.executable, '-B', os.path.join(HERE, 'cinc_c3_snapshot.py'), root, out, os.path.join(ARGS.scratch, f'c3log_{tag}')],
                           check=True, cwd=ARGS.scratch, stdout=lf, stderr=subprocess.STDOUT, env=child_env())
        outs[tag] = torch.load(out)
    a, b = outs['mine'], outs['src']
    sd = a['sd'].keys() == b['sd'].keys() and all(torch.equal(a['sd'][k], b['sd'][k]) for k in a['sd'])
    same = {k: bool(torch.equal(a[k], b[k])) for k in ('x', 'y', 'out_eval', 'out_train')}
    gr = a['grads'].keys() == b['grads'].keys() and all(torch.equal(a['grads'][k], b['grads'][k]) for k in a['grads'])
    nz = sum(float(v.abs().sum()) > 0 for v in a['grads'].values())
    report('c3', sd and all(same.values()) and gr and nz > 10 and a['info'] == b['info'],
           f"CinC Spikformer-style: state_dict ({len(a['sd'])}) {sd}; {same}; grads ({len(a['grads'])}, {nz} non-zero) {gr}; data info equal {a['info'] == b['info']}")


@check('c4')
def c4():
    mine = one(parse(os.path.join(HERE, 'results', f'check-cinc-spkssa-{DATE}', 'raw.txt')), cond='spk_ssa')
    base = one([r for r in parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-cinc', 'raw.txt')) if int(r.get('fold', 0)) == 0], cond='spk_ssa', seed='42.0')
    va, vb = float(str(mine['best_epoch_val']).split(':')[1]), float(str(base['best_epoch_val']).split(':')[1])
    ds, dv = 100 * (mine['score'] - base['score']), 100 * (va - vb)
    report('c4', abs(ds) <= 0.2 and abs(dv) <= 0.2, f"reproduction test {100 * mine['score']:.4f} vs reused {100 * base['score']:.4f} ({ds:+.4f} %p); best validation "
                                                    f"{100 * va:.4f} vs {100 * vb:.4f} ({dv:+.4f}); best epoch {mine['best_epoch_val']} vs {base['best_epoch_val']}; "
                                                    f"params {int(mine['params'])} vs {int(base['params'])}")


@check('c5')
def c5():
    rows = list(csv.DictReader(open(os.path.join(HERE, 'results', f'smoke-mitbih-tune-{DATE}', 'energy_recount.csv'))))
    raw = parse(os.path.join(HERE, 'results', f'smoke-mitbih-tune-{DATE}', 'raw.txt'))
    ok, msgs, n_add = len(rows) == len(raw) == 2, [], 0
    for e in rows:
        r = one(raw, cond=e['cond'])
        if e['rule'] == 'integer':
            n_add += 1
            corr = r['total_fft_uJ'] - r['syn_energy_uJ'] + float(e['syn_int_uJ'])
            good = e['cond'].endswith('_add') and abs(corr - float(e['total_fft_uJ_corr'])) < 1e-4 and float(e['syn_int_uJ']) < float(e['syn_syops_uJ'])
        else:
            good = not e['cond'].endswith('_add') and abs(float(e['total_fft_uJ_corr']) - r['total_fft_uJ']) < 1e-6
        ok &= good
        msgs.append(f"{e['tag']}: rule {e['rule']}, syn syops {float(e['syn_syops_uJ']):.3f} / integer {float(e['syn_int_uJ']):.3f}, total "
                    f"{r['total_fft_uJ']:.3f} -> {float(e['total_fft_uJ_corr']):.3f} uJ {'ok' if good else 'BAD'}")
    report('c5', ok and n_add == 1, ' | '.join(msgs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks', 'check_tune2.txt'))
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--src9094', default=None)
    ap.add_argument('--k3_before', default=None)
    ARGS = ap.parse_args()
    os.makedirs(ARGS.scratch, exist_ok=True)
    for fn in (c0, c1, c2, c3, c4, c5):
        if ARGS.only and fn.__name__ not in ARGS.only:
            continue
        fn()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    with open(ARGS.out, 'a' if ARGS.only else 'w') as f:
        for name, ok, msg in RESULTS:
            f.write(f"{'PASS' if ok else 'FAIL'} {name}: {msg}\n")
        f.write(f'{n_ok}/{len(RESULTS)} passed\n')
    print(f'{n_ok}/{len(RESULTS)} passed -> {ARGS.out}')
    sys.exit(0 if n_ok == len(RESULTS) and len(RESULTS) > 0 else 1)
