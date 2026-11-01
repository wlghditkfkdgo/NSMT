"""Checks for the ETT tuning and the MIT-BIH extension of the fusion MUX (2026-11-01, PROJECT_LOG, exp/f-lif-pop-v3). Fail-closed.

  k0  unchanged earlier results: (a) the edited energy_recount_ett.py re-run (in --scratch) on the 10-31 suites flif-v1-ett-spkmux
      and check-ett-spkssa-20261031 reproduces every existing column of their energy_recount.csv; (b) the edited run_ett_spk.sh with
      the default LR / LAYERS gives the same training arguments as the 10-31 run (logargs.txt of the smoke run vs the 10-31 run of the
      same condition / cell / seed; only date, paths, tag and the epoch count of the 1-epoch smoke may differ)
  k1  ADD energy (smoke suite smoke-ett-tune-<date>, recounted): the ADD row uses the integer rule
      total_fft_uJ - syn_energy_uJ + body_syn_int_uJ + head_AC_corr * 0.9e-6 (recomputed here; body_syn_int < body_syn, the
      syops-rule MACs are gone), its per-step head input is integer; for the 0/1 rows the original formula holds and
      body_syn_uJ == body_syn_int_uJ
  k2  knobs: the smoke runs' logargs carry the requested lr / time_num_layers; the 2-layer run has more parameters than the
      1-layer runs; RESULT lines carry tag= lr= layers=
  k3  MIT-BIH: the Spikformer-style model built by this branch's code and by the code of the baseline commit 26915f45b (git archive,
      --src26915) has the same initial state_dict, test batch, eval / training outputs and CE gradients (mit_k3_snapshot.py, two
      processes, `python -B`); the SpkMixer MIT-BIH smoke runs reported 11 tokens and 5 classes
  k4  MIT-BIH reproduction: this branch's Spikformer-style seed 42 run (suite check-mitbih-spkssa-<date>) is within 0.2 %p
      macro-F1 of the reused baseline row (0.924183)
usage: python check_tune_mit.py --scratch <dir> --src26915 <exported classification dir> [--only k0 k3] [--date 20261101]
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

RESULTS = []
DATA_MIT = '/home/yschoi/CLS_spiking_transformer/data'


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


def logargs(run_dir):
    out = {}
    for line in open(os.path.join(run_dir, 'logargs.txt')):
        m = re.match(r'^(\S+?)-{2,}(.*)$', line.rstrip('\n'))
        if m:
            out[m.group(1)] = m.group(2)
    return out


def run_dir(row):
    return row['dir'].rsplit('/log', 1)[0] if row['dir'].endswith('/log') else row['dir']


@check('k0')
def k0():
    msgs, ok = [], True
    for s in ('flif-v1-ett-spkmux', 'check-ett-spkssa-20261031'):
        os.makedirs(os.path.join(ARGS.scratch, 'k0', s), exist_ok=True)
        subprocess.run(['cp', os.path.join(HERE, 'results', s, 'raw.txt'), os.path.join(ARGS.scratch, 'k0', s, 'raw.txt')], check=True)
        subprocess.run([sys.executable, os.path.join(HERE, 'energy_recount_ett.py'), '--suite', s, '--res_root', os.path.join(ARGS.scratch, 'k0')],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE)
        old = list(csv.DictReader(open(os.path.join(HERE, 'results', s, 'energy_recount.csv'))))
        new = list(csv.DictReader(open(os.path.join(ARGS.scratch, 'k0', s, 'energy_recount.csv'))))
        same = len(old) == len(new) > 0 and all(all(o[k] == n[k] for k in o) for o, n in zip(old, new))
        ok &= same; msgs.append(f'{s}: {len(new)} rows, existing columns identical {same}')
    smoke = [r for r in parse(os.path.join(HERE, 'results', f'smoke-ett-tune-{ARGS.date}', 'raw.txt')) if r['cond'] == 'spkmx_lif_muxlin_or']
    ref = [r for r in parse(os.path.join(HERE, 'results', 'flif-v1-ett-spkmux', 'raw.txt'))
           if r['cond'] == 'spkmx_lif_muxlin_or' and r['ds'] == 'ETTh2' and int(r['pl']) == 96 and int(r['seed']) == 42]
    if len(smoke) != 1 or len(ref) != 1:
        raise RuntimeError(f'{len(smoke)} smoke / {len(ref)} reference rows')
    a, b = logargs(run_dir(smoke[0])), logargs(run_dir(ref[0]))
    allowed = {'date', 'save_result_path', 'save_log_path', 'save_model_state_path', 'tag', 'epoch', 'log_dir', 'txt', 'set_args'}
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    args_ok = set(diff) <= allowed and len(a) > 20 and a.get('epoch') == '1'
    ok &= args_ok; msgs.append(f'default-knob runner arguments vs 10-31: differing keys {diff} (allowed {args_ok})')
    report('k0', ok, ' | '.join(msgs))


@check('k1')
def k1():
    p = os.path.join(HERE, 'results', f'smoke-ett-tune-{ARGS.date}', 'energy_recount.csv')
    rows = list(csv.DictReader(open(p)))
    raw = {(r['cond'], r.get('tag')): r for r in parse(os.path.join(HERE, 'results', f'smoke-ett-tune-{ARGS.date}', 'raw.txt'))}
    ok, msgs, n_add = len(rows) == len(raw) == 3, [], 0
    for e in rows:
        r = raw[(e['cond'], e.get('tag'))]
        if e['rule'] == 'integer':
            n_add += 1
            corr = float(r['total_fft_uJ']) - float(r['syn_energy_uJ']) + float(e['body_syn_int_uJ']) + float(e['head_AC_corr']) * 0.9e-6
            good = (e['cond'].endswith('_add') and abs(corr - float(e['total_fft_uJ_corr'])) < 1e-4
                    and float(e['body_syn_int_uJ']) < float(e['body_syn_uJ']))
        else:
            corr = float(r['total_fft_uJ']) - float(e['head_MAC']) * 4.6e-6 + float(e['head_AC_corr']) * 0.9e-6
            good = (not e['cond'].endswith('_add') and abs(corr - float(e['total_fft_uJ_corr'])) < 1e-4 and e['binary'] == 'True'
                    and (e['cond'] == 'spk_ssa' or e['body_syn_uJ'] == e['body_syn_int_uJ']))
        ok &= good
        msgs.append(f"{e['tag']}: rule {e['rule']}, logged {float(r['total_fft_uJ']):.2f} -> corrected {float(e['total_fft_uJ_corr']):.2f} uJ "
                    f"(body syops {float(e['body_syn_uJ']):.2f} / integer {float(e['body_syn_int_uJ']):.2f}) {'ok' if good else 'BAD'}")
    report('k1', ok and n_add == 1, f'{len(rows)} rows, {n_add} ADD | ' + ' | '.join(msgs))


@check('k2')
def k2():
    rows = parse(os.path.join(HERE, 'results', f'smoke-ett-tune-{ARGS.date}', 'raw.txt'))
    ok, msgs = len(rows) == 3, []
    for r in rows:
        la = logargs(run_dir(r))
        m = re.fullmatch(re.escape(r['cond']) + r'_lr([0-9.e-]+)_L(\d+)', str(r['tag']))
        good = (abs(float(la['lr']) - float(r['lr'])) < 1e-12 and int(la['time_num_layers']) == int(r['layers'])
                and bool(m) and abs(float(m.group(1)) - float(r['lr'])) < 1e-12 and int(m.group(2)) == int(r['layers']))
        ok &= good
        msgs.append(f"{r['tag']}: logargs lr {la['lr']} layers {la['time_num_layers']}, params {int(r['params'])} {'ok' if good else 'BAD'}")
    p = {r['tag']: int(r['params']) for r in rows}
    two = [v for k, v in p.items() if k.endswith('_L2')]; one = [v for k, v in p.items() if k.startswith('spkmx') and k.endswith('_L1')]
    more = bool(two) and bool(one) and all(t > o for t in two for o in one)
    report('k2', ok and more, ' | '.join(msgs) + f' | 2-layer > 1-layer parameters: {more}')


@check('k3')
def k3():
    os.makedirs(ARGS.scratch, exist_ok=True)
    outs = {}
    for tag, root in (('mine', HERE), ('src', ARGS.src26915)):
        out = os.path.join(ARGS.scratch, f'k3_{tag}.pt')
        subprocess.run([sys.executable, '-B', os.path.join(HERE, 'mit_k3_snapshot.py'), root, out, os.path.join(ARGS.scratch, f'k3log_{tag}'), DATA_MIT],
                       check=True, cwd=ARGS.scratch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outs[tag] = torch.load(out)
    a, b = outs['mine'], outs['src']
    sd = a['sd'].keys() == b['sd'].keys() and all(torch.equal(a['sd'][k], b['sd'][k]) for k in a['sd'])
    same = {k: bool(torch.equal(a[k], b[k])) for k in ('x', 'y', 'out_eval', 'out_train')}
    gr = a['grads'].keys() == b['grads'].keys() and all(torch.equal(a['grads'][k], b['grads'][k]) for k in a['grads'])
    nz = sum(float(v.abs().sum()) > 0 for v in a['grads'].values())
    outs_smoke = sorted(glob.glob(os.path.join(HERE, 'log', f'smoke-mitbih-spkmux-{ARGS.date}', '*.stdout')))
    shapes = []
    for f in outs_smoke:
        t = open(f, errors='ignore').read()
        m = re.search(r'number of parameters : (\d+)', t)
        shapes.append(bool(m) and ('Traceback' not in t))
    raw = parse(os.path.join(HERE, 'results', f'smoke-mitbih-spkmux-{ARGS.date}', 'raw.txt'))
    smoke_ok = len(raw) == 2 and all(shapes) and len(shapes) == 2
    ok = sd and all(same.values()) and gr and nz > 10 and a['class_counts'] == b['class_counts'] and smoke_ok
    report('k3', ok, f"spk_ssa MIT-BIH: state_dict ({len(a['sd'])}) {sd}; {same}; grads ({len(a['grads'])}, {nz} non-zero) {gr}; "
                     f"class counts {a['class_counts']}; SpkMixer MIT-BIH smoke RESULT rows {len(raw)} (no traceback {all(shapes)})")


@check('k4')
def k4():
    mine = [r for r in parse(os.path.join(HERE, 'results', f'check-mitbih-spkssa-{ARGS.date}', 'raw.txt')) if r['cond'] == 'spk_ssa']
    base = [r for r in parse(os.path.join(HERE, 'results', 'flif-v1-mitbih-spkmux', 'baseline_spk_ssa_raw.txt')) if int(r['seed']) == 42]
    if len(mine) != 1 or len(base) != 1:
        raise RuntimeError(f'{len(mine)} reproduction rows, {len(base)} baseline rows')
    a, b = float(mine[0]['f1']), float(base[0]['f1'])
    report('k4', abs(a - b) * 100 <= 0.2, f'reproduction macro-F1 {100 * a:.2f} vs reused {100 * b:.2f} ({100 * (a - b):+.2f} %p); accuracy '
                                         f'{100 * float(mine[0]["overall_acc"]):.2f} vs {100 * float(base[0]["overall_acc"]):.2f}; energy '
                                         f'{float(mine[0]["total_fft_uJ"]):.3f} vs {float(base[0]["total_fft_uJ"]):.3f} uJ')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks', 'check_tune_mit.txt'))
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--src26915', default=None)
    ap.add_argument('--date', default='20261101')
    ARGS = ap.parse_args()
    for fn in (k0, k1, k2, k3, k4):
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
