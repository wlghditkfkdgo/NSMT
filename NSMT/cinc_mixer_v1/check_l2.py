"""Checks of the MSE-loss option of train_forecast.py (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG L2 pre-registration). Fail closed: any failure stops with exit 1.
  k1  reported loss of train_epoch: with lr 0 on a small Linear, train_loss='mse' returns the batch mean of the squared error, 'l1' and the default return the absolute error;
      the gradients differ between the two; the default argument is L1.
  k2  run folders (--after): --repro <suite>: the L1 run of ETTh2 H96 seed 42 repeated with this code equals the tuned row of flif-v1-ett-spkmux-tune in every RESULT field
      but time_s, sha, dir, so the tuned L1 rows are valid controls.
  k3  run folders (--after): --smoke <suite>: for every run of the suite, the logged best epoch is the argmin of val_mse (tag ending in _mse) or of val_loss (otherwise),
      and the logged best value equals that column at that epoch (1e-6).
usage: python -B check_l2.py                                  (k1, no GPU)
       python -B check_l2.py --repro check-ett-l2-repro-20261107 --smoke smoke-ett-l2-20261107
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv, _argv = sys.argv[:1], sys.argv          # train_forecast only parses in main(); keep ours


def rows(path):
    out = []
    for line in open(path):
        if line.startswith('SKIP'):
            sys.exit(f'FAIL: SKIP line in {path}: {line[:160]}')
        if line.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in line.split()[1:] if '=' in t))
    return out


def k1():
    import torch
    import torch.nn as nn
    import train_forecast as tf

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 4, bias=False)

        def forward(self, x):
            return self.fc(x)

    def batches():
        g = torch.Generator().manual_seed(0)
        return [(torch.randn(8, 6, generator=g), torch.randn(8, 3, 4, generator=g), None, None) for _ in range(3)]

    class Wrap(nn.Module):
        """Output [B, H, C] as the forecaster; y is cut to the last H steps by train_epoch."""
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = Net()

        def forward(self, x):
            return self.net(x).unsqueeze(1).expand(-1, 3, -1)

    def expected(fn):
        m = Wrap()
        v = [fn(m(x), y.float()[:, -3:, :]).item() for x, y, _, _ in batches()]
        return sum(v) / len(v)

    grads = {}
    for kind, fn in (('l1', nn.L1Loss()), ('mse', nn.MSELoss())):
        m = Wrap()
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        got = tf.train_epoch(m, batches(), opt, 3, 'cpu', kind)['loss']
        exp = expected(fn)
        if not abs(got - exp) < 1e-6:
            sys.exit(f'FAIL k1: train_epoch {kind} loss {got} != expected {exp}')
        grads[kind] = m.net.fc.weight.grad.clone()
    m = Wrap()
    got = tf.train_epoch(m, batches(), torch.optim.SGD(m.parameters(), lr=0.0), 3, 'cpu')['loss']
    if not abs(got - expected(nn.L1Loss())) < 1e-6:
        sys.exit(f'FAIL k1: default loss {got} is not L1')
    if not (grads['l1'].abs().sum() > 0 and grads['mse'].abs().sum() > 0) or torch.equal(grads['l1'], grads['mse']):
        sys.exit('FAIL k1: gradients are zero or identical')
    print('k1 ok: train_epoch returns L1 by default and MSE with train_loss=mse; gradients non-zero and different')


def k2(suite):
    tune = {(r['ds'], r['pl'], r['tag'], r['seed']): r for r in rows(os.path.join(HERE, 'results', 'flif-v1-ett-spkmux-tune', 'raw.txt'))}
    got = rows(os.path.join(HERE, 'results', suite, 'raw.txt'))
    if len(got) != 1:
        sys.exit(f'FAIL k2: {len(got)} rows in {suite}, expected 1')
    r = got[0]
    ref = tune.get((r['ds'], r['pl'], r['tag'], r['seed']))
    if ref is None:
        sys.exit(f'FAIL k2: no tuned row for {r["ds"]} {r["pl"]} {r["tag"]} {r["seed"]}')
    skip = {'time_s', 'sha', 'dir'}
    if set(r) != set(ref):
        sys.exit(f'FAIL k2: fields differ: {sorted(set(r) ^ set(ref))}')
    bad = {k: (r[k], ref[k]) for k in r if k not in skip and r[k] != ref[k]}
    if bad:
        sys.exit(f'FAIL k2: {bad}')
    print(f'k2 ok: {len(r) - len(skip)} fields of the repeated L1 run equal the tuned row (mse {r["mse"]}, best {r["best_epoch_val_epochs"]})')


def k3(suite):
    rr = rows(os.path.join(HERE, 'results', suite, 'raw.txt'))
    if not rr:
        sys.exit(f'FAIL k3: no rows in {suite}')
    for r in rr:
        col = 'val_mse' if r['tag'].endswith('_mse') else 'val_loss'
        log = list(csv.DictReader(open(os.path.join(r['dir'], 'log', 'best_log_0.csv'))))
        be, val, ep = r['best_epoch_val_epochs'].split(':')
        v = [float(x[col]) for x in log]
        if len(v) != int(ep):
            sys.exit(f'FAIL k3: {r["tag"]}: {len(v)} log rows, {ep} epochs')
        if abs(v[int(be)] - float(val)) > 1e-6 or abs(min(v) - float(val)) > 1e-6:
            sys.exit(f'FAIL k3: {r["tag"]}: best {be} {val} is not the argmin of {col} {v}')
    print(f'k3 ok: {len(rr)} runs select the best epoch by the expected validation column')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro')
    ap.add_argument('--smoke')
    a = ap.parse_args(_argv[1:])
    if a.repro or a.smoke:
        if a.repro:
            k2(a.repro)
        if a.smoke:
            k3(a.smoke)
    else:
        k1()
