"""MODI 2026-11-26: checks for the order / accumulation check (order_check.py, oc_data.py). Fail closed: any failure exits non-zero.
usage (classification folder): python check_oc.py
"""
import sys

import numpy as np
import torch

import oc_data as D
import order_check as O

FAILS = []


def ok(name, cond, info=''):
    print(f'{"PASS" if cond else "FAIL"} {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # a. shuffle_tokens: same tokens per sample, new order, last kept when asked, reproducible from the generator
    g = torch.Generator(device=dev); g.manual_seed(0)
    x = torch.randn(64, 25, 7, device=dev)
    s = O.shuffle_tokens(x, g, False)
    same_set = all(torch.equal(torch.sort(x[i, :, 0])[0], torch.sort(s[i, :, 0])[0]) for i in range(64))
    rows_kept = all(any(torch.equal(s[i, j], x[i, k]) for k in range(25)) for i in range(4) for j in range(25))
    moved = (s != x).any(-1).float().mean().item()
    ok('a1 shuffle keeps each sample\'s tokens', same_set and rows_kept)
    ok('a2 shuffle moves tokens', moved > 0.8, f'moved {moved:.3f}')
    g2 = torch.Generator(device=dev); g2.manual_seed(0)
    ok('a3 shuffle reproducible', torch.equal(O.shuffle_tokens(x, g2, False), s))
    perms = {tuple(torch.argsort(s[i, :, 0]).tolist()) for i in range(64)}
    ok('a4 permutation differs across samples', len(perms) > 60, f'{len(perms)} distinct of 64')
    sl = O.shuffle_tokens(x, g, True)
    ok('a5 keep_last keeps the target token', torch.equal(sl[:, -1], x[:, -1]) and (sl[:, :-1] != x[:, :-1]).any().item())
    ok('a6 one context epoch: nothing to shuffle', torch.equal(O.shuffle_tokens(x[:, :2], g, True), x[:, :2]))
    # b. conditions
    bad = 0
    for ds, cond in (('SC', 'gru_x'), ('SC', 'gru_orig_L4'), ('SLEEPCTX', 'gru_orig'), ('MITBIH', 'tf_orig_k5'), ('SC', 'lstm_orig')):
        try:
            O.parse_cond(ds, cond)
        except SystemExit:
            bad += 1
    ok('b1 bad conditions rejected', bad == 5)
    # c. Speech Commands tokens
    xs, ys, ss = D.load_sc()
    ok('c1 SC split counts', np.bincount(ss).tolist() == [84843, 9981, 11005], str(np.bincount(ss).tolist()))
    d = O.Data('SC', O.parse_cond('SC', 'gru_orig'), 0, dev)
    i = int(d.split['test'][3])
    raw = torch.from_numpy(xs[i, :, :100].astype(np.float32))
    tok = d.x[i].cpu().reshape(25, 4, 40)                                              # token j = frames 4j..4j+3
    tr = torch.from_numpy(ss == 0)
    m = torch.from_numpy(xs[:, :, :100].astype(np.float32))[tr].mean((0, 2))
    sd = torch.from_numpy(xs[:, :, :100].astype(np.float32))[tr].std((0, 2))
    ok('c2 SC token j = frames 4j..4j+3 (normalised)', torch.allclose(tok, ((raw - m[:, None]) / sd[:, None]).T.reshape(25, 4, 40), atol=1e-4))
    dk = O.Data('SC', O.parse_cond('SC', 'gru_orig_k10'), 0, dev)
    ok('c3 SC centre crop k10 = tokens 7..16', torch.equal(dk.x[i], d.x[i, 7:17]))
    del d, dk
    # d. MIT-BIH
    xtr, ytr, val, xte, yte = D.load_mitbih()
    ok('d1 MIT-BIH counts', (len(xtr), len(xte)) == (76612, 32833), f'{len(xtr)} / {len(xte)}')
    fr = [float(val[ytr == c].mean()) for c in range(5)]
    ok('d2 MIT-BIH validation stratified 10 %', all(abs(f - 0.1) < 0.002 for f in fr), str([round(f, 4) for f in fr]))
    dm = O.Data('MITBIH', O.parse_cond('MITBIH', 'gru_orig'), 0, dev)
    j = int(dm.split['test'][5])
    ok('d3 MIT-BIH token = 17 consecutive points', torch.equal(dm.x[j].cpu().reshape(-1), torch.from_numpy(xte[j - len(xtr)])))
    ok('d4 MIT-BIH splits disjoint', len(set(dm.split['train']) & set(dm.split['val'])) == 0 and len(dm.split['train']) + len(dm.split['val']) == len(xtr))
    del dm
    # e. Sleep-EDF
    x, lab, tgt, subj, rec, off = D.load_sleep()
    ok('e1 Sleep-EDF-20 target counts = published', np.bincount(lab[tgt], minlength=5).tolist() == D.SL_EXPECT)
    t = np.where(tgt)[0]
    tested = []
    for f in range(5):
        a, b, c = D.sleep_fold(subj[t], f)
        sa, sb, sc = set(subj[t][a]), set(subj[t][b]), set(subj[t][c])
        ok(f'e2 fold {f} subjects disjoint, 14 / 2 / 4', not (sa & sb or sa & sc or sb & sc) and (len(sa), len(sb), len(sc)) == (14, 2, 4))
        tested += sorted(sc)
    ok('e3 every subject tested once', sorted(tested) == list(range(20)))
    dl = O.Data('SLEEPCTX', O.parse_cond('SLEEPCTX', 'gru_orig_L16'), 2, dev)
    ctx = dl.ctx.cpu().numpy()
    ok('e4 context = the 15 raw epochs before the target, target last', (ctx[:, -1] == t).all() and (np.diff(ctx, axis=1) == 1).all())
    ok('e5 context inside the night', (rec[ctx] == rec[t][:, None]).all())
    xb, yb = dl.tokens(dl.split['test'][:4])
    ok('e6 SLEEPCTX batch [B, 16, 3000], label of the last epoch', tuple(xb.shape) == (4, 16, 3000)
       and torch.equal(yb.cpu(), torch.from_numpy(lab[t[dl.split['test'][:4]]])))
    di = O.Data('SLEEPIN', O.parse_cond('SLEEPIN', 'gru_orig'), 2, dev)
    xi, _ = di.tokens(di.split['test'][:4])
    ok('e7 SLEEPIN token j = seconds j of the target epoch', torch.equal(xi.reshape(4, -1), dl.tokens(dl.split['test'][:4])[0][:, -1]))
    # f. models: order matters to both reference models; without the position table the Transformer is order-blind
    torch.manual_seed(0)
    for kind in ('gru', 'tf'):
        mdl = O.RefModel('SC', dict(model=kind, order='orig', k=None, L=None), 25, 160).to(dev).eval()
        z = torch.randn(8, 25, 160, device=dev)
        p = torch.randperm(25, device=dev)
        with torch.no_grad():
            diff = (mdl(z) - mdl(z[:, p])).abs().max().item()
        ok(f'f1 {kind} output depends on token order', diff > 1e-4, f'{diff:.2e}')
        if kind == 'tf':
            with torch.no_grad():
                mdl.pos.zero_()
                diff0 = (mdl(z) - mdl(z[:, p])).abs().max().item()
            ok('f2 tf without position table is order-blind (mean pooling)', diff0 < 1e-4, f'{diff0:.2e}')
    mctx = O.RefModel('SLEEPCTX', dict(model='tf', order='orig', k=None, L=16), 16, 3000).to(dev).eval()
    with torch.no_grad():
        o = mctx(xb)
    ok('f3 SLEEPCTX model output [B, 5]', tuple(o.shape) == (4, 5))
    # g. evaluation is reproducible (fixed permutation seed)
    mm = O.RefModel('SLEEPIN', dict(model='gru', order='orig', k=None, L=None), 30, 100).to(dev)
    e1 = O.evaluate(mm, di, di.split['val'][:3000], True, False)
    e2 = O.evaluate(mm, di, di.split['val'][:3000], True, False)
    ok('g1 shuffled evaluation reproducible', e1 == e2)
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
