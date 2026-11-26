"""MODI 2026-11-26 (evening): checks for the two new diagnostic levels of order_check.py (SLEEPPSD = Sleep-EDF epoch spectra,
MBIP = MIT-BIH inter-patient beats with past beats). Fail closed: any failure exits non-zero.
usage (classification folder): python check_oc2.py
"""
import json
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
    from scipy.signal import butter, lfilter
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # a. MIT-BIH reader (header first values and checksums are checked inside build_mitdb, which stops on a mismatch)
    m = D.load_mitdb()
    pr = json.loads(str(m['per_rec']))
    ok('a1 record 100 = N 2239, S 33, V 1 (PhysioNet beat table)', pr['100'] == [2239, 33, 1, 0, 0], str(pr['100']))
    ok('a2 DS2 S beats of records 232 / 222 = 1382 / 209 (de Chazal 2004 Table VII)', (pr['232'][1], pr['222'][1]) == (1382, 209))
    t1 = np.sum([pr[str(r)] for r in D.DS1], 0).tolist(); t2 = np.sum([pr[str(r)] for r in D.DS2], 0).tolist()
    ok('a3 DS1 / DS2 class totals = the widely quoted counts', (t1, t2) == ([45866, 944, 3788, 415, 8], [44259, 1837, 3221, 388, 7]),
       f'{t1} / {t2} (de Chazal Table II sums {D.DECHAZAL["DS1"]} / {D.DECHAZAL["DS2"]})')
    ok('a4 DS1 and DS2 disjoint, 22 + 22, no paced record', len(set(D.DS1) | set(D.DS2)) == 44 and not ({102, 104, 107, 217} & set(D.DS1 + D.DS2)))
    # b. window alignment and causal filter
    fs, n, sig = D.read_mit_header(232)
    raw = D.read_mit_212(232, len(sig), n)
    c = [s['desc'] for s in sig].index('MLII')
    hb, ha = butter(2, 0.5, btype='high', fs=D.MIT_FS)
    x = lfilter(hb, ha, (raw[:, c] - sig[c]['zero']) / sig[c]['gain'])
    j = int(np.where(m['rec'] == 232)[0][500]); p = int(m['pos'][j])
    ok('b1 stored window = filtered MLII R-90 .. R+144 averaged in pairs', np.allclose(m['w'][j], x[p - 90:p + 144].reshape(-1, 2).mean(1), atol=1e-5))
    y2 = (raw[:, c] - sig[c]['zero']) / sig[c]['gain']; y2 = y2.copy(); y2[p + 144:] += 5.0
    x2 = lfilter(hb, ha, y2)
    ok('b2 high-pass is causal (changing the signal after R+144 leaves the window unchanged)', np.array_equal(x2[:p + 144], x[:p + 144]))
    ok('b3 pre-RR = distance to the previous beat annotation', abs(m['rr'][j] - (p - m['pos'][j - 1]) / 360.0) < 1e-6)
    # c. MBIP data
    d17 = O.Data('MBIP', O.parse_cond('MBIP', 'gru_orig_L17'), 0, dev)
    t = d17.t; rec = m['rec']
    sp = {k: set(np.unique(d17.rec_t[v])) for k, v in d17.split.items()}
    ok('c1 train = DS1 minus validation, validation = 201 / 118 / 223 / 114, test = DS2',
       sp['train'] == set(D.DS1) - set(O.MB_VAL) and sp['val'] == set(O.MB_VAL) and sp['test'] == set(D.DS2))
    ctx = d17.ctx.cpu().numpy()
    ok('c2 context = the 16 beats before the target, same record, target last', (ctx[:, -1] == t).all() and (np.diff(ctx, 1) == 1).all()
       and (rec[ctx] == rec[t][:, None]).all())
    ok('c3 targets are N / S / V / F only', set(np.unique(m['cls'][t])) <= {0, 1, 2, 3})
    cnt = {k: np.bincount(m['cls'][t[v]], minlength=4).tolist() for k, v in d17.split.items()}
    print('    targets per split N S V F:', cnt)
    # c4 (first version bounded the loss by 17 per record and missed the last beat of a record, whose window can pass the
    # signal end; corrected 11-26 after counting the reasons): every DS2 N/S/V/F beat that is not a target is one of the first
    # 17 beats of its record or has its window past the signal end, at most one such end beat per record.
    idx = np.zeros(len(rec), dtype=np.int64)
    for r in np.unique(rec):
        w = np.where(rec == r)[0]; idx[w] = np.arange(len(w))
    lost = np.isin(rec, D.DS2) & (m['cls'] <= 3) & ~np.isin(np.arange(len(rec)), t)
    start, end = lost & (idx < 17), lost & (idx >= 17) & ~m['ok']
    per_end = np.bincount(rec[end] - 100, minlength=200)
    ok('c4 DS2 beats that are not targets: first 17 of a record or window past the end (<= 1 per record), nothing else',
       int((lost & ~start & ~end).sum()) == 0 and per_end.max() <= 1, f'{int(lost.sum())} lost = {int(start.sum())} start + {int(end.sum())} end')
    xb, yb = d17.tokens(d17.split['test'][:3])
    ok('c5 MBIP L17 batch [B, 17, 118]; token = window / scale + pre-RR', tuple(xb.shape) == (3, 17, 118)
       and abs(float(xb[0, -1, -1]) - float(m['rr'][t[d17.split['test'][0]]])) < 1e-6)
    wz = xb[..., :117]
    j0 = int(t[d17.split['test'][0]])
    ok('c5b waveform part of a token = its own window standardised (mean 0, std 1; no record statistics)',
       torch.allclose(wz.mean(-1), torch.zeros_like(wz.mean(-1)), atol=1e-4) and torch.allclose(wz.std(-1, unbiased=False), torch.ones_like(wz.mean(-1)), atol=1e-3)
       and np.allclose(xb[0, -1, :117].cpu().numpy(), (m['w'][j0] - m['w'][j0].mean()) / (m['w'][j0].std() + 1e-6), atol=1e-4))
    drr = O.Data('MBIP', O.parse_cond('MBIP', 'gru_orig_L1_rr'), 0, dev)
    xr, _ = drr.tokens(drr.split['test'][:3])
    i0 = t[drr.split['test'][0]]
    ok('c6 rr baseline token [B, 1, 119], last value = mean pre-RR of the 16 beats before', tuple(xr.shape) == (3, 1, 119)
       and abs(float(xr[0, 0, -1]) - float(m['rr'][i0 - 16:i0].mean())) < 1e-5)
    trc = np.bincount(m['cls'][t[d17.split['train']]], minlength=4)
    ok('c7 balanced class weights from training targets', torch.allclose(d17.cw.cpu(), torch.tensor(trc.sum() / (4 * trc), dtype=torch.float32)))
    bad = 0
    for cond in ('gru_orig_L18', 'gru_orig_L2_rr', 'gru_orig', 'tf_shuf_L1_xx'):
        try:
            O.parse_cond('MBIP', cond)
        except SystemExit:
            bad += 1
    ok('c8 bad MBIP conditions rejected', bad == 4)
    # d. SLEEPPSD data
    psd = D.load_sleep_psd()
    xs, lab, tgt, subj, recs, off = D.load_sleep()
    ok('d1 one spectrum per kept epoch, 60 bins', psd.shape == (len(xs), 60))
    dp = O.Data('SLEEPPSD', O.parse_cond('SLEEPPSD', 'gru_orig_L16'), 2, dev)
    dc = O.Data('SLEEPCTX', O.parse_cond('SLEEPCTX', 'gru_orig_L16'), 2, dev)
    ok('d2 same targets, splits and context as SLEEPCTX', np.array_equal(dp.t, dc.t) and torch.equal(dp.ctx, dc.ctx)
       and all(np.array_equal(dp.split[k], dc.split[k]) for k in dp.split))
    tt = dp.t[dp.split['train']]
    mu, sd = psd[tt].mean(0), psd[tt].std(0)
    xp, yp = dp.tokens(dp.split['test'][:2])
    ok('d3 token = normalised spectrum of the epoch (per bin, training targets)', np.allclose(xp[0].cpu().numpy(), (psd[dp.ctx[dp.split['test'][0]].cpu().numpy()] - mu) / sd, atol=1e-4))
    del dc
    # e. models
    for ds, cond, L, f in (('MBIP', 'tf_orig_L17', 17, 118), ('MBIP', 'gru_orig_L1_rr', 1, 119), ('SLEEPPSD', 'tf_orig_L16', 16, 60)):
        c_ = O.parse_cond(ds, cond)
        mdl = O.RefModel(ds, c_, L, f).to(dev).eval()
        with torch.no_grad():
            o = mdl(torch.randn(4, L, f, device=dev))
        ok(f'e1 {ds} {cond} output [B, {O.NCLS[ds]}], reads the last token', tuple(o.shape) == (4, O.NCLS[ds]) and mdl.last)
    g = torch.Generator(device=dev); g.manual_seed(0)
    s = O.shuffle_tokens(xb, g, True)
    ok('e2 MBIP shuffle keeps the current beat last', torch.equal(s[:, -1], xb[:, -1]))
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
