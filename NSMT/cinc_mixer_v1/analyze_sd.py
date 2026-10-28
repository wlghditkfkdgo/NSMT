"""Post-hoc analysis of the spiking-difference fusion suite (PROJECT_LOG 2026-10-28; not pre-registered, no training, no new
evaluation): reads the saved outputs of the finished CinC fold-0 runs only.

Conditions per neuron n in (lif, flif, pop_lif, pop_flif, popg_lif, popg_flif): Neo-mixer OR, spike linear fusion (mixlin_n), spike xor
(mixlinxor_n), spiking difference (mixlinsd_n), charge difference (3) (mixlinmx_n), and Spikformer-style (spk_ssa); seeds 42 / 7 / 2026.
Run directories come from the RESULT lines (dir=) of the baseline copies (results/BASELINE_COPIES.txt) and of
results/flif-v1-rhythm-sd-cinc/raw.txt; every (cond, seed) must be found exactly once (fail closed).
Per run: log/test_probs.npz (record-level class probabilities of the best-validation weights), log/best_log_0.csv (per-epoch train loss
and validation score). The CinC score recomputed from test_probs must equal the stored score (1e-6, fail closed).
Reported:
  A  per class (N, AF, Other, Noisy): F1 (mean of seeds), recall / precision (pooled over seeds), one-vs-rest AUC (ranking, independent
     of the decision threshold), best F1 if that class's threshold were moved (oracle on test: an upper bound, reported only)
  B  paired bootstrap over the 1,706 test records (2,000 resamples, fixed generator): 95% interval of the seed-mean score difference --
     the uncertainty from which records are in the test set, with the three trained models of each condition fixed
  C  prediction agreement between conditions (same record, same seed) next to the agreement of one condition with itself across seeds
  D  training curves: validation score at epochs 4 / 9 / 19 / 29, best epoch, train loss at the best and the last epoch
  E  test score of the last-epoch weights (last_score of the RESULT line) next to the best-validation weights -- the registered rule is
     the best-validation weights; the last-epoch column only shows how much the checkpoint choice moves the numbers -- and the
     energy parts (AC / MAC operations, synaptic and neuron energy) of the RESULT lines
usage: python analyze_sd.py [--out results/sd-analysis-20261028]
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import rankdata

import summarize_rhythm as S

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
NS = ('lif', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
CLASSES = ('N', 'AF', 'Other', 'Noisy')
KINDS = ('or', 'lin', 'xor', 'sd', 'mx')


def cond(kind, n):
    if kind == 'or':
        return f'mix_{n}_bidir' if n.startswith('popg') else S._mix_cond(n)
    return {'lin': 'mixlin_', 'xor': 'mixlinxor_', 'sd': 'mixlinsd_', 'mx': 'mixlinmx_'}[kind] + n


def run_dirs():
    rows = S._all_fold0() + S._fold0(os.path.join(HERE, 'results', 'flif-v1-rhythm-sd-cinc', 'raw.txt'))
    need = ['spk_ssa'] + [cond(k, n) for n in NS for k in KINDS]
    found = {}
    for r in rows:
        if r['cond'] not in need:
            continue
        key = (r['cond'], int(r['seed']))
        if key in found and found[key]['dir'] != r['dir']:
            raise SystemExit(f'{key}: two different run directories')
        found[key] = r
    missing = [(c, s) for c in need for s in SEEDS if (c, s) not in found]
    if missing:
        raise SystemExit(f'missing runs: {missing}')
    return found, need


def f1_all(Y, pred):
    c = np.zeros((4, 4)); np.add.at(c, (Y, pred), 1)
    tp = np.diag(c)
    return 2 * tp / np.maximum(c.sum(0) + c.sum(1), 1), c


def score_batch(Y, pred, idx):
    """CinC score (mean F1 of N, AF, Other) for every bootstrap row of record indices idx [B, n]."""
    y, p = Y[idx], pred[idx]
    f = []
    for k in range(3):
        tp = ((y == k) & (p == k)).sum(1); fp = ((y != k) & (p == k)).sum(1); fn = ((y == k) & (p != k)).sum(1)
        f.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
    return np.mean(f, axis=0)


def auc(score, pos):
    """one-vs-rest ROC AUC by ranks (ties averaged)."""
    ranks = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold_f1(score, pos):
    order = np.argsort(-score)
    tp = np.cumsum(pos[order]); fp = np.cumsum(~pos[order]); fn = pos.sum() - tp
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1)).max())


def curve(d):
    rows = list(csv.DictReader(open(os.path.join(d, 'log', 'best_log_0.csv'))))
    val = np.array([float(r['val_score']) for r in rows]); loss = np.array([float(r['train_loss']) for r in rows])
    b = int(np.argmax(val))                                            # first maximum, as the trainer keeps the first best
    return dict(val_at={e: float(val[e]) for e in (4, 9, 19, 29) if e < len(val)}, best_epoch=b, best_val=float(val[b]),
                loss_best=float(loss[b]), loss_last=float(loss[-1]), epochs=len(val))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'sd-analysis-20261028'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    found, need = run_dirs()
    runs, U0 = {}, None
    for (c, s), r in found.items():
        z = np.load(os.path.join(r['dir'], 'log', 'test_probs.npz'))
        P, Y, U = z['P'], z['Y'].astype(int), z['units']
        if U0 is None:
            U0, Y0 = U, Y
        if not (np.array_equal(U, U0) and np.array_equal(Y, Y0)):
            raise SystemExit(f'{c} seed {s}: different test records or labels')
        pred = P.argmax(1)
        f, _ = f1_all(Y, pred)
        if abs(f[:3].mean() - float(r['score'])) > 1e-6:
            raise SystemExit(f'{c} seed {s}: recomputed score {f[:3].mean()} vs stored {r["score"]}')
        runs[(c, s)] = dict(P=P, pred=pred, f1=f, curve=curve(r['dir']), dir=r['dir'])
    Y = Y0
    rec = {'n_test_records': int(len(Y)), 'class_counts': np.bincount(Y, minlength=4).tolist(),
           'runs': {f'{c}/s{s}': v['dir'] for (c, s), v in runs.items()}}
    L = [f'# Spiking-difference fusion, post-hoc analysis (CinC fold 0, test records {len(Y)}, class counts '
         + ', '.join(f'{k} {n}' for k, n in zip(CLASSES, np.bincount(Y, minlength=4))) + ')', '']
    # A: per class
    A = {}
    for c in need:
        cm = sum(f1_all(Y, runs[(c, s)]['pred'])[1] for s in SEEDS)
        A[c] = dict(f1=np.mean([runs[(c, s)]['f1'] for s in SEEDS], 0).tolist(),
                    recall=(np.diag(cm) / cm.sum(1)).tolist(), precision=(np.diag(cm) / np.maximum(cm.sum(0), 1)).tolist(),
                    pred_share=(cm.sum(0) / cm.sum()).tolist(),
                    auc=[float(np.mean([auc(runs[(c, s)]['P'][:, k], Y == k) for s in SEEDS])) for k in range(4)],
                    oracle_f1=[float(np.mean([best_threshold_f1(runs[(c, s)]['P'][:, k], Y == k) for s in SEEDS])) for k in range(4)])
    rec['A_per_class'] = A
    true_share = np.bincount(Y, minlength=4) / len(Y)
    L += ['## A. Per class (F1 = mean of seeds; recall / precision / predicted share pooled over seeds; AUC and oracle-threshold F1 = mean of seeds), %', '',
          f'true share: ' + ' / '.join(f'{k} {100 * v:.1f}' for k, v in zip(CLASSES, true_share)), '',
          '| condition | F1 N / AF / Other / Noisy | recall | precision | predicted share | AUC | oracle-threshold F1 |', '|---|---|---|---|---|---|---|']
    fmt = lambda v, m=100, p=1: ' / '.join(f'{m * x:.{p}f}' for x in v)
    for c in need:
        r = A[c]
        L.append(f"| {c} | {fmt(r['f1'])} | {fmt(r['recall'])} | {fmt(r['precision'])} | {fmt(r['pred_share'])} | {fmt(r['auc'], 1, 3)} | {fmt(r['oracle_f1'])} |")
    # B: paired bootstrap
    rng = np.random.default_rng(20261028)
    idx = rng.integers(0, len(Y), size=(2000, len(Y)))
    sc = {c: np.mean([score_batch(Y, runs[(c, s)]['pred'], idx) for s in SEEDS], 0) for c in need}
    pairs = []
    for n in NS:
        pairs += [(cond('sd', n), 'spk_ssa'), (cond('sd', n), cond('mx', n)), (cond('sd', n), cond('lin', n)),
                  (cond('sd', n), cond('xor', n)), (cond('sd', n), cond('or', n))]
    pairs += [(cond('mx', 'lif'), 'spk_ssa')]
    B = {}
    L += ['', '## B. Paired bootstrap over test records (2,000 resamples; seed-mean score difference A - B, %p)', '',
          '| A | B | observed | 95% interval | share of resamples with A > B |', '|---|---|---|---|---|']
    for x, y in pairs:
        obs = float(np.mean([runs[(x, s)]['f1'][:3].mean() - runs[(y, s)]['f1'][:3].mean() for s in SEEDS]))
        dd = sc[x] - sc[y]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        B[f'{x} - {y}'] = dict(observed=obs, lo=float(lo), hi=float(hi), p_gt=float((dd > 0).mean()))
        L.append(f'| {x} | {y} | {100 * obs:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | {100 * (dd > 0).mean():.1f}% |')
    rec['B_bootstrap'] = B
    # C: agreement
    C = {}
    L += ['', '## C. Prediction agreement (share of test records with the same predicted class, mean of seeds), %', '',
          '| neuron | sd vs itself across seeds | (3) vs itself | sd vs (3) | sd vs spike linear | sd vs Spikformer-style | (3) vs Spikformer-style | Spikformer-style vs itself |',
          '|---|---|---|---|---|---|---|---|']
    self_agree = lambda c: float(np.mean([(runs[(c, s)]['pred'] == runs[(c, t)]['pred']).mean() for s, t in ((42, 7), (42, 2026), (7, 2026))]))
    same = lambda x, y: float(np.mean([(runs[(x, s)]['pred'] == runs[(y, s)]['pred']).mean() for s in SEEDS]))
    spk_self = self_agree('spk_ssa')
    for n in NS:
        sd, mx = cond('sd', n), cond('mx', n)
        C[n] = dict(sd_self=self_agree(sd), mx_self=self_agree(mx), sd_mx=same(sd, mx), sd_lin=same(sd, cond('lin', n)),
                    sd_spk=same(sd, 'spk_ssa'), mx_spk=same(mx, 'spk_ssa'), spk_self=spk_self)
        v = C[n]
        L.append(f"| {n} | {100 * v['sd_self']:.1f} | {100 * v['mx_self']:.1f} | {100 * v['sd_mx']:.1f} | {100 * v['sd_lin']:.1f} | "
                 f"{100 * v['sd_spk']:.1f} | {100 * v['mx_spk']:.1f} | {100 * spk_self:.1f} |")
    rec['C_agreement'] = C
    # D: curves
    D = {}
    L += ['', '## D. Training curves (validation score %, mean of seeds; train loss = CAL-weighted CE)', '',
          '| condition | val @ epoch 4 / 9 / 19 / 29 | best epoch (per seed) | best val | train loss at best / last epoch |', '|---|---|---|---|---|']
    for c in need:
        cs = [runs[(c, s)]['curve'] for s in SEEDS]
        D[c] = cs
        va = ' / '.join(f"{100 * np.mean([q['val_at'][e] for q in cs]):.2f}" for e in (4, 9, 19, 29))
        L.append(f"| {c} | {va} | {', '.join(str(q['best_epoch']) for q in cs)} | {100 * np.mean([q['best_val'] for q in cs]):.2f} | "
                 f"{np.mean([q['loss_best'] for q in cs]):.3f} / {np.mean([q['loss_last'] for q in cs]):.3f} |")
    rec['D_curves'] = D
    # E: last-epoch weights and energy parts
    E = {}
    L += ['', '## E. Checkpoint choice and energy parts (mean of seeds)', '',
          '| condition | test score, best-validation weights | test score, last-epoch weights | last - best | AC ops M | MAC ops M | synaptic µJ | neuron µJ | total µJ |',
          '|---|---|---|---|---|---|---|---|---|']
    for c in need:
        m = lambda k: float(np.mean([float(found[(c, s)][k]) for s in SEEDS]))
        E[c] = {k: m(k) for k in ('score', 'last_score', 'AC_op_M', 'MAC_op_M', 'syn_energy_uJ', 'neuron_model_fft_uJ', 'total_fft_uJ')}
        e = E[c]
        L.append(f"| {c} | {100 * e['score']:.2f} | {100 * e['last_score']:.2f} | {100 * (e['last_score'] - e['score']):+.2f} | {e['AC_op_M']:.2f} | "
                 f"{e['MAC_op_M']:.2f} | {e['syn_energy_uJ']:.1f} | {e['neuron_model_fft_uJ']:.1f} | {e['total_fft_uJ']:.1f} |")
    rec['E_checkpoint_energy'] = E
    json.dump(rec, open(os.path.join(a.out, 'analysis.json'), 'x'), indent=1)
    open(os.path.join(a.out, 'analysis.md'), 'x').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
