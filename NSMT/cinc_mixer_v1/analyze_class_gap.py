"""Where does the macro-F1 gap of the LIF charge-difference Neo-mixer (mixlinmx_lif) to Spikformer-style (spk_ssa) come from?
(MODI_PROJECT_LOG 2026-10-26; report only, no training, no new evaluation.)

Reads the saved test outputs of every run (CinC folds 0-4 x seeds 42 / 7 / 2026): log/confusion_matrix.csv and log/test_probs.npz
(record-level class probabilities P, labels Y, record ids) of the best-validation weights, for spk_ssa, mixlinmx_lif and
mixlin_lif. The run directories come from the RESULT lines (dir=) of the suites (fold 0: stage 9, membrane fusion, linear fusion;
folds 1-4: folds suite incl. raw_recovered.txt); every (cond, fold, seed) must be found exactly once (fail closed).
Reported, pooled over the 15 runs of each model and per fold:
  A  per class (N, AF, Other, Noisy): recall, precision, F1, share of records predicted as that class vs its true share
  B  pooled confusion matrix (row = true class, share of the row)
  C  one-vs-rest AUC per class from P (ranking, independent of the decision threshold) -- separates "the model ranks the class
     worse" from "the argmax decision is biased"
  D  the best achievable F1 per class if the class's own decision threshold were moved (oracle on test, so an upper bound of
     what re-weighting could do; reported, not used)
  E  record length: share of each class in short (< 30 s, tiled), medium (30-45 s), long (> 45 s) records and per-bin recall
  F  paired records: per true class, records the two models disagree on (one right, one wrong)
usage: python analyze_class_gap.py [--out results/class-gap-20261026]
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ('N', 'AF', 'Other', 'Noisy')
CONDS = ('spk_ssa', 'mixlinmx_lif', 'mixlin_lif')
SEEDS = (42, 7, 2026)
SOURCES = {0: ('flif-v1-rhythm-cinc', 'flif-v1-rhythm-mem-cinc', 'flif-v1-rhythm-lin-cinc'),
           1: ('flif-v1-rhythm-folds-cinc',)}


def run_dirs():
    found = {}
    for fold_key, suites in SOURCES.items():
        for suite in suites:
            for name in ('raw.txt', 'raw_recovered.txt'):
                path = os.path.join(HERE, 'results', suite, name)
                if not os.path.exists(path):
                    continue
                for line in open(path):
                    if not line.startswith('RESULT '):
                        continue
                    d = dict(re.findall(r'(\S+?)=(\S+)', line))
                    if d['cond'] not in CONDS or int(d.get('crop', 0)) != 0:
                        continue
                    fold = int(d['fold'])
                    if (fold == 0) != (fold_key == 0):
                        continue
                    key = (d['cond'], fold, int(d['seed']))
                    if key in found:
                        raise SystemExit(f'{key} twice')
                    found[key] = d['dir']
    missing = [(c, f, s) for c in CONDS for f in range(5) for s in SEEDS if (c, f, s) not in found]
    if missing:
        raise SystemExit(f'missing runs: {missing}')
    return found


def auc(score, pos):
    """one-vs-rest ROC AUC by ranks (ties averaged)."""
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    s = np.asarray(score)
    for v in np.unique(s):                                             # average ranks of ties
        idx = s == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold_f1(score, pos):
    """max F1 over thresholds on one class's probability (one-vs-rest)."""
    order = np.argsort(-score)
    tp = np.cumsum(pos[order]); fp = np.cumsum(~pos[order]); fn = pos.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1.max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'class-gap-20261026'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    import rhythm_data as rd
    sig, off, labels, names = rd.cinc_cache(125.)
    seconds = (off[1:] - off[:-1]) / 125.
    dirs = run_dirs()
    runs = {}
    for (c, f, s), d in dirs.items():
        z = np.load(os.path.join(d, 'log', 'test_probs.npz'))
        P, Y, U = z['P'], z['Y'], z['units']
        if not np.array_equal(Y, labels[U]):
            raise SystemExit(f'{d}: labels of the saved records differ from the cache')
        cm = np.loadtxt(os.path.join(d, 'log', 'confusion_matrix.csv'), delimiter=',', skiprows=1)[:, 1:]
        pred = P.argmax(1)
        cm2 = np.zeros((4, 4)); np.add.at(cm2, (Y, pred), 1)
        if not np.array_equal(cm, cm2):
            raise SystemExit(f'{d}: confusion_matrix.csv differs from argmax of test_probs')
        runs[(c, f, s)] = dict(P=P, Y=Y, U=U, pred=pred, cm=cm)
    rec = {'runs': {f'{c}/f{f}/s{s}': d for (c, f, s), d in dirs.items()}}
    lines = []
    # A, B
    for c in CONDS:
        cm = sum(runs[(c, f, s)]['cm'] for f in range(5) for s in SEEDS)
        rec_c = {}
        tp = np.diag(cm); recall = tp / cm.sum(1); precision = tp / np.maximum(cm.sum(0), 1)
        f1 = 2 * tp / np.maximum(cm.sum(1) + cm.sum(0), 1)
        per_run_f1 = np.array([[2 * runs[(c, f, s)]['cm'][k, k] / max(runs[(c, f, s)]['cm'][k].sum() + runs[(c, f, s)]['cm'][:, k].sum(), 1)
                                for k in range(4)] for f in range(5) for s in SEEDS])
        rec_c['pooled_confusion'] = cm.tolist()
        rec_c['recall'], rec_c['precision'], rec_c['f1_pooled'] = recall.tolist(), precision.tolist(), f1.tolist()
        rec_c['f1_mean_of_runs'] = per_run_f1.mean(0).tolist(); rec_c['f1_sd_of_runs'] = per_run_f1.std(0, ddof=1).tolist()
        rec_c['pred_share'] = (cm.sum(0) / cm.sum()).tolist(); rec_c['true_share'] = (cm.sum(1) / cm.sum()).tolist()
        # C, D
        aucs, oracle = [], []
        for k in range(4):
            a_k = [auc(runs[(c, f, s)]['P'][:, k], runs[(c, f, s)]['Y'] == k) for f in range(5) for s in SEEDS]
            o_k = [best_threshold_f1(runs[(c, f, s)]['P'][:, k], runs[(c, f, s)]['Y'] == k) for f in range(5) for s in SEEDS]
            aucs.append((float(np.mean(a_k)), float(np.std(a_k, ddof=1)))); oracle.append(float(np.mean(o_k)))
        rec_c['auc'] = aucs; rec_c['oracle_threshold_f1'] = oracle
        # E: record length bins
        bins = {'short (<30 s, tiled)': (0, 30.), 'medium (30-45 s)': (30., 45.), 'long (>45 s)': (45., 1e9)}
        lb = {}
        for bname, (lo, hi) in bins.items():
            row = {}
            for k in range(4):
                hit, n = 0, 0
                for f in range(5):
                    for s in SEEDS:
                        r = runs[(c, f, s)]
                        m = (r['Y'] == k) & (seconds[r['U']] >= lo) & (seconds[r['U']] < hi)
                        hit += int((r['pred'][m] == k).sum()); n += int(m.sum())
                row[CLASSES[k]] = (hit / n if n else float('nan'), n // len(SEEDS))
            lb[bname] = row
        rec_c['recall_by_length'] = lb
        rec[c] = rec_c
    # F: disagreements spk_ssa vs mixlinmx_lif, per true class (same records per fold)
    dis = {}
    for k in range(4):
        only_a, only_b, both_wrong, n = 0, 0, 0, 0
        for f in range(5):
            for s in SEEDS:
                ra, rb = runs[('spk_ssa', f, s)], runs[('mixlinmx_lif', f, s)]
                if not np.array_equal(ra['U'], rb['U']):
                    raise SystemExit(f'fold {f}: different test records')
                m = ra['Y'] == k
                ca, cb = ra['pred'][m] == k, rb['pred'][m] == k
                only_a += int((ca & ~cb).sum()); only_b += int((~ca & cb).sum()); both_wrong += int((~ca & ~cb).sum()); n += int(m.sum())
        dis[CLASSES[k]] = dict(n=n, only_spikformer_right=only_a, only_mixer_right=only_b, both_wrong=both_wrong)
    rec['disagreements_spk_vs_mixlinmx'] = dis
    # where do the mixer's Other / Noisy errors go, and where do Spikformer's
    json.dump(rec, open(os.path.join(a.out, 'class_gap.json'), 'x'), indent=1)
    def pct(v):
        return f'{100 * v:.1f}'
    for c in CONDS:
        r = rec[c]
        lines.append(f'== {c} (15 runs, pooled test records {int(np.sum(r["pooled_confusion"]))})')
        lines.append('   class   recall precision F1(pooled) F1(mean±sd of runs)  pred share / true share   AUC (mean±sd)   oracle-threshold F1')
        for k, name in enumerate(CLASSES):
            lines.append(f'   {name:<6} {pct(r["recall"][k]):>6} {pct(r["precision"][k]):>9} {pct(r["f1_pooled"][k]):>10} '
                         f'{pct(r["f1_mean_of_runs"][k]):>8} ± {pct(r["f1_sd_of_runs"][k]):<5} {pct(r["pred_share"][k]):>12} / {pct(r["true_share"][k]):<6} '
                         f'{r["auc"][k][0]:.3f} ± {r["auc"][k][1]:.3f}   {pct(r["oracle_threshold_f1"][k])}')
        cm = np.array(r['pooled_confusion']); rows = cm / cm.sum(1, keepdims=True)
        lines.append('   confusion (row = true, share of the row): ' + ' | '.join(
            f'{CLASSES[i]} -> ' + ' '.join(f'{CLASSES[j]} {pct(rows[i, j])}' for j in range(4)) for i in range(4)))
        lines.append('   recall by record length: ' + ' | '.join(
            f'{b}: ' + ' '.join(f'{k} {pct(v[0])} (n={v[1]})' for k, v in row.items()) for b, row in r['recall_by_length'].items()))
    lines.append('== disagreements spk_ssa vs mixlinmx_lif (15 paired runs): ' + ' | '.join(
        f"{k}: n {v['n']}, only Spikformer right {v['only_spikformer_right']}, only mixer right {v['only_mixer_right']}, both wrong {v['both_wrong']}"
        for k, v in dis.items()))
    open(os.path.join(a.out, 'class_gap.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
