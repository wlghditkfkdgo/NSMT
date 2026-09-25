"""Safety supplement for the frozen 2J / 2K checkpoints (prereg 2L, D-AU).

Audits 39 and 40 found that the 2J and 2K evaluators checked state safety on the first four
batches only (256 of 1000 sequences), ignored the diagnostic's own `finite` flag, passed a
missing bound, and -- for the random control -- measured error and diagnostics on different
random masks. The recorded performance numbers stand; what was missing is the safety evidence.

This script supplies it without touching any decision:

  * every target row comes from the existing records, and its checkpoint hash, its config
    manifest and `epochs_run == 12` are checked before its split is read;
  * the WHOLE split is passed once, in the original order and batch size, with the random
    generator reseeded to the training seed; each batch is ONE forward that yields both the
    recall-error sums and every state;
  * the recomputed recall MSE must equal the recorded one (|diff| <= 1e-12) -- that proves the
    safety numbers belong to the trajectory the verdict was computed on;
  * safe = every state finite AND max|u| < the frozen bound; no bound = not safe;
  * per D-AN / D-AR, a model that fails safety or provenance withholds every verdict it enters.

Descriptive only (same forward, never used for a verdict): the kept kernel mass sum(m b)/sum(b)
per condition, and for pearson the rate of exact ties at the top-k cut on the trained model.

The record is created with 'x' before any split is read, so the pass cannot run twice.

Usage:
    python hard_safety.py --twoj results/hardsel-014614/hard_selection_record.json \
                          --twok results/hardctrl-152631/hard_control_record.json --out hardsafety-XXXXXX
"""
import sys
import glob
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK                                        # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from test import EVENT_KINDS, kind_mask                        # noqa: E402
import layers                                                  # noqa: E402
from hard_selection import load, sha256, SOURCES               # noqa: E402

SEEDS = (7, 13, 21, 42, 123, 256, 512, 1024)
REGISTERED = {'mode': 'hard', 'hard_axis': 'shared', 'n_train': 2048, 'n_val': 256, 'batch_size': 64,
              'epoch': 12, 'readout': 'spike', 'data_seed': 20260921, 'n_confirm': 1000}
BOUND = 305.0375175476074
TOL = 1e-12
ANALYSIS = ('hard_selection.py', 'hard_confirm.py', 'hard_control.py', 'hard_safety.py',
            'stage_decomposition.py')


def find(suite, seed, stat, q):
    runs = glob.glob(str(TASK / 'log' / suite / 'recall' / '*' / '*' / f'seed{seed}_*hard-shared-{stat}-q{q:g}_k3'))
    res = glob.glob(str(TASK / 'results' / suite / f'*hard-shared-{stat}-q{q:g}_k3_seed{seed}.json'))
    if len(runs) != 1 or len(res) != 1:
        raise SystemExit(f'[safety] {suite} seed {seed} {stat} q={q:g}: {len(runs)} dirs, {len(res)} results')
    return runs[0], res[0]


def manifest(config, result, stat, q, seed):
    """Every registered field, plus the frozen bound and a completed 12-epoch run."""
    want = {**REGISTERED, 'hard_stat': stat, 'hard_q': q, 'seed': seed}
    bad = [f'{k}: {getattr(config, k, None)!r} != {v!r}' for k, v in want.items() if getattr(config, k, None) != v]
    if getattr(config, 'g11_bound', None) != BOUND:
        bad.append(f'g11_bound: {getattr(config, "g11_bound", None)!r} != {BOUND!r}')
    epochs = json.load(open(result)).get('train', {}).get('epochs_run')
    if epochs != REGISTERED['epoch']:
        bad.append(f'epochs_run: {epochs!r} != {REGISTERED["epoch"]}')
    return bad


@torch.no_grad()
def single_pass(model, loader, config, stat, q, seed):
    """One forward per batch over the whole split: error sums, states, kept mass, ties."""
    neuron = model.embedding.neuron
    sel, b = neuron.selector, neuron.b
    sel.reseed_hard(seed)                                      # random: the original evaluation's stream
    model.eval()
    sums = {name: torch.zeros(3, dtype=torch.float64) for name in EVENT_KINDS}
    peak, finite, nonfinite_batches, seqs = 0., True, 0, 0
    kept_mass, kept_n = 0., 0
    ties, tie_n, mask_mismatch = 0, 0, 0
    for x, y, truth, kind in loader:
        x, y = x.float(), y.float()
        output, aux = model(x, mode=config.mode, truth=truth if truth.numel() else None, kind=None,
                            return_aux=True)
        error = (output - y).double()                          # test.evaluate, same order of sums
        for name, want in EVENT_KINDS.items():
            m = kind_mask(kind, want)
            sums[name][0] += (error * m).square().sum()
            sums[name][1] += (error.abs() * m).sum()
            sums[name][2] += m.sum()
        state = aux['state']                                   # [T, B, D, K]; state[t] = u(t+1)
        seqs += x.shape[0]
        ok = bool(torch.isfinite(state).all() and torch.isfinite(output).all())
        if not ok:
            finite = False
            nonfinite_batches += 1                             # never folded into the max
        else:
            peak = max(peak, state.abs().max().item())
        for n in range(1, len(aux['coeff'])):
            c = aux['coeff'][n]                                # [B, D, n] = b * m
            bh = b[1:n + 1].flip(0)
            kept_mass += (c.sum(-1) / bh.sum()).mean().item()
            kept_n += 1
        if stat == 'pearson' and q < 1.:
            current = model.embedding.current(layers.to_patches(x, config.patch_size))
            zero = torch.zeros_like(state[0])
            for n in range(1, current.shape[0]):
                xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
                hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                                    for j in range(n)], dim=-2)
                m, sim = sel.hard_mask(xi, hist)
                mask_mismatch += int(not torch.equal(m, (aux['coeff'][n] > 0).to(m.dtype)))
                k = max(1, int(round(q * n)))
                if k < n:
                    srt = sim[:, 0, :].sort(dim=-1, descending=True).values
                    ties += int((srt[:, k - 1] == srt[:, k]).sum())
                tie_n += x.shape[0]
    res = {name: (t[0] / t[2]).item() for name, t in sums.items() if t[2] > 0}
    safe = finite and peak < BOUND
    return {'recall_mse': res['recall'], 'copy_mse': res['copy'], 'recall_first_mse': res['recall_first'],
            'all_mse': res['all'], 'sequences': seqs, 'finite': finite, 'nonfinite_batches': nonfinite_batches,
            'max_abs_state': peak, 'bound': BOUND, 'safe': safe,
            'kept_mass_frac': kept_mass / max(kept_n, 1),
            'tie_rate': (ties / tie_n) if tie_n else None, 'tie_queries': tie_n, 'mask_reconstruction_mismatch': mask_mismatch}


def targets(twoj, twok):
    """(split, label, seed, stat, q, run, result, recorded mse, recorded ckpt hash) from the records."""
    rows = []
    j = json.load(open(twoj))['confirm']
    for r in j['rows']:
        run, res = find(j['suite'], r['seed'], 'pearson', float(r['q']))
        rows.append(('confirm2', '2J', r['seed'], 'pearson', float(r['q']), run, res, r['recall_mse'], r['checkpoint_sha256']))
    k = json.load(open(twok))
    for r in k['rows']:
        stat, q = {'pearson': ('pearson', .5), 'q1': ('pearson', 1.), 'recent': ('recent', .5),
                   'random': ('random', .5)}[r['condition']]
        suite = k['twoj_suite'] if r['condition'] in ('pearson', 'q1') else k['control_suite']
        run, res = find(suite, r['seed'], stat, q)
        rows.append(('confirm3', '2K/' + r['condition'], r['seed'], stat, q, run, res, r['recall_mse'], r['checkpoint_sha256']))
    return rows


def main():
    parser = argparse.ArgumentParser(description='prereg 2L D-AU: safety supplement for frozen checkpoints')
    parser.add_argument('--twoj', required=True)
    parser.add_argument('--twok', required=True)
    parser.add_argument('--out', required=True, help='new results/ directory name for the record')
    cli = parser.parse_args()
    out = TASK / 'results' / cli.out
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'hard_safety_record.json'

    rows = targets(TASK / cli.twoj, TASK / cli.twok)
    head = {'prereg': '2L D-AU', 'twoj': cli.twoj, 'twok': cli.twok, 'registered': REGISTERED, 'bound': BOUND,
            'source_sha256': {f: sha256(TASK / f) for f in SOURCES},
            'analysis_sha256': {f: sha256(HERE / f) for f in ANALYSIS}}
    with open(record_path, 'x') as handle:                     # 분할을 읽기 전에 기록한다
        json.dump({**head, 'status': 'opening confirm2 and confirm3 for the safety pass'}, handle, indent=2)

    results = []
    print(f"{'split':<9} {'label':<11} {'seed':>5} {'recorded':>10} {'recomputed':>11} {'|diff|':>9} "
          f"{'seqs':>5} {'max|u|':>7} {'finite':>6} {'safe':>5} {'kept mass':>9} {'tie':>9} {'manifest':>8}")
    for split, label, seed, stat, q, run, res, mse_rec, ckpt_rec in rows:
        ckpt = sha256(Path(run) / 'model_state' / 'best+model.pt')
        model, config = load(run)
        bad = manifest(config, res, stat, q, seed)
        if ckpt != ckpt_rec:
            bad.append('checkpoint sha256 differs from the record')
        _, loader = data_provider(config, split)
        r = single_pass(model, loader, config, stat, q, seed)
        diff = abs(r['recall_mse'] - mse_rec)
        reproduced = diff <= TOL
        row = {'split': split, 'label': label, 'seed': seed, 'stat': stat, 'q': q, 'run': run,
               'checkpoint_sha256': ckpt, 'manifest_errors': bad, 'recorded_recall_mse': mse_rec,
               'mse_abs_diff': diff, 'reproduced': reproduced, **r,
               'ok': r['safe'] and reproduced and not bad}
        results.append(row)
        tie = '-' if r['tie_rate'] is None else f"{r['tie_rate']:.2e}"
        print(f"{split:<9} {label:<11} {seed:>5} {mse_rec:>10.6f} {r['recall_mse']:>11.6f} {diff:>9.1e} "
              f"{r['sequences']:>5} {r['max_abs_state']:>7.2f} {str(r['finite']):>6} {str(r['safe']):>5} "
              f"{r['kept_mass_frac']:>9.4f} {tie:>9} {'OK' if not bad else 'FAIL':>8}")
        if bad:
            print(f"          manifest: {bad}")

    # D-AU 5: 사전등록 판정에 그대로 적용
    def blocked(split, labels):
        return sorted({(r['label'], r['seed']) for r in results if r['split'] == split and r['label'] in labels and not r['ok']})
    verdicts = {
        '2J D-AN (q=0.5 vs q=1, confirm2)': blocked('confirm2', {'2J'}),
        '2K D-AR pearson - recent (confirm3)': blocked('confirm3', {'2K/pearson', '2K/recent'}),
        '2K D-AR pearson - random (confirm3)': blocked('confirm3', {'2K/pearson', '2K/random'}),
        '2K secondary recent - q1': blocked('confirm3', {'2K/recent', '2K/q1'}),
        '2K secondary random - q1': blocked('confirm3', {'2K/random', '2K/q1'}),
    }
    print()
    for name, bl in verdicts.items():
        print(f"[safety] {name}: {'recorded verdict stands (all models safe, reproduced, manifest OK)' if not bl else 'WITHHELD by ' + str(bl)}")
    ties = [r['tie_rate'] for r in results if r['tie_rate'] is not None]
    mism = sum(r['mask_reconstruction_mismatch'] for r in results)
    print(f"[safety] pearson top-k boundary ties on trained hard models: max rate {max(ties):.3e} over "
          f"{len(ties)} evaluations; mask reconstruction mismatches {mism}")
    for lab in ('2J', '2K/q1', '2K/pearson', '2K/recent', '2K/random'):
        v = [r['kept_mass_frac'] for r in results if r['label'] == lab]
        print(f"[safety] kept kernel mass fraction {lab:<11} mean {np.mean(v):.4f}  (descriptive, D-AU 6)")

    json.dump({**head, 'status': 'done', 'rows': results, 'verdicts': {k: v for k, v in verdicts.items()},
               'tie_max': max(ties), 'mask_reconstruction_mismatch': mism}, open(record_path, 'w'), indent=2)
    print(f'[safety] recorded to {record_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
