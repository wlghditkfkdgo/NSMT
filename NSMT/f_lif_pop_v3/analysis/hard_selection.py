"""Select hard_q on validation and record the decision (prereg 2J, D-AM); confirm later (D-AN).

Mirror of eta_selection.py for mode=hard. Stage 1 only reads the selection suite: one run per
q on seed 7, all trained by scripts/hard_select.sh. The rule is fixed in D-AM before any of
these numbers were seen:

    metric      validation recall MSE (kind > 0, spike readout) -- NOT the all-event loss the
                trainer minimises, and not M_eff
    tie         any candidate within 0.005 of the best takes part; the LARGEST such q wins
                (closest to the plain kernel)
    exclusion   max|u| >= the frozen G11 bound, or a non-finite state, drops the candidate
    baseline    q = 1.0 on the same code path (bitwise full, gate G18) -- reported, never a candidate

The record is created with 'x' so a rerun cannot rewrite the choice, and it carries the
checkpoint, config and source hashes so stage 2 can refuse if anything moved. confirm2 is
not loaded here.

Usage:
    python hard_selection.py --suite hardsel-XXXXXX
"""
import sys
import glob
import json
import math
import hashlib
import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import Config, parse_defaults, TASK                # noqa: E402
from model import LOAD_MODEL                                   # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from test import evaluate, selection_diagnostics               # noqa: E402
from stage_decomposition import decompose                      # noqa: E402

CANDIDATES = (.1, .25, .5)                                     # D-AM grid, from the screening
BASELINE_Q = 1.
SELECTION_SEED = 7
TIE = .005
SOURCES = ('layers.py', 'ours.py', 'model.py', 'train.py', 'test.py', 'config.py',
           'calibrate.py', 'data_provider/synthetic.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_runs(suite):
    """q -> (run dir, result json) for finished seed-7 runs; duplicates are fatal."""
    log_root = TASK / 'log' / suite
    result_root = TASK / 'results' / suite
    runs, skipped = {}, []
    for run in sorted(glob.glob(str(log_root / 'recall' / '*' / '*' / f'seed{SELECTION_SEED}_*hard-*'))):
        name = Path(run).name
        q = float(name.split('-q', 1)[1].split('_')[0])
        done = [f for f in glob.glob(str(result_root / f'*-q{q:g}_*seed{SELECTION_SEED}.json'))
                if json.load(open(f)).get('train', {}).get('epochs_run')]
        if not done:
            skipped.append((q, name, 'no finished result JSON'))
            continue
        if q in runs:
            raise SystemExit(f'[select] duplicate q={q:g}: {runs[q][0]} and {run}')
        runs[q] = (run, done[0])
    return runs, skipped


def load(run):
    base = parse_defaults()
    base.cpu = True
    config = Config()
    config.load_args(run, base)
    config.device = torch.device('cpu')
    config.max_eval_batches = 0
    return LOAD_MODEL[config.model](config, train=False), config


def metrics(model, loader, config, batches=4):
    """One D-AH row at the run's own mode and q."""
    err = evaluate(model, loader, config)[0]
    diag = selection_diagnostics(model, loader, config, batches=batches)
    stage = decompose(model, loader, config, batches=batches)
    bound = getattr(config, 'g11_bound', None)
    finite = all(math.isfinite(v) for v in (err['recall']['mse'], diag['max_abs_state']))
    return {'q': config.hard_q, 'axis': config.hard_axis, 'stat': config.hard_stat,
            'recall_mse': err['recall']['mse'], 'copy_mse': err['copy']['mse'],
            'recall_first_mse': err['recall_first']['mse'],
            'score_top1': stage['score_top1'], 'score_rank': stage['score_rank'],
            'score_rank_chance': stage['score_rank_chance'], 'p_mass': stage['p_mass'],
            'precap_mass': stage['precap_mass'], 'postcap_mass': stage['postcap_mass'],
            'kernel_mass': stage['kernel_mass'], 'uniform_slot': stage['uniform_slot'],
            'm_eff': diag['m_eff'], 'hit': diag['hit'], 'support_p': diag['support_p'],
            'max_abs_state': diag['max_abs_state'], 'bound': bound, 'finite': finite,
            'within_bound': finite and (bound is None or diag['max_abs_state'] < bound)}


def show(rows, keys=('recall_mse', 'copy_mse', 'score_top1', 'score_rank', 'p_mass',
                     'postcap_mass', 'kernel_mass', 'hit', 'max_abs_state')):
    head = f"{'조건':<18} {'q':>5}" + ''.join(f"{k.replace('_mse', ''):>13}" for k in keys)
    print(head + f"{'G11':>6}")
    for r in rows:
        line = f"{r['label']:<18} {r['q']:>5g}" + ''.join(f"{r[k]:>13.6f}" for k in keys)
        print(line + f"{'OK' if r['within_bound'] else 'FAIL':>6}")


def main():
    parser = argparse.ArgumentParser(description='prereg 2J stage 1: select hard_q on validation')
    parser.add_argument('--suite', required=True)
    cli = parser.parse_args()
    record_path = TASK / 'results' / cli.suite / 'hard_selection_record.json'
    if record_path.exists():
        raise SystemExit(f'[select] {record_path} exists; the selection is already recorded')

    runs, skipped = collect_runs(cli.suite)
    for q, name, why in skipped:
        print(f"[select] skipped q={q:g} ({why}): {name}")
    print(f"[select] finished runs: q = {sorted(runs)}")
    missing = [q for q in (BASELINE_Q,) + CANDIDATES if q not in runs]
    if missing:
        raise SystemExit(f'[select] missing finished runs for q = {missing}')

    rows = []
    for q in (BASELINE_Q,) + CANDIDATES:
        model, config = load(runs[q][0])
        _, val = data_provider(config, 'val')
        label = 'baseline q=1 (full)' if q == BASELINE_Q else f'candidate q={q:g}'
        rows.append(dict(metrics(model, val, config), label=label,
                         best_val_loss_all_events=json.load(open(runs[q][1]))['train']['best_val_loss']))

    print(f"\n[select] validation, seed {SELECTION_SEED}, axis {rows[0]['axis']}, stat {rows[0]['stat']}, "
          f"bound {rows[0]['bound']}")
    show(rows)
    cands = [r for r in rows if r['label'].startswith('candidate')]
    eligible = [r for r in cands if r['within_bound']]
    rejected = [r['q'] for r in cands if not r['within_bound']]
    if not eligible:
        raise SystemExit(f'[select] every candidate disqualified (G11 / non-finite): {rejected}')
    best_mse = min(r['recall_mse'] for r in eligible)
    tied = [r for r in eligible if r['recall_mse'] - best_mse < TIE]
    best = max(tied, key=lambda r: r['q'])                    # D-AM: 동률이면 큰 q
    base = rows[0]
    print(f"[select] best val recall MSE = {best_mse:.6f}; within {TIE}: q = {[r['q'] for r in tied]}"
          f" -> selected q = {best['q']:g}" + (f"; disqualified: {rejected}" if rejected else ''))
    print(f"[select] baseline q=1 val recall MSE = {base['recall_mse']:.6f}; selected candidate "
          f"{best['recall_mse']:.6f} ({100 * (best['recall_mse'] / base['recall_mse'] - 1):+.1f}% -- "
          f"validation, seed 7, exploratory; the confirmatory number comes from confirm2)")

    record = {'suite': cli.suite, 'prereg': '2J D-AM', 'selection_seed': SELECTION_SEED, 'split': 'val',
              'candidates': list(CANDIDATES), 'baseline_q': BASELINE_Q, 'tie': TIE,
              'selected_q': best['q'], 'axis': best['axis'], 'stat': best['stat'],
              'disqualified': rejected, 'rows': rows, 'skipped': skipped,
              'checkpoint_sha256': {f'{q:g}': sha256(Path(p[0]) / 'model_state' / 'best+model.pt')
                                    for q, p in sorted(runs.items())},
              'config_sha256': {f'{q:g}': sha256(Path(p[0]) / 'model_state' / 'config.pt')
                                for q, p in sorted(runs.items())},
              'source_sha256': {f: sha256(TASK / f) for f in SOURCES}}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, 'x') as handle:                     # 'x': 재실행이 선택을 덮어쓸 수 없다
        json.dump(record, handle, indent=2)
    print(f"[select] recorded to {record_path}")
    print("[select] confirm2 는 열지 않았다. 2단계(D-AN)는 8 seed 학습 뒤 hard_confirm.py 로 한 번 연다.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
