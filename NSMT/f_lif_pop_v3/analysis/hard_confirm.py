"""Open confirm2 exactly once for the recorded hard_q (prereg 2J, D-AN).

Stage 2 of hard_selection.py. Refuses to run unless the selection record exists, has no
confirm block yet, and its source hashes still match the working tree; refuses if any of the
sixteen runs (eight seeds x {q = 1, selected q}) is missing or unfinished. Then, per seed:

    delta_s = confirm2 recall MSE(selected q) - confirm2 recall MSE(q = 1)      (same init)

Primary test: the 95% t-interval of the eight deltas excludes zero and the mean is negative.
Secondary, only if the primary passes: relative improvement >= 20% (O7-2's number).
A seed that violates G11 or goes non-finite is reported as FAIL and, per D-AN, blocks the
primary verdict -- it is never dropped. Every row carries the D-AH diagnostics.

Usage:
    python hard_confirm.py --selection results/hardsel-XXXXXX/hard_selection_record.json \
                           --suite hardconf-YYYYYY
"""
import sys
import glob
import json
import math
import hashlib
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK                                        # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from hard_selection import load, metrics, sha256, SOURCES      # noqa: E402

SEEDS = (7, 13, 21, 42, 123, 256, 512, 1024)                   # D-AN
BASELINE_Q = 1.
SECONDARY = .20


def collect(suite, qsel):
    """(seed, q) -> run dir for finished runs; missing or duplicate is fatal."""
    log_root = TASK / 'log' / suite
    result_root = TASK / 'results' / suite
    runs = {}
    for run in sorted(glob.glob(str(log_root / 'recall' / '*' / '*' / 'seed*hard-*'))):
        name = Path(run).name
        seed = int(name.split('seed', 1)[1].split('_')[0])
        q = float(name.split('-q', 1)[1].split('_')[0])
        done = [f for f in glob.glob(str(result_root / f'*-q{q:g}_*seed{seed}.json'))
                if json.load(open(f)).get('train', {}).get('epochs_run')]
        if not done:
            continue
        if (seed, q) in runs:
            raise SystemExit(f'[confirm] duplicate run for seed {seed} q={q:g}')
        runs[(seed, q)] = run
    missing = [(s, q) for s in SEEDS for q in (BASELINE_Q, qsel) if (s, q) not in runs]
    if missing:
        raise SystemExit(f'[confirm] unfinished or missing runs: {missing}')
    return runs


def main():
    parser = argparse.ArgumentParser(description='prereg 2J stage 2: one look at confirm2')
    parser.add_argument('--selection', required=True, help='hard_selection_record.json from stage 1')
    parser.add_argument('--suite', required=True, help='suite trained by scripts/hard_confirm_train.sh')
    cli = parser.parse_args()

    record_path = Path(cli.selection)
    if not record_path.exists():
        raise SystemExit(f'[confirm] no selection record at {record_path}; run stage 1 first')
    record = json.load(open(record_path))
    if record.get('confirm'):
        raise SystemExit('[confirm] this selection record already has confirm2 results. The held-out '
                         'split is looked at once (D-AL); a new candidate needs a new appendix and confirm3.')
    moved = {f: (h, sha256(TASK / f)) for f, h in record['source_sha256'].items() if sha256(TASK / f) != h}
    if moved:
        raise SystemExit(f'[confirm] sources changed since selection: {list(moved)}. Refusing.')
    qsel = float(record['selected_q'])
    runs = collect(cli.suite, qsel)
    print(f"[confirm] selection {record['suite']}: q = {qsel:g}, axis {record['axis']}, stat {record['stat']}")
    print(f"[confirm] suite {cli.suite}: {len(runs)} finished runs, seeds {list(SEEDS)}")

    rows, deltas, fails = [], [], []
    print(f"\n{'seed':>5} {'q=1 recall':>11} {'q=sel recall':>13} {'delta':>10} {'q=1 max|u|':>11} "
          f"{'sel max|u|':>11} {'G11':>5} {'axis/stat match':>16}")
    for seed in SEEDS:
        pair = {}
        for q in (BASELINE_Q, qsel):
            model, config = load(runs[(seed, q)])
            if (config.hard_axis, config.hard_stat) != (record['axis'], record['stat']):
                raise SystemExit(f'[confirm] seed {seed} q={q:g} trained with a different axis/stat')
            _, conf2 = data_provider(config, 'confirm2')
            pair[q] = dict(metrics(model, conf2, config), seed=seed, label=f'q={q:g}',
                           checkpoint_sha256=sha256(Path(runs[(seed, q)]) / 'model_state' / 'best+model.pt'))
        ok = pair[BASELINE_Q]['within_bound'] and pair[qsel]['within_bound']
        d = pair[qsel]['recall_mse'] - pair[BASELINE_Q]['recall_mse']
        rows.extend(pair.values())
        deltas.append(d)
        if not ok:
            fails.append(seed)
        print(f"{seed:>5} {pair[BASELINE_Q]['recall_mse']:>11.6f} {pair[qsel]['recall_mse']:>13.6f} {d:>+10.6f} "
              f"{pair[BASELINE_Q]['max_abs_state']:>11.2f} {pair[qsel]['max_abs_state']:>11.2f} "
              f"{'OK' if ok else 'FAIL':>5} {'yes':>16}")

    deltas = np.array(deltas, dtype=np.float64)
    n = len(deltas)
    mean, sd = deltas.mean(), deltas.std(ddof=1)
    half = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    base_mean = float(np.mean([r['recall_mse'] for r in rows if r['label'] == 'q=1']))
    rel = mean / base_mean
    improved = int((deltas < 0).sum())
    print(f"\n[confirm] paired delta (q={qsel:g} - q=1), n={n}: mean {mean:+.6f}, sd {sd:.6f}, "
          f"95% CI [{lo:+.6f}, {hi:+.6f}], relative {100 * rel:+.2f}%, seeds improved {improved}/{n}")
    if fails:
        verdict = f'판정 보류: G11/비유한 FAIL seed {fails} (D-AN: 탈락 seed가 있으면 1차 판정을 하지 않는다)'
        primary = secondary = None
    else:
        primary = bool(hi < 0.)
        secondary = bool(primary and (-rel) >= SECONDARY)
        verdict = (f"1차 {'통과' if primary else '불통'} (CI가 0을 {'제외' if primary else '포함'}); "
                   + (f"2차 {'통과' if secondary else '불통'} (개선 {100 * -rel:.1f}% vs 20%)" if primary
                      else '2차 미판정'))
    print(f'[confirm] {verdict}')
    print('[confirm] 주장 범위(D-AO): 고정 커널 + val에서 고른 top-q 선별 vs 같은 seed의 순수 f-LIF, confirm2 회상 오차. '
          'O7·전체 우위·다른 설정은 주장하지 않는다.')

    record['confirm'] = {
        'suite': cli.suite, 'split': 'confirm2', 'seeds': list(SEEDS), 'selected_q': qsel,
        'rows': rows, 'deltas': deltas.tolist(), 'mean': float(mean), 'sd': float(sd),
        'ci95': [float(lo), float(hi)], 'relative': float(rel), 'improved': improved,
        'fails': fails, 'primary': primary, 'secondary': secondary, 'verdict': verdict,
        'source_sha256_at_confirm': {f: sha256(TASK / f) for f in SOURCES},
    }
    json.dump(record, open(record_path, 'w'), indent=2)
    print(f'[confirm] appended to {record_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
