"""Open confirm3 exactly once for the same-budget controls (prereg 2K, D-AR).

Four conditions per pre-registered seed, all mode=hard at the same kept count:

    pearson   2J's candidate, q = 0.5         checkpoint from the 2J confirmation suite
    q1        plain f-LIF, q = 1 (== full)    checkpoint from the 2J confirmation suite
    recent    the k most recent slots         trained for 2K (scripts/hard_control_train.sh)
    random    k uniform slots per query       trained for 2K; generator reseeded to the seed
                                              right before its evaluation (D-AP)

None of these models has seen confirm3. Primary tests (both pre-specified, both required for
the joint claim): paired pearson - recent and pearson - random, 95% t-interval excluding zero
with a negative mean. Secondary (reported either way): recent - q1, random - q1. pearson - q1
on confirm3 is printed as a description only; it is not a re-confirmation of 2J (D-AS).

The record is created with 'x' before anything is printed as a verdict; a second run refuses.

Usage:
    python hard_control.py --twoj hardconf-015047 --suite hardctrl-XXXXXX
"""
import sys
import glob
import json
import math
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK                                        # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from hard_selection import load, metrics, sha256, SOURCES      # noqa: E402

SEEDS = (7, 13, 21, 42, 123, 256, 512, 1024)                   # D-AQ
Q = .5
PRIMARY = (('pearson', 'recent'), ('pearson', 'random'))
SECONDARY = (('recent', 'q1'), ('random', 'q1'))
DESCRIPTIVE = (('pearson', 'q1'),)


def find(suite, seed, stat, q):
    """Run dir of a finished run, or a fatal error."""
    pat = f'seed{seed}_*hard-shared-{stat}-q{q:g}_k3'
    runs = glob.glob(str(TASK / 'log' / suite / 'recall' / '*' / '*' / pat))
    done = [f for f in glob.glob(str(TASK / 'results' / suite / f'*hard-shared-{stat}-q{q:g}_k3_seed{seed}.json'))
            if json.load(open(f)).get('train', {}).get('epochs_run') == 12]    # 2L D-AV
    if len(runs) != 1 or len(done) != 1:
        raise SystemExit(f'[control] need exactly one finished run for seed {seed} {stat} q={q:g} '
                         f'in {suite}; found {len(runs)} dirs, {len(done)} finished results')
    return runs[0]


def paired(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)
    return {'deltas': d.tolist(), 'mean': float(mean), 'sd': float(sd), 'ci95': [float(mean - half), float(mean + half)],
            'relative': float(mean / np.mean(b)), 'improved': int((d < 0).sum()), 'n': n}


def main():
    parser = argparse.ArgumentParser(description='prereg 2K: one look at confirm3')
    parser.add_argument('--twoj', required=True, help='2J confirmation suite (pearson q=0.5 and q=1)')
    parser.add_argument('--suite', required=True, help='2K control suite (recent and random)')
    cli = parser.parse_args()
    record_path = TASK / 'results' / cli.suite / 'hard_control_record.json'
    if record_path.exists():
        raise SystemExit(f'[control] {record_path} exists: confirm3 has been opened for 2K already (D-AQ).')

    runs = {}
    for seed in SEEDS:
        runs[(seed, 'pearson')] = find(cli.twoj, seed, 'pearson', Q)
        runs[(seed, 'q1')] = find(cli.twoj, seed, 'pearson', 1.)
        runs[(seed, 'recent')] = find(cli.suite, seed, 'recent', Q)
        runs[(seed, 'random')] = find(cli.suite, seed, 'random', Q)
    record = {'prereg': '2K', 'split': 'confirm3', 'seeds': list(SEEDS), 'q': Q,
              'twoj_suite': cli.twoj, 'control_suite': cli.suite,
              'checkpoint_sha256': {f'{s}/{c}': sha256(Path(r) / 'model_state' / 'best+model.pt')
                                    for (s, c), r in sorted(runs.items())},
              'source_sha256': {f: sha256(TASK / f) for f in SOURCES}}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, 'x') as handle:                     # 열기 전에 기록: 두 번째 실행은 거부된다
        json.dump({**record, 'status': 'opening confirm3'}, handle, indent=2)

    rows, mse = [], {c: [] for c in ('pearson', 'q1', 'recent', 'random')}
    fails = {c: [] for c in mse}
    print(f"[control] confirm3, q = {Q}, seeds {list(SEEDS)}")
    print(f"{'seed':>5} {'q1':>10} {'pearson':>10} {'recent':>10} {'random':>10}   max|u| q1/pearson/recent/random")
    for seed in SEEDS:
        line, peaks = {}, {}
        for cond in ('q1', 'pearson', 'recent', 'random'):
            model, config = load(runs[(seed, cond)])
            want = 'pearson' if cond in ('q1', 'pearson') else cond
            if (config.hard_stat, config.hard_axis, config.seed) != (want, 'shared', seed):
                raise SystemExit(f'[control] {cond} seed {seed}: trained with {config.hard_stat}/{config.hard_axis}/seed {config.seed}')
            model.embedding.neuron.selector.reseed_hard(seed)  # random: reproducible draws (D-AP)
            _, conf3 = data_provider(config, 'confirm3')
            r = dict(metrics(model, conf3, config), seed=seed, condition=cond,
                     checkpoint_sha256=record['checkpoint_sha256'][f'{seed}/{cond}'])
            rows.append(r)
            mse[cond].append(r['recall_mse'])
            if not r['within_bound']:
                fails[cond].append(seed)
            line[cond], peaks[cond] = r['recall_mse'], r['max_abs_state']
        print(f"{seed:>5} {line['q1']:>10.6f} {line['pearson']:>10.6f} {line['recent']:>10.6f} {line['random']:>10.6f}"
              f"   {peaks['q1']:.2f}/{peaks['pearson']:.2f}/{peaks['recent']:.2f}/{peaks['random']:.2f}")

    tests = {}
    print()
    for group, pairs in (('primary', PRIMARY), ('secondary', SECONDARY), ('descriptive', DESCRIPTIVE)):
        for a, b in pairs:
            t = paired(mse[a], mse[b])
            blocked = sorted(set(fails[a]) | set(fails[b]))
            t['blocked_by'] = blocked
            t['excludes_zero_negative'] = None if blocked else bool(t['ci95'][1] < 0.)
            t['excludes_zero_positive'] = None if blocked else bool(t['ci95'][0] > 0.)
            tests[f'{a}-{b}'] = {**t, 'group': group}
            verdict = ('판정 보류: FAIL seed ' + str(blocked)) if blocked else \
                      ('CI가 0 제외, 음수 (a가 낮음)' if t['excludes_zero_negative'] else
                       'CI가 0 제외, 양수 (a가 높음)' if t['excludes_zero_positive'] else 'CI가 0 포함')
            print(f"[control] {group:>11} {a:>7} - {b:<7} mean {t['mean']:+.6f} sd {t['sd']:.6f} "
                  f"95% CI [{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}] rel {100 * t['relative']:+.2f}% "
                  f"a<b in {t['improved']}/{t['n']}  -> {verdict}")

    p = [tests[f'{a}-{b}']['excludes_zero_negative'] for a, b in PRIMARY]
    if any(v is None for v in p):
        joint = '판정 보류 (탈락 seed)'
    elif all(p):
        joint = '두 1차 비교 모두 통과: 같은 예산에서 피어슨 선별이 최근·무작위 선별보다 confirm3 회상 오차가 낮다'
    elif any(p):
        passed = [f'{a}-{b}' for (a, b), v in zip(PRIMARY, p) if v]
        joint = f'한쪽만 통과 ({passed}): 그 비교에 대해서만 말한다 (D-AR)'
    else:
        joint = '두 1차 비교 모두 불통: 같은 예산의 내용 무관 선별 대비 우위를 주장하지 않는다'
    print(f'\n[control] {joint}')
    print('[control] 주장 범위(D-AS): q=0.5 공유 축, confirm3 회상 오차. O7·다른 q/축/통계량·다른 과제는 주장하지 않는다. '
          'pearson - q1 은 기술 통계이지 2J의 재확증이 아니다.')

    record.update({'status': 'done', 'rows': rows, 'recall_mse': mse, 'fails': fails, 'tests': tests,
                   'joint': joint, 'source_sha256_at_eval': {f: sha256(TASK / f) for f in SOURCES}})
    json.dump(record, open(record_path, 'w'), indent=2)
    print(f'[control] recorded to {record_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
