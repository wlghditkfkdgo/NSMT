"""Select eta on validation, record the decision, then open confirm exactly once.

Implements prereg 2I. Audit 27 found four ways the first draft failed to enforce it; each
is now a hard error rather than a convention.

  seed completeness   A seed counts only when its result JSON exists with a finished
                      training record. A directory alone means a run that was interrupted,
                      and its partial best+model.pt would otherwise be evaluated silently --
                      which is what happened to seed 1024 in seeds-173737.
  duplicate seeds     Two paths mapping to one seed used to overwrite in a dict. Now fatal.
  one look at confirm The selection table, the chosen eta and the checkpoint/source/config
                      hashes are written BEFORE confirm is touched, with exclusive create,
                      so a rerun cannot quietly re-open the held-out split.
  D-AH reporting      Every row -- candidates and baselines, validation and confirm --
                      carries score_top1, score_rank with its chance reference, p_mass,
                      precap_mass, postcap_mass, kernel_mass, uniform_slot, the per-event
                      kind errors, and the real max|u| with its bound.

Usage:
    python eta_selection.py --suite seeds-XXXXXX            # 1단계: 선택만
    python eta_selection.py --suite seeds-XXXXXX --confirm  # 2단계: 기록된 선택으로 confirm
"""
import sys
import glob
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import Config, parse_defaults, TASK                # noqa: E402
from model import LOAD_MODEL                                   # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from test import evaluate, selection_diagnostics               # noqa: E402
from stage_decomposition import decompose                      # noqa: E402

CANDIDATES = (0., .1, .2, .3, .5)                              # D-AE
SELECTION_SEED = 7                                             # D-AF
SOURCES = ('layers.py', 'ours.py', 'model.py', 'train.py', 'test.py', 'config.py',
           'calibrate.py', 'data_provider/synthetic.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_runs(suite):
    """Seeds whose training actually finished, refusing duplicates (audit 27)."""
    log_root = TASK / 'log' / suite
    result_root = TASK / 'results' / suite
    runs, skipped = {}, []
    for run in sorted(glob.glob(str(log_root / 'recall' / '*' / '*' / 'seed*'))):
        seed = int(Path(run).name.split('seed', 1)[1].split('_')[0])
        done = [f for f in glob.glob(str(result_root / f'*seed{seed}.json'))
                if json.load(open(f)).get('train', {}).get('epochs_run')]
        if not done:
            skipped.append((seed, Path(run).name, 'no finished result JSON'))
            continue
        if seed in runs:
            raise SystemExit(f'[select] duplicate seed {seed}: {runs[seed][0]} and {run}. '
                             f'Resolve before selecting; a dict would silently keep one.')
        runs[seed] = (run, done[0])

    return runs, skipped


def metrics(model, loader, config, eta, batches=4):
    """One row of D-AH at `eta`. None keeps the trained value."""
    sel = model.embedding.neuron.selector
    saved = sel.eta_value.clone()
    sel.eta_value.fill_(float('nan') if eta is None else eta)
    try:
        err = evaluate(model, loader, config)[0]
        diag = selection_diagnostics(model, loader, config, batches=batches)
        stage = decompose(model, loader, config, batches=batches)
    finally:
        sel.eta_value.copy_(saved)
    bound = getattr(config, 'g11_bound', None)

    return {'eta': eta if eta is not None else float(torch.sigmoid(sel.eta_hat)),
            'recall_mse': err['recall']['mse'], 'copy_mse': err['copy']['mse'],
            'recall_first_mse': err['recall_first']['mse'],
            'score_top1': stage['score_top1'], 'score_rank': stage['score_rank'],
            'score_rank_chance': stage['score_rank_chance'], 'p_mass': stage['p_mass'],
            'precap_mass': stage['precap_mass'], 'postcap_mass': stage['postcap_mass'],
            'kernel_mass': stage['kernel_mass'], 'uniform_slot': stage['uniform_slot'],
            'm_eff': diag['m_eff'], 'hit': diag['hit'], 'support_p': diag['support_p'],
            'max_abs_state': diag['max_abs_state'], 'bound': bound,
            'within_bound': bound is None or diag['max_abs_state'] < bound}


def load(run):
    base = parse_defaults()
    base.cpu = True
    config = Config()
    config.load_args(run, base)
    config.device = torch.device('cpu')
    config.max_eval_batches = 0

    return LOAD_MODEL[config.model](config, train=False), config


def show(rows, keys=('recall_mse', 'score_top1', 'score_rank', 'p_mass', 'precap_mass',
                     'postcap_mass', 'max_abs_state')):
    head = f"{'조건':<22} {'eta':>6}" + ''.join(f"{k.replace('_mse', ''):>13}" for k in keys)
    print(head + f"{'G11':>6}")
    for r in rows:
        line = f"{r['label']:<22} {r['eta']:>6.3f}" + ''.join(f"{r[k]:>13.6f}" for k in keys)
        print(line + f"{'OK' if r['within_bound'] else 'FAIL':>6}")


def main():
    parser = argparse.ArgumentParser(description='prereg 2I selection and confirmation')
    parser.add_argument('--suite', required=True)
    parser.add_argument('--confirm', action='store_true',
                        help='stage 2: open the held-out split using the recorded selection')
    cli = parser.parse_args()
    record_path = TASK / 'results' / cli.suite / 'eta_selection_record.json'

    runs, skipped = collect_runs(cli.suite)
    for seed, name, why in skipped:
        print(f"[select] skipped seed {seed} ({why}): {name}")
    print(f"[select] usable seeds: {sorted(runs)}")

    if not cli.confirm:
        # ---- 1단계: seed 7 validation에서 고르고 기록한다 -------------------
        if SELECTION_SEED not in runs:
            raise SystemExit(f'[select] selection seed {SELECTION_SEED} has no finished run')
        model, config = load(runs[SELECTION_SEED][0])
        _, val = data_provider(config, 'val')
        rows = [dict(metrics(model, val, config, None), label='baseline: trained eta')]
        for eta in CANDIDATES:
            label = 'baseline: eta=0' if eta == 0. else f'candidate eta={eta:g}'
            rows.append(dict(metrics(model, val, config, eta), label=label))

        print(f"\n[select] validation, seed {SELECTION_SEED}, bound {rows[0]['bound']}")
        show(rows)
        eligible = [r for r in rows if r['label'].startswith('candidate') and r['within_bound']]
        if not eligible:
            raise SystemExit('[select] every candidate disqualified by G11 (D-AG)')
        best = min(eligible, key=lambda r: (r['recall_mse'], r['eta']))
        rejected = [r['eta'] for r in rows
                    if r['label'].startswith('candidate') and not r['within_bound']]
        print(f"[select] selected eta = {best['eta']:g}"
              + (f"; G11-disqualified: {rejected}" if rejected else ''))

        record = {'suite': cli.suite, 'selection_seed': SELECTION_SEED, 'split': 'val',
                  'candidates': list(CANDIDATES), 'selected_eta': best['eta'],
                  'g11_disqualified': rejected, 'rows': rows, 'seeds': sorted(runs),
                  'skipped': skipped,
                  'checkpoint_sha256': {s: sha256(Path(p[0]) / 'model_state' / 'best+model.pt')
                                        for s, p in sorted(runs.items())},
                  'config_sha256': {s: sha256(Path(p[0]) / 'model_state' / 'config.pt')
                                    for s, p in sorted(runs.items())},
                  'source_sha256': {f: sha256(TASK / f) for f in SOURCES}}
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with open(record_path, 'x') as handle:          # 'x': 재실행이 선택을 덮어쓸 수 없다
            json.dump(record, handle, indent=2)
        print(f"[select] recorded to {record_path}")
        print("[select] confirm 분할은 아직 열지 않았다. --confirm 으로 2단계를 실행한다.")
        return 0

    # ---- 2단계: 기록된 선택으로 confirm을 한 번 연다 -----------------------
    if not record_path.exists():
        raise SystemExit(f'[confirm] no selection record at {record_path}; run stage 1 first')
    record = json.load(open(record_path))
    if record.get('confirm'):
        raise SystemExit('[confirm] this selection record already has confirm results. '
                         'The held-out split is looked at once (D-AI); start a new suite.')
    if sorted(runs) != record['seeds']:
        raise SystemExit(f"[confirm] seed set changed since selection: recorded "
                         f"{record['seeds']}, now {sorted(runs)}. Re-select in a new suite.")
    for seed, (run, _) in sorted(runs.items()):
        live = sha256(Path(run) / 'model_state' / 'best+model.pt')
        if live != record['checkpoint_sha256'][str(seed)]:
            raise SystemExit(f'[confirm] seed {seed} checkpoint changed since selection')

    eta = record['selected_eta']
    print(f"\n[confirm] held-out split, selected eta {eta:g}, {len(runs)} seeds")
    out = []
    for seed, (run, _) in sorted(runs.items()):
        model, config = load(run)
        _, split = data_provider(config, 'confirm')
        row = {'seed': seed,
               'eta0': metrics(model, split, config, 0.),
               'trained': metrics(model, split, config, None),
               'selected': metrics(model, split, config, eta)}
        row['delta'] = row['selected']['recall_mse'] - row['eta0']['recall_mse']
        out.append(row)
        print(f"[confirm] seed {seed:>5}  eta0 {row['eta0']['recall_mse']:.6f}"
              f"  trained {row['trained']['recall_mse']:.6f}"
              f"  selected {row['selected']['recall_mse']:.6f}  delta {row['delta']:+.6f}"
              f"  max|u| {row['selected']['max_abs_state']:.2f}"
              f"  {'OK' if row['selected']['within_bound'] else 'FAIL'}")

    failed = [r['seed'] for r in out if not r['selected']['within_bound']]
    d = np.array([r['delta'] for r in out])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t95 = {8: 2.365, 7: 2.447, 6: 2.571, 5: 2.776}.get(n, 1.96)
    lo, hi = d.mean() - t95 * se, d.mean() + t95 * se
    rel = d.mean() / np.mean([r['eta0']['recall_mse'] for r in out]) * 100
    print(f"\n[confirm] paired delta (selected - eta0): mean {d.mean():+.6f}"
          f"  95% CI [{lo:+.6f}, {hi:+.6f}]  relative {rel:+.2f}%  n={n}")
    print(f"[confirm] {'CI 상한 < 0 → 개선이 유의하다' if hi < 0 else 'CI가 0을 포함 → 유의하지 않다'}")
    if failed:
        print(f"[confirm] G11 FAIL on seeds {failed}. 실패 seed를 빼고 다시 계산하지 않는다 (D-AG).")
    print("[confirm] 이 절차가 말하는 것은 eta 개입의 효과뿐이다 (D-AJ). O7 판정이 아니다.")

    record['confirm'] = {'rows': out, 'paired': {'mean': float(d.mean()),
                                                 'ci95': [float(lo), float(hi)],
                                                 'relative_pct': float(rel), 'n': n},
                         'g11_failed_seeds': failed}
    with open(record_path, 'w') as handle:
        json.dump(record, handle, indent=2)
    print(f"[confirm] appended to {record_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
