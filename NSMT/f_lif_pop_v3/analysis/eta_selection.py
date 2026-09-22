"""Select eta on validation, then evaluate once on the held-out confirm split.

Implements prereg 2I exactly, and nothing more:

  D-AE  candidates {0, 0.1, 0.2, 0.3, 0.5}; eta=0 and the trained eta are baselines, not
        candidates; eta=1.0 is excluded for already failing G11; training stays at the
        trained eta and only the evaluation eta moves, which is a train/test mismatch and
        is labelled as one.
  D-AF  selection is validation recall-only MSE alone, decided once on seed 7 and applied
        to every seed; ties go to the smaller eta.
  D-AG  a candidate that violates G11 on the FULL forward, against the bound frozen at
        calibration, is disqualified whatever its error.
  D-AH  score -> p -> pre-cap -> post-cap and the real max|u| are reported for every row.
  D-AI  confirm (generator offset 30000) is looked at once, after selection.

Usage:
    python eta_selection.py --suite seeds-XXXXXX
"""
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))

from config import Config, parse_defaults                      # noqa: E402
from model import LOAD_MODEL                                   # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from test import evaluate, selection_diagnostics               # noqa: E402

CANDIDATES = (0., .1, .2, .3, .5)                              # D-AE
SELECTION_SEED = 7                                             # D-AF


def load(run, cpu=True):
    base = parse_defaults()
    base.cpu = cpu
    config = Config()
    config.load_args(run, base)
    config.device = torch.device('cpu' if cpu else f'cuda:{config.num_device}')
    config.max_eval_batches = 0
    model = LOAD_MODEL[config.model](config, train=False)

    return model, config


def at_eta(model, loader, config, eta):
    """Evaluate with the selector forced to `eta` (None keeps the trained value)."""
    sel = model.embedding.neuron.selector
    saved = sel.eta_value.clone()
    sel.eta_value.fill_(float('nan') if eta is None else eta)
    try:
        result = evaluate(model, loader, config)[0]
        diag = selection_diagnostics(model, loader, config, batches=4)
    finally:
        sel.eta_value.copy_(saved)

    return result, diag


def row(label, eta, result, diag, bound):
    within = bound is None or diag['max_abs_state'] < bound
    return {'label': label, 'eta': eta, 'recall_mse': result['recall']['mse'],
            'copy_mse': result['copy']['mse'], 'm_eff': diag['m_eff'], 'hit': diag['hit'],
            'kernel_mass': diag['kernel_mass'], 'support_p': diag['support_p'],
            'max_abs_state': diag['max_abs_state'], 'bound': bound, 'within_bound': within}


def main():
    parser = argparse.ArgumentParser(description='prereg 2I eta selection and confirmation')
    parser.add_argument('--suite', required=True)
    parser.add_argument('--out', default=None)
    cli = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / 'forecasting' / 'log' / cli.suite

    runs = {}
    for run in sorted(glob.glob(str(base / 'recall' / '*' / '*' / 'seed*'))):
        seed = int(run.rsplit('seed', 1)[1].split('_')[0])
        runs[seed] = run
    if SELECTION_SEED not in runs:
        raise SystemExit(f'selection seed {SELECTION_SEED} missing from {cli.suite}')

    # ---- 1단계: seed 7 validation에서 후보를 고른다 (D-AF) --------------------
    model, config = load(runs[SELECTION_SEED])
    _, val = data_provider(config, 'val')
    bound = getattr(config, 'g11_bound', None)
    trained = torch.sigmoid(model.embedding.neuron.selector.eta_hat).item()

    table = []
    result, diag = at_eta(model, val, config, None)
    table.append(row('baseline: trained eta', trained, result, diag, bound))
    for eta in CANDIDATES:
        result, diag = at_eta(model, val, config, eta)
        label = 'baseline: eta=0' if eta == 0. else f'candidate eta={eta:g}'
        table.append(row(label, eta, result, diag, bound))

    print(f"[select] suite {cli.suite}, selection seed {SELECTION_SEED}, validation split")
    print(f"[select] trained eta {trained:.6f}, G11 bound {bound}")
    print(f"[select] {'조건':<22} {'eta':>6} {'val recall':>11} {'M_eff':>8} {'hit':>8}"
          f" {'support':>8} {'max|u|':>9} {'G11':>6}")
    for r in table:
        print(f"[select] {r['label']:<22} {r['eta']:>6.3f} {r['recall_mse']:>11.6f}"
              f" {r['m_eff']:>8.4f} {r['hit']:>8.4f} {r['support_p']:>8.4f}"
              f" {r['max_abs_state']:>9.2f} {'OK' if r['within_bound'] else 'FAIL':>6}")

    eligible = [r for r in table if r['label'].startswith('candidate') and r['within_bound']]
    if not eligible:
        raise SystemExit('[select] every candidate was disqualified by G11 (D-AG)')
    best = min(eligible, key=lambda r: (r['recall_mse'], r['eta']))      # 동률이면 작은 eta
    disqualified = [r['eta'] for r in table
                    if r['label'].startswith('candidate') and not r['within_bound']]
    print(f"[select] selected eta = {best['eta']:g} "
          f"(validation recall MSE {best['recall_mse']:.6f})"
          + (f"; disqualified by G11: {disqualified}" if disqualified else ''))

    # ---- 2단계: confirm 분할에서 8 seed paired 비교 (D-AI) --------------------
    print(f"\n[confirm] held-out split, {len(runs)} seeds, selected eta {best['eta']:g}"
          f" vs baseline eta=0 and the trained eta")
    print(f"[confirm] {'seed':>6} {'eta=0':>11} {'trained':>11} {'selected':>11}"
          f" {'sel - eta0':>11} {'max|u|':>9} {'G11':>6}")
    rows = []
    for seed in sorted(runs):
        model, config = load(runs[seed])
        _, confirm = data_provider(config, 'confirm')
        bound = getattr(config, 'g11_bound', None)
        zero = at_eta(model, confirm, config, 0.)
        base_r = at_eta(model, confirm, config, None)
        pick = at_eta(model, confirm, config, best['eta'])
        delta = pick[0]['recall']['mse'] - zero[0]['recall']['mse']
        ok = bound is None or pick[1]['max_abs_state'] < bound
        rows.append({'seed': seed, 'eta0': zero[0]['recall']['mse'],
                     'trained': base_r[0]['recall']['mse'],
                     'selected': pick[0]['recall']['mse'], 'delta': delta,
                     'm_eff': pick[1]['m_eff'], 'max_abs_state': pick[1]['max_abs_state'],
                     'within_bound': ok})
        print(f"[confirm] {seed:>6} {rows[-1]['eta0']:>11.6f} {rows[-1]['trained']:>11.6f}"
              f" {rows[-1]['selected']:>11.6f} {delta:>+11.6f}"
              f" {rows[-1]['max_abs_state']:>9.2f} {'OK' if ok else 'FAIL':>6}")

    d = np.array([r['delta'] for r in rows])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t95 = 2.365 if n == 8 else 1.96                       # t(0.975, df=7)
    lo, hi = d.mean() - t95 * se, d.mean() + t95 * se
    rel = d.mean() / np.mean([r['eta0'] for r in rows]) * 100
    print(f"\n[confirm] paired delta (selected - eta0): mean {d.mean():+.6f}"
          f"  95% CI [{lo:+.6f}, {hi:+.6f}]  relative {rel:+.2f}%  n={n}")
    print(f"[confirm] {'개선이 유의함 (CI 상한 < 0)' if hi < 0 else '유의하지 않음 (CI가 0을 포함)'}")
    print("[confirm] 이 절차가 말하는 것은 eta 개입의 효과뿐이다 (D-AJ). O7 판정이 아니다.")

    if cli.out:
        with open(cli.out, 'w') as handle:
            json.dump({'suite': cli.suite, 'selection': table, 'selected_eta': best['eta'],
                       'disqualified': disqualified, 'confirm': rows,
                       'paired': {'mean': float(d.mean()), 'ci95': [float(lo), float(hi)],
                                  'relative_pct': float(rel), 'n': n}},
                      handle, indent=2)
        print(f"[confirm] written to {cli.out}")


if __name__ == '__main__':
    main()
