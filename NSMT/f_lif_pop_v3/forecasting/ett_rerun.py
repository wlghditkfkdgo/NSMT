"""Prereg 2T: the ETT experiments 2N / 2O / 2P / 2Q (B) rerun with the spikeDE-based neurons. EXPLORATORY.

    python ett_rerun.py --exp 2N --data ETTh1 --suite <training suite> --calibration <calibration suite> [-nd 0]
    python ett_rerun.py --exp 2P --data both  --suite <training suite> --calibration <calibration suite>

Each experiment is evaluated on the split its earlier evaluator used, with the same contrasts and the
same rules: 2N test (ett_test.py, D-BG), 2O future (ett_future.py, D-BN), 2P validation (ett_alpha.py,
D-BV), 2Q (B) validation (ett_readout.py, D-CB). Every test and future period was opened before, so
2N / 2O are second looks: the registry entry is split_registry.reopen (exploratory), never open_once.
GRU and Linear do not depend on the neuron and were not retrained; their per-seed errors come from the
earlier records (fixed SHA256 in rerun_common.LEGACY), on the same windows.

Before anything is read beyond validation (fail closed): every cell of every dataset of the experiment
is a finished run that passes its manifest or a recorded G11 stop; the calibration files, data files and
earlier records match their fixed hashes; and (2N / 2O) every finished run of the opened dataset
re-measures its own best validation MSE within REPRO_TOL. For each contrast the record gives the rerun's
statistics, the earlier record's statistics (recomputed from its per-seed values and checked against the
stored ones), and whether the verdict category under the original rule is the same.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from config import set_random_seed, Config, parse_defaults, TASK
from data_provider.data_factory import data_provider
import rerun_common as rc

sys.path.insert(0, str(TASK.parent / 'analysis'))
from split_registry import reopen                                # noqa: E402

DATASETS = ('ETTh1', 'ETTh2')
INTERACTION = (('pearson_R', 'q1_R'), ('pearson', 'q1'))
J = (('pearson_R', 'q1_R'), ('pearson_R_a1', 'q1_R_a1'))
EXPERIMENTS = {   # name -> (a, b) or ((a, b), (c, d)) for a difference of differences, rule, CI level, role
    '2N': {'split': 'test', 'conds': ('q1', 'pearson'), 'baselines': ('gru',), 'legacy_ci': 'ci95',
           'contrasts': [('pearson-q1', ('pearson', 'q1'), 'dbg95', .95, 'primary'),
                         ('pearson-gru', ('pearson', 'gru'), 'dbg95', .95, 'secondary'),
                         ('q1-gru', ('q1', 'gru'), 'dbg95', .95, 'secondary')]},
    '2O': {'split': 'future', 'conds': ('q1', 'pearson', 'q1_R', 'pearson_R'),
           'baselines': ('gru', 'linear', 'gru_R', 'linear_R'), 'legacy_ci': 'ci',
           'contrasts': [('D_P = pearson_R - pearson', ('pearson_R', 'pearson'), 'dbn975', .975, 'primary'),
                         ('D_q = q1_R - q1', ('q1_R', 'q1'), 'dir95', .95, 'secondary'),
                         ('S_off = pearson - q1', ('pearson', 'q1'), 'dir95', .95, 'secondary'),
                         ('S_on = pearson_R - q1_R', ('pearson_R', 'q1_R'), 'dir95', .95, 'secondary'),
                         ('I = S_on - S_off', INTERACTION, 'dir95', .95, 'secondary'),
                         ('gru_R - gru', ('gru_R', 'gru'), 'dir95', .95, 'baseline only'),
                         ('linear_R - linear', ('linear_R', 'linear'), 'dir95', .95, 'baseline only')]},
    '2P': {'split': 'val', 'conds': ('q1_R', 'pearson_R', 'q1_R_a1', 'pearson_R_a1'),
           'baselines': ('gru_R', 'linear_R'), 'legacy_ci': 'ci',
           'contrasts': [('F_q = q1_R(0.7) - q1_R(1)', ('q1_R', 'q1_R_a1'), 'dbv', .95, 'labelled'),
                         ('F_p = pearson_R(0.7) - pearson_R(1)', ('pearson_R', 'pearson_R_a1'), 'dbv', .95, 'labelled'),
                         ('S_on(1) = pearson_R(1) - q1_R(1)', ('pearson_R_a1', 'q1_R_a1'), 'dir95', .95, 'reported'),
                         ('J = S_on(0.7) - S_on(1)', J, 'dir95', .95, 'reported'),
                         ('q1_R(1) - gru_R', ('q1_R_a1', 'gru_R'), 'dir95', .95, 'reported'),
                         ('q1_R(1) - linear_R', ('q1_R_a1', 'linear_R'), 'dir95', .95, 'reported'),
                         ('q1_R(0.7) - gru_R', ('q1_R', 'gru_R'), 'dir95', .95, 'reported'),
                         ('q1_R(0.7) - linear_R', ('q1_R', 'linear_R'), 'dir95', .95, 'reported')]},
    '2Q': {'split': 'val', 'conds': ('q1_R', 'pearson_R', 'q1_R_lin', 'pearson_R_lin'), 'baselines': (),
           'legacy_ci': 'ci',
           'contrasts': [('H_q = q1_R(linear) - q1_R(flatten)', ('q1_R_lin', 'q1_R'), 'dcb', .95, 'head rule'),
                         ('H_p = pearson_R(linear) - pearson_R(flatten)', ('pearson_R_lin', 'pearson_R'), 'dcb', .95, 'head rule'),
                         ('S(flatten) = pearson_R - q1_R', ('pearson_R', 'q1_R'), 'dir95', .95, 'reported'),
                         ('S(linear) = pearson_R - q1_R', ('pearson_R_lin', 'q1_R_lin'), 'dir95', .95, 'reported')]},
}


def uses_of(spec):
    return [c for pair in spec for c in pair] if isinstance(spec[0], tuple) else list(spec)


def contrast(spec, values, blocked, level):
    """Paired statistics of one contrast, or 'withheld' when a cell it uses is blocked or missing."""
    uses = uses_of(spec)
    block = sorted([list(k) + [v] for k, v in blocked.items() if k[0] in uses])
    missing = [(c, s) for c in uses for s in rc.SEEDS if (c, s) not in values]
    if block or missing:
        return {'uses': uses, 'blocked': block, 'missing': missing, 'status': 'withheld'}
    if isinstance(spec[0], tuple):
        (a, b), (c, d) = spec
        diff = [(values[(a, s)] - values[(b, s)]) - (values[(c, s)] - values[(d, s)]) for s in rc.SEEDS]
        t = rc.paired(diff, [0.] * len(diff), level=level, relative=False)
    else:
        a, b = spec
        t = rc.paired(rc.per_seed(values, a), rc.per_seed(values, b), level=level)
    t.update(uses=uses, blocked=[], status='reported (exploratory)')
    return t


def legacy_stored(exp, data, record, name):
    """The earlier record's own statistics of a contrast (None when it does not exist there)."""
    if exp == '2N':
        return record['tests'].get(name)
    if exp == '2O':
        return record['primary'] if name.startswith('D_P') else record['secondary'].get(name)
    if exp == '2P':
        return record['contrasts'].get(name)
    return record['contrasts'][data].get(name)


def legacy_side(exp, data, plan):
    """Every contrast recomputed from the earlier record's per-seed values, checked against what that
    record stored (mean and CI to 1e-12, and the same withheld set). Any mismatch is fatal."""
    record, name, digest = rc.legacy_record(exp, None if exp == '2Q' else data)
    values, withheld = rc.legacy_values(exp, data)
    blocked = {k: 'withheld in the earlier record' for k in withheld}
    out = {}
    for cname, spec, rule, level, role in plan['contrasts']:
        t = contrast(spec, values, blocked, level)
        stored = legacy_stored(exp, data, record, cname)
        if stored is None:
            raise SystemExit(f'[ett_rerun] earlier {exp} {data} record has no contrast {cname!r}')
        if t['status'] == 'withheld' or stored.get('status') == 'withheld':
            if (t['status'] == 'withheld') != (stored.get('status') == 'withheld'):
                raise SystemExit(f'[ett_rerun] earlier {exp} {data} {cname}: withheld state differs from the record')
        else:
            ci = stored[plan['legacy_ci']]
            if abs(t['mean'] - stored['mean']) > 1e-12 or max(abs(t['ci'][0] - ci[0]), abs(t['ci'][1] - ci[1])) > 1e-12:
                raise SystemExit(f'[ett_rerun] earlier {exp} {data} {cname}: recomputed {t["mean"]} {t["ci"]} '
                                 f'!= stored {stored["mean"]} {ci}')
        out[cname] = t
    return out, values, {'file': name, 'sha256': digest}


def gate(plan, suite, calibration, opening, device):
    """Manifest of every cell of every dataset (no forward); models are built for `opening` only."""
    errors, runs, failures = [], {}, {}
    for data in DATASETS:
        if rc.sha256(rc.DATA_ROOT / f'{data}.csv') != rc.DATA_SHA256[data]:
            errors.append(f'{data}: data CSV sha256 is not the fixed one')
        for cond in plan['conds']:
            spec = rc.ETT_CONDITIONS[cond]
            cal = calibration[rc.calibration_label('ett', cond, data)]
            for seed in rc.SEEDS:
                status, run, result = rc.find_cell(suite, 'ett', data, spec, seed)
                if status == 'g11':
                    bad, record = rc.check_failure(run, 'ett', spec, seed, cal)
                    failures[(data, cond, seed)] = {'run': run, 'record': record,
                                                    'config_sha256': rc.sha256(Path(run) / 'model_state' / 'config.pt')}
                else:
                    if data in opening:
                        model, args = rc.load_model(run, device)
                    else:
                        model, args = None, Config()
                        args.load_args(run, parse_defaults())
                    bad = rc.check_manifest(args, result, 'ett', spec, seed, cal, data=data, cond=cond)
                    runs[(data, cond, seed)] = (model, args, run, result)
                errors += [f'{data} {cond}/{seed}: {b}' for b in bad]
    return errors, runs, failures


def evaluate(model, args, split):
    set_random_seed(args.seed)
    _, loader = data_provider(args, flag=split)
    return rc.one_pass(model, loader, args)


def parse_arguments():
    parser = argparse.ArgumentParser(description='prereg 2T: ETT reruns with the spikeDE-based neurons (exploratory)')
    parser.add_argument('--exp', required=True, choices=list(EXPERIMENTS))
    parser.add_argument('--data', required=True, choices=list(DATASETS) + ['both'])
    parser.add_argument('--suite', required=True, help='the 2T ETT training suite')
    parser.add_argument('--calibration', required=True, help='suite holding calibration_manifest.json')
    parser.add_argument('-nd', '--num_device', type=int, default=0)
    parser.add_argument('--out', default=None, help='record directory under results/')
    return parser.parse_args()


def main():
    cli = parse_arguments()
    plan = EXPERIMENTS[cli.exp]
    split = plan['split']
    if cli.data == 'both' and split != 'val':
        raise SystemExit('[ett_rerun] a test / future period is reopened one dataset at a time')
    opening = DATASETS if cli.data == 'both' else (cli.data,)
    if torch.__version__ != rc.TORCH:
        raise SystemExit(f'[ett_rerun] torch {torch.__version__} != {rc.TORCH}: the declared tie rule is that torch.topk')
    device = torch.device(f'cuda:{cli.num_device}')

    # ---- 1. gate: every cell, every fingerprint (nothing beyond validation is read) ------------------------
    calibration = rc.calibration_manifest(cli.calibration)
    errors, runs, failures = gate(plan, cli.suite, calibration, opening, device)
    legacy = {data: legacy_side(cli.exp, data, plan) for data in opening}     # fatal on any mismatch
    if errors:
        for e in errors:
            print(f'[ett_rerun] manifest {e}')
        raise SystemExit(f'[ett_rerun] {len(errors)} manifest error(s); nothing was evaluated')
    print(f"[ett_rerun] {cli.exp}: manifest OK for {len(runs)} finished runs and {len(failures)} recorded G11 stops "
          f"({', '.join(DATASETS)}); earlier records verified; torch {torch.__version__}")

    repro = {}
    if split != 'val':                                   # 2N / 2O: each model re-measures its validation first
        bad = []
        for (data, cond, seed), (model, args, run, result) in sorted(runs.items()):
            if data not in opening:
                continue
            r = evaluate(model, args, 'val')
            best = json.loads(Path(result).read_text())['train']['best_val_loss']
            repro[f'{data}/{cond}/{seed}'] = abs(r['mse'] - best)
            if not abs(r['mse'] - best) <= rc.REPRO_TOL:
                bad.append(f'{data} {cond}/{seed}: val {r["mse"]} vs best_val_loss {best}')
        if bad:
            for b in bad:
                print(f'[ett_rerun] reproduction {b}')
            raise SystemExit(f'[ett_rerun] {len(bad)} run(s) do not reproduce their validation MSE; {split} NOT opened')
        print(f'[ett_rerun] validation reproduced by all {len(repro)} runs (worst |diff| {max(repro.values()):.2e})')

    # ---- 2. record, then the second look ----------------------------------------------------------------
    out = TASK / 'results' / (cli.out or f'{cli.suite}-{cli.exp}-{split}-{cli.data}')
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'ett_rerun_record.json'
    if split != 'val':
        reopen(f'{cli.data}-{split}-H96', f'prereg 2T: {cli.exp} rerun with the spikeDE-based neurons', record_path)
    head = {'prereg': '2T', 'experiment': cli.exp, 'exploratory': True, 'split': split,
            'target_rows': rc.TARGET_ROWS[split], 'context': 336, 'pred_len': 96, 'datasets': list(opening),
            'suite': cli.suite, 'calibration_suite': cli.calibration,
            'calibration': {k: v for k, v in calibration.items()},
            'torch': torch.__version__, 'device': torch.cuda.get_device_name(device),
            'checkpoint_sha256': {f'{d}/{c}/{s}': rc.sha256(Path(v[2]) / 'model_state' / 'best+model.pt')
                                  for (d, c, s), v in runs.items() if d in opening},
            'config_sha256': {f'{d}/{c}/{s}': rc.sha256(Path(v[2]) / 'model_state' / 'config.pt')
                              for (d, c, s), v in runs.items() if d in opening},
            'data_sha256': {d: rc.DATA_SHA256[d] for d in opening},
            'earlier_records': {d: legacy[d][2] for d in opening},
            'validation_reproduction_abs_diff': repro,
            'source_sha256': {f: rc.sha256(TASK / f) for f in rc.SOURCES}}
    with open(record_path, 'x') as handle:
        json.dump({**head, 'status': f'opening {split} of {list(opening)}'}, handle, indent=2, allow_nan=False)

    cells, per_data = [], {}
    for data in opening:
        values, blocked = {}, {}
        for cond in plan['conds']:
            for seed in rc.SEEDS:
                if (data, cond, seed) in failures:
                    cells.append({'data': data, 'cond': cond, 'seed': seed, 'status': 'g11_failure_in_training',
                                  'g11': failures[(data, cond, seed)]['record'], 'safe': False})
                    blocked[(cond, seed)] = 'G11 failure in training'
                    continue
                model, args, run, result = runs[(data, cond, seed)]
                print(f"{f' {cli.exp} {data} {cond} seed {seed} ({split}) ':=^100s}")
                r = evaluate(model, args, split)
                stored = json.loads(Path(result).read_text())['train']
                r.update(data=data, cond=cond, seed=seed, status='evaluated', epochs_run=stored['epochs_run'],
                         best_val_loss=stored['best_val_loss'], input_scale=args.input_scale,
                         calibration=args.calibration_file, train_seconds=stored.get('seconds'))
                if split == 'val':
                    r['repro_ok'] = abs(r['mse'] - stored['best_val_loss']) <= rc.REPRO_TOL
                cells.append(r)
                values[(cond, seed)] = r['mse']
                if not r['safe']:
                    blocked[(cond, seed)] = 'unsafe in evaluation'
                elif split == 'val' and not r['repro_ok']:
                    blocked[(cond, seed)] = 'validation MSE not reproduced'
        legacy_t, legacy_values, _ = legacy[data]
        for cond in plan['baselines']:                 # GRU / Linear: the earlier record's values (same windows)
            for seed in rc.SEEDS:
                values[(cond, seed)] = legacy_values[(cond, seed)]
        contrasts = {}
        for cname, spec, rule, level, role in plan['contrasts']:
            t = contrast(spec, values, blocked, level)
            t.update(role=role, comparison=rc.compare(rule, t, legacy_t[cname]),
                     earlier={k: legacy_t[cname].get(k) for k in ('mean', 'ci', 'relative', 'a_lower', 'status')})
            contrasts[cname] = t
        level = {c: {'rerun_mean': float(np.mean([values[(c, s)] for s in rc.SEEDS if (c, s) in values]))
                     if any((c, s) in values for s in rc.SEEDS) else None,
                     'earlier_mean': float(np.mean([legacy_values[(c, s)] for s in rc.SEEDS if (c, s) in legacy_values]))
                     if any((c, s) in legacy_values for s in rc.SEEDS) else None,
                     'source': 'earlier record (not retrained)' if c in plan['baselines'] else 'rerun'}
                 for c in plan['conds'] + plan['baselines']}
        per_data[data] = {'contrasts': contrasts, 'blocked': [list(k) + [v] for k, v in sorted(blocked.items())],
                          'means': level}

    extra = {}
    if cli.exp == '2Q' and cli.data == 'both':         # D-CB: the readout decision over both datasets
        def decide(get):
            reasons = []
            for data in DATASETS:
                for name in ('H_q = q1_R(linear) - q1_R(flatten)', 'H_p = pearson_R(linear) - pearson_R(flatten)'):
                    t = get(data, name)
                    if t.get('status') == 'withheld':
                        reasons.append(f'{data} {name}: withheld')
                    elif t['ci'][0] > 0 and t['mean'] >= rc.MDE:
                        reasons.append(f"{data} {name}: mean {t['mean']:+.6f}, CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}]")
            return ('flatten' if reasons else 'linear'), reasons
        new = decide(lambda d, n: per_data[d]['contrasts'][n])
        old = decide(lambda d, n: legacy[d][0][n])
        extra['head_decision'] = {'rerun': new, 'earlier': old,
                                  'agreement': 'same verdict' if new[0] == old[0] else 'different verdict'}

    # ---- 3. report ---------------------------------------------------------------------------------------
    print(f"{f' RESULT {cli.exp} ({split}, exploratory rerun) ':=^100s}")
    for data, block in per_data.items():
        print(f'[ett_rerun] {data} mean MSE: ' + ', '.join(
            f"{c} {m['rerun_mean']:.6f} (earlier {m['earlier_mean']:.6f})" if m['rerun_mean'] is not None
            else f'{c} none' for c, m in block['means'].items()))
        for cname, t in block['contrasts'].items():
            cmp_ = t['comparison']
            if t['status'] == 'withheld':
                print(f"[ett_rerun] {data} {cname:<44} WITHHELD ({len(t['blocked'])} blocked, {len(t['missing'])} missing)"
                      f"  earlier {cmp_['earlier']} -> {cmp_['agreement']}")
                continue
            rel = '' if t['relative'] is None else f"rel {100 * t['relative']:+.2f}%  "
            print(f"[ett_rerun] {data} {cname:<44} mean {t['mean']:+.6f}  {int(100 * t['level'] * 10) / 10:g}% CI "
                  f"[{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}]  {rel}a<b {t['a_lower']}/{t['n']}  -> {cmp_['rerun']} "
                  f"(earlier {cmp_['earlier']}: {cmp_['agreement']})")
    if 'head_decision' in extra:
        print(f"[ett_rerun] D-CB head: rerun {extra['head_decision']['rerun'][0]}, earlier "
              f"{extra['head_decision']['earlier'][0]} -> {extra['head_decision']['agreement']}")
    t_all = [c['ties'] for c in cells if c.get('ties') is not None]
    print(f"[ett_rerun] pearson top-k boundary ties {sum(t[0] for t in t_all)}/{sum(t[1] for t in t_all)}; "
          f"mask mismatches {sum(c.get('mask_mismatch') or 0 for c in cells)}")

    json.dump({**head, 'status': 'done', 'cells': cells, 'per_data': per_data, **extra},
              open(record_path, 'w'), indent=2, allow_nan=False)
    print(f'Record saved to `{record_path}`')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
