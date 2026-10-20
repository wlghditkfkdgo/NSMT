"""Prereg 2T: the recall-task experiments 2J / 2K / 2L / 2M rerun with the spikeDE-based neurons. EXPLORATORY.

    python recall_rerun.py --stage select   --suite <selection suite> --calibration <cal suite>
    python recall_rerun.py --stage confirm2 --selection <select record> --suite <confirm suite> --calibration <cal>
    python recall_rerun.py --stage confirm3 --selection <select record> --confirm-suite <confirm suite> \
                           --suite <control suite> --calibration <cal>
    python recall_rerun.py --stage confirm4 --selection <select record> --suite <confirm suite> --calibration <cal>

select    2J D-AM on validation, seed 7: the lowest validation recall MSE among q {0.1, 0.25, 0.5}; within
          0.005 of the best the largest q wins; an unsafe, non-finite or G11-stopped candidate is out.
confirm2  2J D-AN: delta = recall MSE(selected q) - recall MSE(q = 1) per seed, same initial model (N18);
          primary: 95% CI upper < 0; secondary: primary and >= 20% improvement.
confirm3  2K D-AR at the selected q: pearson - recent and pearson - random (both primary, 95% CI upper < 0);
          recent - q1, random - q1 (secondary); pearson - q1 (descriptive).
confirm4  2M without the oracle (not expressible in the new classes): O7-1 M_eff(pearson) >= 0.5,
          O7-3 pearson - q1 <= -20% with CI upper < 0, and pearson - GRU / GRU - q1 (GRU from the 2M record).
          G14, O7-2 and O7-4 need the oracle and are not run.

2L D-AU is built in: each model passes the whole split in ONE forward per batch for error and state, twice
(the two passes must agree to 1e-12; the random control is reseeded to its seed before each), and
safe = all states finite and max|u| < the frozen bound. Every confirm split was opened before, so each is
a second look recorded with split_registry.reopen. A blocked model withholds every verdict it enters;
no seed is dropped. Each verdict is compared with the earlier record's under the same rule.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK, set_random_seed                       # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
import rerun_common as rc                                      # noqa: E402
from split_registry import reopen                              # noqa: E402

CANDIDATES = (.1, .25, .5)
TIE = .005
SELECTION_SEED = 7
CPU = torch.device('cpu')


def cell(suite, stat, q, seed, calibration):
    """(status, model, args, run, result, manifest errors, G11 record) of one recall run."""
    spec = {**rc.recall_spec(stat, q), 'head_mode': 'flatten', 'alpha': .7}
    cal = calibration[rc.calibration_label('recall', stat=stat, q=q)]
    if cal['refused']:                                         # D-CN: not trained, its comparisons withheld
        bad, info = rc.refused_cell(suite, 'recall', 'recall', spec, seed, cal)
        return {**info, 'errors': bad}
    status, run, result = rc.find_cell(suite, 'recall', 'recall', spec, seed)
    if status == 'g11':
        bad, record = rc.check_failure(run, 'recall', rc.recall_spec(stat, q), seed, cal)
        return {'status': 'g11', 'run': run, 'errors': bad, 'g11': record}
    model, args = rc.load_model(run, CPU)
    bad = rc.check_manifest(args, result, 'recall', rc.recall_spec(stat, q), seed, cal)
    return {'status': 'done', 'model': model, 'args': args, 'run': run, 'result': result, 'errors': bad}


def evaluate(c, split, seed, twice=True):
    """One or two whole passes over `split`; the second must equal the first (2L D-AU)."""
    model, args = c['model'], c['args']
    out = []
    for _ in range(2 if twice else 1):
        model.embedding.neuron.reseed(seed)                   # random: reproducible draws (D-AP)
        set_random_seed(args.seed)
        _, loader = data_provider(args, split)
        out.append(rc.one_pass(model, loader, args))
    r = out[0]
    if twice:
        diff = abs(out[0]['recall_mse'] - out[1]['recall_mse'])
        r.update(second_pass_abs_diff=diff, reproduced=diff <= rc.TWICE_TOL)
    stored = json.loads(Path(c['result']).read_text())['train']
    r.update(seed=seed, epochs_run=stored['epochs_run'], best_val_loss=stored['best_val_loss'],
             train_seconds=stored.get('seconds'), input_scale=args.input_scale, calibration=args.calibration_file,
             checkpoint_sha256=rc.sha256(Path(c['run']) / 'model_state' / 'best+model.pt'))
    return r


def val_reproduction(c, seed):
    """The model's own validation all-event MSE against the trainer's best_val_loss (REPRO_TOL)."""
    r = evaluate(c, 'val', seed, twice=False)
    return abs(r['all_mse'] - r['best_val_loss'])


def contrast(values, blocked, a, b, level=.95):
    block = sorted([list(k) + [v] for k, v in blocked.items() if k[0] in (a, b)])
    if block:
        return {'uses': [a, b], 'blocked': block, 'status': 'withheld'}
    t = rc.paired(rc.per_seed(values, a), rc.per_seed(values, b), level=level)
    t.update(uses=[a, b], blocked=[], status='reported (exploratory)')
    return t


def legacy_check(t, stored, name):
    """A contrast recomputed from the earlier per-seed values must equal what that record stored."""
    if abs(t['mean'] - stored['mean']) > 1e-12 or max(abs(x - y) for x, y in zip(t['ci'], stored['ci95'])) > 1e-12:
        raise SystemExit(f'[recall_rerun] earlier {name}: recomputed {t["mean"]} {t["ci"]} != stored '
                         f'{stored["mean"]} {stored["ci95"]}')


def record_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as handle:                            # 'x': a second run cannot rewrite the record
        json.dump(payload, handle, indent=2, allow_nan=False)


def gate_cells(cells, split):
    """Manifest errors are fatal before any split beyond validation is read; so is a run that does
    not reproduce its own validation MSE. G11 stops are kept (they block, never drop)."""
    errors = [f'{k}: {e}' for k, c in cells.items() for e in c['errors']]
    repro = {}
    if not errors:
        for (cond, seed), c in sorted(cells.items()):
            if c['status'] == 'done':
                repro[f'{cond}/{seed}'] = val_reproduction(c, seed)
        errors += [f'{k}: validation {v:.2e} > {rc.REPRO_TOL}' for k, v in repro.items() if not v <= rc.REPRO_TOL]
    if errors:
        for e in errors:
            print(f'[recall_rerun] gate {e}')
        raise SystemExit(f'[recall_rerun] {len(errors)} gate error(s); {split} was NOT read')
    print(f'[recall_rerun] gate OK: {len(cells)} cells, validation reproduced (worst |diff| '
          f'{max(repro.values()) if repro else float("nan"):.2e}), {sum(c["status"] != "done" for c in cells.values())} '
          f'G11 stops / refused calibrations')
    return repro


def head(stage, cli, calibration):
    return {'prereg': '2T', 'stage': stage, 'exploratory': True, 'suite': cli.suite,
            'calibration_suite': cli.calibration, 'torch': torch.__version__, 'device': 'cpu',
            'calibration': {k: v for k, v in calibration.items() if k.startswith('recall_')},
            'source_sha256': {f: rc.sha256(TASK / f) for f in rc.SOURCES},
            'analysis_sha256': {f: rc.sha256(HERE / f) for f in ('recall_rerun.py', 'split_registry.py')}}


# ---- stages --------------------------------------------------------------------------------------------
def select(cli, calibration):
    path = TASK / 'results' / cli.suite / 'recall_rerun_select.json'
    if path.exists():
        raise SystemExit(f'[recall_rerun] {path} exists; the selection is already recorded')
    cells = {(rc.recall_label('pearson', q), SELECTION_SEED): cell(cli.suite, 'pearson', q, SELECTION_SEED, calibration)
             for q in (1.,) + CANDIDATES}
    errors = [f'{k}: {e}' for k, c in cells.items() for e in c['errors']]
    if errors:
        raise SystemExit('[recall_rerun] manifest errors: ' + '; '.join(errors))
    rows = []
    for q in (1.,) + CANDIDATES:
        key = (rc.recall_label('pearson', q), SELECTION_SEED)
        c = cells[key]
        if c['status'] != 'done':
            rows.append({'q': q, 'label': key[0], 'status': 'g11_failure_in_training' if c['status'] == 'g11' else c['status'],
                         'detail': c.get('g11') or c.get('reason'), 'eligible': False})
            continue
        r = evaluate(c, 'val', SELECTION_SEED)
        r.update(q=q, label=key[0], status='evaluated',
                 repro_abs_diff=abs(r['all_mse'] - r['best_val_loss']))
        r['eligible'] = q < 1. and r['safe'] and r['reproduced'] and r['repro_abs_diff'] <= rc.REPRO_TOL
        rows.append(r)
    cands = [r for r in rows if r['q'] < 1.]
    eligible = [r for r in cands if r['eligible']]
    if not eligible:
        raise SystemExit(f'[recall_rerun] every candidate disqualified: {[(r["q"], r["status"]) for r in cands]}')
    best_mse = min(r['recall_mse'] for r in eligible)
    tied = [r for r in eligible if r['recall_mse'] - best_mse < TIE]
    chosen = max(tied, key=lambda r: r['q'])                    # D-AM: a tie goes to the larger q
    old, name, digest = rc.legacy_record('2J')
    base = next(r for r in rows if r['q'] == 1.)
    print(f"[select] validation, seed {SELECTION_SEED}: " + ', '.join(
        f"q={r['q']:g} {r.get('recall_mse', float('nan')):.6f}{'' if r['eligible'] or r['q'] == 1. else ' (out)'}" for r in rows))
    print(f"[select] best {best_mse:.6f}; within {TIE}: {[r['q'] for r in tied]} -> selected q = {chosen['q']:g} "
          f"(earlier: {old['selected_q']:g} -> {'same selection' if chosen['q'] == old['selected_q'] else 'different selection'})")
    if base.get('status') == 'evaluated':
        print(f"[select] q=1 {base['recall_mse']:.6f}; selected {chosen['recall_mse']:.6f} "
              f"({100 * (chosen['recall_mse'] / base['recall_mse'] - 1):+.1f}%, validation, seed 7, exploratory)")
    record_once(path, {**head('select', cli, calibration), 'split': 'val', 'selection_seed': SELECTION_SEED,
                       'candidates': list(CANDIDATES), 'tie': TIE, 'selected_q': chosen['q'], 'rows': rows,
                       'disqualified': [r['q'] for r in cands if not r['eligible']],
                       'earlier': {'record': name, 'sha256': digest, 'selected_q': old['selected_q'],
                                   'rows': [{k: r[k] for k in ('q', 'recall_mse', 'm_eff', 'max_abs_state')} for r in old['rows']]},
                       'agreement': 'same selection' if chosen['q'] == old['selected_q'] else 'different selection'})
    print(f'[select] recorded to {path}')


def load_selection(path):
    record = json.loads(Path(path).read_text())
    if record.get('prereg') != '2T' or record.get('stage') != 'select':
        raise SystemExit(f'[recall_rerun] {path} is not a 2T selection record')
    moved = [f for f, h in record['source_sha256'].items() if rc.sha256(TASK / f) != h]
    if moved:
        raise SystemExit(f'[recall_rerun] sources changed since the selection: {moved}')
    return record, float(record['selected_q'])


def run_split(cells, split, out_path, purpose, cli, calibration, extra_head):
    """Gate, reopen, record, then every model twice over the split. Returns rows, values, blocked."""
    repro = gate_cells(cells, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reopen(split, purpose, out_path)
    base = {**head(split, cli, calibration), **extra_head, 'split': split, 'validation_reproduction_abs_diff': repro}
    with open(out_path, 'x') as handle:
        json.dump({**base, 'status': f'opening {split}'}, handle, indent=2, allow_nan=False)
    rows, values, blocked = [], {}, {}
    for (cond, seed), c in sorted(cells.items(), key=lambda kv: (rc.SEEDS.index(kv[0][1]), kv[0][0])):
        if c['status'] != 'done':
            status = 'g11_failure_in_training' if c['status'] == 'g11' else c['status']
            rows.append({'condition': cond, 'seed': seed, 'status': status, 'detail': c.get('g11') or c.get('reason'),
                         'safe': False})
            blocked[(cond, seed)] = {'g11_failure_in_training': 'G11 failure in training',
                                     'calibration_refused': 'calibration refused (not trained)'}[status]
            continue
        r = evaluate(c, split, seed)
        r.update(condition=cond, status='evaluated')
        rows.append(r)
        values[(cond, seed)] = r['recall_mse']
        if not (r['safe'] and r['reproduced']):
            blocked[(cond, seed)] = 'unsafe in evaluation' if not r['safe'] else 'second pass differs'
        print(f"[{split}] seed {seed:>4} {cond:<14} recall {r['recall_mse']:.6f}  max|u| {r['max_abs_state']:.2f}  "
              f"safe {r['safe']}  M_eff {r['m_eff']:.4f}  kept {r['kept_mass_frac']:.4f}  "
              f"ties {r['ties']}  2nd-pass |diff| {r['second_pass_abs_diff']:.1e}")
    return base, rows, values, blocked


def confirm2(cli, calibration):
    selection, qsel = load_selection(cli.selection)
    sel = rc.recall_label('pearson', qsel)
    cells = {}
    for seed in rc.SEEDS:
        cells[('q1', seed)] = cell(cli.suite, 'pearson', 1., seed, calibration)
        cells[(sel, seed)] = cell(cli.suite, 'pearson', qsel, seed, calibration)
    path = TASK / 'results' / cli.suite / 'recall_rerun_confirm2.json'
    base, rows, values, blocked = run_split(cells, 'confirm2', path, 'prereg 2T: 2J rerun with the spikeDE-based neurons',
                                            cli, calibration, {'selection': str(cli.selection), 'selected_q': qsel})
    t = contrast(values, blocked, sel, 'q1')
    old, name, digest = rc.legacy_record('2J')
    ov, _ = rc.legacy_values('2J')
    ot = rc.paired(rc.per_seed(ov, 'pearson'), rc.per_seed(ov, 'q1'))
    legacy_check(ot, {'mean': old['confirm']['mean'], 'ci95': old['confirm']['ci95']}, '2J confirm2')
    verdicts = {'primary (D-AN)': rc.compare('dan', t, ot), 'secondary (>= 20%)': rc.compare('dan20', t, ot)}
    print(f"[confirm2] {sel} - q1: " + (f"mean {t['mean']:+.6f}, 95% CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}], "
          f"rel {100 * t['relative']:+.2f}%, lower in {t['a_lower']}/8" if t['status'] != 'withheld' else 'WITHHELD'))
    for k, v in verdicts.items():
        print(f"[confirm2] {k}: rerun {v['rerun']}, earlier {v['earlier']} (q=0.5) -> {v['agreement']}")
    json.dump({**base, 'status': 'done', 'rows': rows, 'blocked': [list(k) + [v] for k, v in sorted(blocked.items())],
               'contrast': t, 'earlier': {'record': name, 'sha256': digest, 'contrast': ot, 'selected_q': old['selected_q']},
               'verdicts': verdicts}, open(path, 'w'), indent=2, allow_nan=False)
    print(f'[confirm2] recorded to {path}')


def confirm3(cli, calibration):
    selection, qsel = load_selection(cli.selection)
    cells = {}
    for seed in rc.SEEDS:
        cells[('q1', seed)] = cell(cli.confirm_suite, 'pearson', 1., seed, calibration)
        cells[('pearson', seed)] = cell(cli.confirm_suite, 'pearson', qsel, seed, calibration)
        cells[('recent', seed)] = cell(cli.suite, 'recent', qsel, seed, calibration)
        cells[('random', seed)] = cell(cli.suite, 'random', qsel, seed, calibration)
    path = TASK / 'results' / cli.suite / 'recall_rerun_confirm3.json'
    base, rows, values, blocked = run_split(cells, 'confirm3', path, 'prereg 2T: 2K rerun with the spikeDE-based neurons',
                                            cli, calibration, {'selection': str(cli.selection), 'selected_q': qsel,
                                                               'confirm_suite': cli.confirm_suite})
    old, name, digest = rc.legacy_record('2K')
    ov, _ = rc.legacy_values('2K')
    tests, verdicts = {}, {}
    for group, pairs, rule in (('primary', (('pearson', 'recent'), ('pearson', 'random')), 'dar'),
                               ('secondary', (('recent', 'q1'), ('random', 'q1')), 'dir95'),
                               ('descriptive', (('pearson', 'q1'),), 'dir95')):
        for a, b in pairs:
            key = f'{a}-{b}'
            t = contrast(values, blocked, a, b)
            ot = rc.paired(rc.per_seed(ov, a), rc.per_seed(ov, b))
            legacy_check(ot, old['tests'][key], f'2K {key}')
            tests[key] = {**t, 'group': group, 'earlier': ot}
            verdicts[key] = rc.compare(rule, t, ot)
            print(f"[confirm3] {group:>11} {key:<15} " + (f"mean {t['mean']:+.6f} 95% CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}] "
                  f"rel {100 * t['relative']:+.2f}% a<b {t['a_lower']}/8" if t['status'] != 'withheld' else 'WITHHELD')
                  + f"  -> {verdicts[key]['rerun']} (earlier {verdicts[key]['earlier']}: {verdicts[key]['agreement']})")
    joint = {s: [verdicts[k][s] for k in ('pearson-recent', 'pearson-random')] for s in ('rerun', 'earlier')}
    print(f"[confirm3] joint (both primaries): rerun {joint['rerun']}, earlier {joint['earlier']} at q=0.5")
    json.dump({**base, 'status': 'done', 'rows': rows, 'blocked': [list(k) + [v] for k, v in sorted(blocked.items())],
               'tests': tests, 'verdicts': verdicts, 'joint': joint,
               'earlier': {'record': name, 'sha256': digest, 'q': old['q'], 'joint': old['joint']}},
              open(path, 'w'), indent=2, allow_nan=False)
    print(f'[confirm3] recorded to {path}')


def confirm4(cli, calibration):
    selection, qsel = load_selection(cli.selection)
    cells = {}
    for seed in rc.SEEDS:
        cells[('q1', seed)] = cell(cli.suite, 'pearson', 1., seed, calibration)
        cells[('pearson', seed)] = cell(cli.suite, 'pearson', qsel, seed, calibration)
    path = TASK / 'results' / cli.suite / 'recall_rerun_confirm4.json'
    base, rows, values, blocked = run_split(cells, 'confirm4', path, 'prereg 2T: 2M (no oracle) rerun with the spikeDE-based neurons',
                                            cli, calibration, {'selection': str(cli.selection), 'selected_q': qsel})
    old, name, digest = rc.legacy_record('2M')
    ov, owithheld = rc.legacy_values('2M')
    if owithheld:
        raise SystemExit(f'[recall_rerun] the 2M record has unsafe models {owithheld}; its GRU values cannot be reused')
    for seed in rc.SEEDS:
        values[('gru', seed)] = ov[('gru', seed)]               # GRU: not retrained, the same confirm4 sequences
    meff = [r['m_eff'] for r in rows if r.get('condition') == 'pearson' and r['status'] == 'evaluated']
    o71_block = sorted([list(k) + [v] for k, v in blocked.items() if k[0] == 'pearson'])
    o71 = {'m_eff_pearson': float(np.mean(meff)) if len(meff) == len(rc.SEEDS) else None, 'blocked': o71_block}
    o71['category'] = 'withheld' if o71_block or o71['m_eff_pearson'] is None else ('pass' if o71['m_eff_pearson'] >= .5 else 'fail')
    old_o71 = 'pass' if old['verdict']['O7-1']['pass'] else 'fail'
    o73 = contrast(values, blocked, 'pearson', 'q1')
    oo73 = rc.paired(rc.per_seed(ov, 'pearson'), rc.per_seed(ov, 'q1'))
    legacy_check(oo73, old['verdict']['O7-3'], '2M O7-3')
    gru = contrast(values, blocked, 'pearson', 'gru')
    ogru = rc.paired(rc.per_seed(ov, 'pearson'), rc.per_seed(ov, 'gru'))
    legacy_check(ogru, old['verdict']['GRU']['pearson_minus_gru'], '2M pearson - gru')
    gq = contrast(values, blocked, 'gru', 'q1')
    ogq = rc.paired(rc.per_seed(ov, 'gru'), rc.per_seed(ov, 'q1'))
    legacy_check(ogq, old['verdict']['GRU']['gru_minus_q1'], '2M gru - q1')
    verdicts = {'O7-1 M_eff >= 0.5': {'rule': 'o71', 'rerun': o71['category'], 'earlier': old_o71,
                                      'agreement': 'same verdict' if o71['category'] == old_o71 else 'different verdict'},
                'O7-3 pearson - q1': rc.compare('o73', o73, oo73),
                'pearson - gru': rc.compare('dir95', gru, ogru), 'gru - q1': rc.compare('dir95', gq, ogq),
                'G14 / O7-2 / O7-4': 'not run: they need the oracle, which the new classes do not express (2T)'}
    print(f"[confirm4] O7-1 M_eff(pearson) {o71['m_eff_pearson']} -> {o71['category']} "
          f"(earlier {old['verdict']['O7-1']['m_eff_pearson']:.4f} {old_o71})")
    for k in ('O7-3 pearson - q1', 'pearson - gru', 'gru - q1'):
        t = {'O7-3 pearson - q1': o73, 'pearson - gru': gru, 'gru - q1': gq}[k]
        print(f"[confirm4] {k:<18} " + (f"mean {t['mean']:+.6f} 95% CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}] "
              f"rel {100 * t['relative']:+.2f}%" if t['status'] != 'withheld' else 'WITHHELD')
              + f"  -> {verdicts[k]['rerun']} (earlier {verdicts[k]['earlier']}: {verdicts[k]['agreement']})")
    json.dump({**base, 'status': 'done', 'rows': rows, 'blocked': [list(k) + [v] for k, v in sorted(blocked.items())],
               'gru_source': {'record': name, 'sha256': digest},
               'O7-1': o71, 'O7-3': o73, 'pearson_minus_gru': gru, 'gru_minus_q1': gq,
               'earlier': {'O7-1': old['verdict']['O7-1'], 'O7-3': oo73, 'pearson_minus_gru': ogru, 'gru_minus_q1': ogq},
               'verdicts': verdicts}, open(path, 'w'), indent=2, allow_nan=False)
    print(f'[confirm4] recorded to {path}')


def main():
    parser = argparse.ArgumentParser(description='prereg 2T: recall-task reruns with the spikeDE-based neurons')
    parser.add_argument('--stage', required=True, choices=['select', 'confirm2', 'confirm3', 'confirm4'])
    parser.add_argument('--suite', required=True)
    parser.add_argument('--calibration', required=True, help='suite holding calibration_manifest.json')
    parser.add_argument('--selection', default=None, help='recall_rerun_select.json (confirm stages)')
    parser.add_argument('--confirm-suite', dest='confirm_suite', default=None, help='confirm3: the q1 / pearson suite')
    cli = parser.parse_args()
    if torch.__version__ != rc.TORCH:
        raise SystemExit(f'[recall_rerun] torch {torch.__version__} != {rc.TORCH}')
    calibration = rc.calibration_manifest(cli.calibration)
    if cli.stage != 'select' and not cli.selection:
        raise SystemExit('[recall_rerun] the confirm stages need --selection')
    if cli.stage == 'confirm3' and not cli.confirm_suite:
        raise SystemExit('[recall_rerun] confirm3 needs --confirm-suite')
    {'select': select, 'confirm2': confirm2, 'confirm3': confirm3, 'confirm4': confirm4}[cli.stage](cli, calibration)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
