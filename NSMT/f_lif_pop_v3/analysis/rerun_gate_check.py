"""Prereg 2T gate checks of the rerun evaluators (rerun_common, ett_rerun, recall_rerun). Reads no split
beyond validation and writes nothing outside a scratch registry.

    python rerun_gate_check.py --main <main-tree f_lif_pop_v3/forecasting> --smoke <smoke suite> -nd <gpu>

(a) the evaluators' one pass reproduces the earlier evaluators on the earlier checkpoints (main tree,
    read only), on validation:
    E1  ETT: the 2O q1_R / pearson_R checkpoints (ETTh1, ETTh2, 8 seeds) give the 2P record's validation
        MSE, firing rate, max|u| (rel 1e-9) and the same tie counts, with no mask mismatch; kept mass to
        rel 1e-7 (one_pass sums the kernel in float64, the earlier ETT evaluators in float32; E3)
    E2  recall: the 2J selection checkpoints (seed 7, q 1 / 0.1 / 0.25 / 0.5) give the 2J record's
        validation recall MSE (1e-12) and max|u| (rel 1e-9); M_eff to 1e-8 (one_pass forms the shares in
        float64 like the 2M evaluator, test.selection_diagnostics in float32; E3)
    E3  the residuals of E1 / E2 are that precision and nothing else: the earlier float32 formulas give the
        records' kept mass and M_eff exactly (|diff| 0) on the same checkpoints
    (The first run of this gate, with 1e-9 for kept mass and M_eff, failed E1 at 9.8e-9 and E2 at 1.8e-9;
    MSE, firing rate and max|u| differed by 0. E3 was added after that run; see PROJECT_LOG 2T.)
(b) the new classes and fault injection, on the smoke suite (1-epoch runs, results not reported):
    F1  a correctly specified smoke run fails its manifest ONLY for its short training (epoch 1 instead of the
        registered budget; ETT: no early-stop evidence; recall: epochs_run != 12)
    F2  the wrong condition spec is refused (manifest errors)
    F3  another calibration file is refused
    F4  a missing run is fatal (find_cell)
    F5  a legacy run checked against a new-neuron condition is refused
    F6  NaN: paired() refuses NaN; a model with a NaN weight is not finite and not safe in one_pass
    F7  a changed earlier record and a changed calibration file are fatal
    F8  reopen: a never-opened split is refused, an opened one is reopened once, a second reopen is refused
    F9  a blocked cell withholds its contrast, and 'withheld' enters the verdict comparison
    N1  new classes in one_pass: the rebuilt pearson masks equal the model's (0 mismatches), q1 keeps the
        whole kernel (kept mass 1), q 0.5 keeps less, and validation reproduces best_val_loss (1e-6)
"""
import sys
import glob
import json
import tempfile
import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK, Config, parse_defaults, set_random_seed   # noqa: E402
from data_provider.data_factory import data_provider              # noqa: E402
import rerun_common as rc                                         # noqa: E402
import split_registry                                             # noqa: E402

RESULTS = []
SEEDS = rc.SEEDS


def gate(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"[rerun-gate] {'PASS' if ok else 'FAIL'} | {name:<3} {detail}")


def rel(a, b):
    return abs(a - b) / max(1., abs(b))


def refuses(fn):
    try:
        fn()
    except (SystemExit, ValueError):
        return True
    return False


def val_pass(model, args):
    set_random_seed(args.seed)
    _, loader = data_provider(args, 'val')
    return rc.one_pass(model, loader, args)


# ---- (a) earlier checkpoints ------------------------------------------------------------------------------
def e1(main, device):
    worst, worst_kept, ties_ok, mism, n = 0., 0., True, 0, 0
    for data in ('ETTh1', 'ETTh2'):
        record, _, _ = rc.legacy_record('2P', data)
        rows = {(r['cond'], r['seed']): r for r in record['rows'] if r.get('status') == 'evaluated'}
        for cond, q in (('q1_R', 1), ('pearson_R', .5)):
            for seed in SEEDS:
                runs = glob.glob(str(Path(main) / 'log' / 'etthard-revin-20260927' / data / '*' / '*+model+myModel+*'
                                     / f'seed{seed}_flatten_spike_heterogeneous_hard-shared-pearson-q{q:g}_revin'))
                if len(runs) != 1:
                    gate('E1', False, f'{data} {cond} seed {seed}: {len(runs)} earlier run directories')
                    return
                if rc.sha256(Path(runs[0]) / 'model_state' / 'best+model.pt') != record['checkpoint_sha256'][f'{cond}/{seed}']:
                    gate('E1', False, f'{data} {cond} seed {seed}: checkpoint differs from the 2P record')
                    return
                model, args = rc.load_model(runs[0], device)
                r, old = val_pass(model, args), rows[(cond, seed)]
                for key in ('mse', 'firing_rate', 'max_abs_state'):
                    worst = max(worst, rel(r[key], old[key]))
                worst_kept = max(worst_kept, rel(r['kept_mass_frac'], old['kept_mass_frac']))
                ties_ok = ties_ok and r['ties'] == old['ties']
                mism += r['mask_mismatch'] or 0
                n += 1
    gate('E1', worst <= 1e-9 and worst_kept <= 1e-7 and ties_ok and mism == 0,
         f'{n} earlier ETT checkpoints on validation vs the 2P record: worst rel diff {worst:.1e} '
         f'(MSE, firing rate, max|u|), kept mass {worst_kept:.1e}, tie counts equal {ties_ok}, mask mismatches {mism}')


def e2(main):
    record, _, _ = rc.legacy_record('2J')
    worst_mse, worst_meff, worst_peak, n = 0., 0., 0., 0
    for row in record['rows']:
        q = float(row['q'])
        runs = glob.glob(str(Path(main) / 'log' / 'hardsel-014614' / 'recall' / '*' / '*'
                             / f'seed7_*hard-shared-pearson-q{q:g}_k3'))
        if len(runs) != 1 or rc.sha256(Path(runs[0]) / 'model_state' / 'best+model.pt') != record['checkpoint_sha256'][f'{q:g}']:
            gate('E2', False, f'q={q:g}: {len(runs)} earlier run directories or a different checkpoint')
            return
        model, args = rc.load_model(runs[0], torch.device('cpu'))
        r = val_pass(model, args)
        worst_mse = max(worst_mse, abs(r['recall_mse'] - row['recall_mse']))
        worst_meff = max(worst_meff, abs(r['m_eff'] - row['m_eff']))
        worst_peak = max(worst_peak, rel(r['max_abs_state'], row['max_abs_state']))
        n += 1
    gate('E2', worst_mse <= 1e-12 and worst_meff <= 1e-8 and worst_peak <= 1e-9,
         f'{n} earlier recall checkpoints on validation vs the 2J record: recall MSE |diff| {worst_mse:.1e}, '
         f'M_eff |diff| {worst_meff:.1e}, max|u| rel {worst_peak:.1e}')


@torch.no_grad()
def e3(main, device):
    """The earlier float32 formulas on one checkpoint each reproduce the records exactly."""
    from test import selection_diagnostics
    record, _, _ = rc.legacy_record('2P', 'ETTh1')
    old = next(r for r in record['rows'] if r['cond'] == 'pearson_R' and r['seed'] == 7)
    run = glob.glob(str(Path(main) / 'log' / 'etthard-revin-20260927' / 'ETTh1' / '*' / '*+model+myModel+*'
                        / 'seed7_flatten_spike_heterogeneous_hard-shared-pearson-q0.5_revin'))[0]
    model, args = rc.load_model(run, device)
    set_random_seed(args.seed)
    _, loader = data_provider(args, 'val')
    kept, kept_n = 0., 0
    for x, _, _, _ in loader:
        _, aux = model(x.float().to(device), mode=args.mode, return_aux=True)
        b, state = model.embedding.neuron.b, aux['state']
        share = torch.zeros(state.shape[1], dtype=torch.float64, device=device)
        for n in range(1, len(aux['coeff'])):                      # ett_alpha.test: the kernel summed in float32
            share += (aux['coeff'][n].double().sum(-1) / b[1:n + 1].sum().double()).mean(-1)
        kept += (share / (len(aux['coeff']) - 1)).sum().item()
        kept_n += state.shape[1]
    d_kept = abs(kept / kept_n - old['kept_mass_frac'])
    record, _, _ = rc.legacy_record('2J')
    row = next(r for r in record['rows'] if float(r['q']) == .5)
    run = glob.glob(str(Path(main) / 'log' / 'hardsel-014614' / 'recall' / '*' / '*' / 'seed7_*hard-shared-pearson-q0.5_k3'))[0]
    model, args = rc.load_model(run, torch.device('cpu'))
    set_random_seed(args.seed)
    _, loader = data_provider(args, 'val')
    d_meff = abs(selection_diagnostics(model, loader, args, batches=10 ** 9)['m_eff'] - row['m_eff'])
    gate('E3', d_kept == 0. and d_meff == 0.,
         f'earlier float32 formulas: ETTh1 pearson_R seed 7 kept mass |diff| {d_kept:.1e}, '
         f'recall q=0.5 M_eff (test.selection_diagnostics) |diff| {d_meff:.1e}')


# ---- (b) new classes and faults ------------------------------------------------------------------------
def smoke_cells(smoke, calibration):
    out = {}
    for cond in ('q1_R', 'pearson_R'):
        spec = rc.ETT_CONDITIONS[cond]
        status, run, result = rc.find_cell(smoke, 'ett', 'ETTh1', spec, 7)
        out[('ett', cond)] = (spec, run, result, calibration[rc.calibration_label('ett', cond, 'ETTh1')])
    for q in (1., .5):
        spec = {**rc.recall_spec('pearson', q), 'head_mode': 'flatten', 'alpha': .7}
        status, run, result = rc.find_cell(smoke, 'recall', 'recall', spec, 7)
        out[('recall', q)] = (rc.recall_spec('pearson', q), run, result,
                              calibration[rc.calibration_label('recall', stat='pearson', q=q)])
    return out


def faults(main, smoke, calibration, device):
    cells = smoke_cells(smoke, calibration)
    loaded = {}
    for key, (spec, run, result, cal) in cells.items():
        loaded[key] = rc.load_model(run, device if key[0] == 'ett' else torch.device('cpu'))

    # F1: only the short training is flagged
    only_short = {}
    for key, (spec, run, result, cal) in cells.items():
        _, args = loaded[key]
        if key[0] == 'ett':
            errs = rc.check_manifest(args, result, 'ett', spec, 7, cal, data='ETTh1', cond=key[1])
            only_short[str(key)] = 'epoch: 1 != 50' in errs and any('without matching evidence' in e for e in errs) \
                and all(e == 'epoch: 1 != 50' or 'without matching evidence' in e for e in errs)
        else:
            errs = rc.check_manifest(args, result, 'recall', spec, 7, cal)
            only_short[str(key)] = errs == ['epoch: 1 != 12', 'epochs_run 1 != 12']
    gate('F1', all(only_short.values()), f'correct specs flag only the 1-epoch training: {only_short}')

    # F2 / F3
    spec, run, result, cal = cells[('ett', 'q1_R')]
    _, args = loaded[('ett', 'q1_R')]
    wrong = rc.check_manifest(args, result, 'ett', rc.ETT_CONDITIONS['pearson_R'], 7, cal, data='ETTh1', cond='q1_R')
    gate('F2', any(e.startswith('neuron:') for e in wrong) and any(e.startswith('hard_q:') for e in wrong),
         f'q1_R run checked as pearson_R: {len(wrong)} errors, e.g. {[e for e in wrong if not "evidence" in e][:2]}')
    other = {**cal, 'file': 'ETTh1_a0.7_norm-frozen_revin_seed7_260927-175652.json'}
    errs = rc.check_manifest(args, result, 'ett', spec, 7, other, data='ETTh1', cond='q1_R')
    gate('F3', sum('calibration' in e for e in errs) >= 2, f'another calibration file: {[e for e in errs if "calibration" in e]}')

    # F4 / F5
    gate('F4', refuses(lambda: rc.find_cell('no-such-suite', 'ett', 'ETTh1', spec, 7)), 'a missing run is fatal')
    legacy_run = glob.glob(str(Path(main) / 'log' / 'etthard-revin-20260927' / 'ETTh1' / '*' / '*+model+myModel+*'
                               / 'seed7_flatten_spike_heterogeneous_hard-shared-pearson-q1_revin'))[0]
    largs = Config()
    largs.load_args(legacy_run, parse_defaults())
    lres = glob.glob(str(Path(main) / 'results' / 'etthard-revin-20260927' / '*_heterogeneous_hard-shared-pearson-q1_revin_seed7.json'))
    lres = [f for f in lres if 'ETTh1' in Path(f).name]
    errs = rc.check_manifest(largs, lres[0], 'ett', spec, 7, cal, data='ETTh1', cond='q1_R')
    gate('F5', any(e.startswith('neuron:') for e in errs), f'legacy run as a new-neuron condition: {[e for e in errs if e.startswith("neuron")]}')

    # F6: NaN
    nan_paired = refuses(lambda: rc.paired([1., float('nan')] + [1.] * 6, [1.] * 8))
    model, args = rc.load_model(cells[('recall', .5)][1], torch.device('cpu'))
    with torch.no_grad():
        model.embedding.emb_linear.weight[0, 0] = float('nan')
    r = val_pass(model, args)
    gate('F6', nan_paired and not r['finite'] and not r['safe'],
         f"paired refuses NaN {nan_paired}; NaN weight -> finite {r['finite']}, safe {r['safe']}")

    # F7: changed fingerprints
    saved = dict(rc.LEGACY)
    rc.LEGACY[('2N', 'ETTh1')] = (saved[('2N', 'ETTh1')][0], '0' * 64)
    legacy_fatal = refuses(lambda: rc.legacy_record('2N', 'ETTh1'))
    rc.LEGACY.clear()
    rc.LEGACY.update(saved)
    manifest_path = TASK / 'results' / smoke / 'calibration_manifest.json'
    original = manifest_path.read_text()
    try:
        tampered = json.loads(original)
        first = next(iter(tampered['entries']))
        tampered['entries'][first]['sha256'] = '0' * 64
        manifest_path.write_text(json.dumps(tampered))
        cal_fatal = refuses(lambda: rc.calibration_manifest(smoke))
    finally:
        manifest_path.write_text(original)
    gate('F7', legacy_fatal and cal_fatal, f'changed earlier record fatal {legacy_fatal}; changed calibration file fatal {cal_fatal}')

    # F8: reopen on a scratch registry (the real one is untouched)
    real = split_registry.REGISTRY
    with tempfile.TemporaryDirectory() as scratch:
        split_registry.REGISTRY = Path(scratch)
        try:
            never = refuses(lambda: split_registry.reopen('never-opened', 'gate', 'x'))
            split_registry.open_once('opened-once', 'gate', 'x')
            first = split_registry.reopen('opened-once', 'gate', 'x').exists()
            second = refuses(lambda: split_registry.reopen('opened-once', 'gate', 'x'))
            entry = json.loads((Path(scratch) / 'opened-once.reopen-2T.json').read_text())
        finally:
            split_registry.REGISTRY = real
    gate('F8', never and first and second and entry['exploratory'] is True and entry['first_opening']['split'] == 'opened-once',
         f'never opened refused {never}; reopened once {first}; second reopen refused {second}; marked exploratory {entry["exploratory"]}')

    # F9: withheld propagation
    from ett_rerun import contrast
    values = {(c, s): 1. + .01 * i for i, (c, s) in enumerate((c, s) for c in ('a', 'b') for s in SEEDS)}
    t = contrast(('a', 'b'), values, {('a', 13): 'G11 failure in training'}, .95)
    cmp_ = rc.compare('dbg95', t, rc.paired(rc.per_seed(values, 'a'), rc.per_seed(values, 'b')))
    gate('F9', t['status'] == 'withheld' and cmp_['rerun'] == 'withheld' and cmp_['agreement'] == 'different verdict',
         f"blocked cell -> {t['status']}; comparison {cmp_}")

    # N1: new classes in one pass
    detail, ok = {}, True
    for key in (('ett', 'q1_R'), ('ett', 'pearson_R'), ('recall', 1.), ('recall', .5)):
        model, args = loaded[key]
        r = val_pass(model, args)
        best = json.loads(Path(cells[key][2]).read_text())['train']['best_val_loss']
        err = abs((r['mse'] if key[0] == 'ett' else r['all_mse']) - best)
        selective = key[1] in ('pearson_R', .5)
        good = err <= rc.REPRO_TOL and r['safe'] and (
            (r['mask_mismatch'] == 0 and r['ties'] is not None and r['kept_mass_frac'] < 1.) if selective
            else abs(r['kept_mass_frac'] - 1.) < 1e-12)
        ok = ok and good
        detail[str(key)] = f"val |diff| {err:.1e}, kept {r['kept_mass_frac']:.4f}, mismatch {r['mask_mismatch']}, ties {r['ties']}"
    gate('N1', ok, f'new classes: {detail}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True, help='main-tree f_lif_pop_v3/forecasting (earlier checkpoints, read only)')
    ap.add_argument('--smoke', required=True, help='smoke suite with calibration_manifest.json and 1-epoch runs')
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    a = ap.parse_args()
    if torch.__version__ != rc.TORCH:
        raise SystemExit(f'torch {torch.__version__} != {rc.TORCH}')
    device = torch.device(f'cuda:{a.num_device}')
    e1(a.main, device)
    e2(a.main)
    e3(a.main, device)
    faults(a.main, a.smoke, rc.calibration_manifest(a.smoke), device)
    print(f'[rerun-gate] {sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
