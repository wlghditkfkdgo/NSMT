import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

import re
import sys
import glob
import json
import argparse
from pathlib import Path

from config import set_random_seed, Config, NSMT
from data_provider.data_factory import data_provider
import layers
from ett_test import sha256, load_model, stopping_evidence, REGISTERED, SEEDS, TORCH, MDE, TASK
from ett_alpha import paired

sys.path.insert(0, str(TASK.parent / 'analysis'))
from split_registry import open_once                            # noqa: E402

# prereg 2R: the one look at the weather test period (targets [42157, 52696), 10,444 windows at
# H96). Two primary questions on data v3 has never read: does the statistical selection lower the
# test error of the plain f-LIF (P1), and does the fractional kernel beat the ordinary LIF (P2,
# each alpha with its own calibration). GRU and Linear are non-spiking references.

SUITE, DATA, SPLIT = 'weather-20260928', 'weather', 'test'
DATA_ROOT = (NSMT / 'forecasting' / 'dataset' / 'weather').resolve()
DATA_SHA256 = '34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63'
CALIBRATION = {.7: ('weather_a0.7_norm-frozen_revin_seed7_260928-033451.json',     # 2R D-CH
                     '2f28660cef72f3541f70cefd9f5dbfb22f8b90ae8b61be0a49f363f13dc30942'),
               1.: ('weather_a1.0_norm-frozen_revin_seed7_260928-033451.json',
                    '3915d27c49f07a9aaab5b30c55070e36750e6a094278c08a882d79a8458f4f98')}
PEARSON = {'hard_axis': 'shared', 'hard_stat': 'pearson'}
SPIKING = {'model': 'myModel', 'mode': 'hard', 'revin': True, 'head_mode': 'flatten', **PEARSON}
CONDITIONS = {
    'q1_R':      {**SPIKING, 'alpha': .7, 'hard_q': 1.},
    'pearson_R': {**SPIKING, 'alpha': .7, 'hard_q': .5},
    'q1_R_a1':   {**SPIKING, 'alpha': 1., 'hard_q': 1.},
    'gru_R':     {'model': 'GRU', 'mode': 'full', 'revin': True, 'head_mode': 'flatten', 'alpha': .7},
    'linear_R':  {'model': 'Linear', 'mode': 'full', 'revin': True, 'head_mode': 'flatten', 'alpha': .7},
}
PRIMARY = {'P1 D_S = pearson_R - q1_R': ('pearson_R', 'q1_R'),
           'P2 D_alpha = q1_R(0.7) - q1_R(1)': ('q1_R', 'q1_R_a1')}
SECONDARY = {f'{s} - {r}': (s, r) for s in ('q1_R', 'pearson_R', 'q1_R_a1') for r in ('gru_R', 'linear_R')}
REPRO_TOL = 1e-6


def variant_of(spec):
    v = (f"heterogeneous_hard-shared-pearson-q{spec['hard_q']:g}" if spec['model'] == 'myModel'
         else 'heterogeneous_full')
    return v + ('_revin' if spec['revin'] else '')


def find_cell(cond, seed, pred_len=96):
    """(status, run dir, result json): 'done' or 'g11' (2P D-BX); anything else stops here."""
    spec = CONDITIONS[cond]
    variant = variant_of(spec)
    runs = glob.glob(str(TASK / 'log' / SUITE / DATA / '*' / f"*+model+{spec['model']}+*+alpha+{spec['alpha']}+*"
                         / f"seed{seed}_{spec['head_mode']}_spike_{variant}"))
    results = glob.glob(str(TASK / 'results' / SUITE / f"{spec['model']}_ett_{DATA}_p{pred_len}_{spec['head_mode']}"
                                                       f"_a{spec['alpha']}_spike_{variant}_seed{seed}.json"))
    if len(runs) != 1 or len(results) > 1:
        raise SystemExit(f"[weather_test] {cond} seed {seed}: {len(runs)} run dirs, {len(results)} results")
    run = Path(runs[0])
    has_ckpt = (run / 'model_state' / 'best+model.pt').exists()
    has_g11 = (run / 'G11_violation.json').exists()
    if has_ckpt and results and not has_g11:
        return 'done', str(run), results[0]
    if has_g11 and not has_ckpt and not results:
        return 'g11', str(run), None
    raise SystemExit(f"[weather_test] {cond} seed {seed}: checkpoint {has_ckpt}, result {bool(results)}, "
                     f"G11 record {has_g11} -- neither a finished run nor a recorded G11 failure")


def calibrated(alpha):
    name, digest = CALIBRATION[alpha]
    return name, digest, json.load(open(TASK / 'results' / 'calibration' / name))['picked']


def check_manifest(args, result, cond, seed, pred_len=96):
    """Every registered field of a finished run; returns the mismatches (empty = OK)."""
    spec = CONDITIONS[cond]
    want = {**REGISTERED, 'data': DATA, 'pred_len': pred_len, 'seed': seed, 'suite': SUITE, **spec}
    bad = [f"{k}: {getattr(args, k, None)!r} != {v!r}" for k, v in want.items() if getattr(args, k, None) != v]
    name, _, picked = calibrated(spec['alpha'])
    if spec['model'] == 'myModel':
        if getattr(args, 'g11_bound', None) != picked['declared_bound']:
            bad.append(f"g11_bound {getattr(args, 'g11_bound', None)!r} != {picked['declared_bound']!r}")
        if getattr(args, 'input_scale', None) != picked['input_scale']:
            bad.append(f"input_scale {getattr(args, 'input_scale', None)!r} != {picked['input_scale']!r}")
    if getattr(args, 'data_path', None) != f'{DATA}.csv':
        bad.append(f"data_path {getattr(args, 'data_path', None)!r} != '{DATA}.csv'")
    if Path(getattr(args, 'root_path', '')).resolve() != DATA_ROOT:
        bad.append(f"root_path {getattr(args, 'root_path', None)!r} != {str(DATA_ROOT)!r}")
    payload = json.load(open(result))
    train, provenance = payload.get('train', {}), payload.get('provenance', {})
    if not 1 <= (train.get('epochs_run') or 0) <= REGISTERED['epoch']:
        bad.append(f"epochs_run {train.get('epochs_run')!r} outside 1..{REGISTERED['epoch']}")
    if payload.get('test') is not None or payload.get('test_skipped') is not True:
        bad.append('test was not skipped in training')
    if provenance.get('checkpoint_sha256') != sha256(Path(args.save_model_state_path) / 'best+model.pt'):
        bad.append('checkpoint sha256 differs from the one the training run recorded')
    if not str(provenance.get('device', '')).startswith('cuda'):
        bad.append(f"trained on {provenance.get('device')!r}, not CUDA")
    if provenance.get('torch') != TORCH:
        bad.append(f"trained with torch {provenance.get('torch')!r} != {TORCH}")
    if (provenance.get('calibration') or {}).get('file') != name:
        bad.append(f"calibration {(provenance.get('calibration') or {}).get('file')!r} != {name!r}")
    bad += stopping_evidence(args, train, cond, DATA, pred_len, seed)

    return bad


def check_failure(run, cond, seed, pred_len=96):
    """A cell that stopped on G11: config, G11 record and stdout describe the same run at the
    fixed operating point (run_id, run_uuid, input_scale, calibration, bound, no-test, stop line)."""
    spec = CONDITIONS[cond]
    name, _, picked = calibrated(spec['alpha'])
    saved = torch.load(Path(run) / 'model_state' / 'config.pt', map_location='cpu')
    want = {**REGISTERED, 'data': DATA, 'pred_len': pred_len, 'seed': seed, 'suite': SUITE, **spec}
    bad = [f"config {k}: {saved.get(k)!r} != {v!r}" for k, v in want.items() if saved.get(k) != v]
    for key, value in (('input_scale', picked['input_scale']), ('calibration_file', name),
                       ('g11_bound', picked['declared_bound']), ('test', False), ('data_path', f'{DATA}.csv')):
        if saved.get(key) != value:
            bad.append(f"config {key} {saved.get(key)!r} != {value!r}")
    if Path(str(saved.get('root_path', ''))).resolve() != DATA_ROOT:                 # audit 59 A29: data origin
        bad.append(f"config root_path {saved.get('root_path')!r} != {str(DATA_ROOT)!r}")
    record = json.load(open(Path(run) / 'G11_violation.json'))
    for key in ('run_id', 'run_uuid'):
        if record.get(key) != saved.get(key):
            bad.append(f"G11 record {key} {record.get(key)!r} != config {saved.get(key)!r}")
    if record.get('bound') != picked['declared_bound'] or record.get('calibration_file') != name:
        bad.append('G11 record bound or calibration differs from the fixed calibration')
    if not record.get('max_abs_state', 0.) >= record.get('bound', float('inf')):
        bad.append('G11 record does not show the bound reached')
    stdout = glob.glob(str(TASK / 'log' / SUITE / f'ett_{DATA}_p{pred_len}_{cond}_seed{seed}_*.stdout'))
    line = re.search(r'FloatingPointError: G11 violated: max\|u\| = ([0-9.]+) reached the bound ([0-9.]+) '
                     r'frozen at calibration \(epoch (\d+), batch (\d+)\)',
                     Path(stdout[0]).read_text()) if len(stdout) == 1 else None
    if not line:
        bad.append(f"{len(stdout)} stdout file(s); no G11 stop line")
    elif (int(line[3]), int(line[4])) != (record.get('epoch'), record.get('batch')) \
            or abs(float(line[1]) - record.get('max_abs_state', -1.)) > 5e-4 or abs(float(line[2]) - record.get('bound', -1.)) > 5e-4:
        bad.append('stdout stop line differs from the G11 record')

    return bad, record


def test(args:Config, model, cond, flag=SPLIT, count_ties=None):
    """Whole split, one forward per batch: MSE/MAE, per-channel MSE, safety, kept mass, ties, firing rate.

    flag='val' is the dry run before the test period is claimed; it writes nothing.
    """

    set_random_seed(args.seed)

    _, loader = data_provider(args, flag=flag)

    mse = MeanSquaredError().to(args.device)
    mae = MeanAbsoluteError().to(args.device)

    spiking = args.model == 'myModel'
    count_ties = cond.startswith('pearson') if count_ties is None else count_ties
    peak, finite, nonfinite, windows = 0., True, 0, 0
    kept, kept_n, ties, tie_n, mismatch, spikes, spike_n = 0., 0, 0, 0, 0, 0., 0
    sse, per_channel_n = None, 0
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(loader):
            x, y, _, _ = batch
            x = x.float().to(args.device)
            y = y.float().to(args.device)

            output, aux = model(x, mode=args.mode, return_aux=True)   # 오차와 상태를 같은 forward에서
            mse.update(output.contiguous(), y.contiguous())
            mae.update(output.contiguous(), y.contiguous())
            err = (output.double() - y.double()).square().sum(dim=(0, 1))   # [C] 채널별
            sse = err if sse is None else sse + err
            per_channel_n += output.shape[0] * output.shape[1]
            windows += x.shape[0]

            if not spiking:
                if not torch.isfinite(output).all():
                    finite, nonfinite = False, nonfinite + 1
                continue

            state = aux['state']                                       # [T, B*C, D, K]
            if not (torch.isfinite(state).all() and torch.isfinite(output).all()):
                finite, nonfinite = False, nonfinite + 1
            else:
                peak = max(peak, state.abs().max().item())
            spikes += aux['spikes'].double().mean(dim=(0, 2)).sum().item()
            spike_n += aux['spikes'].shape[1]

            b = model.embedding.neuron.b
            share = torch.zeros(state.shape[1], dtype=torch.float64, device=args.device)
            for n in range(1, len(aux['coeff'])):
                c = aux['coeff'][n].double()
                share += (c.sum(-1) / b[1:n + 1].sum().double()).mean(-1)
            kept += (share / (len(aux['coeff']) - 1)).sum().item()
            kept_n += state.shape[1]

            if count_ties:
                from ours import window_norm
                selector = model.embedding.neuron.selector
                current = model.embedding.current(layers.to_patches(window_norm(x)[0], args.patch_size))
                zero = torch.zeros_like(state[0])
                for n in range(1, current.shape[0]):
                    xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
                    hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                                        for j in range(n)], dim=-2)
                    m, sim = selector.hard_mask(xi, hist)
                    mismatch += int(not torch.equal(m, (aux['coeff'][n] > 0).to(m.dtype)))
                    k = max(1, int(round(selector.hard_q * n)))
                    if k < n:
                        ordered = sim[:, 0, :].sort(dim=-1, descending=True).values
                        ties += int((ordered[:, k - 1] == ordered[:, k]).sum())
                    tie_n += xi.shape[0]

    bound = getattr(args, 'g11_bound', None)
    test_result = {
        'mse' : mse.compute().item(),
        'mae' : mae.compute().item(),
        'mse_per_channel' : (sse / per_channel_n).tolist(),
        'windows' : windows,
        'finite' : finite,
        'nonfinite_batches' : nonfinite,
        'max_abs_state' : peak if spiking else None,
        'bound' : bound if spiking else None,
        'safe' : finite and (not spiking or (bound is not None and peak < bound)),
        'kept_mass_frac' : kept / kept_n if kept_n else None,
        'firing_rate' : spikes / spike_n if spike_n else None,
        'ties' : [ties, tie_n] if count_ties else None,
        'mask_mismatch' : mismatch if count_ties else None,
    }

    print("Test was successfully done")

    head = ','.join(["split", "mse", "mae", "windows", "safe", "max_abs_state", "bound", "kept_mass_frac",
                     "firing_rate", "ties", "mse_per_channel"])
    results_csv = ','.join([
        flag,
        f"{test_result['mse']:.6f}",
        f"{test_result['mae']:.6f}",
        f"{test_result['windows']}",
        f"{test_result['safe']}",
        f"{test_result['max_abs_state']:.4f}" if spiking else "None",
        f"{bound:.4f}" if spiking else "None",
        f"{test_result['kept_mass_frac']:.4f}" if spiking else "None",
        f"{test_result['firing_rate']:.4f}" if spiking else "None",
        f"{ties}/{tie_n}" if count_ties else "None",
        ' '.join(f"{v:.6f}" for v in test_result['mse_per_channel']), ]
    )

    for k, v in test_result.items():
        if isinstance(v, float):
            print(f" > {k:16s}:{v:>9.6f}")

    if flag != SPLIT:
        return test_result

    with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
        print(head, end="\n", file=log_csv)
        print(results_csv, end="\n", file=log_csv)

    print(f"Final result saved to `{args.save_log_path}`")

    return test_result


def verdict(t):
    """2R D-CI wording."""
    if t['ci'][1] < 0 and t['mean'] <= -MDE:
        return 'lower (97.5% upper < 0 and mean <= -0.005)'
    if t['ci'][0] > 0 and t['mean'] >= MDE:
        return 'higher (97.5% lower > 0 and mean >= +0.005)'
    return 'no difference shown (not evidence of no effect or of equivalence)'


def analyse(rows):
    """Blocked cells, the two primary contrasts (97.5%) and the secondary ones (95%)."""
    blocked = {(r['cond'], r['seed']): ('G11 failure in training' if r['status'] != 'evaluated'
                                        else 'unsafe in evaluation' if not r['safe']
                                        else 'validation MSE not reproduced or not recorded')
               for r in rows if r['status'] != 'evaluated' or not r['safe'] or r.get('repro_ok') is not True}
    mse = {(r['cond'], r['seed']): r['mse'] for r in rows if r['status'] == 'evaluated'}
    out = {}
    for group, table, level in (('primary', PRIMARY, .975), ('secondary', SECONDARY, .95)):
        for name, (a, b) in table.items():
            block = sorted([list(k) + [v] for k, v in blocked.items() if k[0] in (a, b)])
            if block:
                out[name] = {'group': group, 'uses': [a, b], 'blocked': block, 'status': 'withheld'}
                continue
            t = paired([mse[(a, s)] for s in SEEDS], [mse[(b, s)] for s in SEEDS], level=level)
            t.update(group=group, uses=[a, b], blocked=[], status='reported')
            if group == 'primary':
                t['verdict'] = verdict(t)
            out[name] = t

    return blocked, out


def reproduce(runs):
    """2R D-CI: every finished run's validation pass (writes nothing, ties not counted) must give back
    its stored best validation MSE within REPRO_TOL. Runs BEFORE the registry and the test rows
    (audit 60 A32). Returns {(cond, seed): |diff|} and the list of failures."""
    diffs, bad = {}, []
    for (cond, seed), (model, args, _, _, result) in runs.items():
        r = test(args, model, cond, flag='val', count_ties=False)
        stored = json.load(open(result))['train']['best_val_loss']
        diffs[(cond, seed)] = abs(r['mse'] - stored)
        if not np.isfinite(r['mse']) or diffs[(cond, seed)] > REPRO_TOL:
            bad.append(f"{cond}/{seed}: validation {r['mse']!r} vs stored {stored!r}")
    return diffs, bad


def gate(device):
    """Everything that must hold before the registry is locked or a test row is read (2R D-CJ):
    data and calibration fingerprints, and every one of the 40 cells."""
    errors, runs, failures = [], {}, {}
    for alpha, (name, digest) in CALIBRATION.items():
        path = TASK / 'results' / 'calibration' / name
        if not path.exists() or sha256(path) != digest:
            errors.append(f"calibration alpha {alpha}: {name} missing or not the fixed sha256")
    if sha256(DATA_ROOT / f'{DATA}.csv') != DATA_SHA256:
        errors.append('weather.csv sha256 is not the fixed one')
    if errors:
        return errors, runs, failures
    for cond in CONDITIONS:
        for seed in SEEDS:
            status, run, result = find_cell(cond, seed)
            if status == 'g11':
                bad, record = check_failure(run, cond, seed)
                failures[(cond, seed)] = {'run': run, 'record': record,
                                          'config_sha256': sha256(Path(run) / 'model_state' / 'config.pt')}
            else:
                model, args = load_model(run, device)
                bad = check_manifest(args, result, cond, seed)
                runs[(cond, seed)] = (model, args, sha256(Path(run) / 'model_state' / 'best+model.pt'),
                                      sha256(Path(run) / 'model_state' / 'config.pt'), result)
            errors += [f"{cond}/{seed}: {b}" for b in bad]

    return errors, runs, failures


def parse_arguments():

    parser = argparse.ArgumentParser(description='prereg 2R: the one look at the weather test period')

    parser.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, default=0)
    parser.add_argument('--out', dest='out', default=None, help='record directory under results/ (default: <SUITE>-test)')

    return parser.parse_args()


if __name__ == '__main__':

    config = parse_arguments()
    if torch.__version__ != TORCH:
        raise SystemExit(f"[weather_test] torch {torch.__version__} != {TORCH}: the declared tie rule is that torch.topk")
    device = torch.device(f'cuda:{config.num_device}')

    # ---- 1. 개방 전 전체 대조: 40칸이 모두 '완료' 또는 '기록된 G11 실패'이고 지문이 맞아야 한다 ---------
    errors, runs, failures = gate(device)
    if errors:
        for e in errors:
            print(f"[weather_test] manifest {e}")
        raise SystemExit(f"[weather_test] {len(errors)} manifest error(s); the {SPLIT} period of {DATA} was NOT opened")
    print(f"[weather_test] manifest OK: {len(runs)} finished runs, {len(failures)} recorded G11 failures; torch {torch.__version__}")
    diffs, bad = reproduce(runs)                                  # 등록부·test 행보다 먼저 (감사 60 A32)
    if bad:
        for b in bad:
            print(f"[weather_test] reproduction {b}")
        raise SystemExit(f"[weather_test] {len(bad)} validation reproduction failure(s); the {SPLIT} period was NOT opened")
    print(f"[weather_test] validation reproduction OK for {len(diffs)} runs, max |diff| {max(diffs.values()):.2e}")

    # ---- 2. 전역 등록부 잠금 -> 기록 생성 -------------------------------------------------------------
    out = TASK / 'results' / (config.out or f'{SUITE}-{SPLIT}')
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'weather_test_record.json'
    open_once(f'{DATA}-{SPLIT}-H96', 'prereg 2R: statistical selection (P1) and fractional kernel (P2) on weather', record_path)
    head = {'prereg': '2R', 'data': DATA, 'pred_len': 96, 'split': SPLIT, 'rows': [42157, 52696], 'context': 336,
            'suite': SUITE, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(device),
            'checkpoint_sha256': {f'{c}/{s}': v[2] for (c, s), v in runs.items()},
            'config_sha256': {**{f'{c}/{s}': v[3] for (c, s), v in runs.items()},
                              **{f'{c}/{s}': v['config_sha256'] for (c, s), v in failures.items()}},
            'data_sha256': DATA_SHA256, 'calibration': {str(a): list(v) for a, v in CALIBRATION.items()},
            'source_sha256': {f: sha256(TASK / f) for f in ('layers.py', 'ours.py', 'model.py', 'train.py', 'test.py',
                                                           'config.py', 'ett_test.py', 'weather_test.py',
                                                           'data_provider/data_loader.py', 'data_provider/data_factory.py')}}
    with open(record_path, 'x') as handle:
        json.dump({**head, 'status': f'opening {DATA} {SPLIT}'}, handle, indent=2)

    # ---- 3. 모델마다 test 전체를 한 번 ----------------------------------------------------------------
    rows = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            if (cond, seed) in failures:
                rows.append({'cond': cond, 'seed': seed, 'status': 'g11_failure_in_training',
                             'g11': failures[(cond, seed)]['record'], 'safe': False})
                continue
            model, args, _, _, result = runs[(cond, seed)]
            print(f"{f' {DATA} {cond} seed {seed} ':=^100s}")
            r = test(args, model, cond)
            r.update(cond=cond, seed=seed, status='evaluated', epochs_run=json.load(open(result))['train']['epochs_run'],
                     val_repro_abs_diff=diffs[(cond, seed)], repro_ok=diffs[(cond, seed)] <= REPRO_TOL)
            rows.append(r)

    blocked, contrasts = analyse(rows)
    print(f"{' RESULT ':=^100s}")
    means = {c: np.mean([r['mse'] for r in rows if r['cond'] == c and r['status'] == 'evaluated'] or [np.nan])
             for c in CONDITIONS}
    print(f"[weather_test] {DATA} H96 {SPLIT} mean MSE: " + ', '.join(f'{c} {v:.6f}' for c, v in means.items()))
    for name, t in contrasts.items():
        if t['status'] == 'withheld':
            print(f"[weather_test] {t['group']:9s} {name:<34} WITHHELD: {t['blocked'][:2]}")
            continue
        print(f"[weather_test] {t['group']:9s} {name:<34} mean {t['mean']:+.6f}  {100 * t['level']:.1f}% CI "
              f"[{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}]  rel {100 * t['relative']:+.2f}%  a<b {t['a_lower']}/{t['n']}"
              + (f"  -> {t['verdict']}" if 'verdict' in t else ''))
    print(f"[weather_test] blocked cells: {sorted(blocked.items())}")
    t_all = [r['ties'] for r in rows if r.get('ties') is not None]
    print(f"[weather_test] pearson top-k boundary ties {sum(t[0] for t in t_all)}/{sum(t[1] for t in t_all)}; "
          f"mask mismatches {sum(r.get('mask_mismatch') or 0 for r in rows)}")

    json.dump({**head, 'status': 'done', 'rows': rows,
               'mse': {c: {str(r['seed']): r['mse'] for r in rows if r['cond'] == c and r['status'] == 'evaluated'}
                       for c in CONDITIONS},
               'blocked_cells': [list(k) + [v] for k, v in sorted(blocked.items())],
               'contrasts': contrasts}, open(record_path, 'w'), indent=2, allow_nan=False)
    print(f"Record saved to `{record_path}`")
