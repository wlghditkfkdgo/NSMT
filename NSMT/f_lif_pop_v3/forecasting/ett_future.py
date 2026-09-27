import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from scipy import stats

import sys
import glob
import json
import math
import argparse
from pathlib import Path

from config import set_random_seed, Config, TASK
from data_provider.data_factory import data_provider
import layers
from ett_test import (sha256, load_model, stopping_evidence, REGISTERED, SEEDS, TORCH, MDE,
                      DATA_ROOT, DATA_SHA256, CALIBRATION, DATASETS)

sys.path.insert(0, str(TASK.parent / 'analysis'))
from split_registry import open_once                            # noqa: E402

# prereg 2O: the one look at the future period (targets in [14400, 17420), 2,925 windows) for
# the window-normalisation question (D-BJ ~ D-BN). ett_test.py, the 2N evaluator, is left as it
# was run; this file reuses its unchanged helpers and adds the R-on conditions and the Linear.

S2N, S2O = 'etthard-20260925', 'etthard-revin-20260927'
SPLIT = 'future'
PEARSON = {'hard_axis': 'shared', 'hard_stat': 'pearson'}
CONDITIONS = {
    'q1':        {'suite': S2N, 'model': 'myModel', 'mode': 'hard', 'revin': False, 'hard_q': 1., **PEARSON},
    'pearson':   {'suite': S2N, 'model': 'myModel', 'mode': 'hard', 'revin': False, 'hard_q': .5, **PEARSON},
    'gru':       {'suite': S2N, 'model': 'GRU', 'mode': 'full', 'revin': False},
    'linear':    {'suite': S2O, 'model': 'Linear', 'mode': 'full', 'revin': False},
    'q1_R':      {'suite': S2O, 'model': 'myModel', 'mode': 'hard', 'revin': True, 'hard_q': 1., **PEARSON},
    'pearson_R': {'suite': S2O, 'model': 'myModel', 'mode': 'hard', 'revin': True, 'hard_q': .5, **PEARSON},
    'gru_R':     {'suite': S2O, 'model': 'GRU', 'mode': 'full', 'revin': True},
    'linear_R':  {'suite': S2O, 'model': 'Linear', 'mode': 'full', 'revin': True},
}
# R 켠 보정은 별도 고정 artifact다 (2O D-BK). R 없는 조건은 2N의 보정을 그대로 쓴다.
CALIBRATION_R = {'ETTh1': ('ETTh1_a0.7_norm-frozen_revin_seed7_260927-175652.json',
                           '2f7980227440d63e3cf1d103888ec270bbd50711b1c4e48726cabfa27e3d56bf'),
                 'ETTh2': ('ETTh2_a0.7_norm-frozen_revin_seed7_260927-175659.json',
                           '026998d11948c73e1d2735f0736caa437a3b2cf485d9ea4f406547b57eb2e016')}
INPUT_SCALE = {False: 6., True: 10.}


def variant_of(spec):
    if spec['model'] == 'myModel':
        v = f"heterogeneous_hard-shared-pearson-q{spec['hard_q']:g}"
    else:
        v = 'heterogeneous_full'

    return v + ('_revin' if spec['revin'] else '')


def find_run(data, pred_len, cond, seed):
    """run directory and result json; the model name is in the path because GRU and Linear
    share one variant name inside the 2O suite."""
    spec = CONDITIONS[cond]
    variant = variant_of(spec)
    runs = glob.glob(str(TASK / 'log' / spec['suite'] / data / '*' / f"*+model+{spec['model']}+*"
                         / f'seed{seed}_*_{variant}'))
    results = glob.glob(str(TASK / 'results' / spec['suite']
                            / f"{spec['model']}_ett_{data}_p{pred_len}_*_{variant}_seed{seed}.json"))
    if len(runs) != 1 or len(results) != 1:
        raise SystemExit(f"[ett_future] {data} {cond} seed {seed}: {len(runs)} run dirs, {len(results)} results")

    return runs[0], results[0]


def calibration_of(data, revin):
    return (CALIBRATION_R if revin else CALIBRATION)[data]


def check_manifest(args, result, data, pred_len, cond, seed):
    """Every registered field for this condition; returns the mismatches (empty = OK)."""
    spec = CONDITIONS[cond]
    want = {**REGISTERED, 'data': data, 'pred_len': pred_len, 'seed': seed,
            **{k: v for k, v in spec.items() if k != 'suite'}}
    bad = [f"{k}: {getattr(args, k, None)!r} != {v!r}" for k, v in want.items() if getattr(args, k, None) != v]
    calibration = json.load(open(TASK / 'results' / 'calibration' / calibration_of(data, spec['revin'])[0]))['picked']
    if spec['model'] == 'myModel':
        if getattr(args, 'g11_bound', None) != calibration['declared_bound']:
            bad.append(f"g11_bound {getattr(args, 'g11_bound', None)!r} != {calibration['declared_bound']!r}")
        if getattr(args, 'input_scale', None) != INPUT_SCALE[spec['revin']]:
            bad.append(f"input_scale {getattr(args, 'input_scale', None)!r} != {INPUT_SCALE[spec['revin']]}")
    if getattr(args, 'data_path', None) != f'{data}.csv':
        bad.append(f"data_path {getattr(args, 'data_path', None)!r} != '{data}.csv'")
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
    if (provenance.get('calibration') or {}).get('file') != calibration_of(data, spec['revin'])[0]:
        bad.append(f"calibration {(provenance.get('calibration') or {}).get('file')!r} != "
                   f"{calibration_of(data, spec['revin'])[0]!r}")
    bad += stopping_evidence(args, train, cond, data, pred_len, seed)

    return bad


def test(args:Config, model, cond, flag=SPLIT):
    """Whole split, one forward per batch: MSE/MAE, safety, kept mass, ties, firing rate.

    flag='val' is the dry run before the future period is claimed; it writes nothing.
    """

    set_random_seed(args.seed)

    _, loader = data_provider(args, flag=flag)

    mse = MeanSquaredError().to(args.device)
    mae = MeanAbsoluteError().to(args.device)

    spiking = args.model == 'myModel'
    count_ties = cond.startswith('pearson')
    peak, finite, nonfinite, windows = 0., True, 0, 0
    kept, kept_n, ties, tie_n, mismatch, spikes, spike_n = 0., 0, 0, 0, 0, 0., 0
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(loader):
            x, y, _, _ = batch
            x = x.float().to(args.device)
            y = y.float().to(args.device)

            output, aux = model(x, mode=args.mode, return_aux=True)   # 오차와 상태를 같은 forward에서
            mse.update(output.contiguous(), y.contiguous())
            mae.update(output.contiguous(), y.contiguous())
            windows += x.shape[0]

            if not spiking:
                if not torch.isfinite(output).all():
                    finite, nonfinite = False, nonfinite + 1
                continue

            state = aux['state']                                       # [T, B*C, D, K]
            if not (torch.isfinite(state).all() and torch.isfinite(output).all()):
                finite, nonfinite = False, nonfinite + 1               # 최대값에 섞지 않는다
            else:
                peak = max(peak, state.abs().max().item())
            spikes += aux['spikes'].double().mean(dim=(0, 2)).sum().item()   # 시퀀스(=창 x 채널) 가중
            spike_n += aux['spikes'].shape[1]

            b = model.embedding.neuron.b
            share = torch.zeros(state.shape[1], dtype=torch.float64, device=args.device)
            for n in range(1, len(aux['coeff'])):
                c = aux['coeff'][n].double()                           # [B*C, D, n] = m * b
                share += (c.sum(-1) / b[1:n + 1].sum().double()).mean(-1)
            kept += (share / (len(aux['coeff']) - 1)).sum().item()
            kept_n += state.shape[1]

            if count_ties:                                             # 학습된 모델의 top-k 경계 동점
                selector = model.embedding.neuron.selector
                xin = x
                if getattr(args, 'revin', False):                      # 모델이 본 입력 = R 변환 후
                    from ours import window_norm
                    xin = window_norm(x)[0]
                current = model.embedding.current(layers.to_patches(xin, args.patch_size))
                zero = torch.zeros_like(state[0])
                for n in range(1, current.shape[0]):
                    xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
                    hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                                        for j in range(n)], dim=-2)
                    m, sim = selector.hard_mask(xi, hist)
                    mismatch += int(not torch.equal(m, (aux['coeff'][n] > 0).to(m.dtype)))   # 재구성 = 모델의 마스크
                    k = max(1, int(round(selector.hard_q * n)))
                    if k < n:
                        ordered = sim[:, 0, :].sort(dim=-1, descending=True).values
                        ties += int((ordered[:, k - 1] == ordered[:, k]).sum())
                    tie_n += xi.shape[0]

    bound = getattr(args, 'g11_bound', None)
    test_result = {
        'mse' : mse.compute().item(),
        'mae' : mae.compute().item(),
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
                     "firing_rate", "ties"])
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
        f"{ties}/{tie_n}" if count_ties else "None", ]
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


def paired(a, b, level=.95):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n)

    return {'deltas': d.tolist(), 'mean': float(mean), 'sd': float(sd), 'level': level,
            'ci': [float(mean - half), float(mean + half)], 'relative': float(mean / np.mean(b)),
            'a_lower': int((d < 0).sum()), 'n': n}


def parse_arguments():

    parser = argparse.ArgumentParser(description='prereg 2O: one look at the future period for the R question')

    parser.add_argument('--data', dest='data', nargs='?', choices=list(DATASETS), required=True)
    parser.add_argument('--pred_len', dest='pred_len', nargs='?', type=int, default=96)
    parser.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, default=0,
                        help='CUDA device index (default: %(default)s); training ran on CUDA, so does the evaluation')
    parser.add_argument('--out', dest='out', default=None, help='record directory under results/ (default: <S2O>-future-<data>)')

    return parser.parse_args()


if __name__ == '__main__':

    config = parse_arguments()
    if torch.__version__ != TORCH:
        raise SystemExit(f"[ett_future] torch {torch.__version__} != {TORCH}: the declared tie rule is that torch.topk")
    device = torch.device(f'cuda:{config.num_device}')

    # ---- 1. 개방 전 전체 대조: 두 데이터셋 128 run이 모두 끝나고 모두 맞아야 한다 (2O D-BM) --------
    errors, runs = [], {}
    for data in DATASETS:
        for revin in (False, True):
            name, digest = calibration_of(data, revin)
            if sha256(TASK / 'results' / 'calibration' / name) != digest:
                errors.append(f"{data}: calibration {name} sha256 is not the fixed one")
        if sha256(DATA_ROOT / f'{data}.csv') != DATA_SHA256[data]:
            errors.append(f"{data}: data CSV sha256 is not the fixed one")
        for cond in CONDITIONS:
            for seed in SEEDS:
                run, result = find_run(data, config.pred_len, cond, seed)
                model, args = load_model(run, device)
                bad = check_manifest(args, result, data, config.pred_len, cond, seed)
                errors += [f"{data} {cond}/{seed}: {b}" for b in bad]
                if data == config.data:
                    runs[(cond, seed)] = (model, args, sha256(Path(run) / 'model_state' / 'best+model.pt'),
                                          sha256(Path(run) / 'model_state' / 'config.pt'), result)
    if errors:
        for e in errors:
            print(f"[ett_future] manifest {e}")
        raise SystemExit(f"[ett_future] {len(errors)} manifest error(s); the {SPLIT} period of {config.data} was NOT opened")
    print(f"[ett_future] manifest OK for all {len(DATASETS) * len(CONDITIONS) * len(SEEDS)} runs "
          f"({', '.join(DATASETS)} H{config.pred_len}); torch {torch.__version__}")

    # ---- 2-3. 전역 등록부 잠금 -> 기록 생성 -------------------------------------------------------
    out = TASK / 'results' / (config.out or f'{S2O}-{SPLIT}-{config.data}')
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'ett_future_record.json'
    open_once(f'{config.data}-{SPLIT}-H{config.pred_len}', 'prereg 2O: window normalisation R on ETT', record_path)
    head = {'prereg': '2O', 'data': config.data, 'pred_len': config.pred_len, 'split': SPLIT,
            'rows': [14400, 17420], 'context': 336, 'suites': [S2N, S2O],
            'torch': torch.__version__, 'device': torch.cuda.get_device_name(device),
            'checkpoint_sha256': {f'{c}/{s}': v[2] for (c, s), v in runs.items()},
            'config_sha256': {f'{c}/{s}': v[3] for (c, s), v in runs.items()},
            'data_sha256': DATA_SHA256[config.data],
            'calibration': {'R_off': CALIBRATION[config.data], 'R_on': CALIBRATION_R[config.data]},
            'source_sha256': {f: sha256(TASK / f) for f in ('layers.py', 'ours.py', 'model.py', 'train.py', 'test.py',
                                                           'config.py', 'ett_test.py', 'ett_future.py',
                                                           'data_provider/data_loader.py')}}
    with open(record_path, 'x') as handle:
        json.dump({**head, 'status': f'opening {config.data} {SPLIT}'}, handle, indent=2)

    # ---- 4. 모델마다 구간 전체를 한 번 ------------------------------------------------------------
    rows, mse = [], {c: [] for c in CONDITIONS}
    for seed in SEEDS:
        for cond in CONDITIONS:
            model, args, _, _, result = runs[(cond, seed)]
            print(f"{f' {config.data} {cond} seed {seed} ':=^100s}")
            r = test(args, model, cond)
            r.update(cond=cond, seed=seed, epochs_run=json.load(open(result))['train']['epochs_run'])
            rows.append(r)
            mse[cond].append(r['mse'])

    unsafe = sorted({(r['cond'], r['seed']) for r in rows if not r['safe']})

    def blocked(*conds):
        return [u for u in unsafe if u[0] in conds]

    # ---- 5. 판정 (2O D-BN) -------------------------------------------------------------------------
    primary = paired(mse['pearson_R'], mse['pearson'], level=.975)          # 두 데이터셋 Bonferroni
    p_block = blocked('pearson_R', 'pearson')
    if p_block:
        verdict = f'판정 보류: unsafe {p_block}'
    elif primary['ci'][1] < 0 and primary['mean'] <= -MDE:
        verdict = 'R이 pearson의 future 오차를 줄이는 개선 후보 (97.5% 상한 < 0, 평균 <= -0.005)'
    else:
        verdict = '개선 후보 아님 (97.5% 상한 < 0 과 평균 <= -0.005 를 함께 만족하지 않음)'
    interaction = [(pr - qr) - (p - q) for pr, qr, p, q in
                   zip(mse['pearson_R'], mse['q1_R'], mse['pearson'], mse['q1'])]
    secondary = {'D_q = q1_R - q1': paired(mse['q1_R'], mse['q1']),
                 'S_off = pearson - q1': paired(mse['pearson'], mse['q1']),
                 'S_on = pearson_R - q1_R': paired(mse['pearson_R'], mse['q1_R']),
                 'I = S_on - S_off': paired(interaction, [0.] * len(interaction)),
                 'gru_R - gru': paired(mse['gru_R'], mse['gru']),
                 'linear_R - linear': paired(mse['linear_R'], mse['linear'])}
    print(f"{' RESULT ':=^100s}")
    print(f"[ett_future] {config.data} H{config.pred_len} {SPLIT} mean MSE: "
          + ', '.join(f'{c} {np.mean(v):.6f}' for c, v in mse.items()))
    print(f"[ett_future] 1차 D_P = pearson_R - pearson  mean {primary['mean']:+.6f}  97.5% CI "
          f"[{primary['ci'][0]:+.6f}, {primary['ci'][1]:+.6f}]  rel {100 * primary['relative']:+.2f}%  "
          f"a<b {primary['a_lower']}/{primary['n']}  -> {verdict}")
    for name, t in secondary.items():
        rel = '' if name.startswith('I ') else f"rel {100 * t['relative']:+.2f}%  "
        print(f"[ett_future] 보조 {name:<24} mean {t['mean']:+.6f}  95% CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}]  "
              f"{rel}a<b {t['a_lower']}/{t['n']}")
    if unsafe:
        print(f"[ett_future] UNSAFE models: {unsafe}")
    t_all = [r['ties'] for r in rows if r['ties'] is not None]
    print(f"[ett_future] pearson top-k boundary ties {sum(t[0] for t in t_all)}/{sum(t[1] for t in t_all)}; "
          f"mask mismatches {sum(r['mask_mismatch'] or 0 for r in rows)}")

    json.dump({**head, 'status': 'done', 'rows': rows, 'mse': mse, 'unsafe': unsafe,
               'primary': {**primary, 'verdict': verdict, 'blocked': p_block},
               'secondary': secondary}, open(record_path, 'w'), indent=2)
    print(f"Record saved to `{record_path}`")
