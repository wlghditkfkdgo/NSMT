import os

import numpy as np
import torch

from config import Config, parse_arguments, set_random_seed
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
from utils import write_json

# 사건 종류별 분리 보고. 전체 MSE만 쓰면 n_keys가 클수록 copy 가중치가 늘어난다 (prereg D-W).
EVENT_KINDS = {'all': None, 'copy': 0, 'recall': (1, 2), 'recall_first': 2}
# 개입 조건. `uniform`은 `full`과 수학적으로 같으므로 대조군이 아니라 게이트다 (D-H, G7b).
INTERVENTIONS = ('full', 'recent', 'mass_matched', 'oracle')


def kind_mask(kind, want):
    if want is None:
        return torch.ones_like(kind, dtype=torch.bool)
    if isinstance(want, tuple):
        return sum((kind == v) for v in want).bool()

    return kind == want


@torch.no_grad()
def evaluate(model, data_loader, args, mode=None, baselines=False):
    """Pooled and per-event-kind errors, summed over raw elements before dividing."""
    model.eval()
    mode = mode or args.mode
    sums = {name: torch.zeros(3, dtype=torch.float64, device=args.device) for name in EVENT_KINDS}
    extra = torch.zeros(4, dtype=torch.float64, device=args.device)
    example = None
    for i, batch in enumerate(data_loader):
        if args.max_eval_batches and i >= args.max_eval_batches:
            break
        x, y, truth, kind = batch
        x, y = x.float().to(args.device), y.float().to(args.device)
        truth = truth.to(args.device) if truth.numel() else None
        output = model(x, mode=mode, truth=truth)
        error = (output - y).double()
        if args.task == 'recall':
            kind = kind.to(args.device)
            for name, want in EVENT_KINDS.items():
                m = kind_mask(kind, want)
                sums[name][0] += (error * m).square().sum()
                sums[name][1] += (error.abs() * m).sum()
                sums[name][2] += m.sum()
        else:
            sums['all'][0] += error.square().sum()
            sums['all'][1] += error.abs().sum()
            sums['all'][2] += y.numel()
            if baselines:
                for offset, guess in [(0, x[:, -1:]), (2, x.mean(1, keepdim=True))]:
                    gap = (guess - y).double()
                    extra[offset] += gap.square().sum()
                    extra[offset + 1] += gap.abs().sum()
        if example is None:
            example = {'prediction': output[0].cpu().tolist(), 'target': y[0].cpu().tolist()}

    if sums['all'][2] == 0:
        raise ValueError('Empty evaluation')
    result = {}
    for name, total in sums.items():
        if total[2] == 0:
            continue
        result[name] = {'mse': (total[0] / total[2]).item(), 'mae': (total[1] / total[2]).item(),
                        'elements': int(total[2].item())}
    result['loss'] = result['all']['mse']
    result['mse'], result['mae'] = result['all']['mse'], result['all']['mae']
    if not np.isfinite([result['mse'], result['mae']]).all():
        raise FloatingPointError('Nonfinite evaluation')
    if baselines and args.task != 'recall':
        n = sums['all'][2]
        result['baselines'] = {name: {'mse': (extra[k] / n).item(), 'mae': (extra[k + 1] / n).item()}
                               for name, k in [('persistence', 0), ('window_mean', 2)]}

    return result, example


@torch.no_grad()
def selection_diagnostics(model, data_loader, args, mode=None, batches=4):
    """What the selector actually did, on real data.

    m_eff is the POST-cap coefficient ratio that prereg D-Q defines, not the pre-cap formula
    (1-eta)*m0 + eta, which only holds for a pre-cap perfect oracle. It is averaged over the
    queries inside a sequence first and then over sequences, in that order, because queries
    are not uniformly placed in time.
    """
    model.eval()
    mode = mode or args.mode
    per_sequence = {k: [] for k in ('m_eff', 'hit', 'kernel_mass')}
    pooled = {k: [] for k in ('kappa', 'cap_rate', 'would_cap_rate', 'eta', 'support_p',
                              'support_rho', 'support_braw', 'support_c')}
    peak, peak_branch, finite = 0., None, True
    for i, batch in enumerate(data_loader):
        if i >= batches:
            break
        x, y, truth, kind = batch
        x, truth = x.float().to(args.device), truth.to(args.device)
        _, aux = model(x, mode=mode, truth=truth, return_aux=True)
        live = aux['has_history']
        for key in pooled:
            pooled[key].append(aux[key][live].mean().item())
        magnitude = aux['state'].abs()
        peak = max(peak, magnitude.max().item())
        branch = magnitude.amax(dim=(0, 1, 2)).cpu().numpy()
        peak_branch = branch if peak_branch is None else np.maximum(peak_branch, branch)
        finite = finite and bool(torch.isfinite(aux['state']).all())

        b = model.embedding.neuron.b
        for n in range(1, len(aux['coeff'])):
            c = aux['coeff'][n]                                   # [B, D, n]
            if c is None:
                continue
            answer = truth[:, n, :n]                              # [B, n]
            has = answer.any(-1)
            if not has.any():
                continue
            mask = answer.unsqueeze(1).to(c.dtype)
            share = (c * mask).sum(-1) / c.sum(-1).clamp_min(1e-12)
            per_sequence['m_eff'].append(share[has].mean().item())
            top = c.argmax(-1)                                    # 가장 크게 읽은 칸
            per_sequence['hit'].append(answer.gather(1, top[:, :1].clamp_max(n - 1))[has[:, None]
                                                                                    ].float().mean().item())
            bh = b[1:n + 1].flip(0).double()
            per_sequence['kernel_mass'].append(
                ((bh * answer.double()).sum(-1) / bh.sum())[has].mean().item())

    bound = getattr(args, 'g11_bound', None)
    return {**{k: float(np.mean(v)) if v else None for k, v in per_sequence.items()},
            **{k: float(np.mean(v)) for k, v in pooled.items()},
            'max_abs_state': peak, 'branch_abs_max': [float(v) for v in peak_branch],
            'finite': finite, 'g11_bound': bound,
            'within_bound': None if bound is None else bool(peak < bound)}


def test(args: Config):
    set_random_seed(args.seed)
    _, test_loader = data_provider(args, flag='test')
    model = LOAD_MODEL[args.model](args, train=False)

    result, example = evaluate(model, test_loader, args, baselines=True)
    payload = {'run_id': args.run_id, 'mode': args.mode, 'task': args.task, 'seed': args.seed,
               'test': result, 'example': example}
    if args.model == 'myModel':
        payload['diagnostics'] = selection_diagnostics(model, test_loader, args)
        # 같은 checkpoint에 정책만 바꿔 넣는 개입. 학습된 표현은 그대로 둔다.
        payload['interventions'] = {}
        for mode in INTERVENTIONS:
            if mode == 'oracle' and args.task != 'recall':
                continue
            payload['interventions'][mode] = evaluate(model, test_loader, args, mode=mode)[0]
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    write_json(args.result_path, payload)
    print(f"[test] {args.run_id}: mse {result['mse']:.6f}")
    for name in ('copy', 'recall', 'recall_first'):
        if name in result:
            print(f"[test]   {name:12s} mse {result[name]['mse']:.6f}  ({result[name]['elements']} events)")

    return payload


def main():
    args = parse_arguments()
    config = Config()
    if args.config is None:
        raise ValueError('Standalone test needs --config pointing at an existing run directory')
    config.load_args(args.config, args)
    config.result_path = os.path.join(args.config, 'standalone_test.json')

    return test(config)


if __name__ == '__main__':
    main()
