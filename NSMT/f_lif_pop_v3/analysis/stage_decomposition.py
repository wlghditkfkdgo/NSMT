"""Where does the answer signal go: score -> p -> pre-cap -> post-cap?

The audit's conditional rule for the next candidate: if the SCORE already ranks the answer
well and the mass disappears at p, entmax moves ahead of the key-representation ablation;
if the score cannot separate the answer at all, the key representation stays first. Those
are different stages, so they have to be measured separately on the same sample.

Every quantity is averaged over recall queries inside a sequence, then over sequences
(prereg D-Q), on the recall events only (kind > 0, audit A02-DIAG). Run from
f_lif_pop_v3/analysis with the forecasting directory importable.
"""
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))

from config import Config, parse_defaults                      # noqa: E402
from model import LOAD_MODEL                                   # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402

STAGES = ('score_top1', 'score_rank', 'p_top1', 'p_mass', 'precap_mass', 'postcap_mass',
          'kernel_mass', 'uniform_slot')


@torch.no_grad()
def decompose(model, loader, args, batches=4):
    neuron = model.embedding.neuron
    sel, b = neuron.selector, neuron.b
    per_sequence = {k: [] for k in STAGES}
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        x, y, truth, kind = batch
        x = x.float().to(args.device)
        truth, kind = truth.to(args.device), kind.to(args.device)
        import layers
        current = model.embedding.current(layers.to_patches(x, args.patch_size))
        aux = neuron(current, mode=args.mode, return_aux=True)[1]
        state = aux['state']                                    # [T, B, D, K]; state[n] = u_{n+1}
        B = x.shape[0]
        totals = {k: torch.zeros(B, dtype=torch.float64, device=args.device) for k in STAGES}
        counts = torch.zeros(B, dtype=torch.float64, device=args.device)

        for n in range(1, current.shape[0]):
            answer = truth[:, n, :n]                            # [B, n]
            valid = (kind[:, n] > 0) & answer.any(-1)
            if not valid.any():
                continue
            zero = torch.zeros_like(state[0])
            xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
            hist = torch.stack([torch.cat([state[j - 1] if j else zero,
                                           current[j].unsqueeze(-1)], dim=-1)
                                for j in range(n)], dim=-2)
            b_hist = b[1:n + 1].flip(0)
            c, sa = sel(xi, hist, b_hist, b[0].item(), args.mode)
            p, rho, score = sa['p'], sa['rho'], sa['score']      # [B, D, n]
            raw = b_hist * rho
            wide = answer.unsqueeze(1).expand(-1, c.shape[1], -1)   # [B, D, n]
            mask = wide.to(c.dtype)

            order = score.argsort(-1, descending=True)          # 점수 높은 순
            hit_at = wide.gather(-1, order)                     # 순위별로 정답인가
            rank = hit_at.to(torch.float64).argmax(-1).to(torch.float64) / max(n - 1, 1)
            step = {
                'score_top1': wide.gather(-1, score.argmax(-1, keepdim=True)
                                          ).to(torch.float64).mean((1, 2)),
                'score_rank': rank.mean(-1),
                'p_top1': wide.gather(-1, p.argmax(-1, keepdim=True)
                                      ).to(torch.float64).mean((1, 2)),
                'p_mass': ((p * mask).sum(-1) / p.sum(-1).clamp_min(1e-12)).mean(-1).double(),
                'precap_mass': ((raw * mask).sum(-1) / raw.sum(-1).clamp_min(1e-12)).mean(-1).double(),
                'postcap_mass': ((c * mask).sum(-1) / c.sum(-1).clamp_min(1e-12)).mean(-1).double(),
                'kernel_mass': ((b_hist * answer.double()).sum(-1) / b_hist.sum()
                                ).expand(B).clone(),
                'uniform_slot': (answer.sum(-1).double() / n),
            }
            keep = valid.to(torch.float64)
            for k in STAGES:
                totals[k] += step[k] * keep
            counts += keep

        answered = counts > 0
        for k in STAGES:
            per_sequence[k].extend((totals[k][answered] / counts[answered]).cpu().tolist())

    return {k: float(np.mean(v)) for k, v in per_sequence.items()}


def main():
    parser = argparse.ArgumentParser(description='stage decomposition of the answer signal')
    parser.add_argument('--runs', nargs='+', required=True, help='glob(s) for run directories')
    args_cli = parser.parse_args()

    rows = []
    for pattern in args_cli.runs:
        for run in sorted(glob.glob(pattern)):
            base = parse_defaults()
            base.cpu = True
            config = Config()
            config.load_args(run, base)
            config.device = torch.device('cpu')
            model = LOAD_MODEL[config.model](config, train=False)
            _, loader = data_provider(config, 'test')
            label = (f"{config.mode}/"
                     f"{'learned' if config.eta_fixed is None else f'eta{config.eta_fixed:g}'}"
                     f"{'/qk' if config.qk_norm else ''}")
            rows.append((label, decompose(model, loader, config)))

    print("단계별 정답 신호 (recall query 내부 평균 -> sequence 평균, kind>0만)")
    print(f"{'조건':<22} {'score top1':>11} {'score rank':>11} {'p top1':>8} {'p mass':>8}"
          f" {'pre-cap':>8} {'post-cap':>9} {'kernel':>8} {'chance':>8}")
    for label, r in rows:
        print(f"{label:<22} {r['score_top1']:>11.4f} {r['score_rank']:>11.4f} {r['p_top1']:>8.4f}"
              f" {r['p_mass']:>8.4f} {r['precap_mass']:>8.4f} {r['postcap_mass']:>9.4f}"
              f" {r['kernel_mass']:>8.4f} {r['uniform_slot']:>8.4f}")
    print("\nscore rank: 최상위 정답 칸의 정규화 순위 (0=1등, 0.5=무작위)")
    print("kernel: 선택 없이 b_d만 쓸 때의 정답 질량.  chance: 무작위 한 칸이 정답일 확률")


if __name__ == '__main__':
    main()
