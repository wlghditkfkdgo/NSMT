"""Firing-rate calibration (Phase B, pre-registered decision D11).

The three constants that set the operating point -- the branch time constants tau, the
soma mixing weights w and the threshold theta -- were each fixed for their own reason and
nobody fixed the scale that connects them. At the defaults the membrane peaks at about
half the threshold, so the neuron never fires. This script measures that chain on the
TRAIN SPLIT ONLY, picks a single ``input_scale``, and writes it down.

What D11 forbids is worth repeating: no per-constituent gain and no per-constituent
threshold. Tuning each branch into the same firing rate would erase the heterogeneity
that the whole model is about. One scalar, shared by every comparison condition.

Selection rule, as pre-registered
---------------------------------
D11 asks for a firing rate in [0.1, 0.3] with no dead or saturated CONSTITUENT -- that is,
none of the K branches collapsed to zero or blown up. Among the scales that satisfy it,
take the one whose firing rate is closest to 0.2. If none qualifies, report the failure
instead of widening the band.

Per-unit (embedding dim) dead and saturated fractions are reported as diagnostics. They
are not a pass condition here, because D11 does not ask for one; they are what motivates
``--input_norm frozen``.

Usage
-----
    python calibrate.py --task recall --n_keys 3
    python calibrate.py --task ett --data ETTh1
"""

import os
import json
from pathlib import Path

import numpy as np
import torch

import layers
from config import TASK, Config, parse_arguments, set_random_seed, neuron_kwargs
from data_provider.data_factory import data_provider

BAND = (0.1, 0.3)                        # 사전등록 D11의 목표 발화율 구간
TARGET = 0.2                             # 구간 중앙. 동률일 때의 선택 기준
GRID = [1., 2., 3., 4., 6., 8., 10., 12., 14., 17., 20., 25., 30.]


@torch.no_grad()
def probe(embedding, loader, device, scale, max_batches=8, mode='full'):
    """Run a few train batches at one input_scale and summarise the operating point.

    Args
    ----
    embedding : layers.Embedding
    scale : float
        Candidate input_scale. Set on the module, so the linear projection is shared
        across candidates and only the gain moves.
    mode : str, default 'full'
        Calibrate in the neutral limit, which is condition-independent. The sparse
        condition starts at eta = 0.018 and is verified against the same band afterwards.

    Returns
    -------
    dict with firing rate, dead/saturated fraction, per-constituent |u| and max|u|.
    """
    embedding.input_scale = float(scale)
    rate, per_unit, branch, peak, drive = [], [], [], 0., []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0].to(device)                                  # [B, L, C]
        patch = layers.to_patches(x, embedding.emb_linear.in_features)
        spikes, aux = embedding(patch, mode=mode, return_aux=True)
        rate.append(spikes.mean().item())
        per_unit.append(spikes.mean(dim=(0, 1)).cpu().numpy())   # [D] 유닛별 발화율
        branch.append(aux['state'].abs().mean(dim=(0, 1, 2)).cpu().numpy())
        peak = max(peak, aux['state'].abs().max().item())
        drive.append(patch.abs().max().item() * scale)

    per_unit = np.concatenate([u[None] for u in per_unit]).mean(0)

    return {'input_scale': float(scale), 'firing_rate': float(np.mean(rate)),
            'dead_frac': float((per_unit == 0.).mean()),
            'saturated_frac': float((per_unit > 0.9).mean()),
            'unit_rate_min': float(per_unit.min()), 'unit_rate_max': float(per_unit.max()),
            'branch_abs_mean': [float(v) for v in np.mean(branch, axis=0)],
            'max_abs_state': float(peak), 'max_abs_drive': float(np.max(drive))}


def constituents_healthy(row):
    """D11's 'no dead or saturated constituent': every branch alive and none dominating."""
    branch = np.array(row['branch_abs_mean'])

    return bool((branch > 1e-3).all() and np.isfinite(branch).all()
                and branch.max() / branch.min() < 100.)


def choose(rows):
    """Apply the pre-registered rule. Returns (row or None, reason)."""
    ok = [r for r in rows if BAND[0] <= r['firing_rate'] <= BAND[1] and constituents_healthy(r)]
    if not ok:
        return None, (f"no input_scale put the firing rate in {BAND} with every constituent "
                      f"alive; widening the band is NOT allowed by D11")

    return min(ok, key=lambda r: abs(r['firing_rate'] - TARGET)), 'closest to target'



def main():
    args = parse_arguments()
    set_random_seed(args.seed)
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)
    config.dataset = 'recall' if config.task == 'recall' else config.data
    config.data_path = config.data + '.csv'
    config.num_patches = config.seq_len // config.patch_size
    config.device = torch.device('cpu' if config.cpu else f'cuda:{config.num_device}')

    _, loader = data_provider(config, 'train')
    embedding = layers.Embedding(config.patch_size, config.embed_dim, config.input_scale,
                                 input_norm=config.input_norm, **neuron_kwargs(config)).to(config.device)
    if config.input_norm == 'frozen':
        # 여러 배치를 모아 적합한다. 한 배치로는 추정이 흔들린다.
        sample = []
        for i, batch in enumerate(loader):
            if i >= 8:
                break
            sample.append(layers.to_patches(batch[0].to(config.device), config.patch_size))
        embedding.fit_norm(torch.cat(sample, dim=1))
        print(f"[calib] frozen norm fitted: mean in "
              f"[{embedding.norm_mean.min():.3f}, {embedding.norm_mean.max():.3f}], "
              f"std in [{embedding.norm_std.min():.3f}, {embedding.norm_std.max():.3f}]")

    print(f"[calib] band {BAND}, target {TARGET}, mode 'full' (neutral limit)")
    print(f"[calib] {'scale':>6} {'rate':>7} {'dead':>6} {'sat':>6} "
          f"{'unit min/max':>14} {'max|u|':>8}  branch |u| by tau")
    rows = []
    for scale in GRID:
        row = probe(embedding, loader, config.device, scale)
        rows.append(row)
        print(f"[calib] {scale:>6.1f} {row['firing_rate']:>7.4f} {row['dead_frac']:>6.2f} "
              f"{row['saturated_frac']:>6.2f} {row['unit_rate_min']:>6.3f}/{row['unit_rate_max']:<7.3f} "
              f"{row['max_abs_state']:>8.2f}  "
              + " ".join(f"{v:.3f}" for v in row['branch_abs_mean']))

    picked, reason = choose(rows)
    if picked is None:
        print(f"[calib] FAILED: {reason}")
    else:
        print(f"[calib] picked input_scale = {picked['input_scale']} "
              f"(firing rate {picked['firing_rate']:.4f}, {reason})")
        # G11: 학습 전에 상태 상한을 선언해 둔다. 학습 중 이 값을 넘으면 중단한다.
        print(f"[calib] G11 declared bound = 10 * max|I| = {10 * picked['max_abs_drive']:.1f}, "
              f"observed max|u| = {picked['max_abs_state']:.2f}")

    # 조건 간 공유를 위해 sparse 초기값에서도 같은 구간에 있는지 확인한다
    if picked is not None:
        check = probe(embedding, loader, config.device, picked['input_scale'], mode='sparse')
        print(f"[calib] sparse at init (eta = {torch.sigmoid(embedding.neuron.selector.eta_hat).item():.4f}): "
              f"firing rate {check['firing_rate']:.4f}  "
              f"{'inside band' if BAND[0] <= check['firing_rate'] <= BAND[1] else 'OUTSIDE BAND'}")

    out = Path(TASK) / 'results' / 'calibration'
    os.makedirs(out, exist_ok=True)
    stem = f"{config.dataset}_k{config.n_keys}" if config.task == 'recall' else config.dataset
    name = f"{stem}_a{config.alpha}_norm-{config.input_norm}_seed{config.seed}.json"
    payload = {'band': BAND, 'target': TARGET, 'mode': 'full', 'input_norm': config.input_norm,
               'grid': rows,
               'picked': picked, 'reason': reason,
               'neuron': {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in neuron_kwargs(config).items()},
               'task': config.task, 'dataset': config.dataset, 'seed': config.seed}
    with open(out / name, 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(f"[calib] written to `{out / name}`")

    return 0 if picked is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
