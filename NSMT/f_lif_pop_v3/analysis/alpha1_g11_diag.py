#!/usr/bin/env python3
"""2P pilot: why does pearson_R at alpha=1 hit the G11 bound on the first batch? (diagnostic)

Same set-up as calibrate.py: the first 8 shuffled train batches (seed 7), R-transformed, patched;
a fresh Embedding at seed 7 with the frozen norm fitted on them. Then max|u| at initialisation for
alpha {0.7, 1} x q {1, 0.5}, (i) at each alpha's own calibrated scale and (ii) at one common scale,
so the kernel and the scale are seen apart. The bound is each alpha's frozen G11 bound.
Also max|u| per patch step for the pearson models, to see whether it builds up along the sequence.
Section 2 (added after section 1 was seen): the same budget q=0.5 with the content-blind rules
'recent' and 'random' at each alpha's own scale -- does any dropping of past increments grow the
state at alpha=1, or only the similarity-based choice?
CPU (the random rule's generator is a CPU generator). Train split only; no model is trained, nothing is written except this script's own output.
"""
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
import layers                                                   # noqa: E402
from config import TASK, Config, parse_defaults, set_random_seed, neuron_kwargs   # noqa: E402
from data_provider.data_factory import data_provider            # noqa: E402
from ours import window_norm                                    # noqa: E402

CAL = {('ETTh1', .7): 'ETTh1_a0.7_norm-frozen_revin_seed7_260927-175652.json',
       ('ETTh1', 1.): 'ETTh1_a1.0_norm-frozen_revin_seed7_260927-214003.json',
       ('ETTh2', .7): 'ETTh2_a0.7_norm-frozen_revin_seed7_260927-175659.json',
       ('ETTh2', 1.): 'ETTh2_a1.0_norm-frozen_revin_seed7_260927-214002.json'}


def config_for(data, alpha, q, device, stat='pearson'):
    args = parse_defaults()
    set_random_seed(7)
    config = Config()
    for k, v in vars(args).items():
        setattr(config, k, v)
    for k, v in {'task': 'ett', 'data': data, 'dataset': data, 'data_path': f'{data}.csv', 'alpha': alpha,
                 'revin': True, 'input_norm': 'frozen', 'mode': 'hard', 'hard_axis': 'shared',
                 'hard_stat': stat, 'hard_q': q, 'seed': 7, 'device': device}.items():
        setattr(config, k, v)
    config.num_patches = config.seq_len // config.patch_size
    return config


@torch.no_grad()
def main(device):
    out = {}
    for data in ('ETTh1', 'ETTh2'):
        config = config_for(data, .7, 1., device)
        _, loader = data_provider(config, 'train')
        batches = []
        for i, batch in enumerate(loader):
            if i >= 8:
                break
            batches.append(layers.to_patches(window_norm(batch[0].to(device).float())[0], config.patch_size))
        cal = {a: json.load(open(TASK / 'results' / 'calibration' / CAL[(data, a)]))['picked'] for a in (.7, 1.)}
        common = cal[1.]['input_scale']
        for alpha in (.7, 1.):
            for q in (1., .5):
                for label, scale in (('own', cal[alpha]['input_scale']), ('common', common)):
                    c = config_for(data, alpha, q, device)
                    torch.manual_seed(7)
                    emb = layers.Embedding(c.patch_size, c.embed_dim, scale, input_norm='frozen',
                                           **neuron_kwargs(c)).to(device)
                    emb.fit_norm(torch.cat(batches, dim=1))
                    peak, per_step = 0., None
                    for patch in batches:
                        _, aux = emb(patch, mode='hard', return_aux=True)
                        s = aux['state'].abs().amax(dim=(1, 2, 3))            # [T]
                        per_step = s if per_step is None else torch.maximum(per_step, s)
                        peak = max(peak, s.max().item())
                    bound = cal[alpha]['declared_bound']
                    key = f'{data} alpha {alpha:g} q {q:g} scale {scale:g} ({label})'
                    out[key] = {'max_abs_state': peak, 'bound_of_alpha': bound, 'within': peak < bound,
                                'per_step_max': [round(v, 2) for v in per_step.tolist()]}
                    print(f"[g11diag] {key:<42} max|u| {peak:9.2f}  bound {bound:8.1f}  "
                          f"{'within' if peak < bound else 'EXCEEDS'}   step 1/10/20/41: "
                          + ' '.join(f'{per_step[t]:.1f}' for t in (1, 10, 20, 41)))
        # section 2: content-blind rules at the same budget
        for alpha in (.7, 1.):
            for stat in ('recent', 'random'):
                c = config_for(data, alpha, .5, device, stat)
                torch.manual_seed(7)
                emb = layers.Embedding(c.patch_size, c.embed_dim, cal[alpha]['input_scale'], input_norm='frozen',
                                       **neuron_kwargs(c)).to(device)
                emb.fit_norm(torch.cat(batches, dim=1))
                peak, per_step = 0., None
                for patch in batches:
                    _, aux = emb(patch, mode='hard', return_aux=True)
                    s = aux['state'].abs().amax(dim=(1, 2, 3))
                    per_step = s if per_step is None else torch.maximum(per_step, s)
                    peak = max(peak, s.max().item())
                bound = cal[alpha]['declared_bound']
                key = f"{data} alpha {alpha:g} q 0.5 {stat} scale {cal[alpha]['input_scale']:g} (own)"
                out[key] = {'max_abs_state': peak, 'bound_of_alpha': bound, 'within': peak < bound,
                            'per_step_max': [round(v, 2) for v in per_step.tolist()]}
                print(f"[g11diag] {key:<42} max|u| {peak:9.2f}  bound {bound:8.1f}  "
                      f"{'within' if peak < bound else 'EXCEEDS'}   step 1/10/20/41: "
                      + ' '.join(f'{per_step[t]:.1f}' for t in (1, 10, 20, 41)))
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(torch.device('cpu'))                                  # 'random' mask generator lives on CPU
