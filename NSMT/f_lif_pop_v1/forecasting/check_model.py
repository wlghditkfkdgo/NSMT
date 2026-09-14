"""Synthetic checks of the memory mechanism, independent of forecasting scores."""
import argparse
import copy
import json

from config import set_random_seed
import torch
from f_lif_pop_v1.forecasting.ours import myModel
from utils import parameter_hash


def check(device):
    set_random_seed(7)
    models = []
    for heterogeneous in [False, True]:
        for retrieval in [False, True]:
            set_random_seed(7)
            models.append(myModel(seq_len=32, pred_len=8, patch_size=4, embed_dim=8,
                                  head_dim=8, heterogeneous=heterogeneous, retrieval=retrieval).to(device))
    assert len(set(parameter_hash(m) for m in models)) == 1
    assert len(set(sum(p.numel() for p in m.parameters()) for m in models)) == 1
    torch.testing.assert_close(models[0].embedding.lif.beta.mean(), models[2].embedding.lif.beta.mean())
    x = torch.randn(3, 32, 2, device=device)
    model = models[3]
    output, aux = model(x, return_aux=True)
    assert output.shape == (3, 8, 2)
    assert set(aux['spikes'].unique().tolist()) <= {0., 1.}
    # Last two axes are logical neuron and key time, so mask query/key explicitly.
    w = aux['attention']
    for t in range(w.shape[0]):
        assert w[t, ..., t:].count_nonzero() == 0
        if t:
            torch.testing.assert_close(w[t].sum(-1), torch.ones_like(w[t, ..., 0]))
    assert aux['attention'][0].count_nonzero() == 0
    assert aux['evidence'][0].count_nonzero() == 0
    torch.testing.assert_close(model(x), output, rtol=0, atol=0)
    model(torch.randn_like(x))
    torch.testing.assert_close(model(x), output, rtol=0, atol=0)
    # Changing future patches must not change earlier membrane/query states.
    changed = x.clone()
    changed[:, 16:] += 20.
    _, changed_aux = model(changed, return_aux=True)
    for key in ['membrane', 'attention', 'charge', 'spikes']:
        torch.testing.assert_close(aux[key][:4], changed_aux[key][:4], rtol=0, atol=0)
    torch.testing.assert_close(model(x, memory_mode='off'), models[2](x), rtol=0, atol=0)
    zero_strength = copy.deepcopy(model)
    zero_strength.embedding.lif.memory_strength = 0.
    torch.testing.assert_close(zero_strength(x), models[2](x), rtol=0, atol=0)
    _, hom_aux = models[0](x, return_aux=True)
    torch.testing.assert_close(hom_aux['membrane'], hom_aux['membrane'][..., :1].expand_as(hom_aux['membrane']), rtol=0, atol=0)
    assert aux['membrane'].std(-1).mean() > 0
    # Independently reconstruct additive evidence and post-reset values.
    state = torch.zeros_like(aux['membrane'][0])
    currents = model.embedding.proj(x.transpose(1, 2).reshape(6, 32).unfold(-1, 4, 4).permute(1, 0, 2)) * 2.
    beta = model.embedding.lif.beta
    for t in range(8):
        charged = beta * state + (1. - beta) * currents[t, ..., None]
        memory = sum((w[t, ..., j, None] * aux['membrane'][j] for j in range(t)), torch.zeros_like(state))
        evidence = .05 * aux['gate'][t] * memory
        voltage = charged + evidence
        state = voltage - (voltage >= 1.).to(voltage.dtype)
        torch.testing.assert_close(state, aux['membrane'][t], rtol=1e-5, atol=1e-6)
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        gradients[name] = parameter.grad.abs().max().item()
    for key in ['embedding.proj.weight', 'embedding.lif.query.weight', 'embedding.lif.key.weight', 'embedding.lif.gate.weight']:
        assert gradients[key] > 0, key
    assert not torch.allclose(output, model(x, memory_mode='off'))
    last = myModel(seq_len=32, pred_len=8, patch_size=4, embed_dim=8, head_dim=8, head_mode='last').to(device)
    assert last(x).shape == output.shape
    return {'status': 'passed', 'device': device, 'parameters': sum(p.numel() for p in model.parameters()),
            'checks': ['paired_initial_parameters', 'matched_mean_beta', 'binary_spikes', 'causal_read',
                       'no_future_patch_leakage', 'no_cross_window_state', 'gate_zero_baseline',
                       'homogeneous_identity', 'heterogeneous_diversity', 'additive_update_reconstruction',
                       'finite_nonzero_memory_gradients', 'head_shapes'], 'gradient_max': gradients}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(check(args.device), indent=2))
