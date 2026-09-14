"""Synthetic checks of the memory mechanism, independent of forecasting scores."""
import argparse
import copy
import json

from config import set_random_seed
import torch
from f_lif_pop_tcn_v1.forecasting.ours import myModel
from utils import parameter_hash, write_json
from f_lif_pop_tcn_v1.forecasting.layers import PopulationLIF, CausalConv1d


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
    for neuron in zero_strength.modules():
        if isinstance(neuron, PopulationLIF):
            neuron.memory_strength = 0.
    torch.testing.assert_close(zero_strength(x), models[2](x), rtol=0, atol=0)
    _, hom_aux = models[0](x, return_aux=True)
    torch.testing.assert_close(hom_aux['membrane'], hom_aux['membrane'][..., :1].expand_as(hom_aux['membrane']), rtol=0, atol=0)
    assert aux['membrane'].std(-1).mean() > 0
    # Independently reconstruct additive evidence and post-reset values.
    embedding_aux = aux['layers']['embedding']
    state = torch.zeros_like(embedding_aux['membrane'][0])
    w = embedding_aux['attention']
    currents = model.embedding.proj(x.transpose(1, 2).reshape(6, 32).unfold(-1, 4, 4).permute(1, 0, 2)) * 2.
    beta = model.embedding.lif.beta
    for t in range(8):
        charged = beta * state + (1. - beta) * currents[t, ..., None]
        memory = sum((w[t, ..., j, None] * embedding_aux['membrane'][j] for j in range(t)), torch.zeros_like(state))
        evidence = .05 * embedding_aux['gate'][t] * memory
        voltage = charged + evidence
        state = voltage - (voltage >= 1.).to(voltage.dtype)
        torch.testing.assert_close(state, embedding_aux['membrane'][t], rtol=1e-5, atol=1e-6)
    for name, layer_aux in aux['layers'].items():
        assert set(layer_aux['spikes'].unique().tolist()) <= {0., 1.}
        for key in ['membrane', 'attention', 'charge', 'spikes']:
            torch.testing.assert_close(layer_aux[key][:4], changed_aux['layers'][name][key][:4], rtol=0, atol=0)
        assert layer_aux['evidence'][0].count_nonzero() == 0
    # Independently check the dilated convolution's support and left-only padding.
    conv = CausalConv1d(1, 1, kernel_size=3, dilation=2).to(device)
    with torch.no_grad():
        conv.conv.weight.fill_(1.)
        conv.conv.bias.zero_()
    impulse = torch.zeros(1, 1, 10, device=device)
    impulse[..., 3] = 1.
    expected = torch.zeros_like(impulse)
    expected[..., [3, 5, 7]] = 1.
    torch.testing.assert_close(conv(impulse), expected, rtol=0, atol=0)
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        gradients[name] = parameter.grad.abs().max().item()
    for key in ['embedding.proj.weight', 'embedding.lif.query.weight', 'embedding.lif.key.weight', 'embedding.lif.gate.weight']:
        assert gradients[key] > 0, key
    for i in range(2):
        for key in ['conv.conv.weight', 'lif.query.weight', 'lif.key.weight', 'lif.gate.weight']:
            assert gradients[f'blocks.{i}.' + key] > 0
    assert not torch.allclose(output, model(x, memory_mode='off'))
    last = myModel(seq_len=32, pred_len=8, patch_size=4, embed_dim=8, head_dim=8, head_mode='last').to(device)
    assert last(x).shape == output.shape
    for horizon in [96, 720]:
        forecast = myModel(pred_len=horizon).to(device)
        assert forecast(torch.randn(2, 336, 7, device=device)).shape == (2, horizon, 7)
    return {'status': 'passed', 'device': device, 'parameters': sum(p.numel() for p in model.parameters()),
            'checks': ['paired_initial_parameters', 'matched_mean_beta', 'binary_spikes', 'causal_read',
                       'no_future_patch_leakage', 'no_cross_window_state', 'gate_zero_baseline',
                       'homogeneous_identity', 'heterogeneous_diversity', 'additive_update_reconstruction',
                       'finite_nonzero_memory_gradients_all_layers', 'head_shapes_H96_H720',
                       'causal_dilated_convolution_impulse', 'all_layer_future_invariance'], 'gradient_max': gradients}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    result = check(args.device)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
