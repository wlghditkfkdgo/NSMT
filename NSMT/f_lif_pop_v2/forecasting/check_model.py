"""Scientific preflight: exclusions/null reads, gradients, and every backbone's causality."""
import argparse
import copy
import json

from config import set_random_seed, TASK
import torch

from f_lif_pop_v2.forecasting.layers import PopulationLIF, sparsemax
from f_lif_pop_v2.forecasting.ours import myModel
from utils import parameter_hash, write_json


def check(device):
    set_random_seed(7)
    checks = []
    # Independent simplex-projection properties, closed forms, and numerical Jacobian.
    logits = torch.tensor([[0., 0., -4.], [2., 0., -4.], [-10., -20., -1.]], device=device)
    expected = torch.tensor([[.5, .5, 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
    torch.testing.assert_close(sparsemax(logits), expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(sparsemax(logits + 23.), expected, atol=1e-6, rtol=0)
    values = torch.tensor([[.2, .1, -3.]], dtype=torch.float64, requires_grad=True, device=device)
    assert torch.autograd.gradcheck(sparsemax, (values,), eps=1e-6, atol=1e-5)
    checks.append('sparsemax closed forms / shift invariance / finite-difference Jacobian')
    neuron = PopulationLIF().to(device)
    query = torch.tensor([[[1., 1., 1., 1.]]], device=device)
    # Known relevant past slot, amplitude distractor and opposite-state distractor.
    history = torch.stack([query, 10 * query, -query], dim=-2)
    memory, read = neuron.read_memory(query, history)
    torch.testing.assert_close(read['weight'], torch.tensor([[[1., 0., 0.]]], device=device), atol=1e-7, rtol=0)
    torch.testing.assert_close(memory, query, atol=1e-7, rtol=0)
    changed = history.clone(); changed[..., 1, :] *= 2.
    torch.testing.assert_close(neuron.read_memory(query, changed)[0], memory, atol=1e-7, rtol=0)
    # Changing context changes the selected historical index, without a recency rule.
    torch.testing.assert_close(neuron.read_memory(-query, history)[1]['weight'],
                               torch.tensor([[[0., 0., 1.]]], device=device), atol=1e-7, rtol=0)
    empty_memory, empty = neuron.read_memory(query, -history[..., :2, :])
    assert empty_memory.count_nonzero() == empty['weight'].count_nonzero() == empty['gate'].count_nonzero() == 0
    assert read['gate'].item() > empty['gate'].item()
    # The value path must have zero direct gradient for an excluded stored slot.
    values = history.detach().clone().requires_grad_()
    keys = neuron.key(values.detach())
    neuron.read_memory(query, values, keys=keys)[0].sum().backward()
    assert values.grad[..., 1:, :].count_nonzero() == 0
    checks.append('known-slot/context switch/amplitude discrimination/excluded-slot direct read/null read')
    gradients, parameters = {}, {}
    common_hashes = []
    for architecture in ['patch', 'tcn', 'patchtst', 'tsmixer']:
        models = []
        for heterogeneous in [False, True]:
            for policy in ['off', 'dense', 'sparse']:
                set_random_seed(7)
                model = myModel(seq_len=32, pred_len=8, patch_size=4, embed_dim=8, head_dim=8,
                                architecture=architecture, heterogeneous=heterogeneous,
                                retrieval=policy != 'off', read_mode='dense' if policy == 'dense' else 'sparse').to(device)
                models.append(model)
        assert len({parameter_hash(m) for m in models}) == 1
        parameters[architecture] = sum(p.numel() for p in models[0].parameters())
        common_hashes.append(tuple(p.detach().cpu().numpy().tobytes() for name,p in models[0].named_parameters() if not name.startswith('blocks.')))
        model = models[-1]
        x = torch.randn(3, 32, 2, device=device)
        prediction, aux = model(x, return_aux=True)
        assert prediction.shape == (3, 8, 2)
        changed = x.clone(); changed[:, 16:] += 20.
        _, changed_aux = model(changed, return_aux=True)
        for name, layer in aux['layers'].items():
            assert set(layer['spikes'].unique().tolist()) <= {0., 1.}
            for t in range(8):
                assert layer['attention'][t,...,t:].count_nonzero() == 0
                mass = layer['attention'][t].sum(-1)
                torch.testing.assert_close(mass, (mass > 0).to(mass.dtype), atol=1e-5, rtol=0)
            for key in ['membrane', 'attention', 'charge', 'spikes', 'mask']:
                torch.testing.assert_close(layer[key][:4], changed_aux['layers'][name][key][:4], atol=0, rtol=0)
            assert layer['evidence'][0].count_nonzero() == 0
        model(torch.randn_like(x))
        torch.testing.assert_close(model(x), prediction, atol=0, rtol=0)
        torch.testing.assert_close(model(x, memory_mode='off'), models[3](x), atol=0, rtol=0)
        zero = copy.deepcopy(model)
        for module in zero.modules():
            if isinstance(module, PopulationLIF):module.memory_strength=0.
        torch.testing.assert_close(zero(x), models[3](x), atol=0, rtol=0)
        _, homogeneous = models[0](x, return_aux=True)
        for layer in homogeneous['layers'].values():
            torch.testing.assert_close(layer['membrane'],layer['membrane'][...,:1].expand_as(layer['membrane']),atol=0,rtol=0)
        (prediction - torch.randn_like(prediction)).square().mean().backward()
        grad = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (architecture,name)
            grad[name] = parameter.grad.abs().max().item()
        for name in ['embedding.proj.weight','embedding.lif.query.weight','embedding.lif.key.weight','embedding.lif.gate.weight']:
            assert grad[name]>0,(architecture,name)
        gradients[architecture] = grad
        for horizon in [96, 720]:
            forecast = myModel(pred_len=horizon, architecture=architecture).to(device)
            assert forecast(torch.randn(2,336,7,device=device)).shape == (2,horizon,7)
    assert all(x == common_hashes[0] for x in common_hashes)
    checks += ['matched initial parameters for six conditions per backbone',
               'common embedding/readout initialization across backbones',
               'all-layer binary spikes/causal read/no future-patch leakage',
               'no cross-window state/off and gamma0 equivalence/homogeneous identity',
               'finite gradients / nonzero embedding query-key-gate gradients / H96-H720 shapes']
    return {'status':'passed','device':device,'seed':7,'checks':checks,
            'small_model_parameters':parameters,'gradient_max':gradients,
            'scope':'operator and model invariants; end-to-end synthetic forecasting is not trained'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    result=check(args.device)
    write_json(args.output or TASK / 'results' / ('check_model_' + args.device.replace(':','') + '.json'), result)
    print(json.dumps(result,indent=2))
