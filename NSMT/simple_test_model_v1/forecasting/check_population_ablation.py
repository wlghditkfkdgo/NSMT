"""Check that only tuning changes, and the original Gaussian model is preserved."""
import argparse
import json
import subprocess
import types

import run_ett  # Imports pandas before torch and establishes the workspace path.
import torch
from forecasting.simple_test_model_v1 import SimpleTestModelV1, RepeatedScalarCoding


BASE = 'e59ac8ab4646059fbfa94655e7b0000d91f13cff'


def check(device, backend):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    original = types.ModuleType('original_population_model')
    source = subprocess.check_output(['git', 'show', BASE+':NSMT/forecasting/simple_test_model_v1.py'],
                                     cwd=run_ett.NSMT, text=True)
    exec(compile(source, '<original model>', 'exec'), original.__dict__)
    rows = []
    for axis in ('temporal', 'none'):
        config = dict(seq_len=25, pred_len=7, patch_size=4, num_population=7,
                      d_model=16, num_heads=4, head_dim=9, attention_axis=axis, backend=backend)
        models = []
        for cls, extra in ((original.SimpleTestModelV1, {}),
                           (SimpleTestModelV1, {'population_code':'gaussian'}),
                           (SimpleTestModelV1, {'population_code':'repeat'})):
            torch.manual_seed(13)
            models.append(cls(**config, **extra).to(device))
        old, gaussian, repeat = models
        for model in (gaussian, repeat):
            assert model.state_dict().keys() == old.state_dict().keys()
            for key, tensor in old.state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2,25,3,device=device)
        y = torch.randn(2,7,3,device=device)
        outputs = []
        for model in models:
            out, aux = model(x,return_aux=True)
            assert aux['embedding_spikes'].shape == (7,6,7,16)
            assert aux['num_steps'] == 7
            assert aux['embedding_spikes'].any()
            (out-y).square().mean().backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert model.embedding.linear.weight.grad.abs().sum() > 0
            outputs.append(out.detach())
            if model is repeat:
                # Every repeated slot has identical tuning-free features and spikes.
                torch.testing.assert_close(aux['population'], aux['population'][:,:,0:1].expand_as(aux['population']))
                torch.testing.assert_close(aux['embedding_spikes'], aux['embedding_spikes'][:,:,0:1].expand_as(aux['embedding_spikes']))
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        for name, parameter in old.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(gaussian.named_parameters())[name].grad,rtol=0,atol=0)
        with torch.no_grad():
            first = repeat(x)
            repeat(torch.randn(1,25,1,device=device))
            torch.testing.assert_close(first,repeat(x),rtol=0,atol=0)
        rows.append({'axis':axis,'initial_state_identical':True,'parameters':sum(p.numel() for p in gaussian.parameters()),
                     'original_gaussian_forward_and_grad_exact':True,'repeat_nonzero_embedding_gradient':True})
    coder = RepeatedScalarCoding(7,-3,3).to(device)
    values = torch.tensor([[[-10.,-3.,0.,3.,10.]]],device=device)
    expected = torch.tensor([0.,0.,.5,1.,1.],device=device).expand(1,1,7,5)
    torch.testing.assert_close(coder(values),expected,rtol=0,atol=0)
    return {'status':'passed','device':device,'backend':backend,'base':BASE,'checks':rows}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--backend',default='torch')
    args=parser.parse_args()
    print(json.dumps(check(args.device,args.backend),indent=2))
