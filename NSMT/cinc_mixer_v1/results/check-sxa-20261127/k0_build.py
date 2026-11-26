"""Helper of check_sxa.py k0 (2026-11-27): run from a code directory (old HEAD copy or the worktree), build the forecaster for the
earlier conditions at a fixed seed and save every parameter / buffer and the outputs (train mode and eval mode) on a fixed input.
usage: python -B k0_build.py <code_dir> <out.pt>
"""
import os
import sys

code, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, code)
os.chdir(code)
import torch                                   # noqa: E402
from forecast_wrap import Forecaster           # noqa: E402
from spk_mixer import SpkMixer                 # noqa: E402

CONDS = [('muxlin', 'IAND', 2, 1), ('muxlin', 'ADD', 2, 1), ('or', 'IAND', 1, 1), ('sdlin', 'OR', 2, 1), ('muxlin', 'IAND', 2, 3)]
res = {}
for comb, cf, M, K in CONDS:
    torch.manual_seed(1234)
    inner = SpkMixer(seq_len=96, data_patch_size=16, embed_dim=32, num_classes=1, depths=K, mlp_ratios=0.5, mixer='lif', mixer_layers=M,
                     T_h=4, neo_tau=2.0, bias=False, tau=2.0, mixer_comb=comb, connect_f=cf)
    model = Forecaster(inner, 32, 24, bias=False)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(4, 96, 3, generator=g)
    model.train()
    with torch.no_grad():
        ytr = model(x)
    model.eval()
    with torch.no_grad():
        yev = model(x)
    res[f'{comb}_{cf}_M{M}_K{K}'] = dict(state={k: v.clone() for k, v in model.state_dict().items()}, ytr=ytr, yev=yev)
torch.save(res, out)
print('saved', out, len(res))
