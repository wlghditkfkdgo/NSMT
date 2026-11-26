"""Helper of check_s2.py s0 (2026-11-27 stage 2): from a code directory, build the forecaster for earlier conditions and the stage-1
arms at a fixed seed; save parameters / buffers and train / eval outputs on a fixed input.  usage: python -B s0_build.py <code_dir> <out.pt>"""
import os
import sys
code, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, code)
os.chdir(code)
import torch                                   # noqa: E402
from forecast_wrap import Forecaster           # noqa: E402
from spk_mixer import SpkMixer                 # noqa: E402
CONDS = [('muxlin', 'IAND', False), ('muxlin', 'ADD', False), ('muxlin', 'ADD', True), ('or', 'IAND', False), ('sdlin', 'OR', False)]
CONDS += [(c, 'ADD', s) for c in ('sxd', 'sxdpc', 'sxp', 'sxppc') for s in (False, True)]
res = {}
for comb, cf, share in CONDS:
    torch.manual_seed(4321)
    inner = SpkMixer(seq_len=96, data_patch_size=16, embed_dim=32, num_classes=1, depths=1, mlp_ratios=0.5, mixer='lif', mixer_layers=2,
                     T_h=4, neo_tau=2.0, bias=False, tau=2.0, mixer_comb=comb, connect_f=cf, mixer_share=share)
    model = Forecaster(inner, 32, 24, bias=False)
    x = torch.randn(4, 96, 3, generator=torch.Generator().manual_seed(7))
    model.train()
    with torch.no_grad():
        ytr = model(x)
    model.eval()
    with torch.no_grad():
        yev = model(x)
    res[f'{comb}_{cf}_{share}'] = dict(state={k: v.clone() for k, v in model.state_dict().items()}, ytr=ytr, yev=yev)
torch.save(res, out)
print('saved', len(res))
