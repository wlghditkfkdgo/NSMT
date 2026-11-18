"""Snapshot of the EARLIER paths (no new option) for the bitwise-identity check of the neuron implementation / placement edit (2026-11-18,
PROJECT_LOG, exp/f-lif-pop-v3): the SpkMixer paths of snap_mx3.py, the 11-17 mixer variants (pooling, shortcut, reset), the population LIF
SpkMixer with its STFT-test gain table, and the Spikformer-style model (myModel --hippo_ssa, spikingjelly LIF) as built by LOAD_MODEL for
MIT-BIH and inside the ETT forecaster.
usage: python snap_pn.py OUT CODE_DIR     (run on the unmodified copy and on the edited copy; compare with check_pn.py --compare)"""

import sys
import torch
out, code = sys.argv[1], sys.argv[2]
sys.path.insert(0, code)
from config import set_random_seed
from spk_mixer import SpkMixer
from neo_mixer import NeoMixer
from neurons import LIFNeuron, FLIFNeuron, PopulationFLIF
from layers import spikes_and_charge
res = {}
cfgs = {
    'ett_lif': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='lif', mixer_comb='muxlin'),
    'ett_lif_L2': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=1.0, mixer_layers=2, connect_f='IAND', mixer='lif', mixer_comb='muxlin'),
    'ett_lif_336': dict(seq_len=336, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='lif', mixer_comb='muxlin'),
    'ett_lif_token': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='lif', mixer_comb='muxlin', taxis='token'),
    'mit_lif': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='ADD', mixer='lif', mixer_comb='muxlin'),
    'mit_lif_group': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='ADD', mixer='lif', mixer_comb='muxlin', band='group'),
    'mit_lif_or': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='ADD', mixer='lif', mixer_comb='or'),
    'mit_flif': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='OR', mixer='flif', mixer_comb='sdlin'),
    'mit_ms': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='MS', mixer='lif', mixer_comb='muxlin'),
    'cinc_pop_lif': dict(seq_len=1500, data_patch_size=64, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='IAND', mixer='pop_lif',
                         mixer_comb='sdlin', pop_soma_init=0.5, num_classes=4),
    'cinc_pop_flif': dict(seq_len=1500, data_patch_size=64, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='OR', mixer='pop_flif',
                          mixer_comb='sdor', pop_soma_init=1.0, num_classes=4),
    'cinc_popg_lif': dict(seq_len=1500, data_patch_size=64, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='IAND', mixer='popg_lif',
                          mixer_comb='or', num_classes=4),
    'ett_mx_mean': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='lif', mixer_comb='muxlin', mix_pool='mean'),
    'mit_mx_ms': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='ADD', mixer='lif', mixer_comb='muxlin', mixer_ms=True),
    'ett_mx_zero': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='lif', mixer_comb='muxlin', mixer_reset='zero'),
    'ett_pop_lif_stft': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='pop_lif', mixer_comb='muxlin',
                             pop_soma_init=0.5, gain_table=code + '/calib/stft/gains_ETTh2_pop_lif_muxlin_IAND_bdnone_txrep.json'),
}
for key, kw in cfgs.items():
    set_random_seed(42)
    kw = dict(dict(num_classes=5, depths=1, T_h=4, bias=False, in_channels=1), **kw)
    m = SpkMixer(**kw).cuda()
    res[f'{key}/rng'] = torch.get_rng_state().clone()
    x = torch.randn(6, 1, kw['seq_len'], generator=torch.Generator().manual_seed(1)).cuda()
    m.train(); m.train_mode = 'training'
    z = m(x); (z ** 2).sum().backward()
    res[f'{key}/train_out'] = z.detach().cpu()
    for n, p in m.named_parameters():
        res[f'{key}/w/{n}'] = p.detach().cpu().clone()
        res[f'{key}/g/{n}'] = None if p.grad is None else p.grad.detach().cpu().clone()
    m.eval(); m.train_mode = 'testing'
    with torch.no_grad():
        z, h = m(x)
    res[f'{key}/eval_out'] = z.cpu(); res[f'{key}/eval_h'] = h.cpu()
    del m
set_random_seed(42)
m = NeoMixer(seq_len=187, data_patch_size=32, embed_dim=384, num_classes=5, depths=1, mlp_ratios=1.0, mixer='pop_lif', mixer_bidir=True,
             mixer_layers=2, pop_soma_init=0.5).cuda()
x = torch.randn(6, 1, 187, generator=torch.Generator().manual_seed(1)).cuda()
m.train(); m.train_mode = 'training'
z = m(x); z = z[0] if isinstance(z, tuple) else z
(z ** 2).sum().backward()
res['neomixer/train_out'] = z.detach().cpu()
for n, p in m.named_parameters():
    res[f'neomixer/g/{n}'] = None if p.grad is None else p.grad.detach().cpu().clone()
for name, n in (('lif', LIFNeuron(16, tau=2., max_length=20)), ('flif', FLIFNeuron(16, tau=2., alpha=.5, max_length=20)),
                ('pop', PopulationFLIF(16, tau=(4., 8., 16., 32.), max_length=20, alpha=1.))):
    x = (1.5 * torch.randn(20, 3, 16, generator=torch.Generator().manual_seed(2))).requires_grad_(True)
    s = n(x); s.sum().backward()
    res[f'neuron/{name}/spikes'] = s.detach(); res[f'neuron/{name}/grad'] = x.grad.clone()
    s2, h = spikes_and_charge(n, x.detach())
    res[f'neuron/{name}/charge_spikes'] = s2.detach(); res[f'neuron/{name}/charge'] = h.detach()

# 2026-11-18: the Spikformer-style model (spikingjelly LIF) through LOAD_MODEL, MIT-BIH classifier and ETT forecaster
from spikingjelly.clock_driven import functional
from config import Config, parse_arguments
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
TASKS = {'mit': (['--seq_len', '187', '--patch_size', '32', '-emb', '384', '--mlp_ratios', '1', '--time_layers', '2'], 5, 187),
         'ett': (['--seq_len', '96', '--patch_size', '8', '-emb', '64', '--mlp_ratios', '0.5', '--time_layers', '1'], 1, 96)}
for task, (extra, ncls, L) in TASKS.items():
    argv = ['x', '--data', 'X', '--seed', '42', '-s', '--no-bias', '--layers', '1', '--tau', '2.0', '-nh', '8', '--log_dir', '/tmp/snap_pn',
            '--model', 'myModel', '--hippo_ssa', '--hippo_T', '4'] + extra
    old = sys.argv; sys.argv = argv
    try:
        args = Config(); args.set_args(parse_arguments())
    finally:
        sys.argv = old
    args.num_classes, args.num_channels, args.device = ncls, 1, torch.device('cuda')
    set_random_seed(42)
    inner = LOAD_MODEL['myModel'](args, True)
    m = (Forecaster(inner, args.embed_dim, 96, bias=args.bias) if task == 'ett' else inner).cuda()
    res[f'sf_{task}/rng'] = torch.get_rng_state().clone()
    x = (torch.randn(6, 96, 7, generator=torch.Generator().manual_seed(1)) if task == 'ett' else torch.randn(6, 1, L, generator=torch.Generator().manual_seed(1))).cuda()
    m.train(); inner.train_mode = 'training'
    z = m(x); z = z[0] if isinstance(z, tuple) else z
    (z.float() ** 2).sum().backward(); functional.reset_net(m)
    res[f'sf_{task}/train_out'] = z.detach().cpu()
    for n, p in m.named_parameters():
        res[f'sf_{task}/w/{n}'] = p.detach().cpu().clone()
        res[f'sf_{task}/g/{n}'] = None if p.grad is None else p.grad.detach().cpu().clone()
    del m, inner
torch.save(res, out)
print(f'{len(res)} entries -> {out}')
