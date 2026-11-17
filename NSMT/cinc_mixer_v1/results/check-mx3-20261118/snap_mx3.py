"""Snapshot of the EARLIER paths (no new option) for the bitwise-identity check of the mixer-variant edit (2026-11-17).
usage: python snap_mx3.py OUT CODE_DIR     (run on the unmodified copy and on the edited copy; compare with check_mx3.py --compare)"""
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
torch.save(res, out)
print(f'{len(res)} entries -> {out}')
