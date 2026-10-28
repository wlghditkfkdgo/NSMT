"""Stage 7 (MODI_PROJECT_LOG 2026-10-19): the cross-attention module replaced by the neocortex network.

Patch tokens, T = N alignment everywhere: the 11 patches (32 samples, stride 16, end padding -- as the ECG model) are the
11 spiking time steps of every neuron; there is no repetition axis and no second (hippocampus) path.

    beat [B, 1, 187] -> patches [N, B, 1, P] (token axis first)
    embedding: Linear(P -> D) -> BN -> spikingjelly LIF over the token axis           (fixed across conditions)
    L blocks:  x = x * (1 - mixer(x))        mixer = the neocortex network: `mixer_layers` NeoLayers (Linear -> BN ->
                                              neuron over the token axis); bidirectional = a second stack over the
                                              reversed token axis, flipped back and fused by OR (as stage 6);
                                              mixer 'none' = no cross-token mixing (control)
               x = x * (1 - MLP(x))          two SpikLinearLayers with spikingjelly LIF over the token axis (fixed)
    head: Linear(D -> classes) on the token mean (the ECG model's readout); loss = CAL-weighted CE only (no auxiliary loss)

Cross-token information can only travel through the mixer's neuron state (and the final mean).

MODI 2026-10-21 additions (defaults unchanged): mixer_comb 'xattn' / 'xattn_ch' replaces the OR of the two directions by
token-axis cross-attention (bix.TokenXAttn, q = forward stream, k/v = reverse stream); mixers 'pop_lif' (population with
alpha 1), 'pop_rec' (population f-LIF with a gated recurrent soma current) and 'popg_lif' / 'popg_flif' (soma-less
population: every neuron layer of the model split into tau groups 4/8/16/32 that fire themselves; BN weights calibrated
by layers.tau_gain).
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from layers import SpikLinearLayer, build_neo_neuron, GroupNeuron, group_gamma, sj_lif, apply_tau_calibration
from model import Embedding, NeoLayer
from token_order import TokenShuffle, permute_tokens   # MODI 2026-10-28 (shuffle-retrain)
from bix import TokenXAttn, StreamSpikeLinear

__all__ = ['NeoMixer']

MIXERS = ('none', 'lif', 'flif', 'pop_flif', 'lif_rec', 'sj_lif',   # MODI 2026-10-21 (1-A): lif_rec = NeoLayer(recurrent=True), gated W_rec;
          'pop_lif', 'pop_rec', 'popg_lif', 'popg_flif')           # sj_lif = the same spikingjelly single-step LIF loop without recurrence (control)
COMBS = ('or', 'xattn', 'xattn_ch', 'lin',   # lin (MODI 2026-10-22): m = neuron(BN(W [F; R])), learned per-token fusion, no attention
         'linmem', 'linmx', 'xattnmem',     # MODI 2026-10-23 membrane fusion: the last mixer layers' charges H - theta instead of
         'linxor')                          # MODI 2026-10-24: spike control of linmx, neuron(BN(W [F; R; |F - R|])) (|F - R| = F xor R)
                                            # their spikes: linmem neuron(BN(W [H_f; H_r])), linmx neuron(BN(W [H_f; H_r; |H_f - H_r|])),
                                            # xattnmem = BiX with q / k from H_f / H_r (real-valued) and v from the reverse spikes
MEM_COMBS = ('linmem', 'linmx', 'xattnmem')


class MixerStack(nn.Module):
    """`n` NeoLayers over the token axis with the chosen spikeDE neuron (built by layers.build_neo_neuron)."""
    def __init__(self, dim, n, neuron_fn, bias=False, recurrent=None, tau=2.0):
        super().__init__()
        if recurrent is not None:                                      # spikingjelly single-step LIF loop (+ tanh(g) W_rec s_{t-1}, g init 0, if recurrent)
            self.Mem = nn.ModuleList([NeoLayer(dim, tau=tau, bias=bias, recurrent=bool(recurrent)) for _ in range(n)])
        else:
            self.Mem = nn.ModuleList([NeoLayer(dim, bias=bias, neuron=neuron_fn()) for _ in range(n)])

    def forward(self, x):
        for l in self.Mem:
            x = l(x)
        return x

    def forward_charge(self, x):
        """MODI 2026-10-23: the last layer's (spikes, H - theta); the earlier layers run as in forward()."""
        for l in self.Mem[:-1]:
            x = l(x)
        return self.Mem[-1].forward_charge(x)


class Block(nn.Module):
    def __init__(self, dim, mixer, mixer_bidir, mixer_layers, mlp_ratio, neo_kw, T, bias=False, tau=2.0, comb='or', heads=8):
        super().__init__()
        self.mixer_kind, self.bidir, self.comb = mixer, bool(mixer_bidir), comb
        if comb not in COMBS:
            raise ValueError(f'mixer_comb must be one of {COMBS}')
        if comb != 'or' and not (self.bidir and mixer != 'none'):
            raise ValueError('cross-attention between the two streams needs a bidirectional mixer')
        taus = tuple(neo_kw.get('pop_taus', (4., 8., 16., 32.)))
        self.grouped = mixer.startswith('popg_')                        # soma-less population: every neuron layer in tau groups
        if mixer != 'none':
            rec = {'lif_rec': True, 'sj_lif': False}.get(mixer)          # None -> spikeDE neuron from build_neo_neuron
            if self.grouped:
                fn = lambda: GroupNeuron(dim, taus, mixer.split('_', 1)[1], T=T, alpha=neo_kw.get('alpha', .5))
            else:
                fn = None if rec is not None else (lambda: build_neo_neuron(mixer, dim, T, 'ND', **neo_kw))
            self.mixer = MixerStack(dim, mixer_layers, fn, bias, recurrent=rec, tau=tau)
            if self.bidir:
                self.mixer_rev = MixerStack(dim, mixer_layers, fn, bias, recurrent=rec, tau=tau)
            if self.grouped:
                for st in [self.mixer] + ([self.mixer_rev] if self.bidir else []):
                    for l in st.Mem:
                        with torch.no_grad():
                            l.bn.weight.copy_(group_gamma(dim, taus, mixer.split('_', 1)[1]))   # spikeDE lif / flif family
        spikede = ('lif', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
        if comb in MEM_COMBS and (mixer not in spikede or (self.grouped and comb == 'xattnmem')):
            raise NotImplementedError('membrane fusion is defined for the spikeDE mixers (xattnmem not for the tau groups)')
        if comb in ('lin', 'linmem', 'linmx', 'linxor'):                 # MODI 2026-10-22: Linear(2D -> D) on [F; R] -> BN -> spikingjelly LIF
            if self.grouped and comb not in ('lin', 'linmem', 'linmx', 'linxor'):
                raise NotImplementedError(comb)
            width = 3 * dim if comb in ('linmx', 'linxor') else 2 * dim   # MODI 2026-10-23: [H_f; H_r] / [H_f; H_r; |H_f - H_r|]
            if self.grouped:                                             # MODI 2026-10-24: tau groups (spikingjelly LIF per group, as fc1 / fc2)
                self.fuse = StreamSpikeLinear(width, dim, 1, GroupNeuron(dim, taus, 'sj', axis=-1), bias=bias, bn_layout='flat',
                                              gamma=group_gamma(dim, taus))
            else:
                self.fuse = StreamSpikeLinear(width, dim, 1, sj_lif(tau), bias=bias, bn_layout='flat')
        elif comb != 'or':                                               # MODI 2026-10-21: BiX, q = forward stream, k/v = reverse stream
            afn = (lambda: GroupNeuron(dim, taus, 'sj', axis=-1)) if self.grouped else (lambda: sj_lif(tau))
            self.xattn = TokenXAttn(dim, T, K=1, heads=heads, form=('channel' if comb == 'xattn_ch' else 'head'), neuron_fn=afn, bias=bias,
                                    gamma=(group_gamma(dim, taus) if self.grouped else None),
                                    qk_from=('charge' if comb == 'xattnmem' else 'spikes'))
        hidden = int(dim * mlp_ratio)
        if self.grouped:                                                 # MLP neurons in tau groups too (spikingjelly LIF per group)
            self.fc1 = SpikLinearLayer(dim, hidden, tau=tau, lif_bias=bias, neuron=GroupNeuron(hidden, taus, 'sj', axis=-1))
            self.fc2 = SpikLinearLayer(hidden, dim, tau=tau, lif_bias=bias, neuron=GroupNeuron(dim, taus, 'sj', axis=-1))
            with torch.no_grad():
                self.fc1.bn.weight.copy_(group_gamma(hidden, taus)); self.fc2.bn.weight.copy_(group_gamma(dim, taus))
        else:
            self.fc1 = SpikLinearLayer(dim, hidden, tau=tau, lif_bias=bias)       # spikingjelly LIF over the token axis
            self.fc2 = SpikLinearLayer(hidden, dim, tau=tau, lif_bias=bias)

    def mix(self, x):
        if self.mixer_kind == 'none':
            return None
        if self.comb in MEM_COMBS:                                       # MODI 2026-10-23: membrane fusion
            f, hf = self.mixer.forward_charge(x)
            r, hr = self.mixer_rev.forward_charge(x.flip(0))
            r, hr = r.flip(0), hr.flip(0)
            if self.comb == 'linmem':
                return self.fuse(torch.cat([hf, hr], dim=-1))
            if self.comb == 'linmx':
                return self.fuse(torch.cat([hf, hr, (hf - hr).abs()], dim=-1))
            return self.xattn(f, r, hf, hr)
        m = self.mixer(x)
        if self.bidir:
            r = self.mixer_rev(x.flip(0)).flip(0)
            if self.comb == 'or':
                m = torch.maximum(m, r)
            elif self.comb == 'lin':
                m = self.fuse(torch.cat([m, r], dim=-1))
            elif self.comb == 'linxor':                                  # MODI 2026-10-24: |F - R| = F xor R for 0/1 spikes
                m = self.fuse(torch.cat([m, r, (m - r).abs()], dim=-1))
            else:
                m = self.xattn(m, r)
        return m

    def forward(self, x, mix_fn=None):
        m = self.mix(x)
        if mix_fn is not None:
            m = mix_fn(m)
        if m is not None:
            x = x * (1. - m)
        x = x * (1. - self.fc2(self.fc1(x)))
        return x


class NeoMixer(nn.Module):
    def __init__(self, seq_len=187, data_patch_size=32, embed_dim=384, num_classes=5, depths=1, mlp_ratios=1.0,
                 mixer='pop_flif', mixer_bidir=True, mixer_layers=2, neo_tau=2.0, neo_alpha=0.5, fde_method='pred',
                 fde_step=1.0, fde_memory=0, pop_taus=(4., 8., 16., 32.), bias=False, tau=2.0, train_mode='training',
                 in_channels=1, pop_soma_init=None, pos_emb=False, mixer_comb='or', heads=8, tau_calib=True, token_shuffle=False,
                 shuffle_seed=0, **kwargs):
        super().__init__()
        if mixer not in MIXERS:
            raise ValueError(f'mixer must be one of {MIXERS}')
        if mixer == 'none' and mixer_bidir:
            raise ValueError("mixer 'none' has no direction")
        self.train_mode = train_mode
        self.patch_size, self.stride = data_patch_size, data_patch_size // 2
        self.num_patches = int((seq_len - data_patch_size) / self.stride + 1) + 1
        self.T = self.num_patches                                          # T = N
        self.token_shuffle = TokenShuffle(self.num_patches, shuffle_seed) if token_shuffle else None   # MODI 2026-10-28
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.mixer_kind, self.mixer_bidir, self.mixer_layers, self.depths = mixer, bool(mixer_bidir), int(mixer_layers), int(depths)
        self.no_neo = self.hippo_only = False                              # for the shared evaluators
        neo_kw = dict(tau=neo_tau, alpha=neo_alpha, method=fde_method, step=fde_step, memory=fde_memory, pop_taus=tuple(pop_taus),
                      pop_soma_init=pop_soma_init)
        self.in_channels = int(in_channels or 1)                           # stage 9: a token = P steps x C channels
        self.mixer_comb = mixer_comb
        grouped = mixer.startswith('popg_')
        self.encoding = Embedding(num_patches=self.num_patches, patch_size=data_patch_size * self.in_channels, embed_dim=embed_dim, stride=self.stride,
                                  pe=False, bias=bias, tau=tau, pos_emb=pos_emb,   # spikingjelly LIF over the token axis; pos_emb (1-B)
                                  neuron=(GroupNeuron(embed_dim, tuple(pop_taus), 'sj', axis=2) if grouped else None))   # popg: [T, B, D, N'] layout
        if grouped:
            with torch.no_grad():
                self.encoding.emb_bn.weight.copy_(group_gamma(embed_dim, tuple(pop_taus)))
        self.blocks = nn.ModuleList([Block(embed_dim, mixer, mixer_bidir, mixer_layers, mlp_ratios, neo_kw, self.T, bias, tau, comb=mixer_comb, heads=heads)
                                     for _ in range(self.depths)])
        self.head = nn.Linear(embed_dim, num_classes, bias=bias)
        self.apply(self._init_weights)
        self.tau_calibrated = False
        if grouped and tau_calib:                                       # MODI 2026-10-22: layer-wise rate-matched BN gains
            self.tau_calibrated = apply_tau_calibration(self, 'neoMixer', mixer, mixer_comb, pop_taus)

    def _init_weights(self, m):
        if getattr(m, '_skip_global_init', False):                     # NeoLayer W_rec keeps its small init (as model.py)
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def tokens(self, x):
        if x.dim() != 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        if self.in_channels == 1:                                          # unchanged single-channel path (stage 7)
            xp = self.padding_patch_layer(x[:, 0]).unfold(-1, self.patch_size, self.stride)   # [B, N, P]
            tok = xp.transpose(0, 1).unsqueeze(2).contiguous()                               # [N, B, 1, P]
        else:
            C = x.shape[1]                                                 # stage 9: [B, C, L] -> [N, B, 1, C*P]
            p = self.padding_patch_layer(x).unfold(-1, self.patch_size, self.stride)          # [B, C, N, P]
            p = p.permute(0, 2, 1, 3).reshape(B, p.shape[2], C * self.patch_size)
            tok = p.transpose(0, 1).unsqueeze(2).contiguous()
        if self.token_shuffle is not None:                                 # MODI 2026-10-28 (shuffle-retrain)
            tok = permute_tokens(tok, 0, self.token_shuffle.perms(B, tok.device, self.training))
        return tok

    def forward_parts(self, x, tok_fn=None, mix_fns=None):
        """Returns logits, embedding spikes [N, B, 1, D], first block's mixer output (None for 'none'), head input [B, D],
        and the last block's token spikes [N, B, 1, D]. tok_fn edits the token tensor; mix_fns[i] edits block i's mixer output."""
        tok = self.tokens(x)
        if tok_fn is not None:
            tok = tok_fn(tok)
        e = self.encoding(tok)                                             # [N, B, 1, D]
        h, m0 = e, None
        for i, blk in enumerate(self.blocks):
            fn = mix_fns[i] if mix_fns is not None else None
            if i == 0:
                m0 = blk.mix(h)
                if fn is not None:
                    m0 = fn(m0)
                if m0 is not None:
                    h = h * (1. - m0)
                h = h * (1. - blk.fc2(blk.fc1(h)))
            else:
                h = blk(h, fn)
        feat = h.mean(2).mean(0)                                           # token mean -> [B, D]
        return self.head(feat), e, m0, feat, h

    def forward(self, x):
        z, _, _, _, h = self.forward_parts(x)
        if self.train_mode == 'training':
            return z                                                       # plain logits: train.py uses CE only
        return z, h.detach()                                               # test.py's optional t-SNE feature
