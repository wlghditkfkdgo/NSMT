"""SpkMixer (2026-10-29, PROJECT_LOG, exp/f-lif-pop-v3): the Spikformer-style CinC model (model.myModel with hippo_ssa: T_h = 4
exact repeats of the patch tokens, IAND residuals, token-mean head) with ONLY its self-attention replaced by the bidirectional
token-axis Neo-mixer, and the chosen neuron class in EVERY neuron layer (user, 10-28: "뉴런 종류는 아키텍처 전체에 걸쳐서
일치해야" -- the earlier NeoMixer put it in the mixer layers only; and the layout 임베딩 [4, B, N, D] -> mixer [N, B, 4, D]
-> MLP [4, B, N, D]).

    x [B, 1, L] -> tokens [T_h, B, N, P]            exact repeats, as myModel._tokens_mc (check s6: bitwise the same tokens)
    embedding   Linear(P -> D) -> BN -> neuron over the repetition axis (T_h steps)          [T_h, B, N, D]
    block       x_m = x.permute(2, 1, 0, 3)                                                   [N, B, T_h, D]
                  the T_h repeats are T_h independent token sequences through the same mixer weights
                F, H_f = mixer(x_m)        `mixer_layers` NeoLayers (Linear -> BN -> neuron over the TOKEN axis, N steps)
                R, H_r = mixer_rev(flip)   its own weights, flipped back
                m = fuse(.)                                                                    [N, B, T_h, D]
                  or     : F v R                                                                  (no parameters)
                  sdor   : F v R v s_d,   s_d = neuron_d(BN_d(|H_f - H_r|))                       (user's intent, OR form)
                  sdlin  : neuron(BN(W [F v R; s_d])),  W: 2D -> D, 0/1 input (AC)                (user's intent, learned form)
                  linmx  : neuron(BN(W [H_f - th; H_r - th; |H_f - H_r|])), W: 3D -> D, real input  (reference (3), MAC)
                x = connect(x, m.permute(2, 1, 0, 3))    IAND x (1 - m) | OR max(x, m) | ADD x + m   (same for the MLP)
                x = connect(x, fc2(fc1(x)))              SpikLinearLayer x 2, neuron over the repetition axis
    head        Linear(D -> classes) on the token mean, averaged over T_h (myModel._testing)

Neurons: lif / flif / pop_lif / pop_flif = layers.build_neo_neuron (spikeDE classes, NeoNeuron adapter); popg_lif / popg_flif =
layers.GroupNeuron with the spikeDE groups in every layer (the NeoMixer popg used spikingjelly groups outside the mixer).
BN gains of the pop / popg layers come from calib/gains_spkMixer_<mixer>_<comb>_<connect>.json (calibrate_spk.py), applied by
apply_gain_table (fail closed when the table is missing). Loss, optimiser, data and evaluation are the shared ones
(train_rhythm.py / rhythm_eval.py); the model returns plain logits in training.
"""
import json
import os

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from layers import SpikLinearLayer, build_neo_neuron, GroupNeuron, group_gamma
from model import Embedding
from neo_mixer import MixerStack

__all__ = ['SpkMixer', 'apply_gain_table', 'COMBS', 'CONNECTS', 'NEURONS']

NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
COMBS = ('or', 'sdor', 'sdlin', 'linmx')
CONNECTS = ('IAND', 'OR', 'ADD')
CALIBRATED = ('pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')


def gain_table_path(mixer, comb, connect):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib', f'gains_spkMixer_{mixer}_{comb}_{connect}.json')


def apply_gain_table(model, path):
    """Set BN weights from {bn module name: [gain per slice]} (one gain = the whole layer, K gains = K equal slices)."""
    table = json.load(open(path))['gains']                              # FileNotFoundError = fail closed
    mods = dict(model.named_modules())
    with torch.no_grad():
        for bn_name, gains in table.items():
            bn = mods[bn_name]
            if bn.weight.numel() % len(gains):
                raise ValueError(f'{path}: {bn_name} has {bn.weight.numel()} channels, {len(gains)} gains')
            G = bn.weight.numel() // len(gains)
            for k, g in enumerate(gains):
                bn.weight[k * G:(k + 1) * G] = float(g)
    return True


class FuseLinear(nn.Module):
    """Linear(din -> dout) -> BN(dout) over every (token, batch, repeat) row -> neuron over the token axis, on [N, B, K, din]."""
    def __init__(self, din, dout, neuron, bias=False):
        super().__init__()
        self.linear = nn.Linear(din, dout, bias=bias)
        self.bn = nn.BatchNorm1d(dout)
        self.neuron = neuron
        self.dout = dout

    def forward(self, x):
        N, B, K, _ = x.shape
        y = self.bn(self.linear(x).reshape(-1, self.dout)).reshape(N, B, K, self.dout)
        return self.neuron(y.contiguous())


class SpkBlock(nn.Module):
    def __init__(self, dim, mixer, mixer_layers, mlp_ratio, make_neuron, N, T_h, comb, connect, bias=False):
        super().__init__()
        if comb not in COMBS:
            raise ValueError(f'mixer_comb must be one of {COMBS}')
        if connect not in CONNECTS:
            raise ValueError(f'connect_f must be one of {CONNECTS}')
        self.comb, self.connect_f, self.grouped = comb, connect, mixer.startswith('popg_')
        fn = lambda: make_neuron(dim, N, 'ND')                            # mixer / fusion neurons: token axis, N steps
        self.mixer = MixerStack(dim, mixer_layers, fn, bias)
        self.mixer_rev = MixerStack(dim, mixer_layers, fn, bias)
        if comb in ('sdor', 'sdlin'):
            self.sd_bn = nn.BatchNorm1d(dim)
            self.sd_neuron = make_neuron(dim, N, 'ND')
        if comb in ('sdlin', 'linmx'):
            # built and initialised inside a forked RNG (continuing the current state, restored afterwards) and skipped by the
            # global init: at the same seed every other weight (embedding, mixer, MLP, head) is identical across the four
            # fusions and the three residuals (check s3), so the fusion comparisons are paired on the same initial model
            with torch.random.fork_rng(devices=[]):
                self.fuse = FuseLinear((2 if comb == 'sdlin' else 3) * dim, dim, make_neuron(dim, N, 'ND'), bias)
                trunc_normal_(self.fuse.linear.weight, std=.02)
                if self.fuse.linear.bias is not None:
                    nn.init.constant_(self.fuse.linear.bias, 0)
            self.fuse.linear._skip_global_init = True
        hidden = int(dim * mlp_ratio)
        self.fc1 = SpikLinearLayer(dim, hidden, lif_bias=bias, neuron=make_neuron(hidden, T_h, 'ND'))   # repetition axis, T_h steps
        self.fc2 = SpikLinearLayer(hidden, dim, lif_bias=bias, neuron=make_neuron(dim, T_h, 'ND'))

    def connect(self, x, m):
        if self.connect_f == 'IAND':
            return x * (1. - m)
        if self.connect_f == 'OR':
            return torch.maximum(x, m)
        return x + m

    def mix(self, x):
        """x [T_h, B, N, D] -> m [T_h, B, N, D]."""
        xm = x.permute(2, 1, 0, 3).contiguous()                             # [N, B, T_h, D]
        if self.comb == 'or':
            f = self.mixer(xm)
            r = self.mixer_rev(xm.flip(0)).flip(0)
            m = torch.maximum(f, r)
        else:
            f, hf = self.mixer.forward_charge(xm)
            r, hr = self.mixer_rev.forward_charge(xm.flip(0))
            r, hr = r.flip(0), hr.flip(0)
            if self.comb == 'linmx':
                m = self.fuse(torch.cat([hf, hr, (hf - hr).abs()], dim=-1))
            else:
                d = (hf - hr).abs()
                N, B, K, D = d.shape
                sd = self.sd_neuron(self.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
                fr = torch.maximum(f, r)
                m = torch.maximum(fr, sd) if self.comb == 'sdor' else self.fuse(torch.cat([fr, sd], dim=-1))
        return m.permute(2, 1, 0, 3).contiguous()                           # [T_h, B, N, D]

    def forward(self, x, mix_fn=None):
        m = self.mix(x)
        if mix_fn is not None:
            m = mix_fn(m)
        x = self.connect(x, m)
        x = self.connect(x, self.fc2(self.fc1(x)))
        return x


class SpkMixer(nn.Module):
    def __init__(self, seq_len=3750, data_patch_size=64, embed_dim=384, num_classes=4, depths=1, mlp_ratios=1.0, mixer='lif',
                 mixer_layers=2, T_h=4, neo_tau=2.0, neo_alpha=0.5, fde_method='pred', fde_step=1.0, fde_memory=0,
                 pop_taus=(4., 8., 16., 32.), bias=False, tau=2.0, train_mode='training', in_channels=1, pop_soma_init=None,
                 mixer_comb='or', connect_f='IAND', tau_calib=True, **kwargs):
        super().__init__()
        if mixer not in NEURONS:
            raise ValueError(f'mixer must be one of {NEURONS}')
        if int(T_h) < 1:
            raise ValueError('T_h >= 1')
        self.train_mode = train_mode
        self.patch_size, self.stride = data_patch_size, data_patch_size // 2
        self.num_patches = int((seq_len - data_patch_size) / self.stride + 1) + 1
        self.T_h = self.T = int(T_h)                                          # repetition axis (embedding / MLP); the mixer runs N
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.mixer_kind, self.mixer_comb, self.connect_f = mixer, mixer_comb, connect_f
        self.mixer_layers, self.depths = int(mixer_layers), int(depths)
        self.no_neo = self.hippo_only = False                                 # for the shared evaluators
        self.in_channels = int(in_channels or 1)
        taus = tuple(float(t) for t in pop_taus)
        neo_kw = dict(tau=neo_tau, alpha=neo_alpha, method=fde_method, step=fde_step, memory=fde_memory, pop_taus=taus,
                      pop_soma_init=pop_soma_init)
        grouped = mixer.startswith('popg_')
        fam = mixer.split('_', 1)[1] if grouped else None

        def make_neuron(size, steps, layout):
            if grouped:
                return GroupNeuron(size, taus, fam, T=int(steps), alpha=neo_alpha, axis=(2 if layout == 'DN' else -1))
            return build_neo_neuron(mixer, size, int(steps), layout, **neo_kw)

        N, D = self.num_patches, embed_dim
        self.encoding = Embedding(num_patches=N, patch_size=data_patch_size * self.in_channels, embed_dim=D, stride=self.stride,
                                  pe=False, bias=bias, tau=tau, pos_emb=False, neuron=make_neuron(D, self.T_h, 'DN'))
        self.blocks = nn.ModuleList([SpkBlock(D, mixer, self.mixer_layers, mlp_ratios, make_neuron, N, self.T_h, mixer_comb, connect_f, bias)
                                     for _ in range(self.depths)])
        self.head = nn.Linear(D, num_classes, bias=bias)
        self.apply(self._init_weights)
        if grouped:                                                           # family gains per group first (tau_gain table) ...
            with torch.no_grad():
                for name, mod in self.named_modules():
                    if isinstance(mod, nn.BatchNorm1d):
                        mod.weight.copy_(group_gamma(mod.weight.numel(), taus, fam))
        self.tau_calibrated = False
        if mixer in CALIBRATED and tau_calib:                                 # ... then the layer-wise rate-matched table (fail closed)
            self.tau_calibrated = apply_gain_table(self, gain_table_path(mixer, mixer_comb, connect_f))

    def _init_weights(self, m):
        if getattr(m, '_skip_global_init', False):                             # the fusion Linear (initialised in its fork)
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def tokens(self, x):
        """[B, C, L] -> [T_h, B, N, C*P] exact repeats (myModel._tokens_mc)."""
        if x.dim() != 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        xp = self.padding_patch_layer(x)
        p = xp.unfold(dimension=-1, size=self.patch_size, step=self.stride)   # [B, C, N, P]
        tok = p.permute(0, 2, 1, 3).reshape(B, p.shape[2], -1)                  # [B, N, C*P]
        if tok.shape[1] != self.num_patches:
            raise RuntimeError(f'{tok.shape[1]} tokens but the model was built for {self.num_patches}')
        return tok.unsqueeze(0).expand(self.T_h, -1, -1, -1).contiguous()

    def forward_parts(self, x, tok_fn=None, mix_fns=None):
        """Returns logits, embedding spikes [T_h, B, N, D], first block's mixer output [T_h, B, N, D], head input [B, D] and the
        last block's output [T_h, B, N, D]. tok_fn edits the token tensor; mix_fns[i] edits block i's mixer output."""
        tok = self.tokens(x)
        if tok_fn is not None:
            tok = tok_fn(tok)
        e = self.encoding(tok)                                                # [T_h, B, N, D]
        h, m0 = e, None
        for i, blk in enumerate(self.blocks):
            fn = mix_fns[i] if mix_fns is not None else None
            if i == 0:
                m0 = blk.mix(h)
                if fn is not None:
                    m0 = fn(m0)
                h = blk.connect(h, m0)
                h = blk.connect(h, blk.fc2(blk.fc1(h)))
            else:
                h = blk(h, fn)
        feat = h.mean(2)                                                      # token mean [T_h, B, D]
        z = self.head(feat).mean(0)                                           # Spikformer-style readout
        return z, e, m0, feat.mean(0), h

    def forward(self, x):
        z, _, _, _, h = self.forward_parts(x)
        if self.train_mode == 'training':
            return z
        return z, h.detach()
