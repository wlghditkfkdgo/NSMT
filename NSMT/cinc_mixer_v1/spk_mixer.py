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
                  muxor  : s_d ? (F v R) : (F ^ R)  = majority(F, R, s_d)       (10-30 MUX (i); same modules as sdor)
                  muxlin : s_d ? L : (F ^ R),  L = the sdlin output               (10-30 MUX (ii); same modules as sdlin)
                           training arithmetic s * a + (1 - s) * min(F, R), a = max(F, R) or L (min / max split ties 0.5)
                x = connect(x, m.permute(2, 1, 0, 3))    IAND x (1 - m) | OR max(x, m) | ADD x + m   (same for the MLP)
                x = connect(x, fc2(fc1(x)))              SpikLinearLayer x 2, neuron over the repetition axis
    head        Linear(D -> classes) on the token mean, averaged over T_h (myModel._testing)

Neurons: lif / flif / pop_lif / pop_flif = layers.build_neo_neuron (spikeDE classes, NeoNeuron adapter); popg_lif / popg_flif =
layers.GroupNeuron with the spikeDE groups in every layer (the NeoMixer popg used spikingjelly groups outside the mixer).
BN gains of the pop / popg layers come from calib/gains_spkMixer_<mixer>_<comb>_<connect>.json (calibrate_spk.py), applied by
apply_gain_table (fail closed when the table is missing). Loss, optimiser, data and evaluation are the shared ones
(train_rhythm.py / rhythm_eval.py); the model returns plain logits in training.

2026-11-16 (STFT bands, user request 11-13 / 11-16; defaults leave every earlier path unchanged):
  band  'none' | 'group' (A: LIF, channel groups read one STFT band each) | 'member' (B: population, one band per member,
        lowest band -> slowest member); spk_band.py. Only the embedding changes; single input channel.
  taxis 'rep'   the layout above: T_h repeats in the embedding / MLP, the token (= STFT frame) axis only in the mixer
        'token' the token axis is the spiking axis of EVERY neuron layer and there are no repeats (T_h = 1): tokens [N, B, 1, P]
                -> embedding neuron over N -> mixer (forward / reversed) over N -> MLP neurons over N; the outputs are
                returned in the usual [1, B, N, D] layout, so heads and evaluators are unchanged.
  gain_table  explicit BN gain table of a population model (calibrate_stft.py); needed for band / taxis / other tasks.

2026-11-17 (mixer variants, user request 11-17; defaults leave every earlier path unchanged; rep layout, IAND / OR / ADD only):
  mix_pool     'none' | 'max' | 'mean': the T_h repeats are pooled BEFORE the mixer (max = OR of the 4 spike trains, 0/1;
               mean = their average, values k/4), the two directional stacks and the fusion run ONCE on the token sequence
               [N, B, 1, D], and the fusion output is copied to the T_h repeats for the residual and the MLP.
  mixer_ms     membrane shortcut INSIDE each directional stack (needs mixer_layers >= 2): layer 1 u = BN_1(W_1 x), s_1 = LIF_1(u);
               layer i > 1 u <- u + BN_i(W_i s_{i-1}), s_i = LIF_i(u). Same modules / parameters as the plain stack; the fusion
               reads the last layer's spikes and H - theta as before.
  mixer_reset  'spikede' (default) | 'subtract' | 'zero': reset of the neurons of the two directional stacks only (neurons.LIFNeuron;
               LIF only); the fusion, embedding and MLP neurons keep the default.

2026-11-18 (neuron placement, user request 11-18; default 'all' leaves every earlier path unchanged):
  neuron_scope 'all' | 'mixer': with 'mixer' only the token-axis neurons -- the two directional stacks and the fusion neurons
               (s_d and the fusion Linear's neuron; user choice 11-18 "stack + fusion") -- are the population neuron (pop_lif /
               pop_flif); the embedding and MLP neurons (repeat axis, T_h steps) are the single spikeDE neuron of the same family
               (lif / flif). The population soma has no random init, so at the same seed every weight equals the 'all' model's.
               Needs an explicit gain table (only the population layers are calibrated); rep layout, no band, no connect MS.
"""
import json
import os

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from layers import SpikLinearLayer, build_neo_neuron, GroupNeuron, group_gamma
from model import Embedding
from neo_mixer import MixerStack
from spk_band import BANDS, GroupBandEmbedding, MemberBandEmbedding, member_order, stft_band_dims, stft_values

__all__ = ['SpkMixer', 'apply_gain_table', 'COMBS', 'CONNECTS', 'NEURONS']

NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
COMBS = ('or', 'sdor', 'sdlin', 'linmx', 'muxor', 'muxlin')                 # muxor / muxlin: 10-30 fusion MUX
CONNECTS = ('IAND', 'OR', 'ADD', 'MS')                                       # MS: 2026-11-08 membrane shortcut (SpkBlock.forward_ms)
POOLS = ('none', 'max', 'mean')                                              # 2026-11-17 mixer variants
RESETS = ('spikede', 'subtract', 'zero')
CALIBRATED = ('pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
TAXES = ('rep', 'token')                                                     # 2026-11-16
SCOPES = ('all', 'mixer')                                                    # 2026-11-18 neuron placement


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
    def __init__(self, dim, mixer, mixer_layers, mlp_ratio, make_neuron, N, T_h, comb, connect, bias=False, token_axis=False,
                 mix_pool='none', mixer_ms=False, mixer_reset='spikede', make_rest=None):
        super().__init__()
        rest = make_neuron if make_rest is None else make_rest               # 2026-11-18: embedding / MLP neuron (neuron_scope)
        self.token_axis = bool(token_axis)                                    # 2026-11-16: x is [N, B, 1, D] (T_h = the token steps here)
        self.mix_pool, self.mixer_ms, self.mixer_reset = mix_pool, bool(mixer_ms), mixer_reset   # 2026-11-17
        if comb not in COMBS:
            raise ValueError(f'mixer_comb must be one of {COMBS}')
        if connect not in CONNECTS:
            raise ValueError(f'connect_f must be one of {CONNECTS}')
        self.comb, self.connect_f, self.grouped = comb, connect, mixer.startswith('popg_')
        fn = lambda: make_neuron(dim, N, 'ND')                            # mixer / fusion neurons: token axis, N steps
        if mixer_reset != 'spikede':                                          # 2026-11-17: reset of the directional stacks only
            fn = lambda: make_neuron(dim, N, 'ND', reset=mixer_reset)
        self.mixer = MixerStack(dim, mixer_layers, fn, bias)
        self.mixer_rev = MixerStack(dim, mixer_layers, fn, bias)
        if comb in ('sdor', 'sdlin', 'muxor', 'muxlin'):
            self.sd_bn = nn.BatchNorm1d(dim)
            self.sd_neuron = make_neuron(dim, N, 'ND')
        if comb in ('sdlin', 'linmx', 'muxlin'):
            # built and initialised inside a forked RNG (continuing the current state, restored afterwards) and skipped by the
            # global init: at the same seed every other weight (embedding, mixer, MLP, head) is identical across the four
            # fusions and the three residuals (check s3), so the fusion comparisons are paired on the same initial model
            with torch.random.fork_rng(devices=[]):
                self.fuse = FuseLinear((3 if comb == 'linmx' else 2) * dim, dim, make_neuron(dim, N, 'ND'), bias)
                trunc_normal_(self.fuse.linear.weight, std=.02)
                if self.fuse.linear.bias is not None:
                    nn.init.constant_(self.fuse.linear.bias, 0)
            self.fuse.linear._skip_global_init = True
        hidden = int(dim * mlp_ratio)
        self.fc1 = SpikLinearLayer(dim, hidden, lif_bias=bias, neuron=rest(hidden, T_h, 'ND'))   # repetition axis, T_h steps
        self.fc2 = SpikLinearLayer(hidden, dim, lif_bias=bias, neuron=rest(dim, T_h, 'ND'))
        if connect == 'MS':                                                   # 2026-11-08 membrane shortcut; same fork as the fusion Linear
            with torch.random.fork_rng(devices=[]):
                self.ms_mix = SpikLinearLayer(dim, dim, lif_bias=bias, neuron=rest(dim, T_h, 'ND'))
                trunc_normal_(self.ms_mix.linear.weight, std=.02)
                if self.ms_mix.linear.bias is not None:
                    nn.init.constant_(self.ms_mix.linear.bias, 0)
            self.ms_mix.linear._skip_global_init = True

    def connect(self, x, m):
        if self.connect_f == 'IAND':
            return x * (1. - m)
        if self.connect_f == 'OR':
            return torch.maximum(x, m)
        if self.connect_f == 'MS':                                            # 2026-11-08: a spike residual would silently be ADD
            raise RuntimeError('connect_f MS runs through forward_ms')
        return x + m

    def forward_ms(self, u, m):
        """2026-11-08 membrane shortcut: real-valued sums stay in u, only spikes reach a Linear; m = the mixer output spikes.
        u1 = u + BN(W m), s1 = LIF(u1), u2 = u1 + BN(fc2(fc1(s1))), output LIF(u2); returns (spikes, u2)."""
        u = u + self.ms_mix.current(m)
        s1 = self.ms_mix.lif(u.contiguous())
        u = u + self.fc2.current(self.fc1(s1))
        return self.fc2.lif(u.contiguous()), u

    def stack(self, st, x, charge=False):
        """2026-11-17: one directional stack; mixer_ms adds each layer's BN current to a real-valued stream u (layer 1 starts it)
        and every layer's neuron fires on u. Without mixer_ms this is st(x) / st.forward_charge(x) (unchanged)."""
        if not self.mixer_ms:
            return st.forward_charge(x) if charge else st(x)
        u, last = None, len(st.Mem) - 1
        for i, l in enumerate(st.Mem):
            T, B, N, D = x.shape
            c = l.bn(l.w_in(x).reshape(-1, D)).reshape(T, B, N, D)           # NeoLayer.forward's current, same modules
            u = c if u is None else u + c
            if i == last:
                return l.lif.forward_charge(u) if charge else l.lif(u)
            x = l.lif(u)

    def mix(self, x):
        """x [T_h, B, N, D] -> m [T_h, B, N, D] (token_axis: x and m are [N, B, 1, D])."""
        xm = x if self.token_axis else x.permute(2, 1, 0, 3).contiguous()   # [N, B, T_h, D]
        reps = xm.shape[2]                                                    # 2026-11-17 (not K: the fusion reuses that name)
        if self.mix_pool != 'none':                                           # 2026-11-17: one token sequence [N, B, 1, D]
            xm = xm.amax(2, keepdim=True) if self.mix_pool == 'max' else xm.mean(2, keepdim=True)
        if self.comb == 'or':
            f = self.stack(self.mixer, xm)
            r = self.stack(self.mixer_rev, xm.flip(0)).flip(0)
            m = torch.maximum(f, r)
        else:
            f, hf = self.stack(self.mixer, xm, True)
            r, hr = self.stack(self.mixer_rev, xm.flip(0), True)
            r, hr = r.flip(0), hr.flip(0)
            if self.comb == 'linmx':
                m = self.fuse(torch.cat([hf, hr, (hf - hr).abs()], dim=-1))
            else:
                d = (hf - hr).abs()
                N, B, K, D = d.shape
                sd = self.sd_neuron(self.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
                fr = torch.maximum(f, r)
                if self.comb == 'sdor':
                    m = torch.maximum(fr, sd)
                elif self.comb == 'sdlin':
                    m = self.fuse(torch.cat([fr, sd], dim=-1))
                else:                                                         # muxor / muxlin (10-30): s_d ? a : F ^ R
                    a = fr if self.comb == 'muxor' else self.fuse(torch.cat([fr, sd], dim=-1))
                    m = sd * a + (1. - sd) * torch.minimum(f, r)
        if self.mix_pool != 'none':                                           # 2026-11-17: copy back to the T_h repeats
            m = m.expand(-1, -1, reps, -1)
        if self.token_axis:
            return m
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
                 mixer_comb='or', connect_f='IAND', tau_calib=True, band='none', taxis='rep', gain_table=None, mix_pool='none',
                 mixer_ms=False, mixer_reset='spikede', neuron_scope='all', **kwargs):
        super().__init__()
        if neuron_scope not in SCOPES:                                        # 2026-11-18 neuron placement
            raise ValueError(f'neuron_scope must be one of {SCOPES}')
        if neuron_scope != 'all' and (mixer not in ('pop_lif', 'pop_flif') or band != 'none' or taxis != 'rep' or connect_f == 'MS'):
            raise NotImplementedError("neuron_scope 'mixer': pop_lif / pop_flif, rep layout, no band, no connect MS")
        self.neuron_scope = neuron_scope
        if mix_pool not in POOLS or mixer_reset not in RESETS:                # 2026-11-17 mixer variants
            raise ValueError(f'mix_pool must be one of {POOLS}, mixer_reset one of {RESETS}')
        if (mix_pool != 'none' or mixer_ms or mixer_reset != 'spikede') and (connect_f == 'MS' or band != 'none' or taxis != 'rep'):
            raise NotImplementedError('mix_pool / mixer_ms / mixer_reset: rep layout, no band, no connect MS')
        if mixer_ms and int(mixer_layers) < 2:
            raise ValueError('mixer_ms needs mixer_layers >= 2 (the shortcut runs between the layers of a direction)')
        if mixer_reset != 'spikede' and mixer != 'lif':
            raise NotImplementedError("mixer_reset is implemented for mixer 'lif' only")
        if (mix_pool != 'none' or mixer_ms) and mixer.startswith('popg_'):
            raise NotImplementedError('mix_pool / mixer_ms are not implemented for popg neurons')
        self.mix_pool, self.mixer_ms, self.mixer_reset = mix_pool, bool(mixer_ms), mixer_reset
        if mixer not in NEURONS:
            raise ValueError(f'mixer must be one of {NEURONS}')
        if int(T_h) < 1:
            raise ValueError('T_h >= 1')
        if band not in BANDS or taxis not in TAXES:                          # 2026-11-16 STFT bands / spiking axis
            raise ValueError(f'band must be one of {BANDS}, taxis one of {TAXES}')
        if (band != 'none' or taxis != 'rep') and connect_f == 'MS':
            raise NotImplementedError('band / taxis are not implemented with the membrane shortcut')
        if band == 'group' and mixer not in ('lif', 'flif'):
            raise ValueError("band 'group' (A) is for the single neurons lif / flif")
        if band == 'member' and mixer not in ('pop_lif', 'pop_flif'):
            raise ValueError("band 'member' (B) is for the soma populations pop_lif / pop_flif")
        if band != 'none' and int(in_channels or 1) != 1:
            raise NotImplementedError('STFT bands are implemented for one input channel')
        self.band, self.taxis = band, taxis
        if taxis == 'token':
            T_h = 1                                                           # no repeats: the token axis is the only spiking axis
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

        def make_neuron(size, steps, layout, reset='spikede'):
            if grouped:
                return GroupNeuron(size, taus, fam, T=int(steps), alpha=neo_alpha, axis=(2 if layout == 'DN' else -1))
            if reset != 'spikede':                                            # 2026-11-17: the directional stacks only
                return build_neo_neuron(mixer, size, int(steps), layout, reset=reset, **neo_kw)
            return build_neo_neuron(mixer, size, int(steps), layout, **neo_kw)

        make_rest = None
        if neuron_scope == 'mixer':                                           # 2026-11-18: embedding / MLP = the single neuron
            single = mixer.split('_', 1)[1]                                   # pop_lif -> lif, pop_flif -> flif
            make_rest = lambda size, steps, layout: build_neo_neuron(single, size, int(steps), layout, **neo_kw)

        N, D = self.num_patches, embed_dim
        tok_ax = taxis == 'token'
        steps = N if tok_ax else self.T_h                                     # spiking steps of the embedding / MLP neurons
        self.encoding = Embedding(num_patches=N, patch_size=data_patch_size * self.in_channels, embed_dim=D, stride=self.stride,
                                  pe=False, bias=bias, tau=tau, pos_emb=False, neuron=(make_rest or make_neuron)(D, steps, 'DN'))
        self.blocks = nn.ModuleList([SpkBlock(D, mixer, self.mixer_layers, mlp_ratios, make_neuron, N, steps, mixer_comb, connect_f, bias,
                                              token_axis=tok_ax, mix_pool=mix_pool, mixer_ms=mixer_ms, mixer_reset=mixer_reset,
                                              make_rest=make_rest)
                                     for _ in range(self.depths)])
        self.head = nn.Linear(D, num_classes, bias=bias)
        self.apply(self._init_weights)
        if band != 'none':
            # 2026-11-16: the default embedding was built and initialised above, so at the same seed every other weight and the
            # random state after construction equal those of the band-less model; the band Linears come from a forked RNG.
            dims = stft_band_dims(data_patch_size)
            if len(taus) != len(dims) and band == 'member':
                raise ValueError(f'band member needs {len(dims)} members, got taus {taus}')
            with torch.random.fork_rng(devices=[]):
                if band == 'group':
                    enc = GroupBandEmbedding(dims, D, self.encoding.emb_bn, self.encoding.emb_lif, bias)
                else:
                    enc = MemberBandEmbedding(dims, D, self.encoding.emb_lif, member_order(taus), bias)
                for lin in enc.emb_linear:
                    trunc_normal_(lin.weight, std=.02)
                    if lin.bias is not None:
                        nn.init.constant_(lin.bias, 0)
            self.encoding = enc
        if grouped:                                                           # family gains per group first (tau_gain table) ...
            with torch.no_grad():
                for name, mod in self.named_modules():
                    if isinstance(mod, nn.BatchNorm1d):
                        mod.weight.copy_(group_gamma(mod.weight.numel(), taus, fam))
        self.tau_calibrated = False
        if mixer in CALIBRATED and tau_calib:                                 # ... then the layer-wise rate-matched table (fail closed)
            if gain_table is None and (band != 'none' or taxis != 'rep' or mix_pool != 'none' or mixer_ms or neuron_scope != 'all'):
                raise ValueError('a population model with band / taxis / mix_pool / mixer_ms / neuron_scope needs an explicit gain_table')
            self.tau_calibrated = apply_gain_table(self, gain_table if gain_table is not None else gain_table_path(mixer, mixer_comb, connect_f))

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
        tok_ax = self.taxis == 'token'
        if self.band != 'none':                                               # 2026-11-16: 4 STFT bands of every frame [T_h, B, N, dim_i]
            tok = stft_values(tok)
            if tok_ax:
                tok = [t.permute(2, 1, 0, 3).contiguous() for t in tok]       # [N, B, 1, dim_i]
        elif tok_ax:
            tok = tok.permute(2, 1, 0, 3).contiguous()                        # [N, B, 1, P]
        if self.connect_f == 'MS':                                            # 2026-11-08 membrane shortcut
            u, e = self.encoding.forward_ms(tok)
            h, m0 = e, None
            for i, blk in enumerate(self.blocks):
                m = blk.mix(h)
                if mix_fns is not None and mix_fns[i] is not None:
                    m = mix_fns[i](m)
                if i == 0:
                    m0 = m
                h, u = blk.forward_ms(u, m)
        else:
            e = self.encoding(tok)                                            # [T_h, B, N, D]
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
        if tok_ax:                                                            # back to the usual [1, B, N, D]
            h, e, m0 = (z.permute(2, 1, 0, 3).contiguous() for z in (h, e, m0))
        feat = h.mean(2)                                                      # token mean [T_h, B, D]
        z = self.head(feat).mean(0)                                           # Spikformer-style readout
        return z, e, m0, feat.mean(0), h

    def forward(self, x):
        z, _, _, _, h = self.forward_parts(x)
        if self.train_mode == 'training':
            return z
        return z, h.detach()
