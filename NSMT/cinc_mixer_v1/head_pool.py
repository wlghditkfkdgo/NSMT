"""Head input forms for classification (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG head input form pre-registration; user, 11-03:
"다른 `head 레이어 구조` 혹은 `head 입력 형태` 를 탐색해보자" -> two or three candidates retrained directly, CinC -> MIT-BIH only,
real-valued heads as reference rows only).

Both classification models read z = head(h.mean(2)).mean(0), h = the last block's output spikes [T_h, B, N, D] (spk_mixer.SpkMixer.
forward_parts; model.myModel._training / _testing). Every form keeps that head and ADDS a term built with zero weights without
touching the global RNG streams (head_mem.zero_linear / a zero Parameter), so a run starts bit-identical to the run without it
(weights, outputs, gradients, data order) and differs only through the new term during training:
    max     W(h.amax(2))                         D -> C     any spike over the tokens (rare events the token mean dilutes)
    seg4    W([mean of token segments 1..4])     4D -> C    torch.tensor_split into 4 contiguous segments (117 -> 30/29/29/29,
                                                            11 -> 3/3/3/2): coarse token position / order
    layers  W([e.mean(2); x1.mean(2)])           2D -> C    e = embedding spikes, x1 = the last block after token mixing (the MLP
                                                            input): what the residuals (IAND / OR) remove or saturate
    attn    head(sum_t (a_t - a0_t) h_t)         q in R^D   a = softmax over tokens of h . q, a0 = the same softmax of zeros; q = 0
                                                            at start, so the term is exactly 0 (reference only: real-valued)
All terms are computed per T_h step (like the head) and averaged over T_h by the model's own .mean(0).
Taps (one block, --layers 1): e = output of model.encoding; x1, y = input / output of the last block's MLP (spkMixer: blocks[-1].fc1
input and blocks[-1].fc2 output; myModel --hippo_ssa: data_block[-1].mlp); h = connect(x1, y) (spkMixer blocks[-1].connect;
myModel x1 (1 - y)). The head hook checks h.mean(2) == the head input bitwise on every forward (fail closed).
Energy: the new Linear is counted by syops / energy.SynOpRecorder like any Linear; attn's dot products and softmax are not counted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from head_mem import zero_linear

__all__ = ['KINDS', 'PoolTaps', 'enable_head_pool']

KINDS = ('max', 'seg4', 'layers', 'attn')


class PoolTaps:
    """Keeps e, x1 and y of the latest forward (graph attached) and rebuilds h."""
    def __init__(self, model):
        from model import myModel
        from spk_mixer import SpkMixer
        self.e = self.x1 = self.y = None
        if isinstance(model, SpkMixer):
            if len(model.blocks) != 1:
                raise NotImplementedError('head_pool: one block expected')
            blk = model.blocks[-1]
            self.connect = blk.connect
            blk.fc1.register_forward_pre_hook(lambda mod, inp: setattr(self, 'x1', inp[0]))
            blk.fc2.register_forward_hook(lambda mod, inp, out: setattr(self, 'y', out))
        elif isinstance(model, myModel) and model.hippo_ssa:
            if len(model.data_block) != 1 or not model.data_block[-1].use_ssa:
                raise NotImplementedError('head_pool: one SSA block expected')
            self.connect = lambda x, y: x * (1. - y)                      # model.Block.forward (use_ssa)
            def mlp_hook(mod, inp, out):
                self.x1, self.y = inp[0], out
            model.data_block[-1].mlp.register_forward_hook(mlp_hook)
        else:
            raise NotImplementedError('head_pool: spkMixer or myModel --hippo_ssa only')
        model.encoding.register_forward_hook(lambda mod, inp, out: setattr(self, 'e', out))

    def h(self):
        if self.x1 is None or self.y is None:
            raise RuntimeError('head_pool: no block output captured (forward not run?)')
        return self.connect(self.x1, self.y)


def enable_head_pool(model, kind):
    """Attach the taps and the added head term to a freshly built model (before any checkpoint is loaded)."""
    if kind not in KINDS:
        raise ValueError(f'head_pool must be one of {KINDS}')
    if getattr(model, '_pool_taps', None) is not None or getattr(model, '_charge_tap', None) is not None:
        raise RuntimeError('head_pool: another head term is already enabled')
    head = model.head
    if not isinstance(head, nn.Linear):
        raise NotImplementedError(f'head_pool: head is {type(head).__name__}')
    taps = PoolTaps(model)
    model._pool_taps, model.head_pool = taps, kind
    D, C = head.in_features, head.out_features
    if kind == 'attn':
        with torch.random.fork_rng(devices=[]):
            model.pool_q = nn.Parameter(torch.zeros(D))
    else:
        model.pool_readout = zero_linear({'max': 1, 'seg4': 4, 'layers': 2}[kind] * D, C)

    def add_term(mod, inp, out):                                          # head(h.mean(2)): inp[0] is [T_h, B, D]
        h = taps.h()
        if not torch.equal(h.mean(2), inp[0]):
            raise RuntimeError('head_pool: the rebuilt block output does not match the head input')
        if kind == 'max':
            f = h.amax(2)
        elif kind == 'seg4':
            f = torch.cat([s.mean(2) for s in torch.tensor_split(h, 4, dim=2)], dim=-1)
        elif kind == 'layers':
            f = torch.cat([taps.e.mean(2), taps.x1.mean(2)], dim=-1)
        else:
            a = torch.softmax(torch.einsum('tbnd,d->tbn', h, model.pool_q), dim=-1)
            a0 = torch.softmax(torch.zeros_like(a), dim=-1)
            return out + F.linear(torch.einsum('tbn,tbnd->tbd', a - a0, h), mod.weight)
        return out + model.pool_readout(f)
    model._pool_hook = head.register_forward_hook(add_term)
    return model
