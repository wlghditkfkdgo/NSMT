import torch
import torch.nn as nn

from layers import Embedding, to_patches


__all__ = ['myModel', 'GRUBaseline', 'truth_to_oracle_p']


def truth_to_oracle_p(truth, kind, step, embed_dim):
    """Turn the generator's answer set into the oracle policy p for one step.

    truth : [B, T, T] bool, truth[b, n, j] = "j is where y[b, n] was shown"
    kind  : [B, T] int8, 0 = copy, 1 = recall, 2 = recall and first of its run
    step  : n, the event being answered
    returns p : [B, D, n], uniform over the answer slots on recall events

    The policy is uniform -- that is, exactly the eta=0 kernel (gate G7b), injecting nothing
    -- on n = 0 and on every COPY event. The first draft applied the answer set wherever
    truth was non-empty, which also caught copy events inside a key's first run that already
    had an earlier presentation behind them: 610 of them in one test split (audit A12-ORACLE).
    That made the oracle stronger than its own description and inflated the headroom that
    O7's G divides by. Confining it to recall events keeps the oracle's advantage where the
    claim is, and `kind` must be passed for oracle mode for that reason.
    """
    if step == 0:
        return None
    mask = truth[:, step, :step].to(torch.float32)                  # [B, n]
    if kind is not None:
        mask = mask * (kind[:, step] > 0).to(mask.dtype).unsqueeze(-1)
    total = mask.sum(-1, keepdim=True)
    mask = torch.where(total > 0, mask / total.clamp_min(1.), torch.full_like(mask, 1. / step))

    return mask.unsqueeze(1).expand(-1, embed_dim, -1)


def window_norm(x, eps=1e-5):
    #  x: [B, L, C]  ->  z: [B, L, C],  mu, s: [B, 1, C]
    """Reversible input-window normalisation R of prereg 2O D-BK (RevIN without affine).

    Each window and each channel is standardised along time only, with statistics that are
    detached: the model learns on shape, and the level and scale it cannot see are put back
    by window_denorm on the output. Nothing is shared across the batch or across channels,
    and the future is never looked at.
    """
    mu = x.mean(dim=1, keepdim=True).detach()
    s = (x.var(dim=1, keepdim=True, unbiased=False) + eps).sqrt().detach()

    return (x - mu) / s, mu, s


def window_denorm(y, mu, s):
    #  y: [B, pred_len, C]  ->  same, back on the input's scale
    return y * s + mu


class myModel(nn.Module):
    """M1: patch -> population f-LIF -> readout. One backbone, two task heads.

    The heads are deliberately separate. The recall task is supervised per event, so its head
    is a per-event Linear(D, 1): event n is answered from s_n alone, which by gate G6 depends
    only on patches up to n, so the answer cannot be read off a future patch. Forecasting
    keeps the v2 flatten/last head. Mixing the two would let the recall task see the whole
    sequence at once and stop testing recall at all.

    Three readouts, asking three different questions, all with the graph attached so the
    neuron still trains end to end. `spike` is the model. `analog` reads the soma membrane
    after reset, so it asks whether the spike nonlinearity is the bottleneck. `drive` reads
    the branch mixture before the soma at all, so it asks whether the information is in the
    branch states in the first place. Results are labelled with which was run; a probe on
    the detached state would answer a fourth, weaker question and is not what these are.
    """
    def __init__(self, task='recall', seq_len=336, pred_len=96, patch_size=8, embed_dim=32,
                 head_dim=32, head_mode='flatten', readout='spike', input_scale=8.,
                 input_norm='frozen', revin=False, **neuron_args):
        super().__init__()

        if revin and task == 'recall':
            raise ValueError('window normalisation R is defined for forecasting only (prereg 2O)')
        self.revin = bool(revin)
        if seq_len < patch_size or seq_len % patch_size:
            raise ValueError('Require complete chronological non-overlapping patches')
        if head_mode not in ('flatten', 'last', 'linear') or readout not in ('spike', 'analog', 'drive'):
            raise ValueError('Invalid readout configuration')
        self.task, self.readout = task, readout
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_size, self.head_mode = patch_size, head_mode
        self.num_patches = seq_len // patch_size

        self.embedding = Embedding(patch_size, embed_dim, input_scale, input_norm, **neuron_args)
        # 두 head를 항상 함께 만든다. 그래야 초기화가 task에 따라 달라지지 않는다.
        self.recall_head = nn.Linear(embed_dim, 1)
        self.head_compress = nn.Linear(embed_dim, head_dim)
        # 'linear' (prereg 2Q D-CB): model_v1's readout -- the flattened spikes go through ONE
        # non-spiking nn.Linear(embed_dim * num_patches, pred_len). head_compress is still built
        # first so the random stream, and with it every other initial value, matches the
        # 'flatten' run at the same seed; it is then dropped.
        head_in = {'flatten': head_dim * self.num_patches, 'last': head_dim,
                   'linear': embed_dim * self.num_patches}[head_mode]
        self.head = nn.Linear(head_in, pred_len)
        if head_mode == 'linear':
            self.head_compress = nn.Identity()

    def forward(self, x, mode='sparse', truth=None, kind=None, return_aux=False):
        #  x: [B, L, C]  ->  recall: [B, T]   ett: [B, pred_len, C]
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError('Expected [B, seq_len, C] input')
        B, L, C = x.shape
        if self.revin:
            x, mu, s = window_norm(x)                                # 2O D-BK: patch/embedding 앞
        patch = to_patches(x, self.patch_size)                       # [T, B*C, patch_size]

        oracle_p = None
        # mode=hard with hard_stat=oracle (2M) needs the same answer sets as mode=oracle.
        if mode == 'oracle' or (mode == 'hard' and self.embedding.neuron.selector.hard_stat == 'oracle'):
            if truth is None or kind is None:
                raise ValueError('oracle mode needs the generator truth and the event kinds')
            if C != 1:
                raise ValueError('oracle policy is defined for the single-channel recall task')
            embed_dim = self.recall_head.in_features
            oracle_p = [truth_to_oracle_p(truth, kind, n, embed_dim)
                        for n in range(self.num_patches)]

        want = return_aux or self.readout != 'spike'
        analog = False if self.readout == 'spike' else self.readout
        result = self.embedding(patch, mode, oracle_p, want, analog)
        spikes, aux = result if want else (result, {})
        z = spikes if self.readout == 'spike' else aux['analog']      # [T, B*C, D]

        if self.task == 'recall':
            output = self.recall_head(z).squeeze(-1).transpose(0, 1)  # [B*C, T] -> 채널 1개
        else:
            h = self.head_compress(z)                                # 'linear': Identity
            h = h.transpose(0, 1).reshape(B * C, -1) if self.head_mode in ('flatten', 'linear') else h[-1]
            output = self.head(h).reshape(B, C, self.pred_len).transpose(1, 2)
            if self.revin:
                output = window_denorm(output, mu, s)                # 손실·평가는 복원 척도에서

        return (output, aux) if return_aux else output


class GRUBaseline(nn.Module):
    """Compressed-state control (prereg D-H).

    The recall task can be solved by carrying the values in a hidden state and reading them
    out by the current cue, with no per-event selection anywhere. That is not a flaw to hide:
    it is why this baseline is mandatory. The gap to it says how much of the task our neuron's
    selection is actually responsible for.
    """
    def __init__(self, task='recall', seq_len=336, pred_len=96, patch_size=8, embed_dim=32,
                 head_dim=32, head_mode='flatten', num_layers=1, revin=False, **unused):
        super().__init__()

        if revin and task == 'recall':
            raise ValueError('window normalisation R is defined for forecasting only (prereg 2O)')
        if head_mode not in ('flatten', 'last'):                     # 'linear' is the spiking model's readout (2Q)
            raise ValueError(f"GRUBaseline head_mode must be 'flatten' or 'last', not {head_mode!r}")
        self.revin = bool(revin)
        self.task, self.seq_len, self.pred_len = task, seq_len, pred_len
        self.patch_size, self.head_mode = patch_size, head_mode
        self.num_patches = seq_len // patch_size
        self.gru = nn.GRU(patch_size, embed_dim, num_layers=num_layers)   # 인과적: 단방향
        self.recall_head = nn.Linear(embed_dim, 1)
        self.head_compress = nn.Linear(embed_dim, head_dim)
        self.head = nn.Linear(head_dim * (self.num_patches if head_mode == 'flatten' else 1), pred_len)

    def forward(self, x, mode='sparse', truth=None, kind=None, return_aux=False):
        #  x: [B, L, C]  ->  recall: [B, T]   ett: [B, pred_len, C]
        B, L, C = x.shape
        if self.revin:
            x, mu, s = window_norm(x)
        patch = to_patches(x, self.patch_size)
        z, _ = self.gru(patch)                                       # [T, B*C, D]

        if self.task == 'recall':
            output = self.recall_head(z).squeeze(-1).transpose(0, 1)
        else:
            h = self.head_compress(z)
            h = h.transpose(0, 1).reshape(B * C, -1) if self.head_mode == 'flatten' else h[-1]
            output = self.head(h).reshape(B, C, self.pred_len).transpose(1, 2)
            if self.revin:
                output = window_denorm(output, mu, s)

        return (output, {}) if return_aux else output


class LinearBaseline(nn.Module):
    """Linear reference of prereg 2O (Zeng et al., AAAI 2023, the channel-shared Linear).

    One Linear(seq_len -> pred_len) with bias, shared by every channel and applied to each
    channel on its own, so it is channel-independent like the other models. With revin it is
    Linear + R, not NLinear (NLinear subtracts and restores the last value, a different
    operation). 336 * 96 + 96 = 32,352 parameters: small in structure, not in count.
    """
    def __init__(self, task='ett', seq_len=336, pred_len=96, revin=False, **unused):
        super().__init__()

        if task == 'recall':
            raise ValueError('the linear reference is defined for forecasting only')
        self.task, self.seq_len, self.pred_len = task, seq_len, pred_len
        self.revin = bool(revin)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x, mode='full', truth=None, kind=None, return_aux=False):
        #  x: [B, L, C]  ->  [B, pred_len, C]
        if self.revin:
            x, mu, s = window_norm(x)
        output = self.linear(x.transpose(1, 2)).transpose(1, 2)
        if self.revin:
            output = window_denorm(output, mu, s)

        return (output, {}) if return_aux else output
