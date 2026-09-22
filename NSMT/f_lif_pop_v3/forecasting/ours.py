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
                 input_norm='frozen', **neuron_args):
        super().__init__()

        if seq_len < patch_size or seq_len % patch_size:
            raise ValueError('Require complete chronological non-overlapping patches')
        if head_mode not in ('flatten', 'last') or readout not in ('spike', 'analog', 'drive'):
            raise ValueError('Invalid readout configuration')
        self.task, self.readout = task, readout
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_size, self.head_mode = patch_size, head_mode
        self.num_patches = seq_len // patch_size

        self.embedding = Embedding(patch_size, embed_dim, input_scale, input_norm, **neuron_args)
        # 두 head를 항상 함께 만든다. 그래야 초기화가 task에 따라 달라지지 않는다.
        self.recall_head = nn.Linear(embed_dim, 1)
        self.head_compress = nn.Linear(embed_dim, head_dim)
        self.head = nn.Linear(head_dim * (self.num_patches if head_mode == 'flatten' else 1), pred_len)

    def forward(self, x, mode='sparse', truth=None, kind=None, return_aux=False):
        #  x: [B, L, C]  ->  recall: [B, T]   ett: [B, pred_len, C]
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError('Expected [B, seq_len, C] input')
        B, L, C = x.shape
        patch = to_patches(x, self.patch_size)                       # [T, B*C, patch_size]

        oracle_p = None
        if mode == 'oracle':
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
            h = self.head_compress(z)
            h = h.transpose(0, 1).reshape(B * C, -1) if self.head_mode == 'flatten' else h[-1]
            output = self.head(h).reshape(B, C, self.pred_len).transpose(1, 2)

        return (output, aux) if return_aux else output


class GRUBaseline(nn.Module):
    """Compressed-state control (prereg D-H).

    The recall task can be solved by carrying the values in a hidden state and reading them
    out by the current cue, with no per-event selection anywhere. That is not a flaw to hide:
    it is why this baseline is mandatory. The gap to it says how much of the task our neuron's
    selection is actually responsible for.
    """
    def __init__(self, task='recall', seq_len=336, pred_len=96, patch_size=8, embed_dim=32,
                 head_dim=32, head_mode='flatten', num_layers=1, **unused):
        super().__init__()

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
        patch = to_patches(x, self.patch_size)
        z, _ = self.gru(patch)                                       # [T, B*C, D]

        if self.task == 'recall':
            output = self.recall_head(z).squeeze(-1).transpose(0, 1)
        else:
            h = self.head_compress(z)
            h = h.transpose(0, 1).reshape(B * C, -1) if self.head_mode == 'flatten' else h[-1]
            output = self.head(h).reshape(B, C, self.pred_len).transpose(1, 2)

        return (output, {}) if return_aux else output
