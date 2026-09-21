import numpy as np
import torch
from torch.utils.data import Dataset


# 사전등록 §4: 의미 단위(사건) 하나 = patch 하나 = 모델이 지목할 수 있는 기억 한 칸.
# patch 8칸 = [신호표시 n_keys칸 | 값 1칸 | 여분]. raw 길이 = n_events * patch_size = 336.
CUE_MODES = ('onehot', 'code')


def make_run_sequence(rng, n_events, n_keys, run_range, gap_range):
    """Lay out the key runs so that a key reappears after 1-3 intervening runs.

    Returns
    -------
    key : (n_events,) int, which signal is active at each event
    run : (n_events,) int, which run the event belongs to
    """
    keys, lengths, seen, total = [], [], {}, 0
    while total < n_events:
        r = len(keys)
        fresh = [k for k in range(n_keys) if k not in seen]
        if fresh:                                                # 도입부: 아직 안 쓴 신호 먼저
            cand = fresh
        else:
            cand = [k for k in range(n_keys) if gap_range[0] <= r - seen[k] - 1 <= gap_range[1]]
            if not cand:                                         # 신호 수가 많아 창을 못 맞추면 가장 오래된 것
                cand = [min(seen, key=seen.get)]
        k = int(rng.choice(cand))
        keys.append(k)
        lengths.append(int(rng.integers(run_range[0], run_range[1] + 1)))
        seen[k] = r
        total += lengths[-1]

    key = np.concatenate([np.full(L, k) for k, L in zip(keys, lengths)])[:n_events]
    run = np.concatenate([np.full(L, r) for r, L in enumerate(lengths)])[:n_events]

    return key, run


def make_sequence(rng, n_events=42, patch_size=8, n_keys=3, run_range=(2, 5), gap_range=(1, 3),
                  cue_mode='onehot', cue_dim=None, cue_noise=0., distractor=0.,
                  value_range=(-1., 1.)):
    """One recall sequence.

    The value of a signal rides in the input ONLY during that signal's first run; every
    later run carries 0 in the value slot. The target is the signal's value at every
    event, so a re-appearance can only be answered by reading the past. Cue and value sit
    in the SAME channel because the model has one input stream: a separate channel would
    let the readout bypass the memory entirely.

    Args
    ----
    n_events : int, default 42
        Patch-axis length T. Equals ETT's T so the tensor shapes match.
    n_keys : int, default 3
        Number of signals. The difficulty axis (D-H) raises this to 8 and 16.
    cue_mode : {'onehot', 'code'}
        'onehot' is the pre-registered default and needs n_keys <= patch_size - 1.
        'code' uses fixed random +-1 codes and is what the n_keys sweep must use at
        EVERY level, so the encoding itself is not a confound.

    Returns
    -------
    x : (n_events * patch_size, 1) float32, raw single-channel series
    y : (n_events,) float32, target value per event
    truth : (n_events, n_events) bool, truth[n, j] = "j is a place where y[n] was shown"
    recall : (n_events,) bool, events whose answer is NOT in the current input
    """
    assert cue_mode in CUE_MODES
    key, run = make_run_sequence(rng, n_events, n_keys, run_range, gap_range)
    value = rng.uniform(value_range[0], value_range[1], size=n_keys)
    first_run = {k: run[key == k].min() for k in range(n_keys) if (key == k).any()}

    if cue_mode == 'onehot':
        assert n_keys <= patch_size - 1, 'one-hot cue does not fit; use cue_mode="code"'
        cue_dim = n_keys
        code = np.eye(n_keys)
    else:
        cue_dim = cue_dim or min(patch_size - 1, 4)
        code = rng_codes(n_keys, cue_dim)

    x = np.zeros((n_events, patch_size))
    x[:, :cue_dim] = code[key]
    if cue_noise:
        x[:, :cue_dim] += rng.normal(0., cue_noise, size=(n_events, cue_dim))
    shown = np.array([run[n] == first_run[key[n]] for n in range(n_events)])
    x[shown, cue_dim] = value[key[shown]]                        # 값은 첫 등장 구간에만
    if distractor and patch_size > cue_dim + 1:
        x[:, cue_dim + 1:] = rng.normal(0., distractor, size=(n_events, patch_size - cue_dim - 1))

    y = value[key]
    truth = np.zeros((n_events, n_events), dtype=bool)
    for n in range(n_events):
        source = np.flatnonzero(shown & (key == key[n]))
        truth[n, source[source < n]] = True                      # 과거만. 인과성 유지

    return (x.reshape(-1, 1).astype(np.float32), y.astype(np.float32), truth, ~shown)


def rng_codes(n_keys, cue_dim, seed=20260921):
    """Fixed +-1 cue codes, shared by every sequence so a key means the same thing."""
    gen = np.random.default_rng(seed)
    code = gen.choice([-1., 1.], size=(n_keys, cue_dim))
    while len(np.unique(code, axis=0)) < n_keys:                 # 중복 코드가 나오면 다시
        code = gen.choice([-1., 1.], size=(n_keys, cue_dim))

    return code / np.sqrt(cue_dim)


class Dataset_Recall(Dataset):
    """Synthetic recall task, returning the repository's four-value batch interface.

    ETT leaves the last two slots empty; here they carry the supervision the recall
    metrics need (oracle positions and which events actually require recall), so the
    training loop does not need a task-specific branch to fetch them.
    """
    def __init__(self, args, flag='train'):
        assert flag in ['train', 'val', 'test']
        count = {'train': args.n_train, 'val': args.n_val, 'test': args.n_test}[flag]
        offset = {'train': 0, 'val': 1, 'test': 2}[flag]
        rng = np.random.default_rng(args.data_seed + 10000 * offset)   # 분할 간 생성 난수 분리

        self.x, self.y, self.truth, self.recall = [], [], [], []
        for _ in range(count):
            x, y, truth, recall = make_sequence(
                rng, n_events=args.num_patches, patch_size=args.patch_size,
                n_keys=args.n_keys, run_range=tuple(args.run_range),
                gap_range=tuple(args.gap_range), cue_mode=args.cue_mode,
                cue_noise=args.cue_noise, distractor=args.distractor)
            self.x.append(x)
            self.y.append(y)
            self.truth.append(truth)
            self.recall.append(recall)
        self.x = torch.from_numpy(np.stack(self.x))
        self.y = torch.from_numpy(np.stack(self.y))
        self.truth = torch.from_numpy(np.stack(self.truth))
        self.recall = torch.from_numpy(np.stack(self.recall))

    def __getitem__(self, index):
        # x [L,1], y [T], truth [T,T] bool, recall [T] bool
        return self.x[index], self.y[index], self.truth[index], self.recall[index]

    def __len__(self):
        return self.x.shape[0]


def sanity_check(n_seq=200, verbose=True, **kwargs):
    """Assert the properties the task claims, before any of them is relied on."""
    rng = np.random.default_rng(0)
    runs, gaps, recall_frac, source_count = [], [], [], []
    for _ in range(n_seq):
        x, y, truth, recall = make_sequence(rng, **kwargs)
        T = y.shape[0]
        patch = x.reshape(T, -1)
        n_keys = kwargs.get('n_keys', 3)
        cue_dim = n_keys if kwargs.get('cue_mode', 'onehot') == 'onehot' \
            else (kwargs.get('cue_dim') or min(patch.shape[1] - 1, 4))
        cue = patch[:, :cue_dim]
        val = patch[:, cue_dim]

        assert not truth[np.triu_indices(T, 0)].any(), 'truth must be strictly causal'
        assert np.allclose(val[recall], 0.), 'a re-appearance must not carry its value'
        assert np.allclose(val[~recall], y[~recall]), 'a first run must carry its value'
        assert truth[recall].any(axis=-1).all(), 'every recall event needs a source'
        for n in np.flatnonzero(recall):
            assert np.allclose(y[np.flatnonzero(truth[n])], y[n]), 'source value must match target'

        # 코드 모드에서는 argmax가 신호를 복원하지 못하므로 코드와의 거리로 되찾는다
        book = np.eye(n_keys) if kwargs.get('cue_mode', 'onehot') == 'onehot' \
            else rng_codes(n_keys, cue_dim)
        key = np.linalg.norm(cue[:, None, :] - book[None], axis=-1).argmin(-1)
        edge = np.flatnonzero(np.diff(key) != 0) + 1
        runs += np.diff(np.concatenate([[0], edge, [T]])).tolist()
        seen = {}
        for r, start in enumerate(np.concatenate([[0], edge])):
            k = key[start]
            if k in seen:
                gaps.append(r - seen[k] - 1)
            seen[k] = r
        recall_frac.append(recall.mean())
        if recall.any():
            source_count.append(truth.sum(-1)[recall].mean())

    if not gaps:
        # 신호 수가 많고 구간이 길면 T 안에 재등장 자체가 없다. 조용히 넘어가면 안 된다.
        print(f"[recall] NO REAPPEARANCE at keys={kwargs.get('n_keys', 3)}, "
              f"T={kwargs.get('n_events', 42)}: {np.mean(runs):.2f}-event runs leave only "
              f"~{kwargs.get('n_events', 42) / np.mean(runs):.0f} runs, fewer than the key count.")
        return False

    if verbose:
        run_hist = np.bincount(runs)[1:]
        print(f"[recall] {n_seq} sequences, T={y.shape[0]}, keys={kwargs.get('n_keys', 3)}")
        print(f"[recall] run length   : {np.mean(runs):.2f} avg, histogram {run_hist.tolist()}")
        print(f"[recall] gap (runs)   : {np.mean(gaps):.2f} avg, "
              f"min {np.min(gaps)}, max {np.max(gaps)}")
        print(f"[recall] recall events: {100 * np.mean(recall_frac):.1f}% of the sequence")
        print(f"[recall] sources/event: {np.mean(source_count):.2f} correct past slots")
        print(f"[recall] chance mass  : {np.mean(source_count) / (0.5 * y.shape[0]):.3f} "
              f"(uniform read over the average history)")

    return True
