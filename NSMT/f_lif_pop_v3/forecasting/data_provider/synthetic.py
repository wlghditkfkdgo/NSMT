import numpy as np
import torch
from torch.utils.data import Dataset


# 사전등록 §4: 의미 단위(사건) 하나 = patch 하나 = 모델이 지목할 수 있는 기억 한 칸.
# patch 8칸 = [신호표시 n_keys칸 | 값 1칸 | 여분]. raw 길이 = n_events * patch_size = 336.
CUE_MODES = ('onehot', 'code')


def make_run_sequence(rng, n_events, n_keys, run_range, min_gap):
    """Lay out the key runs: introduce every key once, then re-query them uniformly.

    Task revision r2 (audit A03). The r1 layout introduced every fresh key first and then
    only admitted keys whose last run was 1-3 runs back. With 8 keys and only ~12 runs in
    T=42 the keys introduced first fell out of that window permanently and were never
    queried again, so raising n_keys raised the number of distractors but NOT the number of
    keys actually held and recalled: measured 2.52 / 2.59 / 2.55 distinct re-queried keys at
    n_keys 3 / 5 / 8. Here phase 2 samples uniformly among every key last seen at least
    `min_gap` runs ago, so each key keeps an equal chance of being needed again and the
    source lag grows with n_keys, which is what a memory-capacity axis has to do.

    Returns
    -------
    key : (n_events,) int, which signal is active at each event
    run : (n_events,) int, which run the event belongs to
    """
    keys, lengths, seen, total = [], [], {}, 0
    order = rng.permutation(n_keys)
    while total < n_events:
        r = len(keys)
        if r < n_keys:
            k = int(order[r])                                    # 1단계: 모든 key를 한 번씩 소개
        else:
            cand = [j for j in range(n_keys) if r - seen[j] - 1 >= min_gap]
            if not cand:                                         # min_gap이 key 수보다 크면 가장 오래된 것
                cand = [min(seen, key=seen.get)]
            k = int(rng.choice(cand))                            # 2단계: 균등 재질의 -> coverage 보장
        keys.append(k)
        lengths.append(int(rng.integers(run_range[0], run_range[1] + 1)))
        seen[k] = r
        total += lengths[-1]

    key = np.concatenate([np.full(L, k) for k, L in zip(keys, lengths)])[:n_events]
    run = np.concatenate([np.full(L, r) for r, L in enumerate(lengths)])[:n_events]

    return key, run


def make_sequence(rng, n_events=42, patch_size=8, n_keys=3, run_range=(2, 5), min_gap=1,
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
    truth : (n_events, n_events) bool, truth[n, j] = "j is a place where y[n] was shown".
        The source set is the key's FIRST run, i.e. where the value was actually presented,
        not the previous re-appearance (audit A03 asked for this to be stated).
    recall : (n_events,) bool, events whose answer is NOT in the current input
    """
    assert cue_mode in CUE_MODES
    key, run = make_run_sequence(rng, n_events, n_keys, run_range, min_gap)
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
    """Fixed +-1 cue codes, shared by every sequence so a key means the same thing.

    Drawn WITHOUT replacement from the 2**cue_dim distinct sign patterns. Rejection
    sampling does not work here and the earlier draft that used it was never actually
    replaced in the file (audit A11): at cue_dim=4 only 16 patterns exist, so redrawing
    the whole matrix until 16 rows come out distinct has probability 1.2e-7 per attempt.
    """
    if n_keys > 2 ** cue_dim:
        raise ValueError(f'cue_dim={cue_dim} holds at most {2 ** cue_dim} distinct codes, '
                         f'asked for {n_keys}')
    grid = np.array([[1. if (i >> d) & 1 else -1. for d in range(cue_dim)]
                     for i in range(2 ** cue_dim)])
    pick = np.random.default_rng(seed).permutation(len(grid))[:n_keys]

    return grid[pick] / np.sqrt(cue_dim)


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
                min_gap=args.min_gap, cue_mode=args.cue_mode,
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


def task_stats(n_seq=300, seed=20260921, alpha=.7, **kwargs):
    """Measure what the task actually is, with the aggregation unit stated for each number.

    Two "chance" quantities exist and they are NOT the same (audit A04):

    uniform_slot_chance
        A reader that picks one past slot uniformly. Per query that is |A_n| / n, so the
        sequence-level number is the mean of |A_n|/n over queries -- NOT mean(|A_n|) / (T/2),
        which is what the first draft reported. The two differ because queries are not
        uniformly placed in time and drift later as n_keys grows.
    full_kernel_mass
        The fraction of the FULL (eta=0) fractional kernel that sits on the answer slots,
        sum_{j in A_n} b(n-j) / sum_{j<n} b(n-j). A uniform p also produces this kernel, so
        this -- not uniform_slot_chance -- is the baseline the selector has to beat.

    Both are averaged per query inside a sequence first, then across sequences.
    """
    from math import lgamma, exp
    rng = np.random.default_rng(seed)
    gamma = exp(lgamma(alpha + 1.))
    b = np.array([((d + 1.) ** alpha - d ** alpha) / gamma for d in range(kwargs.get('n_events', 42))])

    out = {k: [] for k in ('recall_frac', 'sources', 'slot_chance', 'kernel_mass', 'coverage',
                           'queries_per_key', 'lag_mean', 'lag_max', 'first_query_frac', 'run_len')}
    n_keys = kwargs.get('n_keys', 3)
    for _ in range(n_seq):
        x, y, truth, recall = make_sequence(rng, **kwargs)
        T = y.shape[0]
        idx = np.flatnonzero(recall)
        out['recall_frac'].append(recall.mean())
        out['run_len'] += np.diff(np.flatnonzero(np.diff(np.concatenate([[np.nan], y])) != 0).tolist()
                                  + [T]).tolist() if False else []
        if len(idx) == 0:
            out['coverage'].append(0.)
            continue
        slot, kern, lag, src = [], [], [], []
        for n in idx:
            a = np.flatnonzero(truth[n])
            if n == 0 or a.size == 0:
                continue
            slot.append(a.size / n)                              # |A_n| / n
            kern.append(b[n - a].sum() / b[1:n + 1].sum())        # 정답 slot이 가진 full 커널 질량
            lag.append(float((n - a).mean()))
            src.append(a.size)
        if not slot:
            continue
        out['sources'].append(np.mean(src))
        out['slot_chance'].append(np.mean(slot))
        out['kernel_mass'].append(np.mean(kern))
        out['lag_mean'].append(np.mean(lag))
        out['lag_max'].append(np.max(lag))
        # 실제로 다시 질의된 고유 key 수 / 전체 key 수
        cue_dim = n_keys if kwargs.get('cue_mode', 'onehot') == 'onehot' \
            else (kwargs.get('cue_dim') or min(x.shape[0] // T - 1, 4))
        book = np.eye(n_keys) if kwargs.get('cue_mode', 'onehot') == 'onehot' \
            else rng_codes(n_keys, cue_dim)
        cue = x.reshape(T, -1)[:, :cue_dim]
        key = np.linalg.norm(cue[:, None, :] - book[None], axis=-1).argmin(-1)
        queried = np.unique(key[recall])
        out['coverage'].append(len(queried) / n_keys)
        out['queries_per_key'].append(len(idx) / max(len(queried), 1))
        # 재등장 구간의 첫 칸인가 (전환 반응 속도용)
        first = recall & np.concatenate([[True], key[1:] != key[:-1]])
        out['first_query_frac'].append(first.sum() / max(len(idx), 1))

    return {k: (float(np.mean(v)) if v else float('nan')) for k, v in out.items() if k != 'run_len'}


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
        stats = task_stats(n_seq=min(n_seq, 300), **kwargs)
        print(f"[recall] {n_seq} sequences, T={y.shape[0]}, keys={kwargs.get('n_keys', 3)}, "
              f"revision r2")
        print(f"[recall] run length   : {np.mean(runs):.2f} avg (마지막 run은 잘릴 수 있음), "
              f"histogram {np.bincount(runs)[1:].tolist()}")
        print(f"[recall] recall events: {100 * stats['recall_frac']:.1f}% of the sequence, "
              f"{stats['sources']:.2f} source slots per query")
        print(f"[recall] key coverage : {100 * stats['coverage']:.1f}% of keys re-queried, "
              f"{stats['queries_per_key']:.2f} queries per queried key")
        print(f"[recall] source lag   : {stats['lag_mean']:.2f} avg, {stats['lag_max']:.2f} max "
              f"(events back to the first run)")
        print(f"[recall] first-query  : {100 * stats['first_query_frac']:.1f}% of queries are a "
              f"run's first event")
        print(f"[recall] uniform-slot chance : {stats['slot_chance']:.4f}   "
              f"(mean over queries of |A_n|/n)")
        print(f"[recall] full-kernel mass    : {stats['kernel_mass']:.4f}   "
              f"(what eta=0 already puts on the answer; the baseline to beat)")

    return True
