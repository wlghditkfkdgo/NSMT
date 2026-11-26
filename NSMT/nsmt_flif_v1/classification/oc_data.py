"""MODI 2026-11-26: data for the order / accumulation check (docs/MODI_PROJECT_LOG.md 2026-11-26, user decision 11-26:
"Speech Commands와 Sleep-EDF에서 '순서가 필요한 과제인지' 확인해보자"). Diagnostic reference models only (order_check.py).

Speech Commands v0.02 (Warden 2018): 35 words, official validation_list.txt / testing_list.txt, every other file of the 35 word folders
  is training (expected 84,843 / 9,981 / 11,005). Clip padded / cut to 1 s at 16 kHz, log-mel: 40 bands, 25 ms window (400), 10 ms hop
  (160), centred frames -> [40, 101]; log(mel + 1e-6); stored float16. Normalisation (per band, training clips) is done in order_check.py.
Sleep-EDF-20 (Kemp et al. 2000; PhysioNet sleep-edfx 1.0.0, sleep-cassette subjects 0-19, 39 nights): EEG Fpz-Cz, 100 Hz, 30-s epochs.
  Labels as DeepSleepNet (Supratak et al. 2017, prepare_physionet.py): W/1/2/3/4/R -> W/N1/N2/N3/N3/REM, 'Sleep stage ?' and
  'Movement time' removed, then only the epochs from 30 min (60 epochs, counted among the kept epochs) before the first non-W epoch to
  30 min after the last one are targets. Expected targets: W 8,285, N1 2,804, N2 17,799, N3 5,703, REM 7,717 (42,308; the counts the
  DeepSleepNet / AttnSleep papers report for Sleep-EDF-20 Fpz-Cz). Context epochs are the raw preceding epochs of the same night
  (whatever their label), so the target set is the same for every context length; each night keeps CTX_MAX - 1 = 15 epochs before its
  first target. The EDF reader below is written here (no pyedflib / mne in the environment); check_oc.py tests it.
MIT-BIH (Kachuee et al. 2018 beats, the files the NSMT / MODI runs use): 187 points at 125 Hz, train_7 / test_3 split; validation =
  a stratified 10 % of train (seed 0).
usage (classification folder): python oc_data.py sc | sleep      (builds the caches; MIT-BIH needs none)
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '..', '..'))
SC_DIR = os.path.join(HERE, 'dataset', 'speech_commands')
SC_RAW = os.path.join(SC_DIR, 'raw')
SC_NPZ = os.path.join(SC_DIR, 'sc35_logmel40.npz')
SL_DIR = os.path.join(HERE, 'dataset', 'sleep_edf20')
SL_NPZ = os.path.join(SL_DIR, 'sleepedf20_fpzcz.npz')
MB_DIR = os.path.join(R, 'nsmt_base', 'classification', 'ECG', 'data')
SC_WORDS = sorted(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house',
                   'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree',
                   'two', 'up', 'visual', 'wow', 'yes', 'zero'])
SL_STAGE = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
            'Sleep stage ?': -1, 'Movement time': -1}
SL_EXPECT = [8285, 2804, 17799, 5703, 7717]
CTX_MAX = 16
EP = 3000                                                                       # 30 s at 100 Hz


# ---------------------------------------------------------------- EDF / EDF+ (European Data Format, Kemp et al. 1992 / 2003)
def read_edf(path):
    """-> dict(start=(date, time), labels, fs (per signal), dur, nrec, data(i) -> physical values of signal i, raw_bytes(i))."""
    with open(path, 'rb') as f:
        h = f.read(256)
        hb, nrec, dur, ns = int(h[184:192]), int(h[236:244]), float(h[244:252]), int(h[252:256])
        sh = f.read(ns * 256)
        f.seek(hb)
        raw = np.frombuffer(f.read(), dtype='<i2')

    def fld(off, w):
        return [sh[off + i * w: off + (i + 1) * w].decode('latin-1').strip() for i in range(ns)]
    labels = fld(0, 16)
    o = 16 * ns + 80 * ns + 8 * ns                                               # label, transducer, physical dimension
    pmin, pmax, dmin, dmax = (np.array([float(v) for v in fld(o + k * 8 * ns, 8)]) for k in range(4))
    nsamp = np.array([int(v) for v in fld(o + 4 * 8 * ns + 80 * ns, 8)])        # + 4 x 8-byte fields + prefiltering
    rs = int(nsamp.sum())
    n = len(raw) // rs
    if nrec >= 0 and n != nrec:
        raise SystemExit(f'{path}: {n} data records, header says {nrec}')
    rec = raw[:n * rs].reshape(n, rs)
    col = np.concatenate([[0], np.cumsum(nsamp)])

    def data(i):
        d = rec[:, col[i]:col[i + 1]].reshape(-1).astype(np.float64)
        return (d - dmin[i]) * (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) + pmin[i]

    def raw_bytes(i):
        return rec[:, col[i]:col[i + 1]].reshape(-1).tobytes()
    return dict(start=(h[168:176].decode(), h[176:184].decode()), labels=labels, fs=nsamp / dur, dur=dur, nrec=n, data=data,
                raw_bytes=raw_bytes)


def edf_annotations(edf):
    """EDF+ time-stamped annotation lists of the 'EDF Annotations' signal -> [(onset s, duration s, text)]."""
    i = edf['labels'].index('EDF Annotations')
    b = edf['raw_bytes'](i).decode('latin-1')
    out = []
    for m in re.finditer(r'([+-]\d+(?:\.\d*)?)(?:\x15(\d+(?:\.\d*)?))?\x14([^\x00]*)', b):
        onset, dur, texts = float(m.group(1)), float(m.group(2) or 0), [t for t in m.group(3).split('\x14') if t]
        out += [(onset, dur, t) for t in texts]                                 # the record time-keeping TAL has no text
    return out


def sleep_night(psg, hyp):
    """one night -> (epochs [n, 3000] float32 of Fpz-Cz, label per epoch (-1 = none / removed), target mask)."""
    e, a = read_edf(psg), read_edf(hyp)
    if e['start'] != a['start']:
        raise SystemExit(f'{psg}: PSG start {e["start"]} != hypnogram start {a["start"]}')
    ch = e['labels'].index('EEG Fpz-Cz')
    if e['fs'][ch] != 100:
        raise SystemExit(f'{psg}: Fpz-Cz at {e["fs"][ch]} Hz')
    x = e['data'](ch).astype(np.float32)
    n_ep = len(x) // EP
    x = x[:n_ep * EP].reshape(n_ep, EP)
    lab = np.full(n_ep, -1, dtype=np.int64)
    seen = np.zeros(n_ep, dtype=bool)
    for onset, dur, t in edf_annotations(a):
        if t not in SL_STAGE:
            raise SystemExit(f'{hyp}: unknown annotation {t!r}')
        if onset % 30 or dur % 30:
            raise SystemExit(f'{hyp}: annotation not on the 30-s grid ({onset}, {dur}, {t})')
        s, k = int(onset // 30), int(dur // 30)
        s2 = min(s + k, n_ep)                                                    # DeepSleepNet: drop labels past the signal end
        if s < s2:
            if seen[s:s2].any():
                raise SystemExit(f'{hyp}: overlapping annotations at epoch {s}')
            lab[s:s2], seen[s:s2] = SL_STAGE[t], True
    kept = np.where(lab >= 0)[0]                                                 # unknown / movement removed first
    nw = np.where(lab[kept] != 0)[0]
    lo, hi = max(nw[0] - 60, 0), min(nw[-1] + 60, len(kept) - 1)                 # 30 min of kept epochs on each side
    tgt = np.zeros(n_ep, dtype=bool)
    tgt[kept[lo:hi + 1]] = True
    return x, lab, tgt


def build_sleep():
    files = [l.strip() for l in open(os.path.join(SL_DIR, 'FILES.txt')) if l.strip()]
    psgs = sorted(f for f in files if f.endswith('-PSG.edf'))
    hyps = {f[:6]: f for f in files if f.endswith('-Hypnogram.edf')}
    X, LAB, TGT, SUBJ, NIGHT, OFF = [], [], [], [], [], [0]
    for p in psgs:
        key = p[:6]                                                              # SC4ssN
        x, lab, tgt = sleep_night(os.path.join(SL_DIR, 'raw', p), os.path.join(SL_DIR, 'raw', hyps[key]))
        t = np.where(tgt)[0]
        lo = t[0] - (CTX_MAX - 1)
        if lo < 0:
            raise SystemExit(f'{p}: first target at epoch {t[0]}, fewer than {CTX_MAX - 1} epochs before it')
        x, lab, tgt = x[lo:t[-1] + 1], lab[lo:t[-1] + 1], tgt[lo:t[-1] + 1]
        X.append(x); LAB.append(lab); TGT.append(tgt)
        SUBJ.append(int(key[3:5])); NIGHT.append(int(key[5])); OFF.append(OFF[-1] + len(x))
        print(f'{p}: {len(x)} epochs kept, {int(tgt.sum())} targets', flush=True)
    lab, tgt = np.concatenate(LAB), np.concatenate(TGT)
    cnt = np.bincount(lab[tgt], minlength=5).tolist()
    print('targets per class W N1 N2 N3 REM:', cnt, 'total', sum(cnt), '(expected', SL_EXPECT, sum(SL_EXPECT), ')')
    np.savez(SL_NPZ, x=np.concatenate(X), lab=lab, tgt=tgt, subj=np.array(SUBJ), night=np.array(NIGHT), off=np.array(OFF, dtype=np.int64),
             names=np.array([p[:-8] for p in psgs]))
    return cnt


def load_sleep():
    d = np.load(SL_NPZ)
    rec = np.repeat(np.arange(len(d['subj'])), np.diff(d['off']))               # night index of every kept epoch
    return d['x'], d['lab'], d['tgt'], d['subj'][rec], rec, d['off']


def sleep_fold(subj_of_target, fold):
    """5 folds of 4 subjects (fold f tests subjects 4f..4f+3); validation = the next 2 subjects (cyclic); training = the other 14."""
    test = set(range(4 * fold, 4 * fold + 4))
    val = {(4 * fold + 4) % 20, (4 * fold + 5) % 20}
    s = subj_of_target
    return np.isin(s, sorted(set(range(20)) - test - val)), np.isin(s, sorted(val)), np.isin(s, sorted(test))


# ---------------------------------------------------------------- Speech Commands
def build_sc():
    import torch
    import torchaudio
    from scipy.io import wavfile
    lists = {k: set(open(os.path.join(SC_RAW, f'{k}_list.txt')).read().split()) for k in ('validation', 'testing')}
    names = sorted(f'{w}/{f}' for w in SC_WORDS for f in os.listdir(os.path.join(SC_RAW, w)) if f.endswith('.wav'))
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=40)
    X = np.zeros((len(names), 40, 101), dtype=np.float16)
    Y = np.array([SC_WORDS.index(n.split('/')[0]) for n in names], dtype=np.int64)
    S = np.array([1 if n in lists['validation'] else 2 if n in lists['testing'] else 0 for n in names], dtype=np.int64)
    for i0 in range(0, len(names), 2048):
        batch = []
        for n in names[i0:i0 + 2048]:
            fs, w = wavfile.read(os.path.join(SC_RAW, n))
            if fs != 16000 or w.dtype != np.int16:
                raise SystemExit(f'{n}: {fs} Hz {w.dtype}')
            w = w.astype(np.float32) / 32768.0
            batch.append(np.pad(w[:16000], (0, max(0, 16000 - len(w)))))
        m = mel(torch.from_numpy(np.stack(batch)))
        X[i0:i0 + len(batch)] = torch.log(m + 1e-6).numpy().astype(np.float16)
        print(f'{i0 + len(batch)} / {len(names)}', flush=True)
    cnt = np.bincount(S, minlength=3).tolist()
    print('train / validation / test clips:', cnt, '(expected [84843, 9981, 11005])')
    missing = (lists['validation'] | lists['testing']) - set(names)
    print('list entries without a file among the 35 words:', len(missing))
    np.savez(SC_NPZ, x=X, y=Y, split=S, names=np.array(names))
    return cnt


def load_sc():
    d = np.load(SC_NPZ)
    return d['x'], d['y'], d['split']


# ---------------------------------------------------------------- MIT-BIH (Kachuee et al. 2018)
def load_mitbih():
    tr = np.loadtxt(os.path.join(MB_DIR, '5class_mitbih_train_7_187.csv'), delimiter=',', dtype=np.float32)
    te = np.loadtxt(os.path.join(MB_DIR, '5class_mitbih_test_3_187.csv'), delimiter=',', dtype=np.float32)
    ytr = tr[:, 187].astype(np.int64)
    rng = np.random.default_rng(0)
    val = np.zeros(len(tr), dtype=bool)
    for c in np.unique(ytr):                                                     # stratified 10 % validation
        i = np.where(ytr == c)[0]
        val[rng.choice(i, int(round(0.1 * len(i))), replace=False)] = True
    return tr[:, :187], ytr, val, te[:, :187], te[:, 187].astype(np.int64)


if __name__ == '__main__':
    {'sc': build_sc, 'sleep': build_sleep}[sys.argv[1]]()
