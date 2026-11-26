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


# ---------------------------------------------------------------- MODI 2026-11-26 (user decision 11-26 evening): Sleep-EDF epoch
# spectrum front end ((나) ①) and MIT-BIH inter-patient beats with past beats ((가)). Added below; the functions above are unchanged.
SL_PSD_NPZ = os.path.join(SL_DIR, 'sleepedf20_fpzcz_psd.npz')
MIT_DIR = os.path.join(HERE, 'dataset', 'mitdb')
MIT_NPZ = os.path.join(MIT_DIR, 'mitdb_beats_mlii.npz')
# de Chazal, O'Dwyer & Reilly 2004 (IEEE TBME 51(7)); paced records 102, 104, 107, 217 left out.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
# MIT annotation codes (WFDB ecgcodes.h) -> AAMI EC57 class as in de Chazal 2004 Table I: N = N L R e j, S = A a J S, V = V E,
# F = F, Q = / f Q. Other codes are not beats.
MIT_SYM = {1: 'N', 2: 'L', 3: 'R', 4: 'a', 5: 'V', 6: 'F', 7: 'J', 8: 'A', 9: 'S', 10: 'E', 11: 'j', 12: '/', 13: 'Q', 34: 'e', 38: 'f'}
AAMI = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'A': 1, 'a': 1, 'J': 1, 'S': 1, 'V': 2, 'E': 2, 'F': 3, '/': 4, 'f': 4, 'Q': 4}
AAMI_NAMES = ['N', 'S', 'V', 'F', 'Q']
# de Chazal 2004 Table II class totals (summed by the protocol note, docs/MODI_PROJECT_LOG.md 2026-11-26): N, S, V, F, Q
DECHAZAL = {'DS1': [45868, 942, 3787, 415, 8], 'DS2': [44258, 1837, 3221, 388, 7]}
MIT_FS, MIT_PRE, MIT_POST, MIT_K = 360, 90, 144, 16          # window R - 0.25 s .. R + 0.40 s (234 samples), up to 16 past beats


def build_sleep_psd():
    """Welch spectrum of every kept 30-s epoch: 2-s Hann segments (200 points), 50 % overlap, 0.5 Hz bins; bins 0.5..30 Hz (60);
    log10(power in uV^2/Hz + 1e-6). Same epochs and order as sleepedf20_fpzcz.npz."""
    from scipy.signal import welch
    d = np.load(SL_NPZ)
    f, p = welch(d['x'].astype(np.float64), fs=100.0, window='hann', nperseg=200, noverlap=100, axis=-1)
    k = (f >= 0.5) & (f <= 30.0)
    out = np.log10(p[:, k] + 1e-6).astype(np.float32)
    print('epochs', out.shape, 'bins', f[k][0], '..', f[k][-1], 'Hz,', int(k.sum()), 'bins')
    np.savez(SL_PSD_NPZ, psd=out, freqs=f[k])


def load_sleep_psd():
    return np.load(SL_PSD_NPZ)['psd']


def read_mit_header(rec):
    lines = [l for l in open(os.path.join(MIT_DIR, 'raw', f'{rec}.hea')) if not l.startswith('#')]
    name, ns, fs, n = lines[0].split()[:4]
    sig = []
    for l in lines[1:1 + int(ns)]:
        p = l.split()
        sig.append(dict(fmt=p[1], gain=float(p[2].split('(')[0].split('/')[0]), zero=int(p[4]), init=int(p[5]), checksum=int(p[6]),
                        desc=' '.join(p[8:])))
    return int(fs), int(n), sig


def read_mit_212(rec, nsig, n):
    """MIT format 212: two 12-bit two's-complement samples in three bytes, signals interleaved. -> int array [n, nsig]."""
    b = np.fromfile(os.path.join(MIT_DIR, 'raw', f'{rec}.dat'), dtype=np.uint8).astype(np.int32)
    m = (len(b) // 3) * 3
    b = b[:m].reshape(-1, 3)
    s0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s1 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    s = np.stack([s0, s1], 1).reshape(-1)
    s = np.where(s > 2047, s - 4096, s)
    return s[:n * nsig].reshape(n, nsig)


def read_mit_atr(rec):
    """MIT annotation file -> (sample positions, codes) of every annotation (WFDB annotation format: 16-bit words, code = bits
    15-10, interval = bits 9-0; SKIP 59 carries a 32-bit interval, NUM 60 / SUB 61 / CHN 62 change fields, AUX 63 is followed by
    its bytes, padded to an even count; 0/0 ends the file)."""
    b = np.fromfile(os.path.join(MIT_DIR, 'raw', f'{rec}.atr'), dtype=np.uint8)
    i, t, pos, code = 0, 0, [], []
    while i + 1 < len(b):
        w = int(b[i]) | (int(b[i + 1]) << 8); i += 2
        A, I = w >> 10, w & 0x3FF
        if A == 0 and I == 0:
            break
        if A == 59:                                                                # SKIP: PDP-11 long, high word first
            hi = int(b[i]) | (int(b[i + 1]) << 8); lo = int(b[i + 2]) | (int(b[i + 3]) << 8); i += 4
            v = (hi << 16) | lo
            t += v - (1 << 32) if v >= 1 << 31 else v
        elif A == 63:                                                              # AUX string
            i += I + (I & 1)
        elif A in (60, 61, 62):
            pass
        else:
            t += I
            pos.append(t); code.append(A)
    return np.array(pos, dtype=np.int64), np.array(code, dtype=np.int64)


def build_mitdb():
    """Per record: MLII in mV ((adu - zero) / gain), causal 2nd-order Butterworth high-pass at 0.5 Hz (forward only, so the
    filtered value at time t uses no sample after t), beats = annotations whose code maps to an AAMI class. Stored per beat:
    record, R position, AAMI class, pre-RR in s (NaN for the first beat), and the window R - 90 .. R + 144 averaged in pairs to
    180 Hz (117 values); beats whose window leaves the signal are kept with a flag (in_signal = 0) and never used as targets."""
    from scipy.signal import butter, lfilter
    recs = [int(r) for r in open(os.path.join(MIT_DIR, 'RECORDS')).read().split()]
    R_, POS, CLS, RR, W, OK, SYM = [], [], [], [], [], [], []
    hb, ha = butter(2, 0.5, btype='high', fs=MIT_FS)
    per_rec = {}
    for rec in recs:
        fs, n, sig = read_mit_header(rec)
        if fs != MIT_FS or any(s['fmt'] != '212' for s in sig):
            raise SystemExit(f'{rec}: fs {fs} formats {[s["fmt"] for s in sig]}')
        raw = read_mit_212(rec, len(sig), n)
        for c, s in enumerate(sig):                                                # header check: first value and checksum
            if raw[0, c] != s['init'] or ((int(raw[:, c].sum()) + 32768) % 65536) - 32768 != s['checksum']:
                raise SystemExit(f'{rec} signal {c}: first {raw[0, c]} vs {s["init"]}, checksum mismatch')
        pos, code = read_mit_atr(rec)
        keep = np.array([c in MIT_SYM for c in code], dtype=bool)
        pos, sym = pos[keep], [MIT_SYM[c] for c in code[keep]]
        per_rec[rec] = np.bincount([AAMI[s] for s in sym], minlength=5).tolist()
        if 'MLII' not in [s['desc'] for s in sig]:
            continue                                                               # 102, 104 (paced, not in DS1 / DS2)
        c = [s['desc'] for s in sig].index('MLII')
        x = lfilter(hb, ha, (raw[:, c] - sig[c]['zero']) / sig[c]['gain'])
        for j, p in enumerate(pos):
            a, z = p - MIT_PRE, p + MIT_POST
            ok = a >= 0 and z <= n
            w = x[a:z].reshape(-1, 2).mean(1) if ok else np.zeros((MIT_PRE + MIT_POST) // 2)
            R_.append(rec); POS.append(p); CLS.append(AAMI[sym[j]]); SYM.append(sym[j]); OK.append(ok); W.append(w.astype(np.float32))
            RR.append((p - pos[j - 1]) / MIT_FS if j > 0 else np.nan)
    rec_a = np.array(R_)
    for name, recs_ in (('DS1', DS1), ('DS2', DS2)):
        tot = np.sum([per_rec[r] for r in recs_], axis=0).tolist()
        print(f'{name}: annotated beats per AAMI class N S V F Q = {tot} (total {sum(tot)}); de Chazal 2004 Table II: '
              f'{DECHAZAL[name]} (total {sum(DECHAZAL[name])})')
    np.savez(MIT_NPZ, rec=rec_a, pos=np.array(POS), cls=np.array(CLS), sym=np.array(SYM), rr=np.array(RR, dtype=np.float32),
             w=np.stack(W), ok=np.array(OK), per_rec=json_dump(per_rec))
    return per_rec


def json_dump(d):
    import json
    return np.array(json.dumps(d))


def load_mitdb():
    d = np.load(MIT_NPZ)
    return {k: d[k] for k in d.files}


if __name__ == '__main__':
    {'sc': build_sc, 'sleep': build_sleep, 'sleep_psd': build_sleep_psd, 'mitdb': build_mitdb}[sys.argv[1]]()
