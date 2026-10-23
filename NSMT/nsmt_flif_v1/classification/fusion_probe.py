"""Fusion probe (read-only diagnostic, 2026-10-23; no training): does the OR fusion of the bidirectional Neo-mixer throw away
class information that is still present in the membrane potential of the last neuron layer of each direction?

For every trained CinC 2017 model (fold 0, best-validation weights = model_state/checkpoint.pth, see train_rhythm.py: it is
saved at every validation improvement and reloaded for the 'best' final test; best+model.pt is the last epoch):
  * capture, at the last layer of the forward stack blk.mixer and of the reverse stack blk.mixer_rev, the spikes F / R and the
    pre-threshold charge H (the value that is thresholded), stored as H - threshold, R and H_r flipped back to token order;
      members fire (LIF / f-LIF):  H_t = u_prev + (I_t - u_prev) / tau  (u_prev[0] = 0, u_prev[t] = state after step t-1)
      soma (population with soma): H_t = V_prev + (a_t - V_prev) / tau_s (V = soma after reset, a = soma drive, V_prev[0] = 0)
  * fail-closed checks on every batch (any failure aborts the model): (1) the spikes implied by H (arctan_spike: H - thr >= 0)
    equal the layer's output spikes exactly (per neuron part and assembled, both directions); (2) max(F, R) equals the m0 of
    forward_parts exactly; (3) the logits of the instrumented forward_parts equal the normal forward's to 1e-6; plus: every
    captured module ran exactly once, H finite, m0 binary;
  * token-mean features per window: or = mean m0, spk = [mean F; mean R], mem = [mean H_f; mean H_r],
    memxor = [mem; mean |H_f - H_r|], spkmem = [spk; mem], input = mean embedding spikes e, head = the model's head input;
  * linear probes (StandardScaler + LogisticRegression(class_weight='balanced', max_iter=3000)), C in {0.001, 0.01, 0.1, 1}
    chosen on a stratified 10% inner hold-out of the probe-train records (one centred window per training record) by the CinC
    score, refit on all probe-train records, evaluated on the validation records (all windows, window probabilities averaged
    per record as rhythm_eval.predict). The test split is never built.
usage: python fusion_probe.py --models all            (or cond:seed ...)   outputs results/fusion-probe-20261023/<cond>_seed<s>.json
       python fusion_probe.py --summarize                                   summary.csv / summary.txt
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional          # noqa: E402
from load_model import LOAD_MODEL                         # noqa: E402
from layers import NeoNeuron, GroupNeuron                 # noqa: E402
from stream_mixer import StreamNeurons                    # noqa: E402
import rhythm_data as rd                                   # noqa: E402
import rhythm_eval as re_                                  # noqa: E402

TAG = 'fusion-probe-20261023'
OUT_DIR = os.path.join(HERE, 'results', TAG)
CACHE_DIR = os.path.join(HERE, 'log', TAG)
LOG_ROOTS = [os.path.join(HERE, 'log', 'flif-v1-rhythm-cinc', 'CINC2017'), os.path.join(HERE, 'log', 'flif-v1-rhythm-bix-cinc', 'CINC2017')]
RAWS = [os.path.join(HERE, 'results', 'flif-v1-rhythm-cinc', 'raw.txt'), os.path.join(HERE, 'results', 'flif-v1-rhythm-bix-cinc', 'raw.txt')]
CONDS = {                                                  # cond -> (model, mixer) expected in config.pt
    'mix_lif_bidir': ('neoMixer', 'lif'), 'mix_flif_bidir': ('neoMixer', 'flif'),
    'mix_pop_bidir': ('neoMixer', 'pop_flif'), 'mix_pop_lif_bidir': ('neoMixer', 'pop_lif'),
    'mix_popg_lif_bidir': ('neoMixer', 'popg_lif'), 'mix_popg_flif_bidir': ('neoMixer', 'popg_flif'),
    'mix_popk_lif_bidir': ('streamMixer', 'popk_lif'), 'mix_popk_flif_bidir': ('streamMixer', 'popk_flif')}
SEEDS = (42, 7, 2026)
CS = (0.001, 0.01, 0.1, 1.0)
FSETS = ('or', 'spk', 'mem', 'memxor', 'spkmem', 'input', 'head')
BASE = ('Fm', 'Rm', 'Hf', 'Hr', 'Xor', 'Om', 'Em', 'feat')
COMPOSE = {'or': ('Om',), 'spk': ('Fm', 'Rm'), 'mem': ('Hf', 'Hr'), 'memxor': ('Hf', 'Hr', 'Xor'),
           'spkmem': ('Fm', 'Rm', 'Hf', 'Hr'), 'input': ('Em',), 'head': ('feat',)}
HIST_LO, HIST_HI, HIST_BINS = -40., 40., 800
LOGIT_TOL = 1e-6


class CheckFailed(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


SCRIPT_SHA = sha256(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------------------------ run lookup
def parse_result(line):
    return dict(t.split('=', 1) for t in line.strip().split(' ')[1:] if '=' in t)


def find_run(cond, seed):
    """exactly one outer dir with '+<cond>+lr0.001+seed<s>+', exactly one inner dir with logargs.txt + model_state files,
    exactly one RESULT line (cond, seed) over both raw.txt files whose dir= is that inner dir."""
    key = f'+{cond}+lr0.001+seed{seed}+'
    outer = [d for root in LOG_ROOTS for d in glob.glob(os.path.join(root, '*', '*')) if os.path.isdir(d) and key in os.path.basename(d)]
    if len(outer) != 1:
        raise CheckFailed(f'{cond} seed {seed}: {len(outer)} outer run dirs {outer}')
    inner = [d for d in glob.glob(os.path.join(outer[0], '*')) if os.path.isfile(os.path.join(d, 'logargs.txt'))]
    if len(inner) != 1:
        raise CheckFailed(f'{cond} seed {seed}: {len(inner)} inner run dirs')
    run = inner[0]
    for f in ('config.pt', 'checkpoint.pth', 'best+model.pt'):
        if not os.path.isfile(os.path.join(run, 'model_state', f)):
            raise CheckFailed(f'{run}: model_state/{f} missing')
    hits = []
    for raw in RAWS:
        for line in open(raw):
            if line.startswith('RESULT '):
                r = parse_result(line)
                if r.get('cond') == cond and r.get('seed') == str(seed):
                    hits.append((raw, r))
    if len(hits) != 1:
        raise CheckFailed(f'{cond} seed {seed}: {len(hits)} RESULT lines')
    raw, r = hits[0]
    if os.path.realpath(r.get('dir', '')) != os.path.realpath(run):
        raise CheckFailed(f'{cond} seed {seed}: RESULT dir {r.get("dir")} != {run}')
    if r.get('data') != 'CINC2017' or r.get('fold') != '0':
        raise CheckFailed(f'{cond} seed {seed}: RESULT data/fold {r.get("data")}/{r.get("fold")}')
    return run, raw, r


def check_config(cond, cfg):
    model, mixer = CONDS[cond]
    bad = []
    if cfg.get('model') != model: bad.append(f"model {cfg.get('model')} != {model}")
    if cfg.get('mixer') != mixer: bad.append(f"mixer {cfg.get('mixer')} != {mixer}")
    if cfg.get('mixer_bidir') is not True: bad.append(f"mixer_bidir {cfg.get('mixer_bidir')}")
    if cfg.get('mixer_comb', 'or') != 'or': bad.append(f"mixer_comb {cfg.get('mixer_comb')}")
    if int(cfg.get('num_layers', -1)) != 1: bad.append(f"num_layers {cfg.get('num_layers')}")
    if str(cfg.get('data')).upper() != 'CINC2017': bad.append(f"data {cfg.get('data')}")
    if int(cfg.get('fold', -1)) != 0: bad.append(f"fold {cfg.get('fold')}")
    if int(cfg.get('crop_tokens', 0) or 0) != 0: bad.append(f"crop_tokens {cfg.get('crop_tokens')}")
    if bad:
        raise CheckFailed(f'{cond}: config ' + '; '.join(bad))


# ------------------------------------------------------------------------------------------------------------- capture
def last_neuron(stack):
    """the neuron module of the last layer of a mixer stack and its NeoNeuron parts + concat axis."""
    if hasattr(stack, 'Mem'):                                  # neo_mixer.MixerStack of NeoLayers
        layer = stack.Mem[-1]
        if not layer.spikede:
            raise CheckFailed('last NeoLayer is not a spikeDE neuron layer')
        cont = layer.lif
    else:                                                      # stream_mixer._Stack of StreamSpikeLinear
        cont = stack.layers[-1].neuron
    if isinstance(cont, NeoNeuron):
        parts, axis = [cont], None
    elif isinstance(cont, GroupNeuron):
        if cont.kind not in ('lif', 'flif'):
            raise CheckFailed(f'GroupNeuron kind {cont.kind}')
        parts, axis = list(cont.parts), cont.axis
    elif isinstance(cont, StreamNeurons):
        parts, axis = list(cont.parts), 2
    else:
        raise CheckFailed(f'unknown last neuron {type(cont).__name__}')
    for p in parts:
        if not isinstance(p, NeoNeuron) or p.layout != 'ND':
            raise CheckFailed(f'part {type(p).__name__} is not NeoNeuron(ND)')
    return cont, parts, axis


class Capture:
    def __init__(self, model):
        blk = model.blocks[0]
        if not (blk.bidir and blk.comb == 'or'):
            raise CheckFailed('block is not bidirectional OR')
        self.on = False
        self.stacks = {'f': blk.mixer, 'r': blk.mixer_rev}
        self.parts, self.axis, self.info = {}, {}, {}
        self.part_out, self.stack_out, self.calls = {}, {}, {}
        for d, st in self.stacks.items():
            cont, parts, axis = last_neuron(st)
            self.parts[d], self.axis[d] = parts, axis
            self.info[d] = dict(container=type(cont).__name__, n_parts=len(parts), axis=axis,
                                parts=[self._describe(p.neuron) for p in parts])
            for i, p in enumerate(parts):
                self._wrap(p, (d, i))
            st.register_forward_hook(self._hook(d))

    @staticmethod
    def _describe(n):
        return dict(cls=type(n).__name__, size=int(n.size), has_soma=bool(n.has_soma), members_fire=bool(n.members_fire),
                    taus=[float(t) for t in n.tau.reshape(-1).tolist()], threshold=float(n.threshold),
                    alpha=float(getattr(n, 'alpha', 1.)), tau_s=(float(n.soma.tau_s) if n.has_soma else None),
                    soma_threshold=(float(n.soma.threshold) if n.has_soma else None))

    def _hook(self, d):
        def h(mod, inp, out):
            if self.on:
                self.calls[('stack', d)] = self.calls.get(('stack', d), 0) + 1
                self.stack_out[d] = out
        return h

    def _wrap(self, mod, key):
        cap = self

        def fwd(x):
            if not cap.on:
                return NeoNeuron.forward(mod, x)                     # the original code path
            if x.ndim != 4 or mod.layout != 'ND':
                raise CheckFailed(f'{key}: unexpected input {tuple(x.shape)} / layout {mod.layout}')
            T, B, N1, D = x.shape
            cur = x.reshape(T, B * N1, D)                            # as NeoNeuron('ND')
            n = mod.neuron
            if n.has_soma:
                out, aux = n(cur, return_aux=True, analog='drive')
                V, a = aux['voltage'], aux['analog']                 # soma after reset, soma drive: [T, B', D]
                Vp = torch.cat([torch.zeros_like(V[:1]), V[:-1]], 0)
                Hm = (Vp + (a - Vp) / n.soma.tau_s) - n.soma.threshold   # Soma.forward: charge - threshold
            else:
                if not (n.members_fire and n.num_members == 1):
                    raise CheckFailed(f'{key}: members_fire {n.members_fire} K {n.num_members}')
                out, aux = n(cur, return_aux=True)
                u = aux['state']                                     # state after each step: [T, B', D, 1]
                up = torch.cat([torch.zeros_like(u[:1]), u[:-1]], 0)
                drift = (cur.unsqueeze(-1) - up) / n.tau             # LIFNeuron.rhs
                look = up + 1. * drift
                Hm = (look - n.threshold).reshape(T, B * N1, -1)
            del aux
            s = out.reshape(T, B, N1, D)
            Hm = Hm.reshape(T, B, N1, D)
            cap.calls[key] = cap.calls.get(key, 0) + 1
            cap.part_out[key] = (s, Hm)
            return s

        mod.forward = fwd                                            # instance attribute, the class is untouched

    def start(self):
        self.part_out, self.stack_out, self.calls = {}, {}, {}
        self.on = True

    def collect(self):
        """-> {d: (S, Hm)} in stack output order (R / H_r still reversed); checks (1) and the call counts."""
        self.on = False
        res = {}
        for d in ('f', 'r'):
            if self.calls.get(('stack', d)) != 1:
                raise CheckFailed(f'stack {d} ran {self.calls.get(("stack", d))} times')
            Ss, Hs = [], []
            for i, p in enumerate(self.parts[d]):
                if self.calls.get((d, i)) != 1:
                    raise CheckFailed(f'part {d}{i} ran {self.calls.get((d, i))} times')
                s, Hm = self.part_out[(d, i)]
                if not torch.isfinite(Hm).all():
                    raise CheckFailed(f'part {d}{i}: non-finite H')
                if not torch.equal((Hm >= 0).to(s.dtype), s):        # (1) per part: arctan_spike fires at >= 0
                    raise CheckFailed(f'check1 part {d}{i}: spikes implied by H differ in {int(((Hm >= 0).to(s.dtype) != s).sum())} entries')
                Ss.append(s); Hs.append(Hm)
            ax = self.axis[d]
            S = Ss[0] if len(Ss) == 1 else torch.cat(Ss, dim=ax)
            H = Hs[0] if len(Hs) == 1 else torch.cat(Hs, dim=ax)
            if not torch.equal(S, self.stack_out[d]):                # (1) assembled = the stack's output spikes
                raise CheckFailed(f'check1 stack {d}: assembled spikes differ from the stack output')
            res[d] = (S, H)
        return res


# -------------------------------------------------------------------------------------------------------- statistics
class HStats:
    def __init__(self):
        self.n = 0; self.fire = 0; self.near = 0; self.s = 0.; self.ss = 0.; self.mn = float('inf'); self.mx = -float('inf')
        self.hist = torch.zeros(HIST_BINS, dtype=torch.float64); self.below = 0; self.above = 0

    def add(self, Hm):
        h = Hm.float()
        self.n += h.numel(); self.fire += int((h >= 0).sum()); self.near += int(((h > -.5) & (h < 0)).sum())
        self.s += float(h.double().sum()); self.ss += float(h.double().square().sum())
        self.mn = min(self.mn, float(h.min())); self.mx = max(self.mx, float(h.max()))
        self.hist += torch.histc(h, bins=HIST_BINS, min=HIST_LO, max=HIST_HI).double().cpu()
        self.below += int((h < HIST_LO).sum()); self.above += int((h > HIST_HI).sum())

    def summary(self):
        mean = self.s / self.n
        c = np.concatenate([[self.below], self.hist.numpy(), [self.above]]).cumsum() / self.n
        edges = np.concatenate([[HIST_LO], np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)[1:], [np.inf]])
        q = {f'q{int(p * 1000):03d}': float(edges[int(np.searchsorted(c, p))]) for p in (.001, .01, .05, .25, .5, .75, .95, .99, .999)}
        return dict(n=self.n, fire_rate=self.fire / self.n, near_share=self.near / self.n, mean=mean,
                    std=float(np.sqrt(max(self.ss / self.n - mean * mean, 0.))), min=self.mn, max=self.mx,
                    share_below_hist=self.below / self.n, share_above_hist=self.above / self.n,
                    quantiles_upper_bin_edge=q)


# --------------------------------------------------------------------------------------------------------- extraction
@torch.no_grad()
def extract(model, cap, ds, bs, name):
    feats = {k: [] for k in BASE}
    ys, us = [], []
    st = {d: HStats() for d in ('f', 'r')}
    st_parts = {(d, i): HStats() for d in ('f', 'r') for i in range(len(cap.parts[d]))}
    max_logit_diff = 0.
    nb = 0
    functional.reset_net(model)
    for bi, (x, y, u) in enumerate(rd.loader(ds, bs, shuffle=False)):
        x = x.float().cuda()
        z_ref = re_._logits(model, x)                                # normal forward (capture off)
        functional.reset_net(model)
        cap.start()
        z, e, m0, feat, h = model.forward_parts(x)                   # instrumented
        functional.reset_net(model)
        res = cap.collect()                                          # (1)
        F, Hf = res['f']
        R, Hr = res['r'][0].flip(0), res['r'][1].flip(0)             # reverse stack -> token order
        if not torch.equal(torch.maximum(F, R), m0):                 # (2)
            raise CheckFailed(f'{name} batch {bi}: check2 max(F, R) != m0')
        if not bool(((m0 == 0) | (m0 == 1)).all()):
            raise CheckFailed(f'{name} batch {bi}: m0 not binary')
        dz = float((z.float() - z_ref.float()).abs().max())
        if not np.isfinite(dz) or dz > LOGIT_TOL:                    # (3)
            raise CheckFailed(f'{name} batch {bi}: check3 logits differ by {dz}')
        max_logit_diff = max(max_logit_diff, dz)
        fl = lambda t: t.mean(0).flatten(1).float().cpu().numpy()   # token mean, stream / N' axis flattened in
        feats['Fm'].append(fl(F)); feats['Rm'].append(fl(R)); feats['Hf'].append(fl(Hf)); feats['Hr'].append(fl(Hr))
        feats['Xor'].append(fl((Hf - Hr).abs())); feats['Om'].append(fl(m0)); feats['Em'].append(fl(e))
        feats['feat'].append(feat.reshape(feat.shape[0], -1).float().cpu().numpy())
        ys.append(y.numpy()); us.append(u.numpy())
        st['f'].add(Hf); st['r'].add(Hr)
        for (d, i), s_ in st_parts.items():
            s_.add(cap.part_out[(d, i)][1])
        cap.part_out, cap.stack_out = {}, {}
        nb += 1
    out = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise CheckFailed(f'{name}: non-finite feature {k}')
    out['y'] = np.concatenate(ys).astype(np.int64); out['u'] = np.concatenate(us).astype(np.int64)
    stats = {d: st[d].summary() for d in st}
    stats['parts'] = {f'{d}{i}': st_parts[(d, i)].summary() for (d, i) in st_parts}
    return out, stats, dict(batches=nb, max_logit_abs_diff=max_logit_diff)


# -------------------------------------------------------------------------------------------------------------- probes
def aggregate(p, u, y):
    """per-record mean of window probabilities, as rhythm_eval.predict."""
    uniq, inv = np.unique(u, return_inverse=True)
    P = np.zeros((len(uniq), p.shape[1])); np.add.at(P, inv, p)
    P /= np.bincount(inv).astype(float)[:, None]
    Y = np.zeros(len(uniq), dtype=int); Y[inv] = y
    return P, Y


def fit(X, y, C):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        pipe.fit(X, y)
    lr = pipe[-1]
    if list(lr.classes_) != [0, 1, 2, 3]:
        raise CheckFailed(f'classes {lr.classes_}')
    n_iter = int(np.max(lr.n_iter_))
    warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
    return pipe, dict(n_iter=n_iter, converged=bool(n_iter < 3000 and not warned))


def probe(tr, va, fset):
    from sklearn.model_selection import train_test_split
    Xtr = np.concatenate([tr[k] for k in COMPOSE[fset]], 1).astype(np.float64)
    Xva = np.concatenate([va[k] for k in COMPOSE[fset]], 1).astype(np.float64)
    ytr = tr['y']
    i_in, i_ho = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=0)
    inner = []
    for C in CS:
        pipe, cv = fit(Xtr[i_in], ytr[i_in], C)
        m, _ = re_.metrics(pipe.predict_proba(Xtr[i_ho]), ytr[i_ho], 'CINC2017')
        inner.append(dict(C=C, score=m['score'], macro_f1=m['macro_f1'], acc=m['acc'], **cv))
    best = max(range(len(CS)), key=lambda j: (inner[j]['score'], -j))   # ties -> the smaller C
    C = CS[best]
    pipe, cv = fit(Xtr, ytr, C)
    P, Y = aggregate(pipe.predict_proba(Xva), va['u'], va['y'])
    m, conf = re_.metrics(P, Y, 'CINC2017')
    return dict(dim=int(Xtr.shape[1]), C=C, inner_score=inner[best]['score'], val_score=m['score'], val_macro_f1=m['macro_f1'],
                val_acc=m['acc'], val_f1=[m[f'f1_{i}'] for i in range(4)], n_val_records=int(len(Y)), refit=cv, inner=inner,
                val_confusion=conf.tolist())


# --------------------------------------------------------------------------------------------------------- one model
def run_model(cond, seed, bs=None, reuse_cache=False):
    t0 = time.time()
    name = f'{cond}_seed{seed}'
    rec = dict(name=name, cond=cond, seed=seed, script=os.path.abspath(__file__), script_sha256=SCRIPT_SHA, status='started',
               host=platform.node(), torch=torch.__version__, cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        import sklearn
        rec['sklearn'] = sklearn.__version__
        run, raw, result = find_run(cond, seed)
        rec.update(run_dir=run, raw_file=raw, result_line_score_test=float(result['score']), result_best_epoch_val=result.get('best_epoch_val'))
        cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
        check_config(cond, cfg)
        ckpt = os.path.join(run, 'model_state', 'checkpoint.pth')
        rec.update(checkpoint='model_state/checkpoint.pth', checkpoint_sha256=sha256(ckpt),
                   checkpoint_reason='train_rhythm.py saves checkpoint.pth at every validation-score improvement and reloads it for the '
                                     "'best' final test; best+model.pt is the last epoch",
                   cfg=dict(model=cfg['model'], mixer=cfg['mixer'], mixer_bidir=cfg['mixer_bidir'], mixer_comb=cfg.get('mixer_comb', 'or (absent)'),
                            num_layers=cfg['num_layers'], time_num_layers=cfg['time_num_layers'], embed_dim=cfg['embed_dim'],
                            patch_size=cfg['patch_size'], neo_alpha=cfg.get('neo_alpha'), pop_taus=cfg.get('pop_taus'),
                            pop_soma_init=cfg.get('pop_soma_init'), best_epoch=cfg.get('best_epoch'), best_val_score=cfg.get('best_val_score'),
                            eval_bs=cfg.get('eval_bs')))
        args = argparse.Namespace(**cfg)
        bs_ref = int(getattr(args, 'eval_bs', 32) or 32)                 # the model's own val metrics: the training-time eval batch
        bs = int(bs or 128)                                              # feature extraction (per-sample in eval mode; larger = faster)
        rec['batch_size_extract'], rec['batch_size_model_val'] = bs, bs_ref
        # data (the test split is never built)
        sig, off, labels, names = rd.cinc_cache(args.fs_target)
        tr, va, te = rd.cinc_split(labels, int(args.fold), float(args.val_frac))
        if tuple(cfg['data_info']['split_sizes']) != (len(tr), len(va), len(te)):
            raise CheckFailed(f"split sizes {(len(tr), len(va), len(te))} != config {cfg['data_info']['split_sizes']}")
        if np.intersect1d(np.concatenate([tr, va]), te).size or np.intersect1d(tr, va).size:
            raise CheckFailed('split overlap')
        win, hop = int(round(args.win_s * args.fs_target)), int(round(args.hop_s * args.fs_target))
        if win != int(args.seq_len):
            raise CheckFailed(f'win {win} != seq_len {args.seq_len}')
        ds_tr = rd.CincEval(sig, off, labels, tr, win, hop, single=True)
        ds_va = rd.CincEval(sig, off, labels, va, win, hop)
        rec.update(win=win, hop=hop, n_probe_train_records=int(len(tr)), n_probe_train_windows=len(ds_tr),
                   n_val_records=int(len(va)), n_val_windows=len(ds_va))
        # model
        model = LOAD_MODEL[args.model](args, True)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model = model.cuda().eval(); model.train_mode = 'testing'
        functional.reset_net(model)
        rec['tokens'] = int(model.num_patches)
        tm = time.time()
        P, Y, _ = re_.predict(model, ds_va, 'cuda:0', bs=bs_ref)             # the model's own validation metrics (no capture)
        mv, _ = re_.metrics(P, Y, 'CINC2017')
        rec['model_val'] = dict(score=mv['score'], macro_f1=mv['macro_f1'], acc=mv['acc'], f1=[mv[f'f1_{i}'] for i in range(4)], n=mv['n'])
        d_ref = abs(mv['score'] - float(cfg['best_val_score']))
        rec['check_ref_val'] = dict(model_val_score=mv['score'], config_best_val_score=float(cfg['best_val_score']), abs_diff=d_ref,
                                    passed=bool(d_ref <= 1e-6))
        rec['seconds_model_val'] = time.time() - tm
        cache = os.path.join(CACHE_DIR, f'{name}.npz')
        cap = Capture(model)
        rec['capture'] = cap.info
        if reuse_cache and os.path.isfile(cache):
            z = np.load(cache, allow_pickle=False)
            if str(z['ckpt_sha256']) != rec['checkpoint_sha256']:
                raise CheckFailed('cache checkpoint sha mismatch')
            ftr = {k: z['tr_' + k] for k in BASE + ('y', 'u')}; fva = {k: z['va_' + k] for k in BASE + ('y', 'u')}
            meta = json.loads(str(z['meta']))
            rec.update(h_stats=meta['h_stats'], checks=meta['checks'], seconds_extract=meta['seconds_extract'], cache_reused=True)
        else:
            te_ = time.time()
            ftr, s_tr, c_tr = extract(model, cap, ds_tr, bs, name + ' probe-train')
            fva, s_va, c_va = extract(model, cap, ds_va, bs, name + ' val')
            rec['seconds_extract'] = time.time() - te_
            rec['h_stats'] = dict(probe_train=s_tr, val=s_va)
            rec['checks'] = dict(check1_spikes_from_H='passed', check2_or_equals_m0='passed', check3_logits_tol=LOGIT_TOL,
                                 check3_max_logit_abs_diff=max(c_tr['max_logit_abs_diff'], c_va['max_logit_abs_diff']),
                                 batches=c_tr['batches'] + c_va['batches'], passed=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache, ckpt_sha256=np.array(rec['checkpoint_sha256']),
                     meta=np.array(json.dumps(dict(h_stats=rec['h_stats'], checks=rec['checks'], seconds_extract=rec['seconds_extract']))),
                     **{'tr_' + k: v for k, v in ftr.items()}, **{'va_' + k: v for k, v in fva.items()})
            rec['cache'] = cache
        if fva['u'].size != len(ds_va) or len(np.unique(fva['u'])) != len(va) or ftr['u'].size != len(tr):
            raise CheckFailed('window / record counts')
        if not np.array_equal(np.sort(np.unique(fva['u'])), np.sort(va)) or not np.array_equal(np.sort(ftr['u']), np.sort(tr)):
            raise CheckFailed('records differ from the split')
        rec['firing'] = {s: dict(F=rec['h_stats'][s]['f']['fire_rate'], R=rec['h_stats'][s]['r']['fire_rate'],
                                 or_rate=float((ftr if s == 'probe_train' else fva)['Om'].mean())) for s in ('probe_train', 'val')}
        rec['near_threshold'] = {s: dict(F=rec['h_stats'][s]['f']['near_share'], R=rec['h_stats'][s]['r']['near_share'])
                                 for s in ('probe_train', 'val')}
        del model, cap
        torch.cuda.empty_cache()
        tp = time.time()
        rec['probes'] = {}
        for fs in FSETS:
            tf = time.time()
            rec['probes'][fs] = probe(ftr, fva, fs)
            rec['probes'][fs]['seconds'] = time.time() - tf
        rec['seconds_probe'] = time.time() - tp
        rec['status'] = 'ok'
    except CheckFailed as ex:
        rec['status'] = 'aborted'; rec['abort_reason'] = str(ex)
    rec['seconds'] = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f'{name}.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    if rec['status'] == 'ok':
        p = rec['probes']
        print(f"[{time.strftime('%H:%M:%S')}] {name} ok  model {rec['model_val']['score']:.4f} (ref diff {rec['check_ref_val']['abs_diff']:.2e})  "
              + ' '.join(f"{k}={p[k]['val_score']:.4f}" for k in FSETS)
              + f"  F {rec['firing']['val']['F']:.4f} R {rec['firing']['val']['R']:.4f}  dz {rec['checks']['check3_max_logit_abs_diff']:.1e}"
              + f"  extract {rec.get('seconds_extract', 0):.0f}s probe {rec['seconds_probe']:.0f}s total {rec['seconds']:.0f}s", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {name} ABORTED: {rec['abort_reason']}  ({rec['seconds']:.0f}s)", flush=True)
    return rec


# ------------------------------------------------------------------------------------------------------------ summary
def summarize():
    recs = {}
    for c in CONDS:
        for s in SEEDS:
            p = os.path.join(OUT_DIR, f'{c}_seed{s}.json')
            recs[(c, s)] = json.load(open(p)) if os.path.isfile(p) else None
    ok = lambda r: r is not None and r.get('status') == 'ok'

    def val(c, s, fs, key='val_score'):                                # fail closed: missing / aborted / non-finite -> nan
        r = recs[(c, s)]
        if not ok(r):
            return float('nan')
        v = r['model_val'][{'val_score': 'score', 'val_macro_f1': 'macro_f1', 'val_acc': 'acc'}[key]] if fs == 'model' else r['probes'][fs][key]
        return float(v) if v is not None and np.isfinite(v) else float('nan')

    def mean3(v):
        return float(np.mean(v)) if len(v) == len(SEEDS) and all(np.isfinite(v)) else float('nan')

    rows = []
    for c in CONDS:
        for fs in ('model',) + FSETS:
            sc = [val(c, s, fs) for s in SEEDS]; mf = [val(c, s, fs, 'val_macro_f1') for s in SEEDS]
            row = dict(cond=c, feature_set=fs, n_ok=int(sum(np.isfinite(sc))), mean_val_score=mean3(sc), mean_val_macro_f1=mean3(mf))
            for s, a, b in zip(SEEDS, sc, mf):
                row[f'val_score_seed{s}'] = a; row[f'val_macro_f1_seed{s}'] = b
            if fs != 'model':
                for s in SEEDS:
                    r = recs[(c, s)]
                    row[f'C_seed{s}'] = r['probes'][fs]['C'] if ok(r) else ''
                    row[f'dim_seed{s}'] = r['probes'][fs]['dim'] if ok(r) else ''
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT_DIR, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()})
    pairs = [('mem', 'spk'), ('memxor', 'spk'), ('spkmem', 'spk'), ('spk', 'or'), ('or', 'input'), ('head', 'model')]
    L = [f'fusion probe summary ({TAG}); CinC 2017 fold 0, validation records; score = mean F1 of N / AF / Other; x100 = score points',
         f'script sha256 {SCRIPT_SHA}', 'seeds ' + ', '.join(map(str, SEEDS)) + '; a missing or aborted seed makes the mean nan', '']
    for c in CONDS:
        L.append(f'== {c}  (ok seeds {sum(ok(recs[(c, s)]) for s in SEEDS)}/{len(SEEDS)})')
        for fs in ('model',) + FSETS:
            sc = [val(c, s, fs) for s in SEEDS]
            dims = [recs[(c, s)]['probes'][fs]['dim'] for s in SEEDS if ok(recs[(c, s)])] if fs != 'model' else []
            Cs = [recs[(c, s)]['probes'][fs]['C'] for s in SEEDS if ok(recs[(c, s)])] if fs != 'model' else []
            L.append(f'  {fs:7s} val score mean {100 * mean3(sc):6.2f}   per seed ' + ' '.join(f'{100 * v:6.2f}' for v in sc)
                     + (f'   dim {dims[0]}  C {Cs}' if dims else ''))
        L.append('  paired differences (x100; mean | per seed ' + '/'.join(map(str, SEEDS)) + ')')
        for a, b in pairs:
            d = [val(c, s, a) - val(c, s, b) for s in SEEDS]
            L.append(f'    {a:>6s} - {b:<6s} {100 * mean3(d):+6.2f} | ' + ' '.join(f'{100 * v:+6.2f}' for v in d))
        fr = lambda k, s2, d2: [recs[(c, s)][k]['val'][d2] if ok(recs[(c, s)]) else float('nan') for s in SEEDS]
        L.append('  val firing rate F ' + ' '.join(f'{v:.4f}' for v in fr('firing', 'val', 'F')) + '   R ' + ' '.join(f'{v:.4f}' for v in fr('firing', 'val', 'R'))
                 + '   OR ' + ' '.join(f'{v:.4f}' for v in fr('firing', 'val', 'or_rate')))
        L.append('  val near-threshold share (-0.5 < H - thr < 0) F ' + ' '.join(f'{v:.4f}' for v in fr('near_threshold', 'val', 'F'))
                 + '   R ' + ' '.join(f'{v:.4f}' for v in fr('near_threshold', 'val', 'R')))
        hs = [(recs[(c, s)]['h_stats']['val']['f'], recs[(c, s)]['h_stats']['val']['r']) for s in SEEDS if ok(recs[(c, s)])]
        if hs:
            L.append('  val H - thr  F mean/std/min/max ' + ' | '.join(f"{h[0]['mean']:.2f}/{h[0]['std']:.2f}/{h[0]['min']:.1f}/{h[0]['max']:.1f}" for h in hs)
                     + '   R ' + ' | '.join(f"{h[1]['mean']:.2f}/{h[1]['std']:.2f}/{h[1]['min']:.1f}/{h[1]['max']:.1f}" for h in hs))
        nc = [(fs, s) for s in SEEDS if ok(recs[(c, s)]) for fs in FSETS
              if not recs[(c, s)]['probes'][fs]['refit']['converged']]
        L.append('  refit not converged (max_iter 3000): ' + (', '.join(f'{fs}@{s}' for fs, s in nc) if nc else 'none'))
        ref = [recs[(c, s)]['check_ref_val']['abs_diff'] if ok(recs[(c, s)]) else float('nan') for s in SEEDS]
        L.append('  |model val score - config best_val_score| ' + ' '.join(f'{v:.2e}' for v in ref))
        bad = [f"{s}: {recs[(c, s)]['abort_reason'] if recs[(c, s)] else 'missing'}" for s in SEEDS if not ok(recs[(c, s)])]
        if bad:
            L.append('  NOT OK: ' + '; '.join(bad))
        L.append('')
    open(os.path.join(OUT_DIR, 'summary.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=[], help="'all' or cond:seed ...")
    ap.add_argument('--bs', type=int, default=128, help='feature-extraction batch size (the model val reference uses the config eval_bs)')
    ap.add_argument('--reuse_cache', action='store_true')
    ap.add_argument('--skip_done', action='store_true', help='skip models whose json has status ok')
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    todo = [(c, s) for c in CONDS for s in SEEDS] if a.models == ['all'] else [(m.split(':')[0], int(m.split(':')[1])) for m in a.models]
    for c, _ in todo:
        if c not in CONDS:
            raise SystemExit(f'unknown condition {c}')
    t0 = time.time()
    for i, (c, s) in enumerate(todo):
        p = os.path.join(OUT_DIR, f'{c}_seed{s}.json')
        if a.skip_done and os.path.isfile(p) and json.load(open(p)).get('status') == 'ok':
            print(f'{c}_seed{s} done, skipped', flush=True); continue
        print(f'[{i + 1}/{len(todo)}] {c} seed {s}', flush=True)
        run_model(c, s, bs=a.bs or None, reuse_cache=a.reuse_cache)
    if todo:
        print(f'all {len(todo)} models in {time.time() - t0:.0f}s', flush=True)
    if a.summarize:
        summarize()


if __name__ == '__main__':
    main()
