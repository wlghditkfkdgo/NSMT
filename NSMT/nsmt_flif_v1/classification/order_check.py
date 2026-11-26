"""MODI 2026-11-26: order / accumulation check with diagnostic reference models (not result-table baselines; docs/MODI_PROJECT_LOG.md
2026-11-26, user decision 11-26). Does the TASK need the order of its tokens (B), and does it need a long span of input (A)?

(B) the same model is trained and tested with the tokens of every sample in their original order ('orig') or shuffled ('shuf': a new
    random permutation for every sample at every draw, in training, validation and test). The tokens themselves are unchanged, so the
    difference is the value of the token order for that model. SLEEPCTX shuffles only the context epochs; the target epoch stays last.
(A) the model is trained and tested on a shorter input: Speech Commands the centre k tokens, SLEEPCTX the target with L - 1 epochs before it.
Report only: a model trained in the original order is also tested with shuffled tokens (5 draws; the PPT-style "Permuted" test).

data / tokens (token sizes give 10-40 tokens per sample, the PPT rule, except SLEEPCTX where a token is one 30-s epoch):
  SC        log-mel 40 x 100 frames -> 25 tokens of 4 frames (40 ms), token = 160 values; accuracy (official split)
  MITBIH    187 points -> 11 tokens of 17 points; macro-F1 (train_7 / test_3, validation 10 % of train)
  SLEEPIN   one 30-s epoch -> 30 tokens of 1 s (100 points); macro-F1 and accuracy (subject 5-fold, oc_data.sleep_fold)
  SLEEPCTX  the target epoch and the L - 1 raw epochs before it (L = 16 / 4 / 1), token = one epoch (3000 points); macro-F1 and accuracy
models: token encoder (MLP for SC / MIT-BIH, small CNN for 1-s EEG tokens, TinySleepNet-style CNN for 30-s epochs) + either
  gru: 2-layer unidirectional GRU (128), read the last step, or
  tf : 2-layer Transformer encoder (128, 4 heads, pre-norm) with a learned absolute position embedding, mean over tokens (SLEEPCTX: the
       last token = the target epoch).
training: AdamW lr 1e-3, weight decay 1e-2, 1 epoch linear warm-up then cosine to 0 (per step), batch 256 (SLEEPCTX 64), 30 epochs, plain
  cross entropy, gradient clip 1.0; validation every epoch; the best validation epoch (SC accuracy, others macro-F1) is tested once.
logging as NSMT/neorecall_v1 (AGENTS.md): <log_dir>/{logargs.txt, log/best_log_0.csv, log/final+result.csv, log/train_0, log/val_0,
  model_state/config.pt, model_state/best+model.pt}; the runner (scripts/run_oc.sh) writes the RESULT line.
usage: python order_check.py --ds SC --cond gru_orig --seed 42 --log_dir <dir>   [--fold f for SLEEP*] [--epochs N] [--max_batches N]
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

import oc_data as D

NCLS = {'SC': 35, 'MITBIH': 5, 'SLEEPIN': 5, 'SLEEPCTX': 5, 'SLEEPPSD': 5, 'MBIP': 4}   # MODI 2026-11-26 (evening): + SLEEPPSD, MBIP
CTXL = {'SLEEPCTX': 16, 'SLEEPPSD': 16, 'MBIP': 17}    # levels whose condition carries _L<L> (L = tokens incl. the target)
MB_VAL = [201, 118, 223, 114]   # MBIP validation records: DS1 ranked by S beats, ranks 2 / 4 / 6 / 8 (fixed before training)
EVAL_SEED = 12345                                     # permutations of validation / test draws (fixed, so evaluation is reproducible)


def parse_cond(ds, cond):
    """gru_orig | gru_shuf | tf_orig | tf_shuf, + _k<k> (SC) or _L<L> (SLEEPCTX / SLEEPPSD / MBIP, required);
    MBIP also gru_orig_L1_rr (the current beat + the mean pre-RR of the 16 beats before it, MODI 2026-11-26 evening)."""
    p = cond.split('_')
    rr = ds == 'MBIP' and len(p) == 4 and p[3] == 'rr'
    if rr:
        p = p[:3]
    if p[0] not in ('gru', 'tf') or p[1] not in ('orig', 'shuf') or len(p) > 3:
        raise SystemExit(f'bad cond {cond}')
    c = dict(model=p[0], order=p[1], k=None, L=None)
    if ds == 'MBIP':
        c['rr'] = rr
    if len(p) == 3:
        if p[2].startswith('k') and ds == 'SC':
            c['k'] = int(p[2][1:])
        elif p[2].startswith('L') and ds in CTXL:
            c['L'] = int(p[2][1:])
        else:
            raise SystemExit(f'bad cond {cond} for {ds}')
    if ds in CTXL and c['L'] is None:
        raise SystemExit(f'{ds} needs _L<L>')
    if c['L'] is not None and not 1 <= c['L'] <= CTXL[ds]:
        raise SystemExit(f'L must be 1..{CTXL[ds]}')
    if rr and c['L'] != 1:
        raise SystemExit('_rr only with L1')
    if c['k'] is not None and not 1 <= c['k'] <= 25:
        raise SystemExit('k must be 1..25')
    return c


# ---------------------------------------------------------------- data on the GPU
class Data:
    """holds every sample on the device; tokens(idx) -> [B, N, F] float tensor in the original order."""

    def __init__(self, ds, c, fold, dev):
        self.ds, self.c, self.dev = ds, c, dev
        if ds == 'SC':
            x, y, s = D.load_sc()
            x = torch.from_numpy(x[:, :, :100].astype(np.float32))                    # [n, 40, 100] (last frame dropped)
            tr = torch.from_numpy(s == 0)
            m, sd = x[tr].mean((0, 2), keepdim=True), x[tr].std((0, 2), keepdim=True)
            x = ((x - m) / sd).transpose(1, 2).reshape(len(x), 25, 4 * 40)          # token = 4 frames x 40 bands, time-major
            if c['k']:
                a = (25 - c['k']) // 2
                x = x[:, a:a + c['k']]
            self.x, self.y = x.to(dev), torch.from_numpy(y).to(dev)
            self.split = {k: np.where(s == v)[0] for k, v in (('train', 0), ('val', 1), ('test', 2))}
        elif ds == 'MITBIH':
            xtr, ytr, val, xte, yte = D.load_mitbih()
            x = np.concatenate([xtr, xte]).reshape(-1, 11, 17)
            self.x, self.y = torch.from_numpy(x).to(dev), torch.from_numpy(np.concatenate([ytr, yte])).to(dev)
            n = len(xtr)
            self.split = {'train': np.where(~val)[0], 'val': np.where(val)[0], 'test': n + np.arange(len(xte))}
        elif ds == 'SLEEPPSD':                                                         # MODI 2026-11-26 (evening): epoch spectra
            _, lab, tgt, subj, rec, off = D.load_sleep()
            psd = D.load_sleep_psd()
            t = np.where(tgt)[0]
            trm, vam, tem = D.sleep_fold(subj[t], fold)
            mu, sd = psd[t[trm]].mean(0, keepdims=True), psd[t[trm]].std(0, keepdims=True)   # per bin, training target epochs
            self.xe = torch.from_numpy((psd - mu) / sd).to(dev)                         # [n_epochs, 60]
            self.t, self.y = t, torch.from_numpy(lab).to(dev)[torch.from_numpy(t).to(dev)]
            self.split = {'train': np.where(trm)[0], 'val': np.where(vam)[0], 'test': np.where(tem)[0]}
            ctx = t[:, None] - np.arange(c['L'] - 1, -1, -1)[None]
            if (rec[ctx] != rec[t][:, None]).any():
                raise SystemExit('context crosses a night boundary')
            self.ctx = torch.from_numpy(ctx).to(dev)
        elif ds == 'MBIP':                                                             # MODI 2026-11-26 (evening): MIT-BIH beats
            m = D.load_mitdb()
            rec, cls, rr, ok = m['rec'], m['cls'], m['rr'], m['ok']
            K = D.MIT_K
            idx_in_rec = np.zeros(len(rec), dtype=np.int64)
            for r in np.unique(rec):
                w = np.where(rec == r)[0]; idx_in_rec[w] = np.arange(len(w))
            okp = np.array([ok[max(0, i - K):i + 1].all() for i in range(len(rec))])        # this beat and the 16 before it in the signal
            tgt = (idx_in_rec >= K + 1) & (cls <= 3) & okp & np.isin(rec, D.DS1 + D.DS2)
            t = np.where(tgt)[0]
            if (rec[t - K] != rec[t]).any() or np.isnan(rr[t[:, None] - np.arange(K + 1)[None]]).any():
                raise SystemExit('MBIP context crosses a record or lacks a pre-RR')
            trr = np.isin(rec[t], D.DS1) & ~np.isin(rec[t], MB_VAL)
            # each beat window standardised by its own mean and std (no patient or record statistics; chosen after a validation-only
            # probe in which one global scale left the models unable to find S in unseen patients, docs/MODI_PROJECT_LOG.md 11-26)
            w = m['w']; wz = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-6)
            B = np.concatenate([wz, np.nan_to_num(rr)[:, None]], 1).astype(np.float32)   # token = 117 points + pre-RR (s)
            self.B = torch.from_numpy(B).to(dev)
            self.mrr = torch.from_numpy(np.array([rr[i - K:i].mean() for i in t], dtype=np.float32)).to(dev)   # mean pre-RR, 16 beats before
            self.t, self.y, self.rec_t = t, torch.from_numpy(cls[t]).to(dev), rec[t]
            self.split = {'train': np.where(trr)[0], 'val': np.where(np.isin(rec[t], MB_VAL))[0], 'test': np.where(np.isin(rec[t], D.DS2))[0]}
            self.ctx = torch.from_numpy(t[:, None] - np.arange(c['L'] - 1, -1, -1)[None]).to(dev)
            cnt = np.bincount(cls[t[trr]], minlength=4).astype(np.float64)
            self.cw = torch.tensor(cnt.sum() / (4 * cnt), dtype=torch.float32, device=dev)   # balanced class weights (training)
        else:
            x, lab, tgt, subj, rec, off = D.load_sleep()
            t = np.where(tgt)[0]
            trm, vam, tem = D.sleep_fold(subj[t], fold)
            mu, sd = x[t[trm]].mean(), x[t[trm]].std()
            self.xe = torch.from_numpy((x - mu) / sd).to(dev)                          # [n_epochs, 3000]
            self.lab, self.rec = torch.from_numpy(lab).to(dev), rec
            self.t = t                                                                 # sample i = target epoch t[i]
            self.y = self.lab[torch.from_numpy(t).to(dev)]
            self.split = {'train': np.where(trm)[0], 'val': np.where(vam)[0], 'test': np.where(tem)[0]}
            L = c['L'] or 1
            ctx = t[:, None] - np.arange(L - 1, -1, -1)[None]                          # [n, L], last column = target
            if (rec[ctx] != rec[t][:, None]).any():
                raise SystemExit('context crosses a night boundary')
            self.ctx = torch.from_numpy(ctx).to(dev)

    def tokens(self, idx):
        i = torch.as_tensor(idx, device=self.dev)
        if self.ds in ('SC', 'MITBIH'):
            return self.x[i], self.y[i]
        if self.ds == 'SLEEPIN':
            return self.xe[self.ctx[i, -1]].reshape(len(i), 30, 100), self.y[i]
        if self.ds == 'MBIP':
            x = self.B[self.ctx[i]]                                                    # [B, L, 118]
            if self.c['rr']:
                x = torch.cat([x, self.mrr[i].view(-1, 1, 1)], -1)                     # [B, 1, 119]
            return x, self.y[i]
        return self.xe[self.ctx[i]], self.y[i]                                         # [B, L, 3000] (SLEEPPSD: [B, L, 60])


def shuffle_tokens(x, gen, keep_last):
    """new random permutation of the token axis for every sample; keep_last: the last token stays in place."""
    B, N = x.shape[:2]
    n = N - 1 if keep_last else N
    if n < 2:
        return x
    perm = torch.argsort(torch.rand(B, n, generator=gen, device=x.device), dim=1)
    if keep_last:
        perm = torch.cat([perm, torch.full((B, 1), N - 1, device=x.device, dtype=perm.dtype)], 1)
    return torch.gather(x, 1, perm.view(B, N, *([1] * (x.dim() - 2))).expand_as(x))


# ---------------------------------------------------------------- models
class MLPTok(nn.Module):
    def __init__(self, f, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, x):                                                              # [B, N, f]
        return self.net(x)


class CNN1s(nn.Module):
    """1-s EEG token (100 points) -> d."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
                                 nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
                                 nn.Conv1d(64, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):                                                              # [B, N, 100]
        B, N, f = x.shape
        return self.net(x.reshape(B * N, 1, f)).reshape(B, N, -1)


class CNN30s(nn.Module):
    """30-s EEG epoch (3000 points at 100 Hz) -> d; TinySleepNet-style (Supratak & Guo 2020) representation part."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 128, 50, stride=6), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(8, 8), nn.Dropout(0.5),
                                 nn.Conv1d(128, 128, 8, padding='same'), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128, 128, 8, padding='same'), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128, 128, 8, padding='same'), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.MaxPool1d(4, 4), nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 15, d))

    def forward(self, x):                                                              # [B, L, 3000]
        B, L, f = x.shape
        return self.net(x.reshape(B * L, 1, f)).reshape(B, L, -1)


class RefModel(nn.Module):
    def __init__(self, ds, c, n_tok, f, d=128):
        super().__init__()
        self.kind, self.last = c['model'], ds in ('SLEEPCTX', 'SLEEPPSD', 'MBIP')   # read the target token
        self.enc = CNN30s(d) if ds == 'SLEEPCTX' else CNN1s(d) if ds == 'SLEEPIN' else MLPTok(f, d)
        if self.kind == 'gru':
            self.seq = nn.GRU(d, d, num_layers=2, batch_first=True, dropout=0.1)
        else:
            self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, 2)
            self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, NCLS[ds])

    def forward(self, x):
        h = self.enc(x)
        if self.kind == 'gru':
            o, _ = self.seq(h)
            z = o[:, -1]
        else:
            o = self.seq(h + self.pos)
            z = self.norm(o[:, -1] if self.last else o.mean(1))
        return self.head(z)


# ---------------------------------------------------------------- train / evaluate
def metrics(y, p):
    y, p = y.cpu().numpy(), p.cpu().numpy()
    return dict(acc=100.0 * float((y == p).mean()), mf1=100.0 * float(f1_score(y, p, average='macro', labels=np.unique(y))))


@torch.no_grad()
def evaluate(model, data, idx, shuf, keep_last, draw=0):
    model.eval()
    bs = 256 if data.ds == 'SLEEPCTX' else 1024
    gen = torch.Generator(device=data.dev); gen.manual_seed(EVAL_SEED + draw)
    ys, ps = [], []
    for a in range(0, len(idx), bs):
        x, y = data.tokens(idx[a:a + bs])
        if shuf:
            x = shuffle_tokens(x, gen, keep_last)
        ps.append(model(x).argmax(1)); ys.append(y)
    return metrics(torch.cat(ys), torch.cat(ps))


@torch.no_grad()
def mbip_metrics(model, data, idx, shuf):
    """test S / V sensitivity, positive predictivity, F1 (%), and S sensitivity in records 232 and 222 (DS2)."""
    model.eval()
    gen = torch.Generator(device=data.dev); gen.manual_seed(EVAL_SEED)
    ps, ys = [], []
    for a in range(0, len(idx), 1024):
        x, y = data.tokens(idx[a:a + 1024])
        if shuf:
            x = shuffle_tokens(x, gen, True)
        ps.append(model(x).argmax(1)); ys.append(y)
    p, y = torch.cat(ps).cpu().numpy(), torch.cat(ys).cpu().numpy()
    out = {}
    for k, name in ((1, 'S'), (2, 'V')):
        tp, fn, fp = int(((p == k) & (y == k)).sum()), int(((p != k) & (y == k)).sum()), int(((p == k) & (y != k)).sum())
        se = 100.0 * tp / max(1, tp + fn); pp = 100.0 * tp / max(1, tp + fp)
        out.update({f'{name}_se': se, f'{name}_pp': pp, f'{name}_f1': 2 * se * pp / (se + pp) if se + pp > 0 else 0.0})
    r = data.rec_t[idx]
    for rec in (232, 222):
        m = (r == rec) & (y == 1)
        out[f'S_se_{rec}'] = 100.0 * float((p[m] == 1).mean()) if m.any() else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=list(NCLS)); ap.add_argument('--cond', required=True)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-2); ap.add_argument('--batch_size', type=int, default=0)
    ap.add_argument('--log_dir', required=True); ap.add_argument('--max_batches', type=int, default=0)   # smoke only
    ap.add_argument('--test_draws', type=int, default=5)
    a = ap.parse_args()
    c = parse_cond(a.ds, a.cond)
    a.batch_size = a.batch_size or (64 if a.ds == 'SLEEPCTX' else 256)
    sel = 'acc' if a.ds == 'SC' else 'mf1'
    keep_last = a.ds in ('SLEEPCTX', 'SLEEPPSD', 'MBIP')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    data = Data(a.ds, c, a.fold, dev)
    x0, _ = data.tokens(data.split['train'][:2])
    model = RefModel(a.ds, c, x0.shape[1], x0.shape[2]).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    os.makedirs(os.path.join(a.log_dir, 'log'), exist_ok=True); os.makedirs(os.path.join(a.log_dir, 'model_state'), exist_ok=True)
    cfg = dict(vars(a), **c, n_tokens=int(x0.shape[1]), token_dim=list(x0.shape[2:]), params=npar, selection=sel, keep_last=keep_last,
               n_train=len(data.split['train']), n_val=len(data.split['val']), n_test=len(data.split['test']))
    with open(os.path.join(a.log_dir, 'logargs.txt'), 'w') as f:
        f.write(json.dumps(cfg, indent=1) + '\n')
    torch.save(cfg, os.path.join(a.log_dir, 'model_state', 'config.pt'))
    print(json.dumps(cfg), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tr = data.split['train']
    spe = math.ceil(len(tr) / a.batch_size)
    if a.max_batches:
        spe = min(spe, a.max_batches)
    total, warm = a.epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wtr, wva = SummaryWriter(os.path.join(a.log_dir, 'log', 'train_0')), SummaryWriter(os.path.join(a.log_dir, 'log', 'val_0'))
    gtr = torch.Generator(device=dev); gtr.manual_seed(a.seed)
    rows, best, best_ep = [], -1.0, -1
    for ep in range(a.epochs):
        model.train()
        order = np.random.permutation(tr)
        tl, tn = 0.0, 0
        for b in range(spe):
            x, y = data.tokens(order[b * a.batch_size:(b + 1) * a.batch_size])
            if c['order'] == 'shuf':
                x = shuffle_tokens(x, gtr, keep_last)
            loss = F.cross_entropy(model(x), y, weight=data.cw if a.ds == 'MBIP' else None)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss) * len(y); tn += len(y)
        vm = evaluate(model, data, data.split['val'], c['order'] == 'shuf', keep_last)
        rows.append(dict(epoch=ep, train_loss=tl / tn, val_acc=vm['acc'], val_mf1=vm['mf1'], lr=opt.param_groups[0]['lr']))
        wtr.add_scalar('loss', tl / tn, ep); wva.add_scalar('acc', vm['acc'], ep); wva.add_scalar('mf1', vm['mf1'], ep)
        if vm[sel] > best:
            best, best_ep = vm[sel], ep
            torch.save(model.state_dict(), os.path.join(a.log_dir, 'model_state', 'best+model.pt'))
        print(f'epoch {ep} loss {tl / tn:.4f} val acc {vm["acc"]:.2f} mf1 {vm["mf1"]:.2f}{" *" if best_ep == ep else ""}', flush=True)
    wtr.close(); wva.close()
    with open(os.path.join(a.log_dir, 'log', 'best_log_0.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    model.load_state_dict(torch.load(os.path.join(a.log_dir, 'model_state', 'best+model.pt'), map_location=dev))
    te = evaluate(model, data, data.split['test'], c['order'] == 'shuf', keep_last)
    res = dict(test_acc=te['acc'], test_mf1=te['mf1'], best_epoch=best_ep, best_val=best)
    if c['order'] == 'orig':                                                           # report only: PPT-style permuted test
        d = [evaluate(model, data, data.split['test'], True, keep_last, draw=k) for k in range(a.test_draws)]
        res.update(ptest_acc=float(np.mean([m['acc'] for m in d])), ptest_mf1=float(np.mean([m['mf1'] for m in d])))
    else:
        res.update(ptest_acc=float('nan'), ptest_mf1=float('nan'))
    if a.ds == 'MBIP':                                                                 # MODI 2026-11-26 (evening): per-class, per-record
        res.update(mbip_metrics(model, data, data.split['test'], c['order'] == 'shuf'))
    res.update(params=npar, time_s=round(time.time() - t0, 1),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else 0.0)
    with open(os.path.join(a.log_dir, 'log', 'final+result.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res)); w.writeheader(); w.writerow({k: repr(v) if isinstance(v, float) else v for k, v in res.items()})
    print('FINAL ' + ' '.join(f'{k}={v!r}' if isinstance(v, float) else f'{k}={v}' for k, v in res.items()), flush=True)


if __name__ == '__main__':
    main()
