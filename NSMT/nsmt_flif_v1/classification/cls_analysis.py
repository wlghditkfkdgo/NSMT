"""What does the aligned neocortex learn on MIT-BIH? (MODI_PROJECT_LOG 2026-10-19; no training, trained weights only)

For one trained run (the last-epoch weights the framework tests, model_state/best+model.pt) on the 30% test file:

  A1 replay tests  -- the hippocampus cross-attention reads the neocortex output (mx). Replace mx with
       shift        the neocortex output of another beat in the batch (the ECG framework's shuffle_replay)
       same_class   the neocortex output of another beat of the SAME true class
       other_class  the neocortex output of a beat of a DIFFERENT true class
       zero / ones  an all-0 / all-1 neocortex output (no key/value / the gate fully open)
     and report accuracy, macro-F1 and the share of changed predictions. If same_class ~ normal while other_class
     drops, the classifier uses class information carried by the neocortex for that beat.
  A2 linear probes -- class-balanced logistic regression (train: up to 1000 beats per class from the training file,
     fixed draw; test: the full test file), on the raw beat, the neocortex embedding spikes, the neocortex output
     spikes (11 tokens x 384) and the hippocampus features that feed the classifier head. Macro-F1 / accuracy.
  A3 token-axis memory -- replace neocortex input token j by the same token of another beat (the hippocampus keeps
     the original) and measure the share of neocortex spikes that change at every token t (t >= j only can change:
     check c5), plus the share of changed predictions. Averaged by lag d = t - j this shows how far information
     travels along the aligned axis (spikeDE LIF vs f-LIF vs the spikingjelly LIF).
  A4 firing by class -- mean neocortex firing per token for each class (which beat segments drive the neocortex).

Conditions without a neocortex (no_neo, hippo_only) get the raw and hippocampus probes only.
usage: python cls_analysis.py --run_dir <run dir holding model_state/> --name <cond>_seed<s> --out_dir <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from einops import rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                    # noqa: E402
from load_model import LOAD_MODEL                                    # noqa: E402

N_CLASSES = 5


def macro_f1(pred, y, k=N_CLASSES):
    f = []
    for c in range(k):
        tp = ((pred == c) & (y == c)).sum().item(); fp = ((pred == c) & (y != c)).sum().item(); fn = ((pred != c) & (y == c)).sum().item()
        f.append(0. if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), f


def load_csv(path):
    d = pd.read_csv(path, header=None).values
    return torch.tensor(d[:, :-1], dtype=torch.float32).unsqueeze(1), torch.tensor(d[:, -1], dtype=torch.long)


class Runner:
    def __init__(self, m):
        self.m = m
        self.has_neo = not (getattr(m, 'no_neo', False) or getattr(m, 'hippo_only', False))

    @torch.no_grad()
    def forward(self, x, neo_tok_fn=None, mx_fn=None):
        """The ECG model's testing forward (roll_mode none), split open. Returns logits, neocortex embedding spikes
        e [N, B, 1, D], neocortex output o [N, B, 1, D] (None without a neocortex), head input h [B, D]."""
        m = self.m
        xx = x.unsqueeze(0).repeat(m.T, 1, 1, 1)
        xx = rearrange(xx, 't b c l -> t (b c) l', c=1)
        xx = m.padding_patch_layer(xx)
        patched = xx.unfold(dimension=-1, size=m.patch_size, step=m.stride)          # [T, B, N, P]
        x_hippo = m.encoding(patched)
        e = o = None
        if self.has_neo:
            neo_in = patched.transpose(0, 2).contiguous().mean(2, keepdim=True)     # [N, B, 1, P]
            if neo_tok_fn is not None:
                neo_in = neo_tok_fn(neo_in)
            e, o = m.neo_forward(neo_in)                                              # stage 6: bidirectional-aware
            mx = o if mx_fn is None else mx_fn(o)
        else:
            mx = None
        for dblk in m.data_block:
            x_hippo = dblk(x_hippo, mx, self_kv=getattr(m, 'hippo_only', False))
        h = x_hippo.mean(2)                                                            # [T, B, D]
        z = m.head(h).mean(0)
        functional.reset_net(m)
        return z, e, o, h.mean(0)


def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


def partner(y, same, g):
    """For every beat an index of another beat in the batch with the same / a different label (itself if none)."""
    idx = torch.arange(len(y))
    out = idx.clone()
    for i in range(len(y)):
        cand = torch.nonzero((y == y[i]) if same else (y != y[i])).flatten()
        cand = cand[cand != i]
        if len(cand):
            out[i] = cand[torch.randint(len(cand), (1,), generator=g)].item()
    return out


def logreg(xtr, ytr, xte, yte, l2=1e-3, iters=300):
    """Multinomial logistic regression, full-batch LBFGS on GPU; features standardised with training statistics;
    class-balanced loss weights."""
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    cnt = torch.bincount(ytr, minlength=N_CLASSES).float()
    w = (cnt.sum() / (N_CLASSES * cnt.clamp(min=1))).to(xtr.device)
    lin = torch.nn.Linear(xtr.shape[1], N_CLASSES).to(xtr.device)
    torch.nn.init.zeros_(lin.weight); torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1., max_iter=iters, history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(xtr), ytr, weight=w) + l2 * lin.weight.square().sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        p = lin(xte).argmax(1)
    f1, per = macro_f1(p.cpu(), yte.cpu())
    return dict(macro_f1=f1, acc=(p.cpu() == yte.cpu()).float().mean().item(), **{f'f1_{c}': v for c, v in enumerate(per)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--data', default=os.path.join(HERE, '..', '..', 'nsmt_base', 'classification', 'ECG', 'data'))
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--reach_n', type=int, default=4096, help='test beats used for A3')
    ap.add_argument('--probe_per_class', type=int, default=1000)
    a = ap.parse_args()
    torch.manual_seed(0)
    out = os.path.join(a.out_dir, a.name); os.makedirs(out, exist_ok=True)

    cfg = torch.load(os.path.join(a.run_dir, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg)
    m = LOAD_MODEL['myModel'](args, True)
    state = torch.load(os.path.join(a.run_dir, 'model_state', 'best+model.pt'), map_location='cpu')
    m.load_state_dict(state)                                                           # strict
    m = m.cuda().eval(); m.train_mode = 'testing'
    R = Runner(m)

    xte, yte = load_csv(os.path.join(a.data, '5class_mitbih_test_3_187.csv'))
    xtr, ytr = load_csv(os.path.join(a.data, '5class_mitbih_train_7_187.csv'))
    n = len(yte)

    # the split-open forward must reproduce the model's own logits
    xb = xte[:a.bs].cuda()
    with torch.no_grad():
        z_ref = m(xb)[0]; functional.reset_net(m)
    z_new = R.forward(xb)[0]
    parity = dict(bitwise=bool(torch.equal(z_ref, z_new)), max_abs=float((z_ref - z_new).abs().max()))
    if not torch.allclose(z_ref, z_new, atol=1e-5):
        raise RuntimeError(f'split forward differs from model forward: {parity}')

    meta = dict(name=a.name, run_dir=a.run_dir, has_neo=R.has_neo, parity=parity, n_test=n,
                neo_neuron=getattr(args, 'neo_neuron', 'sj_lif'), neo_where=getattr(args, 'neo_where', 'both'),
                no_neo=getattr(args, 'no_neo', False), hippo_only=getattr(args, 'hippo_only', False))

    # ---- normal pass: logits, features, firing by class
    preds, H, O, E = [], [], [], []
    for sl in batches(n, a.bs):
        z, e, o, h = R.forward(xte[sl].cuda())
        preds.append(z.argmax(1).cpu()); H.append(h.cpu())
        if R.has_neo:
            O.append(o[:, :, 0].transpose(0, 1).to(torch.uint8).cpu()); E.append(e[:, :, 0].transpose(0, 1).to(torch.uint8).cpu())
    p0 = torch.cat(preds); H = torch.cat(H)
    f1, per = macro_f1(p0, yte)
    rows = [dict(test='normal', acc=(p0 == yte).float().mean().item(), macro_f1=f1, changed=0.,
                 **{f'f1_{c}': v for c, v in enumerate(per)})]
    meta['normal_macro_f1'] = f1

    if R.has_neo:
        O = torch.cat(O); E = torch.cat(E)                                               # [n, N, D] uint8
        # A4 firing by class (per token, mean over units)
        fr = []
        for c in range(N_CLASSES):
            sel = yte == c
            fr.append(dict(cls=c, n=int(sel.sum()), **{f'out_t{t}': O[sel, t].float().mean().item() for t in range(O.shape[1])},
                           **{f'emb_t{t}': E[sel, t].float().mean().item() for t in range(E.shape[1])}))
        pd.DataFrame(fr).to_csv(os.path.join(out, 'firing_by_class.csv'), index=False, float_format='%.6f')
        meta['rate_out'] = O.float().mean().item(); meta['rate_emb'] = E.float().mean().item()

        # ---- A1 replay tests
        g = torch.Generator().manual_seed(0)
        def run_mx(mode):
            ps = []
            for sl in batches(n, a.bs):
                yb = yte[sl]
                if mode == 'shift':
                    perm = torch.roll(torch.arange(len(yb)), shifts=1)
                elif mode in ('same_class', 'other_class'):
                    perm = partner(yb, mode == 'same_class', g)
                else:
                    perm = None
                if mode == 'zero':
                    fn = lambda o: torch.zeros_like(o)
                elif mode == 'ones':
                    fn = lambda o: torch.ones_like(o)
                else:
                    pc = perm.cuda()
                    fn = lambda o, pc=pc: o[:, pc]
                ps.append(R.forward(xte[sl].cuda(), mx_fn=fn)[0].argmax(1).cpu())
            p = torch.cat(ps)
            f, pr = macro_f1(p, yte)
            return dict(test=mode, acc=(p == yte).float().mean().item(), macro_f1=f, changed=(p != p0).float().mean().item(),
                        **{f'f1_{c}': v for c, v in enumerate(pr)})
        for mode in ('shift', 'same_class', 'other_class', 'zero', 'ones'):
            rows.append(run_mx(mode))
    pd.DataFrame(rows).to_csv(os.path.join(out, 'replay.csv'), index=False, float_format='%.6f')

    # ---- A2 probes
    rng = np.random.RandomState(0)
    pick = np.concatenate([rng.permutation(np.nonzero(ytr.numpy() == c)[0])[:a.probe_per_class] for c in range(N_CLASSES)])
    pick.sort()
    xs, ys = xtr[pick], ytr[pick]
    Htr, Otr, Etr = [], [], []
    for sl in batches(len(ys), a.bs):
        z, e, o, h = R.forward(xs[sl].cuda())
        Htr.append(h.cpu())
        if R.has_neo:
            Otr.append(o[:, :, 0].transpose(0, 1).to(torch.uint8).cpu()); Etr.append(e[:, :, 0].transpose(0, 1).to(torch.uint8).cpu())
    feats = {'raw': (xs[:, 0], xte[:, 0]), 'hippo_head_input': (torch.cat(Htr), H)}
    if R.has_neo:
        feats['neo_emb_spikes'] = (torch.cat(Etr).flatten(1).float(), E.flatten(1).float())
        feats['neo_out_spikes'] = (torch.cat(Otr).flatten(1).float(), O.flatten(1).float())
        feats['neo_out_token_rate'] = (torch.cat(Otr).float().mean(2), O.float().mean(2))      # 11 numbers per beat
    prow = []
    for k, (ftr, fte) in feats.items():
        r = logreg(ftr.cuda(), ys.cuda(), fte.cuda(), yte)
        prow.append(dict(feature=k, dim=ftr.shape[1], n_train=len(ys), **r))
    pd.DataFrame(prow).to_csv(os.path.join(out, 'probe.csv'), index=False, float_format='%.6f')

    # ---- A3 token-axis memory
    if R.has_neo:
        nr = min(a.reach_n, n)
        Nt = O.shape[1]
        ch_out = np.zeros((Nt, Nt)); ch_emb = np.zeros((Nt, Nt)); pred_rows = []
        for j in range(Nt):
            cnt_o = np.zeros(Nt); cnt_e = np.zeros(Nt); tot = 0; ps = []
            for sl in batches(nr, a.bs):
                xb = xte[sl].cuda()
                perm = torch.roll(torch.arange(xb.shape[0]), shifts=1).cuda()
                def tok(neo_in, j=j, perm=perm):
                    t = neo_in.clone(); t[j] = neo_in[j][perm]; return t
                z, e, o, _ = R.forward(xb, neo_tok_fn=tok)
                o0 = O[sl].cuda().transpose(0, 1).float(); e0 = E[sl].cuda().transpose(0, 1).float()     # [N, b, D]
                cnt_o += (o[:, :, 0] != o0).float().mean(2).sum(1).cpu().numpy()
                cnt_e += (e[:, :, 0] != e0).float().mean(2).sum(1).cpu().numpy()
                tot += xb.shape[0]; ps.append(z.argmax(1).cpu())
            ch_out[j] = cnt_o / tot; ch_emb[j] = cnt_e / tot
            p = torch.cat(ps)
            f, _ = macro_f1(p, yte[:nr])
            pred_rows.append(dict(token=j, changed_pred=(p != p0[:nr]).float().mean().item(), macro_f1=f,
                                  macro_f1_normal=macro_f1(p0[:nr], yte[:nr])[0]))
        pd.DataFrame(ch_out, columns=[f't{t}' for t in range(Nt)]).rename_axis('j').to_csv(os.path.join(out, 'reach_out.csv'), float_format='%.6f')
        pd.DataFrame(ch_emb, columns=[f't{t}' for t in range(Nt)]).rename_axis('j').to_csv(os.path.join(out, 'reach_emb.csv'), float_format='%.6f')
        pd.DataFrame(pred_rows).to_csv(os.path.join(out, 'reach_pred.csv'), index=False, float_format='%.6f')
        lag = []
        for d in range(Nt):
            vo = [ch_out[j, j + d] for j in range(Nt - d)]; ve = [ch_emb[j, j + d] for j in range(Nt - d)]
            lag.append(dict(lag=d, out=float(np.mean(vo)), emb=float(np.mean(ve)), n_pairs=len(vo)))
        pd.DataFrame(lag).to_csv(os.path.join(out, 'reach_lag.csv'), index=False, float_format='%.6f')
        meta['acausal_max'] = float(max(ch_out[j, t] for j in range(Nt) for t in range(j)) if Nt > 1 else 0.)
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
