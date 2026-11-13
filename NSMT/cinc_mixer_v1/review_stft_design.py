"""Design review checks of the STFT + population proposal (2026-11-13, exp/f-lif-pop-v3; PROJECT_LOG). No training, CPU. usage (from cinc_mixer_v1/): CUDA_VISIBLE_DEVICES= python -B review_stft_design.py
(1) population LIF with equal branch taus collapses to one branch; (2) per-frame rfft is an invertible linear map of the patch; (3) frame counts; (4) share of signal power per STFT bin on ETTh2 / ETTm2 / MIT-BIH."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
torch.manual_seed(0)
from layers import build_neo_neuron
# (1a) the real class: K = 4 branches with tau 2 vs K = 1 branch with tau 2, same input
T, B, N, D = 24, 3, 2, 16
x = torch.randn(T, B, N, D) * 1.5
a = build_neo_neuron('pop_lif', D, T, 'ND', tau=2., pop_taus=(2., 2., 2., 2.))
b = build_neo_neuron('pop_lif', D, T, 'ND', tau=2., pop_taus=(2.,))
with torch.no_grad():
    sa, sb = a(x), b(x)
print('(1a) pop_lif taus (2,2,2,2) vs K=1 tau 2, same input: identical spikes', bool(torch.equal(sa, sb)), '| rate %.4f' % sa.mean().item(),
      '| soma weights', [round(float(w), 3) for w in a.n.soma.weight.flatten()[:4]] if hasattr(a, 'n') else '')
c = build_neo_neuron('lif', D, T, 'ND', tau=2.)
with torch.no_grad():
    sc = c(x)
print('     plain LIF tau 2 on the same input: rate %.4f, same spikes as the K=1 population: %s' % (sc.mean().item(), bool(torch.equal(sc, sb))))
# (1b) different inputs per branch, same tau: sum_k w_k v_k(I_k) == v(sum_k w_k I_k)   (branch update v <- v + (I - v) / tau, no reset)
K, tau = 4, 2.
I = torch.randn(T, K, dtype=torch.float64); w = torch.rand(K, dtype=torch.float64)
def branch(i, tau):
    v, out = torch.zeros((), dtype=torch.float64), []
    for t in range(i.shape[0]):
        v = v + (i[t] - v) / tau; out.append(v)
    return torch.stack(out)
lhs = sum(w[k] * branch(I[:, k], tau) for k in range(K)); rhs = branch((I * w).sum(1), tau)
print('(1b) equal tau, different inputs: max |sum_k w_k v_k - v(sum_k w_k I_k)| = %.2e' % (lhs - rhs).abs().max().item())
lhs2 = sum(w[k] * branch(I[:, k], (4., 8., 16., 32.)[k]) for k in range(K)); rhs2 = branch((I * w).sum(1), 2.)
print('     different taus (4,8,16,32): the same difference = %.2e (does not collapse)' % (lhs2 - rhs2).abs().max().item())
# (2) per-frame rfft (real, imag) of a patch of P samples is an invertible linear map
for P in (6, 8, 32):
    E = torch.eye(P, dtype=torch.float64); F = torch.fft.rfft(E, dim=1)
    M = torch.cat([F.real, F.imag], 1)
    print(f'(2) P={P}: bins {P // 2 + 1}, real-valued features {M.shape[1]} (non-zero columns {(M.abs().sum(0) > 1e-12).sum().item()}), rank {torch.linalg.matrix_rank(M).item()}')
# (3) frames
for name, L, nfft, hop in (('ETT 4 bins', 96, 6, 3), ('ETT 4 bins', 96, 6, 4), ('ETT patch-aligned', 96, 8, 4), ('MIT-BIH 4 bins', 187, 6, 3), ('MIT-BIH 4 bins, 11 frames', 187, 6, 16), ('MIT-BIH patch-aligned', 187, 32, 16), ('CinC 4 bins', 3750, 6, 3), ('CinC patch-aligned', 3750, 64, 32)):
    f = torch.stft(torch.zeros(L), nfft, hop_length=hop, window=torch.ones(nfft), center=True, return_complex=True)
    print(f'(3) {name}: L {L}, n_fft {nfft}, hop {hop} -> bins {f.shape[0]}, frames {f.shape[1]}' + (' (hop > window: samples skipped)' if hop > nfft else ''))
# (4) power share per bin
import pandas as pd
def share(X, nfft, hop):
    S = torch.stft(torch.as_tensor(X, dtype=torch.float32), nfft, hop_length=hop, window=torch.ones(nfft), center=False, return_complex=True)   # [n, bins, frames]
    p = (S.abs() ** 2); p[:, 1:nfft // 2 + (0 if nfft % 2 == 0 else 1)] *= 2          # one-sided: double the bins that have a mirror
    p = p.sum((0, 2)); return (p / p.sum()).numpy()
root = '../forecasting/dataset/ETT-small'
for ds, n_train in (('ETTh2', 12 * 30 * 24), ('ETTm2', 12 * 30 * 24 * 4)):
    d = pd.read_csv(f'{root}/{ds}.csv').iloc[:n_train, 1:].values.astype('float32')
    d = (d - d.mean(0)) / d.std(0)
    idx = np.arange(0, len(d) - 96, 24)
    W = np.stack([d[i:i + 96] for i in idx])                              # [n, 96, 7]
    W = (W - W.mean(1, keepdims=True)) / np.sqrt(W.var(1, keepdims=True) + 1e-5)   # window normalisation as the Forecaster
    X = W.transpose(0, 2, 1).reshape(-1, 96)
    for nfft, hop in ((6, 3), (8, 4)):
        print(f'(4) {ds} train windows (window-normalised), n_fft {nfft}: power share per bin %', np.round(100 * share(X, nfft, hop), 1))
m = pd.read_csv('/home/yschoi/CLS_spiking_transformer/data/5class_mitbih_train_7_187.csv', header=None).values[:, :187].astype('float32')
for nfft, hop in ((6, 3), (32, 16)):
    s = 100 * share(m, nfft, hop)
    print(f'(4) MIT-BIH train beats, n_fft {nfft}: power share per bin %', np.round(s, 1) if nfft == 6 else np.round(s[:8], 1), '' if nfft == 6 else f'... bins 8-16 sum {s[8:].sum():.2f}')
    if nfft == 32:
        cum = np.cumsum(s); print('     octave bands [0], [1], [2-3], [4-16] %:', np.round([s[0], s[1], s[2:4].sum(), s[4:].sum()], 1))
