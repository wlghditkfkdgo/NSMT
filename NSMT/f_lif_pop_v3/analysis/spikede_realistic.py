"""Original spikeDE package (fcd743b, unmodified files) vs f_lif_pop_v3 neurons.py at training-like sizes.

usage: python spikede_realistic.py <spikeDE root> <f_lif_pop_v3/forecasting>   (snn_recall, GPU)

torch 1.12 has no torch.compile; an identity shim is installed BEFORE importing spikeDE (compile only
changes speed). Input: N(0, s^2) currents, B=896 (128 windows x 7 channels), D=32, T=42."""
import sys, torch
torch.compile = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)   # shim
sys.path.insert(0, sys.argv[2]); sys.path.insert(0, sys.argv[1])   # 원본 spikeDE 패키지를 먼저 찾게 한다
from spikeDE.neuron import LIFNeuron as RefLIF                   # 설치된 원본 패키지 (spikede.py 복사본이 아님)
from spikeDE.solver import pred_integrate_tuple
import neurons as N

def ref(x, tau, alpha):
    T = x.shape[0]
    neuron = RefLIF(tau=tau, threshold=1., surrogate_grad_scale=5.)
    t_grid = torch.arange(T + 2, dtype=x.dtype, device=x.device)
    seen = []
    def rhs(t, values):
        n = int(round(float(t)))
        seen.append(values[0])
        return neuron(values[0], x[n] if n < T else torch.zeros_like(x[0]))
    y0 = (torch.zeros_like(x[0]), torch.zeros_like(x[0]))
    hist = pred_integrate_tuple(rhs, y0, [alpha], t_grid)
    return torch.stack(seen[1:]), torch.stack(hist[-1][:-1]).to(x.dtype)

def ours(x, tau, alpha):
    n = N.build_neuron('flif', x.shape[-1], tau=tau, alpha=alpha, max_length=x.shape[0], dtype=x.dtype).to(x.device)
    out, aux = n(x, return_aux=True)
    return aux['state'][..., 0], out

g = torch.Generator().manual_seed(0)
for dev in ('cpu', 'cuda'):
    for dtype in (torch.float64, torch.float32):
        for tau, alpha, scale in ((2., .7, 3.), (8., .7, 6.), (2., .5, 3.), (2., 1., 3.)):
            x = (scale * torch.randn(42, 896, 32, generator=g, dtype=torch.float64)).to(dev, dtype)
            with torch.no_grad():
                rs, rk = ref(x, tau, alpha); os_, ok = ours(x, tau, alpha)
            rel = float((rs - os_).abs().max() / rs.abs().max().clamp_min(1))
            flips = int((rk != ok).sum()); first = (rk != ok).nonzero()[:, 0].min().item() if flips else None
            print(f'{dev:4s} {str(dtype)[6:]:8s} tau {tau:3g} alpha {alpha:3g}: max rel state diff {rel:.1e}, '
                  f'spikes {int(rk.sum())}, differing spikes {flips} ({flips/rk.numel():.2e} of all), first at t={first}')
