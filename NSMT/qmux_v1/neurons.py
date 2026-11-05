# nsmt_flif_v1 copy of NSMT/f_lif_pop_v3/forecasting/neurons.py (repo commit cd9b4cfec, 2026-10-14).
# The ONLY change: the line `from layers import arctan_spike, Soma` is replaced by inline copies of
# ArcTanSpike/arctan_spike (v3 layers.py:51-76) and Soma (v3 layers.py:347-376), because this folder
# has its own layers.py (the NSMT one). Everything else is byte-identical to the v3 file, except (MODI 2026-10-31) the `reset`
# option of LIFNeuron / FLIFNeuron / PopulationFLIF / Soma (default 'spikede' = unchanged; see LIFNeuron._apply_reset).
"""Neuron family: LIF -> f-LIF -> population / selective / selective population f-LIF.

Reference implementation: PhysAGI/spikeDE, commit fcd743befe504b1a471fa81887e6af7d6789da2e
(upstream HEAD on 2026-10-14, MIT licence), the code of "Fractional-order Spiking Neural
Network" (Ge et al., ICLR 2026, arXiv:2507.16937). analysis/spikede_parity.py runs the
original spikeDE classes against the ones here.

spikeDE splits a neuron into a right-hand side and a solver (spikeDE/neuron.py, solver.py):

    drift_k = (I_k - v_k) / tau
    s_k     = H(v_k + 1.0 * drift_k - theta)       spike on a one-step Euler look-ahead, dt = 1
    F_k     = drift_k - theta * s_k.detach() / tau  the reset lives inside the dynamics

    LIF   (odeint + euler) : v_{k+1} = v_k + h F_k
    f-LIF (fdeint + pred)  : v_{k+1} = sum_{j<=k} b_{k-j} F_j,
                             b_d = h^a / Gamma(a+1) * [(d+1)^a - d^a]     Caputo, Adams-Bashforth
    f-LIF (fdeint + gl)    : v_{k+1} = h^a F_k - sum_{m=1..k} c_{k+1-m} v_m,
                             c_0 = 1, c_j = (1 - (1+a)/j) c_{j-1}         Grunwald-Letnikov

with v_0 = 0, and the spike reported for step k is the one decided from v_k, before the
update. `pred` is the default of the paper's own training script and the scheme v3 has used
since v3-A. The surrogate gradient is spikeDE's arctan, (kappa/2) / (1 + (pi/2 kappa x)^2).

Each class adds one thing:

    LIFNeuron                     the spikeDE LIF, Euler-integrated
    FLIFNeuron(LIFNeuron)         the same right-hand side, fractional solver (pred | gl)
    PopulationFLIF(FLIFNeuron)    K members with their own tau on the same input current.
                                  soma='lif': the members do not fire, one LIF soma integrates
                                  them (the v3 arrangement, layers.Soma); soma='none': the members
                                  are K independent f-LIF neurons, K spike trains out
    SelectiveFLIF(FLIFNeuron)     past increments enter with c = m * b, m a 0/1 top-q mask by the
                                  similarity of xi = [v, I] (pred only)
    SelectivePopulationFLIF(SelectiveFLIF, PopulationFLIF)    both

Neutral limits hold exactly (check_neurons.py): FLIFNeuron(alpha=1) == LIFNeuron, under conditions that depend
on the solver (audits 78 and 81, A35): method='pred' only with the full history (memory=None; any truncation
drops old increments, since every b_d = h at alpha=1); method='gl' with memory=None or memory >= 2 (at alpha=1
c_1 = -1 and c_j = 0 for j >= 2, so only the previous state enters), but not with memory=1,
SelectiveFLIF(q=1) == FLIFNeuron, PopulationFLIF(K=1, soma='none') == FLIFNeuron,
SelectivePopulationFLIF(q=1) == PopulationFLIF, and with soma='lif' the two population classes
reproduce layers.PopulationNeuron in mode 'full' and mode 'hard'.

All classes take a [T, B, D] input current and return [T, B, out_features] spikes, with the
same aux dictionary as layers.PopulationNeuron so the rest of the pipeline is unchanged.
"""
import math

import torch
import torch.nn as nn

# ---- inlined from f_lif_pop_v3/forecasting/layers.py (unchanged) ------------------------------
class ArcTanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        """
        x: (...) membrane minus threshold
        return: (...) Heaviside spike in {0., 1.}
        """
        ctx.save_for_backward(x)
        ctx.scale = scale

        return (x >= 0.).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        """
        gradient: (...)
        return: (...) arctan surrogate (s/2) / (1 + (pi/2 * s * x)^2)
        """
        x, = ctx.saved_tensors
        s = ctx.scale

        return gradient * (s / 2.) / (1. + (math.pi / 2. * s * x).square()), None


def arctan_spike(x, scale=5.):
    return ArcTanSpike.apply(x, scale)


class Soma(nn.Module):
    """Single spiking site of the population (prereg D1, D-D).

    The K branches never reset; they only hold memory and the retrieval key. Firing and
    subtractive reset happen here once per patch, which is the DH-SNN arrangement
    (Zheng et al., Nat. Commun. 2024):

        a_n = sum_k w_k u(n+1,k),   v_n = v_{n-1} + (a_n - v_{n-1}) / tau_s
        s_n = H(v_n - theta),       v_n <- v_n - theta * s_n

    D-D is the u(n+1) in that first line: the soma reads the state that ALREADY contains
    I_n. Reading u(n) instead would leave the last patch unable to reach the last spike
    (gate G15) and delay every selection by one step.
    """
    def __init__(self, embed_dim, num_population=4, tau_s=2., threshold=1., surrogate_scale=5., reset='subtract'):
        super().__init__()

        if tau_s <= 1. or threshold <= 0.:
            raise ValueError('Require tau_s > 1 and threshold > 0')
        if reset not in ('subtract', 'zero'):                         # MODI 2026-10-31: 'zero' = hard reset of the soma
            raise ValueError("Soma reset must be 'subtract' or 'zero'")
        self.reset_mode = reset   # not 'reset': spikingjelly reset_net calls m.reset()
        self.weight = nn.Parameter(torch.full((embed_dim, num_population), 1. / num_population))
        self.tau_s = float(tau_s)
        self.threshold = float(threshold)
        self.surrogate_scale = float(surrogate_scale)                # O2: pinned 코드가 넘기는 5.0

    def forward(self, u, v):                                         # u: [B, D, K]   v: [B, D]
        a = (self.weight * u).sum(-1)                                # 구성원 혼합 -> 소마 입력 전류
        charge = v + (a - v) / self.tau_s
        s = arctan_spike(charge - self.threshold, self.surrogate_scale)

        if self.reset_mode == 'zero':                                     # MODI 2026-10-31
            return charge * (1. - s.detach()), s, a
        return charge - self.threshold * s.detach(), s, a            # detach_reset

# ---- end of inlined helpers ---------------------------------------------------------------

__all__ = ['LIFNeuron', 'FLIFNeuron', 'PopulationFLIF', 'SelectiveFLIF', 'SelectivePopulationFLIF',
           'NEURONS', 'NEUTRAL', 'build_neuron', 'pred_coefficients', 'gl_coefficients', 'select_mask']

STATS = ('pearson', 'cosine', 'recent', 'random')
AXES = ('unit', 'shared', 'input')
PARTS = ('all', 'input')
KEEP = ('kappa', 'cap_rate', 'would_cap_rate', 'eta', 'support_p', 'support_rho', 'support_braw',
        'support_c', 'support_size', 'score_std')                   # layers.PopulationNeuron의 aux 키


def pred_coefficients(alpha, length, step=1., dtype=torch.float32):
    """b_d = h^a / Gamma(a+1) [(d+1)^a - d^a], d = 0..length-1 (spikeDE AdamsBashforthSNN).
    Built in float64 and cast; with step = 1 this is bitwise layers.fractional_coefficients."""
    if not 0. < alpha <= 1.:
        raise ValueError('Require 0 < alpha <= 1')
    d = torch.arange(length, dtype=torch.float64)
    b = ((d + 1.).pow(alpha) - d.pow(alpha)) / math.exp(math.lgamma(alpha + 1.))
    if step != 1.:
        b = b * step ** alpha

    return b.to(dtype)


def gl_coefficients(alpha, length, dtype=torch.float32):
    """c_0 = 1, c_j = (1 - (1+a)/j) c_{j-1}, j = 1..length (spikeDE GrunwaldLetnikovSNN)."""
    c = [1.]
    for j in range(1, length + 1):
        c.append((1. - (1. + alpha) / j) * c[-1])

    return torch.tensor(c, dtype=torch.float64).to(dtype)


@torch.no_grad()
def select_mask(xi, xi_hist, q, stat='pearson', axis='shared', generator=None):
    #  xi: [B, D, F]   xi_hist: [B, D, J, F]   ->   m: [B, D, J] in {0, 1},  sim: [B, D, J]
    """Keep the round(q J) past slots most similar to the present; layers.Selector.hard_mask.

    axis 'shared' flattens every unit's descriptor into one vector (one decision for the whole
    population of units), 'unit' decides per unit, 'input' looks at the input current only.
    'recent' and 'random' keep the same number of slots without looking at the content. The
    random mask comes from a CPU generator and is moved to the data's device (the legacy code
    left it on the CPU). No gradient: m is a hard 0/1.
    """
    B, D, J, F = xi_hist.shape
    if q >= 1.:                                                      # 중립 극한: 통계를 계산하지 않는다
        ones = xi_hist.new_ones(B, D, J)
        return ones, ones
    keep = max(1, int(round(q * J)))                                 # 파이썬 round: 짝수 쪽으로
    if stat in ('recent', 'random'):
        if stat == 'recent':
            sim = torch.arange(J, dtype=xi_hist.dtype).expand(B, 1, J)
        else:
            sim = torch.rand(B, 1, J, generator=generator).to(xi_hist.dtype)
        sim = sim.to(xi_hist.device)
        idx = sim.topk(keep, dim=-1).indices
        m = torch.zeros_like(sim).scatter_(-1, idx, 1.)
        return m.expand(B, D, J), sim.expand(B, D, J)
    if axis == 'unit':
        q_, k_ = xi.unsqueeze(-2), xi_hist
    elif axis == 'shared':
        q_ = xi.reshape(B, 1, 1, D * F)
        k_ = xi_hist.permute(0, 2, 1, 3).reshape(B, 1, J, D * F)
    else:
        q_ = xi[..., -1].reshape(B, 1, 1, D)
        k_ = xi_hist[..., -1].permute(0, 2, 1).reshape(B, 1, J, D)
    if stat == 'pearson':
        q_, k_ = q_ - q_.mean(-1, keepdim=True), k_ - k_.mean(-1, keepdim=True)
    num = (q_ * k_).sum(-1)
    den = (q_.square().sum(-1).sqrt() * k_.square().sum(-1).sqrt()).clamp_min(1e-12)
    sim = num / den                                                  # [B, D or 1, J]
    idx = sim.topk(keep, dim=-1).indices
    m = torch.zeros_like(sim).scatter_(-1, idx, 1.)

    return m.expand(B, D, J), sim.expand(B, D, J)


class LIFNeuron(nn.Module):
    """spikeDE LIFNeuron, Euler-integrated: tau dv/dt = -v + I, reset -theta*s/tau inside F.

    Args
    ----
    size : int
        D, the number of neurons (units).
    tau : float
        Membrane time constant (spikeDE default 0.5; the paper's vision models use 2.0).
    threshold : float
        theta.
    surrogate_scale : float
        kappa of the arctan surrogate (spikeDE passes 5.0).
    step : float
        h, the integration step. The spike look-ahead keeps dt = 1 as in spikeDE.
    max_length : int
        Longest sequence; sizes the coefficient table b (all h for LIF, kept for the
        diagnostics that read neuron.b).
    """
    def __init__(self, size, tau=2., threshold=1., surrogate_scale=5., step=1., max_length=64,
                 dtype=torch.float32, reset='spikede'):
        super().__init__()

        # MODI 2026-10-31: reset = how far a spike lowers the membrane at its step: 'spikede' (default, spikeDE: -theta*s/tau
        # inside F), 'subtract' (exactly theta), 'zero' (to exactly 0, the spikingjelly hard reset); see _apply_reset
        if reset not in ('spikede', 'subtract', 'zero'):
            raise ValueError("reset must be 'spikede', 'subtract' or 'zero'")
        if reset != 'spikede' and type(self).__name__.startswith('Selective'):
            raise NotImplementedError('reset options are defined for the LIF / f-LIF / population classes')
        self.reset_mode = reset   # not 'reset': spikingjelly reset_net calls m.reset()

        tau = torch.as_tensor(tau, dtype=torch.float64).reshape(-1)
        if tau.numel() < 1 or not bool((tau > 0.).all()):
            raise ValueError('Require tau > 0')
        if threshold <= 0. or step <= 0. or max_length < 1:
            raise ValueError('Require threshold > 0, step > 0, max_length >= 1')
        self.register_buffer('tau', tau.to(dtype))
        self._check_members()
        self.size = int(size)
        self.threshold = float(threshold)
        self.surrogate_scale = float(surrogate_scale)
        self.step = float(step)
        self.max_length = int(max_length)
        self.members_fire = True                                     # spikeDE: 각 뉴런이 스스로 발화·리셋
        self.register_buffer('b', pred_coefficients(1., max_length, step, dtype))   # alpha=1: 모두 h

    # ---- structure -------------------------------------------------------------------------
    def _check_members(self):
        if self.tau.numel() != 1:
            raise ValueError(f'{type(self).__name__} is a single neuron: give one tau '
                             f'(a population is PopulationFLIF)')

    @property
    def num_members(self):
        return self.tau.numel()

    @property
    def has_soma(self):
        return False

    @property
    def out_features(self):
        return self.size * (self.num_members if self.members_fire else 1)

    # ---- right-hand side (spikeDE LIFNeuron.forward) ---------------------------------------
    def rhs(self, u, x):
        #  u: [B, D, K] v_k   x: [B, D] I_k   ->   F_k: [B, D, K],  s_k: [B, D, K] or None
        """F_k and the member spikes; members that do not fire (soma='lif') have no reset term."""
        drift = (x.unsqueeze(-1) - u) / self.tau                     # (-v + I) / tau
        if not self.members_fire:
            return drift, None
        look = u + 1. * drift                                        # v + dt * drift, dt = 1 (spikeDE)
        s = arctan_spike(look - self.threshold, self.surrogate_scale)
        if self.reset_mode != 'spikede':                                  # MODI 2026-10-31: reset applied after the update
            return drift, s

        return drift - (s.detach() * self.threshold) / self.tau, s

    def reset_gain(self):
        """Coefficient of F_t in v_{t+1} (Euler: h)."""
        return self.step

    def _apply_reset(self, nxt, s, history):
        """MODI 2026-10-31: 'subtract' / 'zero' lower the updated membrane of the spiking members by theta / by all of it
        (spike detached, as detach_reset). The stored increment F_t (f-LIF pred history) is corrected by the same amount over
        its coefficient, so v_{t+1} = sum_j b_{t-j} F_j still holds for every later step."""
        if self.reset_mode == 'spikede' or s is None:
            return nxt
        sd = s.detach()
        delta = sd * (self.threshold if self.reset_mode == 'subtract' else nxt)
        if history['F']:
            history['F'][-1] = history['F'][-1] - delta / self.reset_gain()
        return nxt - delta

    # ---- integrator --------------------------------------------------------------------------
    @staticmethod
    def new_history():
        return {'F': [], 'v': [], 'keys': [], 'input': []}

    def _check_length(self, T):
        if T > self.max_length:
            raise ValueError(f'T={T} exceeds the coefficient table ({self.max_length})')

    @staticmethod
    def _stats(c=None, b_hist=None, m=None, sim=None, like=None):
        """aux statistics in layers.Selector's format: zeros at t=0, neutral without selection."""
        if c is None:                                                # 이력 없음 (t=0)
            zeros = like.new_zeros(like.shape[:2])
            return {k: zeros for k in KEEP}
        J = c.shape[-1]
        zeros = torch.zeros_like(c[..., 0])
        if m is None:                                                # 선택 없음: layers의 mode='full'
            ones = torch.ones_like(zeros)
            return {'eta': zeros, 'kappa': ones, 'cap_rate': zeros, 'would_cap_rate': zeros,
                    'support_size': ones * J, 'score_std': zeros, 'support_p': ones,
                    'support_rho': ones, 'support_braw': ones, 'support_c': ones}
        frac = m.mean(-1)                                            # layers의 mode='hard'
        return {'eta': zeros, 'kappa': c.sum(-1) / b_hist.sum(), 'cap_rate': zeros,
                'would_cap_rate': zeros, 'support_size': m.sum(-1),
                'score_std': sim.std(-1, unbiased=False), 'support_p': frac, 'support_rho': frac,
                'support_braw': frac, 'support_c': frac}

    def window_start(self, t):
        """First history index that still enters at step t (spikeDE get_memory_bounds)."""
        return 0

    def has_history(self, t):
        return t > self.window_start(t)

    def _neutral_stats(self, u, t):
        """Statistics of a step whose past enters unselected (no coefficient tensor exists)."""
        J = t - self.window_start(t)
        if J < 1:
            return self._stats(like=u)
        c = u.new_ones(*u.shape[:2], J)
        return self._stats(c=c, b_hist=c[0, 0])

    def integrate(self, t, u, F, x, history, oracle_p=None):
        """Euler (spikeDE euler_integrate_tuple): v_{t+1} = v_t + h F_t."""
        return u + self.step * F, None, self._neutral_stats(u, t)

    # ---- output ------------------------------------------------------------------------------
    def initial_soma(self, x):
        return None

    def emit(self, nxt, s, soma):
        #  -> spikes [B, out_features], new soma state, soma drive (or None)
        """Members fire themselves: the output is their spikes, K trains side by side."""
        return s.reshape(s.shape[0], -1), soma, None

    # ---- the sequence --------------------------------------------------------------------------
    def forward(self, x, mode=None, oracle_p=None, return_aux=False, analog=False):
        #  x: [T, B, D] input current   ->   spikes: [T, B, out_features]
        """`mode` is accepted for layers.Embedding's call signature and ignored: the selection
        of these classes is fixed at construction. `analog` ('analog' | 'drive') reads the soma
        membrane after reset or the soma input, with the graph attached (soma='lif' only)."""
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError('Expected nonempty [T, B, D] current')
        T, B, D = x.shape
        if D != self.size:
            raise ValueError(f'Expected D={self.size}, got {D}')
        self._check_length(T)
        if analog and not self.has_soma:
            raise ValueError("analog/drive readouts read the soma; this neuron has none (soma='lif' does)")

        u = x.new_zeros(B, D, self.num_members)                      # v_0 = 0
        soma = self.initial_soma(x)
        history = self.new_history()
        spikes, live = [], []
        logs = {k: [] for k in ('state', 'voltage', 'coeff') + KEEP}
        has_history = torch.zeros(T, dtype=torch.bool, device=x.device)
        for t in range(T):
            F, s = self.rhs(u, x[t])
            nxt, coeff, stats = self.integrate(t, u, F, x[t], history, oracle_p)
            nxt = self._apply_reset(nxt, s, history)                 # MODI 2026-10-31 (no-op for 'spikede')
            out, soma, drive = self.emit(nxt, s, soma)
            spikes.append(out)
            if analog:
                live.append(drive if analog == 'drive' else soma)
            if return_aux:
                logs['state'].append(nxt.detach())
                logs['voltage'].append((soma if soma is not None else nxt).detach())
                logs['coeff'].append(None if coeff is None else coeff.detach())
                for key in KEEP:
                    logs[key].append(stats[key])
            has_history[t] = self.has_history(t)
            u = nxt

        out = torch.stack(spikes)                                    # [T, B, out_features]
        if not return_aux:
            return (out, {'analog': torch.stack(live)}) if analog else out
        aux = {'spikes': out.detach(), 'coeff': logs['coeff'], 'has_history': has_history}
        if analog:
            aux['analog'] = torch.stack(live)
        aux.update({k: torch.stack(logs[k]) for k in ('state', 'voltage') + KEEP})

        return out, aux


class FLIFNeuron(LIFNeuron):
    """spikeDE f-LIF: the LIFNeuron right-hand side under a fractional (Caputo) solver.

    Args (in addition to LIFNeuron)
    ----
    alpha : float
        Fractional order in (0, 1]. alpha = 1 is the Euler LIF for 'pred' with the full history and for 'gl'
        with memory=None or memory >= 2 (see the module docstring).
    method : str
        'pred' (Adams-Bashforth predictor on the increments F, the paper scripts' default and
        v3's scheme) or 'gl' (Grunwald-Letnikov on the states, the README example).
    memory : int or None
        spikeDE's short-memory option: only the last `memory` steps enter. None = full history.
    """
    def __init__(self, size, alpha=.7, method='pred', memory=None, **kwargs):
        super().__init__(size, **kwargs)

        if not 0. < alpha <= 1.:
            raise ValueError('Require 0 < alpha <= 1')
        if method not in ('pred', 'gl'):
            raise ValueError("method must be 'pred' or 'gl'")
        if memory is not None and memory < 1:
            raise ValueError('memory must be None (full history) or >= 1')
        self.alpha = float(alpha)
        self.method = method
        self.memory = None if memory is None else int(memory)
        dtype = self.tau.dtype
        self.register_buffer('b', pred_coefficients(alpha, self.max_length, self.step, dtype))
        if method == 'gl' and self.reset_mode != 'spikede':
            raise NotImplementedError("reset options are defined for method 'pred'")
        if method == 'gl':
            self.register_buffer('c', gl_coefficients(alpha, self.max_length, dtype))
            self.h_alpha = self.step ** self.alpha

    def reset_gain(self):
        """Coefficient of F_t in v_{t+1} (pred: b_0)."""
        return self.b[0].item()

    def window_start(self, t):
        """spikeDE get_memory_bounds: the first history index that still enters at step t."""
        return 0 if self.memory is None else max(0, t + 1 - self.memory)

    def history_coefficients(self, t, start, u, x, history, b_hist, past, oracle_p=None):
        #  -> c: [B, D, J] weights of the stored increments, extra term or None, aux statistics
        """Without selection every stored increment keeps its kernel weight b(t - j)."""
        c = b_hist.expand(*u.shape[:2], b_hist.numel()).clone()      # layers의 mode='full'과 같은 경로

        return c, None, self._stats(c=c, b_hist=b_hist)

    def integrate(self, t, u, F, x, history, oracle_p=None):
        if self.method == 'gl':
            return self._integrate_gl(t, u, F, history)
        history['F'].append(F)
        b0 = self.b[0].item()
        nxt = b0 * F                                                 # 현재 스텝 항 b_0 F_t
        start = self.window_start(t)
        coeff, stats = None, self._stats(like=u)
        if t > start:
            past = torch.stack(history['F'][start:t], dim=-2)        # [B, D, J, K]
            b_hist = self.b[1:t - start + 1].flip(0)                 # b(t-j), 오래된 slot이 앞
            coeff, extra, stats = self.history_coefficients(t, start, u, x, history, b_hist, past, oracle_p)
            nxt = nxt + torch.einsum('bdj,bdjk->bdk', coeff, past)   # 과거 증분 재합산
            if extra is not None:
                nxt = nxt + extra

        return nxt, coeff, stats

    def _integrate_gl(self, t, u, F, history):
        """spikeDE GrunwaldLetnikovSNN: v_{t+1} = h^a F_t - sum_{j=start}^{t-1} c_{t-j} v_{j+1}."""
        nxt = self.h_alpha * F
        if t > 0:
            conv = 0
            for j in range(self.window_start(t), t):
                conv = conv + self.c[t - j] * history['v'][j]        # history['v'][j] = v_{j+1}
            nxt = nxt - conv
        history['v'].append(nxt)

        return nxt, None, self._neutral_stats(u, t)


class PopulationFLIF(FLIFNeuron):
    """K f-LIF members with their own time constants on the same input current.

    soma='lif' (default, the v3 arrangement, DH-SNN style): the members are non-resetting
    fractional branches and do not fire. One LIF soma per unit mixes them,
        a_t = sum_k w_k v_{t+1,k},  V_t = V_{t-1} + (a_t - V_{t-1}) / tau_s,
        s_t = H(V_t - theta),       V_t <- V_t - theta s_t      (layers.Soma, w learnable, 1/K)
    soma='none': the members are K independent spikeDE f-LIF neurons with their own reset;
    the output is K spike trains per unit, out_features = D * K.

    heterogeneous=False replaces the K time constants by their harmonic mean (the same total
    leak), as layers.PopulationNeuron does.
    """
    def __init__(self, size, tau=(4., 8., 16., 32.), soma='lif', tau_s=2., heterogeneous=True, **kwargs):
        super().__init__(size, tau=tau, **kwargs)

        if soma not in ('lif', 'none'):
            raise ValueError("soma must be 'lif' or 'none'")
        if not heterogeneous:
            K = self.num_members
            self.tau = torch.full_like(self.tau, float(K / (1. / self.tau).sum()))
        self.soma_mode = soma
        self.members_fire = soma == 'none'
        if soma == 'lif':
            self.soma = Soma(size, self.num_members, tau_s, self.threshold, self.surrogate_scale,
                             reset='zero' if self.reset_mode == 'zero' else 'subtract')   # MODI 2026-10-31: the soma keeps subtract unless 'zero'

    def _check_members(self):
        pass                                                         # K >= 1

    @property
    def has_soma(self):
        return self.soma_mode == 'lif'

    def initial_soma(self, x):
        return x.new_zeros(x.shape[1], x.shape[2]) if self.has_soma else None

    def emit(self, nxt, s, soma):
        if not self.has_soma:
            return super().emit(nxt, s, soma)
        v, spikes, drive = self.soma(nxt, soma)                      # 갱신된 가지 상태 u(t+1)을 읽는다 (D-D)

        return spikes, v, drive


class SelectiveFLIF(FLIFNeuron):
    """f-LIF whose past increments enter with c(t, j) = m(t, j) b(t - j), m in {0, 1}.

    m keeps the round(q J) past slots whose descriptor xi_j = [v_j, I_j] is most similar to the
    present xi_t (stat 'pearson' | 'cosine'), or the same number of slots without looking at
    the content ('recent' | 'random', controls). The present increment always enters with b_0.

    part='all' (default, the v3 design) drops whole increments. part='input' drops only their
    input part I_j/tau and keeps the leak and reset parts of every past slot (the stabilised
    candidate of the 2Q (A) diagnostic, not yet trained). q = 1 is exactly FLIFNeuron.
    """
    def __init__(self, size, q=.5, stat='pearson', axis='shared', part='all', seed=0, **kwargs):
        super().__init__(size, **kwargs)

        if self.method != 'pred':
            raise ValueError("selection acts on the stored increments F_j: method must be 'pred'")
        if not 0. < q or stat not in STATS or axis not in AXES or part not in PARTS:
            raise ValueError(f'q > 0, stat in {STATS}, axis in {AXES}, part in {PARTS}')
        self.q, self.stat, self.axis, self.part = float(q), stat, axis, part
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)

    def reseed(self, seed=None):
        """Reset the 'random' control's generator so an evaluation is reproducible."""
        self.generator.manual_seed(self.seed if seed is None else int(seed))

    def integrate(self, t, u, F, x, history, oracle_p=None):
        self._xi = torch.cat([u, x.unsqueeze(-1)], dim=-1)          # 질의는 갱신 전 상태로 (G6)
        if self.part == 'input':
            history['input'].append((x.unsqueeze(-1) / self.tau).expand_as(F))
        out = super().integrate(t, u, F, x, history, oracle_p)
        history['keys'].append(self._xi)

        return out

    def history_coefficients(self, t, start, u, x, history, b_hist, past, oracle_p=None):
        keys = torch.stack(history['keys'][start:t], dim=-2)         # [B, D, J, K+1]
        m, sim = select_mask(self._xi, keys, self.q, self.stat, self.axis, self.generator)
        c = b_hist * m                                               # 사전등록 2J: c = m * b
        extra = None
        if self.part == 'input' and self.q < 1.:
            # 버린 칸의 누설·리셋 부분은 원래 무게 b로 남긴다: sum (b - c)(F_j - I_j/tau)
            past_in = torch.stack(history['input'][start:t], dim=-2)
            extra = torch.einsum('bdj,bdjk->bdk', b_hist - c, past - past_in)
        stats = self._stats(c=c, b_hist=b_hist, m=m, sim=sim) if self.q < 1. else self._stats(c=c, b_hist=b_hist)

        return c, extra, stats


class SelectivePopulationFLIF(SelectiveFLIF, PopulationFLIF):
    """Population f-LIF with the shared selection: xi_j = [v_{j,1..K}, I_j] for every unit.
    With soma='lif', axis='shared', stat='pearson', part='all' this is layers.PopulationNeuron
    in mode 'hard'."""


NEURONS = {'lif': LIFNeuron, 'flif': FLIFNeuron, 'pop_flif': PopulationFLIF,
           'sel_flif': SelectiveFLIF, 'sel_pop_flif': SelectivePopulationFLIF}
NEUTRAL = {'sel_flif': 'flif', 'sel_pop_flif': 'pop_flif'}          # q = 1: 보정은 중립 극한과 공유
_BASE_ARGS = ('tau', 'threshold', 'surrogate_scale', 'step', 'max_length', 'dtype')
_ARGS = {LIFNeuron: _BASE_ARGS,
         FLIFNeuron: _BASE_ARGS + ('alpha', 'method', 'memory'),
         PopulationFLIF: _BASE_ARGS + ('alpha', 'method', 'memory', 'soma', 'tau_s', 'heterogeneous'),
         SelectiveFLIF: _BASE_ARGS + ('alpha', 'method', 'memory', 'q', 'stat', 'axis', 'part', 'seed')}
_ARGS[SelectivePopulationFLIF] = tuple(dict.fromkeys(_ARGS[PopulationFLIF] + _ARGS[SelectiveFLIF]))


def build_neuron(name, size, **kwargs):
    """The class `name` with only the keyword arguments it accepts."""
    if name not in NEURONS:
        raise ValueError(f'neuron must be one of {tuple(NEURONS)}')
    cls = NEURONS[name]

    return cls(size, **{k: v for k, v in kwargs.items() if k in _ARGS[cls]})
