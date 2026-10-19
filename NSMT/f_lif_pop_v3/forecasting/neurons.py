"""Neuron family on the original spikeDE code: LIF -> f-LIF -> population / selective / soma.

Every spiking computation here is a call into spikede.py, a verbatim copy of PhysAGI/spikeDE
(commit fcd743befe504b1a471fa81887e6af7d6789da2e, MIT; "Fractional-order Spiking Neural Network",
Ge et al., ICLR 2026). spikede_copy.py rebuilds and checks that copy against upstream.

spikeDE splits a neuron into two parts:
    neuron  spikede.LIFNeuron.forward(v_k, I_k) -> (dv/dt, s_k); it never changes v itself
                drift = (I - v)/tau,  s = H(v + drift - theta),  dv/dt = drift - theta s.detach()/tau
    solver  the time loop that turns dv/dt into v_{k+1}
                LIF   spikede.euler_integrate_tuple          v_{k+1} = v_k + h f_k
                f-LIF spikede.snn_solve + AdamsBashforthSNN  v_{k+1} = sum_j b_{j,k+1} f_j   (pred)
                      spikede.snn_solve + GrunwaldLetnikovSNN                                (gl)
The fractional order alpha lives in the solver's weights only.

Behaviour components (one behaviour each):
    Population   K spikeDE LIFNeuron members with their own tau on the same input current
    Selective    a spikeDE AdamsBashforthSNN whose convolution keeps only the selected past
                 increments, sum_j m_j b_{j,k+1} f_j (m: top-q mask by the similarity of [v, I])
    Soma         outside the solver: reads the members' UPDATED membrane u_{k+1}, mixes it with a
                 learnable per-unit weight and fires as a spikeDE LIFNeuron under spikeDE Euler

Neuron classes (all inherit FLIFNeuron, the components are instances made in __init__):
    LIFNeuron                         spikeDE LIF + Euler
    FLIFNeuron(LIFNeuron)             spikeDE LIF + fractional solver
    PopulationFLIF                    + Population                       (K firing members)
    SelectiveFLIF                     + Selective
    PopulationSelectiveFLIF           + Population + Selective           (K firing members)
    FLIFSoma, PopulationFLIFSoma, SelectiveFLIFSoma, PopulationSelectiveFLIFSoma
                                      the same with a Soma: the members are silent (threshold
                                      SILENT, so s = 0 and the reset term vanishes) and only the
                                      soma fires. PopulationSelectiveFLIFSoma is the v3 idea.

All classes take a [T, B, D] input current and return [T, B, out_features] spikes, with the aux
dictionary of layers.PopulationNeuron so the pipeline is unchanged. The legacy v3 neuron
(layers.PopulationNeuron) is not touched; against it these classes agree to rounding, not
bitwise, and the soma resets by theta/tau_s (spikeDE) instead of theta (v3).
"""
import math

import torch
import torch.nn as nn

import spikede

__all__ = ['Population', 'Selective', 'Soma', 'LIFNeuron', 'FLIFNeuron', 'PopulationFLIF', 'SelectiveFLIF',
           'PopulationSelectiveFLIF', 'FLIFSoma', 'PopulationFLIFSoma', 'SelectiveFLIFSoma',
           'PopulationSelectiveFLIFSoma', 'NEURONS', 'NEUTRAL', 'DEFAULT_SOMA', 'neuron_class', 'build_neuron',
           'pred_coefficients', 'select_mask', 'SILENT']

SILENT = 1e30                                                        # 발화하지 않는 구성원의 임계값 (결정 2-a)
STATS = ('pearson', 'cosine', 'recent', 'random')
AXES = ('unit', 'shared', 'input')
PARTS = ('all', 'input')
KEEP = ('kappa', 'cap_rate', 'would_cap_rate', 'eta', 'support_p', 'support_rho', 'support_braw',
        'support_c', 'support_size', 'score_std')                   # layers.PopulationNeuron의 aux 키


def pred_coefficients(alpha, length, step=1., dtype=torch.float32):
    """b_d = h^a / Gamma(a+1) [(d+1)^a - d^a], d = 0..length-1: the pred kernel as a table.
    Diagnostics only (neuron.b, aux['coeff'] without selection); the solver computes its own
    weights in AdamsBashforthSNN.compute_weights_for_layer, equal to this up to rounding."""
    if not 0. < alpha <= 1.:
        raise ValueError('Require 0 < alpha <= 1')
    d = torch.arange(length, dtype=torch.float64)
    b = ((d + 1.).pow(alpha) - d.pow(alpha)) / math.exp(math.lgamma(alpha + 1.))
    if step != 1.:
        b = b * step ** alpha

    return b.to(dtype)


@torch.no_grad()
def select_mask(xi, xi_hist, q, stat='pearson', axis='shared', generator=None):
    #  xi: [B, D, F]   xi_hist: [B, D, J, F]   ->   m: [B, D, J] in {0, 1},  sim: [B, D, J]
    """Keep the round(q J) past slots most similar to the present; layers.Selector.hard_mask.

    axis 'shared' flattens every unit's descriptor into one vector (one decision for the whole
    population of units), 'unit' decides per unit, 'input' looks at the input current only.
    'recent' and 'random' keep the same number of slots without looking at the content. The
    random mask comes from a CPU generator and is moved to the data's device. No gradient.
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


def stats_of(c=None, b_hist=None, m=None, sim=None, like=None):
    """aux statistics in layers.Selector's format: zeros without history, neutral without selection."""
    if c is None:                                                    # 이력 없음 (t=0)
        zeros = like.new_zeros(like.shape[:2])
        return {k: zeros for k in KEEP}
    J = c.shape[-1]
    zeros = torch.zeros_like(c[..., 0])
    if m is None:                                                    # 선택 없음: layers의 mode='full'
        ones = torch.ones_like(zeros)
        return {'eta': zeros, 'kappa': ones, 'cap_rate': zeros, 'would_cap_rate': zeros,
                'support_size': ones * J, 'score_std': zeros, 'support_p': ones,
                'support_rho': ones, 'support_braw': ones, 'support_c': ones}
    frac = m.mean(-1)                                                # layers의 mode='hard'
    return {'eta': zeros, 'kappa': c.sum(-1) / b_hist.sum(), 'cap_rate': zeros,
            'would_cap_rate': zeros, 'support_size': m.sum(-1),
            'score_std': sim.std(-1, unbiased=False), 'support_p': frac, 'support_rho': frac,
            'support_braw': frac, 'support_c': frac}


# ==== behaviour components =====================================================================
class Population(nn.Module):
    """Population behaviour: K spikeDE LIFNeuron members, each with its own tau, same input current.

    heterogeneous=False replaces the K time constants by their harmonic mean (the same total
    leak), as layers.PopulationNeuron does. threshold=SILENT makes the members non-firing.
    """
    def __init__(self, tau=(4., 8., 16., 32.), heterogeneous=True, threshold=1., surrogate_scale=5.):
        super().__init__()

        taus = [float(t) for t in torch.as_tensor(tau, dtype=torch.float64).reshape(-1)]
        if not taus or min(taus) <= 0.:
            raise ValueError('Require at least one tau, all > 0')
        if not heterogeneous:
            taus = [len(taus) / sum(1. / t for t in taus)] * len(taus)
        self.cells = nn.ModuleList(spikede.LIFNeuron(tau=t, threshold=threshold, surrogate_grad_scale=surrogate_scale)
                                   for t in taus)

    @property
    def num_members(self):
        return len(self.cells)


class Selective(spikede.AdamsBashforthSNN):
    """Selective behaviour as a spikeDE solver: AdamsBashforthSNN whose convolution keeps only the
    selected past increments,

        v_{k+1} = b_{k,k+1} f_k + sum_{j<k} m_j b_{j,k+1} f_j,   m_j in {0, 1} (select_mask),

    the present increment always entering. The query xi_k = [v_k(all members), I_k] uses the
    PRE-update state. part='all' drops whole increments (v3); part='input' drops only their input
    part I_j/tau and keeps the leak and reset parts (the 2Q (A) candidate). q = 1 is AdamsBashforthSNN
    itself: every call goes to the parent unchanged.

    The neuron drives it: `start` before a sequence, `observe` from its right-hand side at each step.
    """
    def __init__(self, q=.5, stat='pearson', axis='shared', part='all', seed=0):
        if not 0. < q or stat not in STATS or axis not in AXES or part not in PARTS:
            raise ValueError(f'q > 0, stat in {STATS}, axis in {AXES}, part in {PARTS}')
        self.q, self.stat, self.axis, self.part = float(q), stat, axis, part
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)

    def reseed(self, seed=None):
        """Reset the 'random' control's generator so an evaluation is reproducible."""
        self.generator.manual_seed(self.seed if seed is None else int(seed))

    def start(self, length, taus):
        """A new sequence of `length` steps for members with time constants `taus`."""
        self.length, self.taus = int(length), list(taus)
        self.keys, self.inputs = [], []
        self.coeff, self.stats = {}, {}
        self._mask_step, self._mask = None, None

    def observe(self, k, v, x):
        #  v: [B, D, K] the members' v_k   x: [B, D] I_k
        """Called by the right-hand side at step k, before the solver uses step k."""
        self.query = torch.cat([v, x.unsqueeze(-1)], dim=-1)        # xi_k, 갱신 전 상태 (G6)
        self.keys.append(self.query)
        self.inputs.append(x)

    def mask(self, k, start_idx, weights):
        """m over the past slots start_idx..k-1, chosen once per step and shared by all members."""
        if self._mask_step != k:
            keys = torch.stack(self.keys[start_idx:k], dim=-2)       # [B, D, J, K+1]
            m, sim = select_mask(self.query, keys, self.q, self.stat, self.axis, self.generator)
            b_hist = weights[:-1]                                    # b_{j,k+1}, j = start..k-1
            c = b_hist * m
            self.coeff[k] = c.detach()
            self.stats[k] = stats_of(c=c, b_hist=b_hist, m=m, sim=sim)
            self._mask_step, self._mask = k, m
        return self._mask

    def compute_convolution(self, k, start_idx, weights, history_i, config, layer_idx):
        if self.q >= 1. or k >= self.length or k <= start_idx:     # 선택 없음 / 덧붙인 마지막 스텝 / 과거 없음
            return super().compute_convolution(k, start_idx, weights, history_i, config, layer_idx)
        m = self.mask(k, start_idx, weights)
        if self.part == 'input':
            # 모든 칸의 증분을 원래 무게 b 로 더한 뒤 (spikeDE 그대로), 버린 칸의 입력 부분 I_j/tau 만 뺀다:
            # 누설·리셋 부분은 모든 칸에 남는다.   sum_j b_j f_j - sum_{j<k} b_j (1 - m_j) I_j / tau
            convolution_sum = super().compute_convolution(k, start_idx, weights, history_i, config, layer_idx)
            tau = self.taus[layer_idx]
            for j in range(start_idx, k):
                local_idx = j - start_idx
                convolution_sum = convolution_sum - weights[local_idx] * (1. - m[..., local_idx]) * self.inputs[j] / tau
            return convolution_sum
        convolution_sum = 0
        for j in range(start_idx, k + 1):                            # spikeDE 와 같은 순서의 합
            local_idx = j - start_idx
            w = weights[local_idx] if j == k else weights[local_idx] * m[..., local_idx]
            convolution_sum = convolution_sum + w * history_i[j]
        return convolution_sum


class Soma(nn.Module):
    """Soma: the firing site outside the solver (DH-SNN arrangement, v3 D-D timing).

        a_k = sum_m w[d, m] u_{k+1, m}      per-unit learnable mix of the members' UPDATED membrane
                                            (w: [D, K], initial 1/K, no sign constraint)
        V, s : spikeDE LIFNeuron(tau_s, threshold) driven by a_k under spikeDE Euler (h = 1)
               -> s_k = H(V_k + (a_k - V_k)/tau_s - theta),  V_{k+1} = that - theta s_k / tau_s
    """
    def __init__(self, size, num_members=1, tau_s=2., threshold=1., surrogate_scale=5., dtype=torch.float32):
        super().__init__()

        if tau_s <= 0. or threshold <= 0.:
            raise ValueError('Require tau_s > 0 and threshold > 0')
        self.weight = nn.Parameter(torch.full((size, num_members), 1. / num_members, dtype=dtype))
        self.neuron = spikede.LIFNeuron(tau=float(tau_s), threshold=float(threshold),
                                        surrogate_grad_scale=float(surrogate_scale))

    def forward(self, u):
        #  u: [T, B, D, K] the members' u_1..u_T (after each solver step)   ->   s, V, a: [T, B, D]
        a = (self.weight * u).sum(-1)                                # 구성원 혼합 = soma 입력 전류
        T = a.shape[0]

        def rhs(t, y):                                               # spikeDE 우변, k 번째 입력 a_k
            return self.neuron(y[0], a[int(round(float(t)))])

        zero = a.new_zeros(a.shape[1:])
        t_grid = torch.arange(T + 1, dtype=a.dtype, device=a.device)
        hist = spikede.euler_integrate_tuple(rhs, (zero, zero), t_grid, 1)   # spikeDE Euler, h = 1
        spikes = torch.stack(hist[1]).to(a.dtype)                    # s_0..s_{T-1}
        voltage = torch.stack(hist[0])                               # V_1..V_T (리셋 이후)

        return spikes, voltage, a


# ==== neurons ==================================================================================
class LIFNeuron(nn.Module):
    """LIF: spikeDE LIFNeuron (right-hand side) integrated by spikeDE's Euler solver.

    Args
    ----
    size : int             D, the number of units.
    tau : float or None    membrane time constant of the single member; None when a Population
                           supplies the members (subclasses).
    threshold : float      theta (SILENT: the member never fires).
    surrogate_scale : float  kappa of spikeDE's arctan surrogate (spikeDE passes 5.0).
    step : float           h, the solver step. spikeDE's spike look-ahead keeps dt = 1.
    max_length : int       longest sequence; sizes the diagnostic kernel table b.
    """
    alpha, method, memory = 1., 'euler', None

    def __init__(self, size, tau=2., threshold=1., surrogate_scale=5., step=1., max_length=64,
                 dtype=torch.float32):
        super().__init__()

        if threshold <= 0. or step <= 0. or max_length < 1:
            raise ValueError('Require threshold > 0, step > 0, max_length >= 1')
        if tau is not None and torch.as_tensor(tau).numel() != 1:
            raise ValueError(f'{type(self).__name__} has one member: give one tau (several taus are a Population)')
        if tau is not None and float(torch.as_tensor(tau).reshape(-1)[0]) <= 0.:
            raise ValueError('Require tau > 0')
        self.size = int(size)
        self.threshold = float(threshold)
        self.surrogate_scale = float(surrogate_scale)
        self.step = float(step)
        self.max_length = int(max_length)
        self.dtype = dtype
        self.member = None if tau is None else spikede.LIFNeuron(
            tau=float(torch.as_tensor(tau).reshape(-1)[0]), threshold=self.threshold,
            surrogate_grad_scale=self.surrogate_scale)
        self.population = None                                       # 구성 요소: 하위 클래스가 인스턴스를 넣는다
        self.selective = None
        self.soma = None
        self.register_buffer('b', pred_coefficients(1., max_length, step, dtype))   # alpha=1: 모두 h

    # ---- structure -------------------------------------------------------------------------
    @property
    def cells(self):
        return list(self.population.cells) if self.population is not None else [self.member]

    @property
    def num_members(self):
        return len(self.cells)

    @property
    def tau(self):
        return torch.tensor([c.tau for c in self.cells], dtype=self.b.dtype, device=self.b.device)

    @property
    def has_soma(self):
        return self.soma is not None

    @property
    def out_features(self):
        return self.size if self.has_soma else self.size * self.num_members

    def solver(self):
        """None = spikeDE Euler; otherwise a spikeDE SNNFractionalMethod instance."""
        return None

    def window_start(self, t):
        """First history index that still enters at step t (spikeDE get_memory_bounds)."""
        return 0 if self.memory is None else max(0, t + 1 - self.memory)

    def reseed(self, seed=None):
        if self.selective is not None:
            self.selective.reseed(seed)

    # ---- the sequence --------------------------------------------------------------------------
    def integrate(self, x):
        #  x: [T, B, D]   ->   u: [T, B, D, K] u_1..u_T (graph attached),  member spikes [T, B, D, K]
        """The members under the spikeDE solver. The states are read as the solver hands them to the
        right-hand side (pred keeps no state history); one extra step with zero input exposes u_T and
        cannot change an earlier step."""
        T, B, D = x.shape
        cells, K = self.cells, self.num_members
        zero, seen = x.new_zeros(B, D), []
        if self.selective is not None:
            self.selective.start(T, [c.tau for c in cells])

        def rhs(t, y):                                               # spikeDE ode_func
            k = int(round(float(t) / self.step))
            current = x[k] if k < T else zero
            seen.append(torch.stack(y[:K], dim=-1))                  # v_k (갱신 전)
            if self.selective is not None:
                self.selective.observe(k, seen[-1], current)
            out = [cell(v_m, current) for cell, v_m in zip(cells, y[:K])]
            return tuple(o[0] for o in out) + tuple(o[1] for o in out)

        y0 = tuple(x.new_zeros(B, D) for _ in range(2 * K))
        t_grid = torch.arange(T + 2, dtype=x.dtype, device=x.device) * self.step
        method = self.solver()
        if method is None:
            hist = spikede.euler_integrate_tuple(rhs, y0, t_grid, K)
        else:
            hist = spikede.snn_solve(rhs, y0, [self.alpha] * K, t_grid, method, memory=self.memory)
        u = torch.stack(seen[1:T + 1])                               # u_1..u_T (갱신 후)
        s = torch.stack([torch.stack(hist[K + m][:T]) for m in range(K)], dim=-1).to(x.dtype)

        return u, s

    def step_record(self, t, like):
        """aux coefficient and statistics of step t."""
        start = self.window_start(t)
        if t <= start:
            return None, stats_of(like=like)
        if self.selective is not None and t in self.selective.coeff:
            return self.selective.coeff[t], self.selective.stats[t]
        if self.method == 'pred':                                    # 선택 없음: 커널 그대로 (진단용 표)
            b_hist = self.b[1:t - start + 1].flip(0)
            c = b_hist.expand(*like.shape[:2], b_hist.numel()).clone()
            return c, stats_of(c=c, b_hist=b_hist)
        c = like.new_ones(*like.shape[:2], t - start)                # Euler / gl: 계수 텐서 없음
        return None, stats_of(c=c, b_hist=c[0, 0])

    def forward(self, x, mode=None, oracle_p=None, return_aux=False, analog=False):
        #  x: [T, B, D] input current   ->   spikes: [T, B, out_features]
        """`mode` / `oracle_p` are accepted for layers.Embedding's call signature and ignored: the
        behaviour is fixed at construction. `analog` ('analog' | 'drive') returns the soma membrane
        after reset or the soma input with the graph attached (classes with a soma only)."""
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError('Expected nonempty [T, B, D] current')
        T, B, D = x.shape
        if D != self.size:
            raise ValueError(f'Expected D={self.size}, got {D}')
        if T > self.max_length:
            raise ValueError(f'T={T} exceeds max_length ({self.max_length})')
        if analog and not self.has_soma:
            raise ValueError('analog/drive readouts read the soma; this neuron has none')

        u, member_spikes = self.integrate(x)                         # f-LIF (solver)
        if self.has_soma:
            out, voltage, drive = self.soma(u)                       # soma: 갱신 후 막전위 u_{k+1} 를 읽는다
        else:
            out, voltage, drive = member_spikes.reshape(T, B, D * self.num_members), u, None
        live = (drive if analog == 'drive' else voltage) if analog else None
        if not return_aux:
            return (out, {'analog': live}) if analog else out

        records = [self.step_record(t, x[0]) for t in range(T)]
        aux = {'spikes': out.detach(), 'coeff': [r[0] for r in records], 'state': u.detach(),
               'voltage': voltage.detach(),
               'has_history': torch.tensor([t > self.window_start(t) for t in range(T)], device=x.device)}
        aux.update({k: torch.stack([r[1][k] for r in records]) for k in KEEP})
        if analog:
            aux['analog'] = live

        return out, aux


class FLIFNeuron(LIFNeuron):
    """f-LIF: the same spikeDE LIFNeuron right-hand side under spikeDE's fractional solver.

    alpha : fractional order in (0, 1]; 1 is the Euler LIF.
    method : 'pred' (AdamsBashforthSNN, the paper scripts' default) or 'gl' (GrunwaldLetnikovSNN).
    memory : spikeDE's short memory, only the last `memory` steps enter; None = full history.
    """
    def __init__(self, size, alpha=.7, method='pred', memory=None, **kwargs):
        super().__init__(size, **kwargs)

        if not 0. < alpha <= 1.:
            raise ValueError('Require 0 < alpha <= 1')
        if method not in ('pred', 'gl'):
            raise ValueError("method must be 'pred' or 'gl'")
        if memory is not None and memory < 1:
            raise ValueError('memory must be None (full history) or >= 1')
        self.alpha, self.method = float(alpha), method
        self.memory = None if memory is None else int(memory)
        self.register_buffer('b', pred_coefficients(alpha, self.max_length, self.step, self.dtype))

    def solver(self):
        if self.selective is not None:
            return self.selective
        return spikede.AdamsBashforthSNN() if self.method == 'pred' else spikede.GrunwaldLetnikovSNN()

    def _require_pred(self):
        if self.method != 'pred':
            raise ValueError("selection acts on the stored increments f_j: method must be 'pred'")


class PopulationFLIF(FLIFNeuron):
    """f-LIF + Population: K firing f-LIF members, K spike trains per unit (out_features = D K)."""
    def __init__(self, size, tau=(4., 8., 16., 32.), heterogeneous=True, **kwargs):
        super().__init__(size, tau=None, **kwargs)
        self.population = Population(tau, heterogeneous, self.threshold, self.surrogate_scale)


class SelectiveFLIF(FLIFNeuron):
    """f-LIF + Selective."""
    def __init__(self, size, q=.5, stat='pearson', axis='shared', part='all', seed=0, **kwargs):
        super().__init__(size, **kwargs)
        self._require_pred()
        self.selective = Selective(q, stat, axis, part, seed)


class PopulationSelectiveFLIF(FLIFNeuron):
    """f-LIF + Population + Selective: K firing members sharing one selection."""
    def __init__(self, size, tau=(4., 8., 16., 32.), heterogeneous=True, q=.5, stat='pearson', axis='shared',
                 part='all', seed=0, **kwargs):
        super().__init__(size, tau=None, **kwargs)
        self._require_pred()
        self.population = Population(tau, heterogeneous, self.threshold, self.surrogate_scale)
        self.selective = Selective(q, stat, axis, part, seed)


class FLIFSoma(FLIFNeuron):
    """f-LIF with soma: a silent f-LIF member, then the Soma reads its updated membrane and fires."""
    def __init__(self, size, tau_s=2., threshold=1., surrogate_scale=5., dtype=torch.float32, **kwargs):
        super().__init__(size, threshold=SILENT, surrogate_scale=surrogate_scale, dtype=dtype, **kwargs)
        self.soma = Soma(size, 1, tau_s, threshold, surrogate_scale, dtype)


class PopulationFLIFSoma(FLIFNeuron):
    """f-LIF + Population + Soma: K silent members mixed by one soma per unit (v3 'full')."""
    def __init__(self, size, tau=(4., 8., 16., 32.), heterogeneous=True, tau_s=2., threshold=1.,
                 surrogate_scale=5., dtype=torch.float32, **kwargs):
        super().__init__(size, tau=None, threshold=SILENT, surrogate_scale=surrogate_scale, dtype=dtype, **kwargs)
        self.population = Population(tau, heterogeneous, SILENT, surrogate_scale)
        self.soma = Soma(size, self.population.num_members, tau_s, threshold, surrogate_scale, dtype)


class SelectiveFLIFSoma(FLIFNeuron):
    """f-LIF + Selective + Soma."""
    def __init__(self, size, q=.5, stat='pearson', axis='shared', part='all', seed=0, tau_s=2., threshold=1.,
                 surrogate_scale=5., dtype=torch.float32, **kwargs):
        super().__init__(size, threshold=SILENT, surrogate_scale=surrogate_scale, dtype=dtype, **kwargs)
        self._require_pred()
        self.selective = Selective(q, stat, axis, part, seed)
        self.soma = Soma(size, 1, tau_s, threshold, surrogate_scale, dtype)


class PopulationSelectiveFLIFSoma(FLIFNeuron):
    """The v3 idea: f-LIF + Population + Selective + Soma. K silent members share one selection on
    xi = [v_{1..K}, I]; one soma per unit mixes their updated membrane and fires."""
    def __init__(self, size, tau=(4., 8., 16., 32.), heterogeneous=True, q=.5, stat='pearson', axis='shared',
                 part='all', seed=0, tau_s=2., threshold=1., surrogate_scale=5., dtype=torch.float32, **kwargs):
        super().__init__(size, tau=None, threshold=SILENT, surrogate_scale=surrogate_scale, dtype=dtype, **kwargs)
        self._require_pred()
        self.population = Population(tau, heterogeneous, SILENT, surrogate_scale)
        self.selective = Selective(q, stat, axis, part, seed)
        self.soma = Soma(size, self.population.num_members, tau_s, threshold, surrogate_scale, dtype)


# ==== registry ===================================================================================
NEURONS = {'lif': (LIFNeuron, None), 'flif': (FLIFNeuron, FLIFSoma),
           'pop_flif': (PopulationFLIF, PopulationFLIFSoma), 'sel_flif': (SelectiveFLIF, SelectiveFLIFSoma),
           'sel_pop_flif': (PopulationSelectiveFLIF, PopulationSelectiveFLIFSoma)}   # (soma 없음, soma 있음)
DEFAULT_SOMA = {'lif': 'none', 'flif': 'none', 'sel_flif': 'none', 'pop_flif': 'lif', 'sel_pop_flif': 'lif'}
NEUTRAL = {'sel_flif': 'flif', 'sel_pop_flif': 'pop_flif'}          # q = 1: 보정은 중립 극한과 공유
_BASE = ('tau', 'threshold', 'surrogate_scale', 'step', 'max_length', 'dtype')
_FRAC = _BASE + ('alpha', 'method', 'memory')
_POP, _SEL, _SOMA = ('heterogeneous',), ('q', 'stat', 'axis', 'part', 'seed'), ('tau_s',)
_ARGS = {LIFNeuron: _BASE, FLIFNeuron: _FRAC, PopulationFLIF: _FRAC + _POP, SelectiveFLIF: _FRAC + _SEL,
         PopulationSelectiveFLIF: _FRAC + _POP + _SEL, FLIFSoma: _FRAC + _SOMA,
         PopulationFLIFSoma: _FRAC + _POP + _SOMA, SelectiveFLIFSoma: _FRAC + _SEL + _SOMA,
         PopulationSelectiveFLIFSoma: _FRAC + _POP + _SEL + _SOMA}


def neuron_class(name, soma=None):
    """The class for a registry name and soma ('lif' | 'none'; None = the name's default)."""
    if name not in NEURONS:
        raise ValueError(f'neuron must be one of {tuple(NEURONS)}')
    soma = DEFAULT_SOMA[name] if soma is None else soma
    if soma not in ('lif', 'none'):
        raise ValueError("soma must be 'lif' or 'none'")
    cls = NEURONS[name][soma == 'lif']
    if cls is None:
        raise ValueError(f"{name} has no soma variant")

    return cls


def build_neuron(name, size, soma=None, **kwargs):
    """The class (name, soma) with only the keyword arguments it accepts."""
    cls = neuron_class(name, soma)

    return cls(size, **{k: v for k, v in kwargs.items() if k in _ARGS[cls]})
