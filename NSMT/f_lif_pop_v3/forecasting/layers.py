import math

import torch
import torch.nn as nn


__all__ = ['PopulationNeuron', 'Embedding', 'to_patches']

SELECTOR_MODES = ('full', 'dense', 'sparse', 'recent', 'mass_matched', 'oracle', 'hard')
HARD_AXES = ('unit', 'shared', 'input')
HARD_STATS = ('pearson', 'cosine', 'recent', 'random')      # recent/random: same-budget controls (2K)


class Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score):
        """
        score: (..., J) content scores over the past patches
        return: (..., J) euclidean projection onto the probability simplex
        """
        # Martins & Astudillo (2016), Alg.1
        score = score - score.max(dim=-1, keepdim=True).values
        ordered = score.sort(dim=-1, descending=True).values
        cumulative = ordered.cumsum(dim=-1)
        ranks = torch.arange(1, score.shape[-1] + 1, device=score.device, dtype=score.dtype)
        support_size = (1 + ranks * ordered > cumulative).sum(dim=-1, keepdim=True)
        threshold = (cumulative.gather(-1, support_size - 1) - 1) / support_size
        output = (score - threshold).clamp_min(0.)
        ctx.save_for_backward(output > 0)

        return output

    @staticmethod
    def backward(ctx, gradient):
        """
        gradient: (..., J)
        return: (..., J), Eq.14 in closed form
        """
        # sort/gather의 backward가 torch1.12 CUDA에서 비결정적이라 Eq.14를 직접 쓴다
        support, = ctx.saved_tensors
        selected = gradient * support
        mean = selected.sum(dim=-1, keepdim=True) / support.sum(dim=-1, keepdim=True)

        return support * (gradient - mean)


def sparsemax(score):
    return Sparsemax.apply(score)


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


def fractional_coefficients(alpha, length, dtype=torch.float32):
    """
    Adams-Bashforth-Moulton weights of the Caputo derivative (prereg D7).

    Args:
        alpha (float): fractional order in (0, 1]. alpha=1 gives b_d == 1 for every d,
                       which is what makes the alpha=1 Euler reduction exact (gate G2).
        length (int): table length; must cover the longest sequence the neuron sees.

    Returns:
        b (torch.Tensor): (length,) with b_d = [(d+1)^alpha - d^alpha] / Gamma(alpha+1).
                          Monotonically decreasing in d, so b_d <= b_0 for d >= 1 -- the
                          reason the R3 cap cannot fire in the neutral limit (gate G12).
    """
    if not 0. < alpha <= 1.:
        raise ValueError('Require 0 < alpha <= 1')
    d = torch.arange(length, dtype=torch.float64)                    # float64로 만들고 마지막에 캐스팅
    b = ((d + 1.).pow(alpha) - d.pow(alpha)) / math.exp(math.lgamma(alpha + 1.))

    return b.to(dtype)


class Selector(nn.Module):
    """Content-based read policy over the past increments (prereg rev.1 R3, D-A, D-K).

    The population shares one score, so every constituent re-reads the SAME past patch;
    only the branch time constant differs. Mass-preserving rescale keeps the kernel's total
    weight, and the cap keeps the moving feedback loop bounded:

        e(n,j)   = -||W_Q xi_n - W_K xi_j||^2 / (d_q * theta)
        p_n      = sparsemax(e_n) | softmax(e_n)
        rho(n,j) = (1 - eta) + eta * B_n * p(n,j) / sum_l b(n-l) p(n,l)
        c(n,j)   = min(b(n-j) * rho(n,j), b_0)

    eta=0 gives rho == 1 and b_d <= b_0, so the cap never fires and the branch is the bare
    f-LIF kernel bit for bit (gate G7, G12). A uniform p gives rho == 1 as well, which is why
    `uniform` is NOT a control mode here but a gate (G7b). The cap is what stops the state
    blowing up: mass conservation alone let an adversarial policy reach max|u| = 397,853
    (log 2026-09-21 22:19). D-A drops the tempered prior pi_k, so rho carries no k index.

    `mass_matched` matches the mass AFTER the cap (audit A01). Matching before it is vacuous:
    sum_j b_j rho_j = (1-eta) B + eta B = B identically, so the control collapses onto `full`
    (reproduced: control kappa 1.000000000000000, |control - full| = 1.11e-16 where the
    selection model's own kappa was 0.5698). kappa is a state-dependent quantity produced by
    the selector, so the honest name for this control is "total mass preserved, slot
    allocation removed" rather than "content-free". Its gradient is kept, so the control can
    also be trained from scratch rather than only injected at test time.
    """
    def __init__(self, num_population=4, query_dim=None, theta=1., eta_init=-4.,
                 eta_fixed=None, cap=True, key_norm='none', qk_norm=False, qk_eps=1e-6,
                 hard_axis='shared', hard_stat='pearson', hard_q=1., hard_seed=0):
        super().__init__()

        if key_norm not in ('none', 'frozen'):
            raise ValueError("key_norm must be 'none' or 'frozen'")
        # 2K same-budget controls: 'recent' keeps the k most recent slots, 'random' keeps k slots
        # drawn uniformly per query from a generator seeded here (fresh draws every forward;
        # an evaluator calls reseed_hard() before evaluating so the run is reproducible).
        # Neither looks at the content, so neither touches parameter creation (gate G19).
        self.hard_rng = torch.Generator().manual_seed(int(hard_seed))
        self.hard_seed = int(hard_seed)
        # mode='hard' (prereg 2J): the user's design. The kernel b(n-j) is kept exactly and each
        # past slot is either included or not, m in {0,1}, decided by a statistic between the
        # current descriptor and the past one. The statistic is never multiplied into a
        # coefficient, so there is no eta, no rescale and nothing for the cap to bound
        # (m*b <= b <= b_0). hard_q is the fraction of past slots kept per query; hard_q >= 1
        # is the neutral limit and must reproduce mode='full' bit for bit (gate G18).
        if hard_axis not in HARD_AXES or hard_stat not in HARD_STATS:
            raise ValueError(f'hard_axis in {HARD_AXES}, hard_stat in {HARD_STATS}')
        if not 0. < float(hard_q):
            raise ValueError('hard_q must be positive')
        self.hard_axis, self.hard_stat, self.hard_q = hard_axis, hard_stat, float(hard_q)
        # QK 정규화: 투영된 q, k를 x/sqrt(||x||^2 + eps^2)로 만든다. 하드 clamp가 아니라
        # soft norm인 이유는 Jacobian 때문이다. x/||x||의 Jacobian은 (I - xx^T)/||x||이라
        # ||x||->0에서 발산하는데, key는 [I 0]으로 초기화되어 k=u_j이고 초기 상태의 norm이
        # 0에 가깝다. 측정: eps=1e-6 하드 clamp에서 key gradient가 1.75e13까지 갔다.
        # soft norm의 Jacobian은 1/eps로 유계다. 시점 n의 q는 갱신 전 상태로만 만들어지므로
        # 인과성은 변하지 않는다.
        self.qk_norm, self.qk_eps = bool(qk_norm), float(qk_eps)
        self.key_norm = key_norm
        self.register_buffer('key_mean', torch.zeros(num_population + 1))
        self.register_buffer('key_std', torch.ones(num_population + 1))

        self.num_population = num_population
        self.query_dim = num_population if query_dim is None else query_dim
        self.theta = float(theta)
        if self.theta <= 0:
            raise ValueError('Require theta > 0')

        # xi = [u_1..u_K ; I] in R^{K+1}. D5: W_Q = W_K = [I 0], 즉 초기에는 상태만 비교한다
        self.query = nn.Linear(num_population + 1, self.query_dim, bias=False)
        self.key   = nn.Linear(num_population + 1, self.query_dim, bias=False)
        with torch.no_grad():
            self.query.weight.zero_()
            self.key.weight.zero_()
            eye = min(self.query_dim, num_population)
            self.query.weight[:eye, :eye] = torch.eye(eye)
            self.key.weight[:eye, :eye] = torch.eye(eye)
        self.eta_hat = nn.Parameter(torch.tensor(float(eta_init)))   # D-B: -4 -> eta ~ 0.018, 중립 출발
        self.cap = bool(cap)
        # eta_fixed는 sigmoid의 큰 logit 근사가 아니라 정확한 덮어쓰기여야 한다 (audit A06).
        # eta=0과 eta=1을 정확히 표현할 수 있어야 중립극한과 완전 배제가 성립한다.
        self.register_buffer('eta_value', torch.tensor(float('nan') if eta_fixed is None
                                                       else float(eta_fixed)))
        if eta_fixed is not None:
            if not 0. <= float(eta_fixed) <= 1.:
                raise ValueError('eta_fixed must lie in [0, 1]')
            self.eta_hat.requires_grad_(False)

    @property
    def eta(self):
        if torch.isnan(self.eta_value):
            return torch.sigmoid(self.eta_hat)

        return self.eta_value

    def reseed_hard(self, seed=None):
        """Reset the 'random' control's generator so an evaluation is reproducible (2K, D-AP)."""
        self.hard_rng.manual_seed(self.hard_seed if seed is None else int(seed))

    @torch.no_grad()
    def hard_mask(self, xi, xi_hist):
        #  xi: [B, D, K+1]   xi_hist: [B, D, J, K+1]   ->  m: [B, D, J] in {0,1},  sim: [B, D, J]
        """Top hard_q fraction of past slots by similarity; same rule as analysis/hard_mask_screen.py.

        Axes (the numbers are component counts, not sample sizes):
            unit    one unit's own xi, K+1 components, one decision per unit
            shared  every unit's xi flattened, D*(K+1), one decision for the population
            input   the I component across units, D, so the state never enters the decision
        pearson centres each descriptor by its own component mean, then cosine. A constant
        descriptor gets similarity 0, not NaN. No gradient: m is a hard 0/1 and sim is only
        reported.
        """
        B, D, J, F = xi_hist.shape
        if self.hard_q >= 1.:                                        # 중립 극한: 통계를 계산하지 않는다
            ones = xi_hist.new_ones(B, D, J)
            return ones, ones
        keep = max(1, int(round(self.hard_q * J)))                   # 고정 예산: 빈 mask가 없다
        if self.hard_stat in ('recent', 'random'):                   # 2K: 내용을 보지 않는 같은 예산 대조
            if self.hard_stat == 'recent':
                sim = torch.arange(J, dtype=xi_hist.dtype).expand(B, 1, J)          # 최근 칸일수록 큼
            else:
                sim = torch.rand(B, 1, J, generator=self.hard_rng).to(xi_hist.dtype)
            idx = sim.topk(keep, dim=-1).indices
            m = torch.zeros_like(sim).scatter_(-1, idx, 1.)
            return m.expand(B, D, J), sim.expand(B, D, J)
        if self.hard_axis == 'unit':
            q, k = xi.unsqueeze(-2), xi_hist
        elif self.hard_axis == 'shared':
            q = xi.reshape(B, 1, 1, D * F)
            k = xi_hist.permute(0, 2, 1, 3).reshape(B, 1, J, D * F)
        else:
            q = xi[..., -1].reshape(B, 1, 1, D)
            k = xi_hist[..., -1].permute(0, 2, 1).reshape(B, 1, J, D)
        if self.hard_stat == 'pearson':
            q, k = q - q.mean(-1, keepdim=True), k - k.mean(-1, keepdim=True)
        num = (q * k).sum(-1)
        den = (q.square().sum(-1).sqrt() * k.square().sum(-1).sqrt()).clamp_min(1e-12)
        sim = num / den                                              # [B, D or 1, J]
        idx = sim.topk(keep, dim=-1).indices
        m = torch.zeros_like(sim).scatter_(-1, idx, 1.)
        return m.expand(B, D, J), sim.expand(B, D, J)

    def forward(self, xi, xi_hist, b_hist, b0, mode='sparse', oracle_p=None):
        #  xi: [B, D, K+1]   xi_hist: [B, D, J, K+1]   b_hist: [J]   ->  c: [B, D, J]
        """
        Every mode leaves through the SAME path p -> rho -> c, so an intervention only
        swaps p. b_hist[i] is b(n-j) for the i-th stored slot (oldest first), i.e.
        flip(b[1..J]); the current-step term b_0 * f_n is added by the caller.
        """
        if mode not in SELECTOR_MODES:
            raise ValueError(f'Unknown selector mode {mode!r}')
        J = xi_hist.shape[-2]
        if J == 0:
            raise ValueError('Empty history is handled by the caller')
        cap = b_hist.new_tensor(float(b0))

        if mode == 'full':                                           # eta=0 경로를 점수 계산 없이 빠르게
            c = b_hist.expand(*xi.shape[:-1], J).clone()
            ones, zeros = torch.ones_like(c[..., 0]), torch.zeros_like(c[..., 0])
            # p는 None이다. b/sum(b)를 p라고 부르면 latent uniform p와 혼동된다 (audit A08).
            return c, {'p': None, 'rho': torch.ones_like(c), 'score': None,
                       'eta': zeros, 'kappa': ones, 'cap_rate': zeros,
                       'would_cap_rate': zeros, 'support_size': ones * J,
                       'score_std': zeros,
                       'support_p': ones, 'support_rho': ones, 'support_braw': ones,
                       'support_c': ones}

        if mode == 'hard':                                           # 사전등록 2J: c = m * b, 곱셈 없음
            m, sim = self.hard_mask(xi, xi_hist)
            c = b_hist * m
            zeros = torch.zeros_like(c[..., 0])
            frac = m.mean(-1)
            # p는 남긴 칸 위의 균등 분포다: 이 설계의 읽기 정책 그 자체이지 latent p가 아니다.
            return c, {'p': m / m.sum(-1, keepdim=True).clamp_min(1.), 'rho': m, 'score': sim,
                       'eta': zeros, 'kappa': c.sum(-1) / b_hist.sum(), 'cap_rate': zeros,
                       'would_cap_rate': zeros, 'support_size': m.sum(-1),
                       'score_std': sim.std(-1, unbiased=False),
                       'support_p': frac, 'support_rho': frac, 'support_braw': frac,
                       'support_c': frac}

        if self.key_norm == 'frozen':
            # 점수를 원시 상태 크기에서 떼어 놓는다. 이것이 없으면 d(score)/d(u)가 |u|에
            # 비례하고, 그 값이 T 스텝 되먹임으로 누적되어 역전파가 폭주한다.
            xi = (xi - self.key_mean) / self.key_std
            xi_hist = (xi_hist - self.key_mean) / self.key_std
        q, kk = self.query(xi), self.key(xi_hist)
        if self.qk_norm:
            e2 = self.qk_eps ** 2
            q = q / (q.square().sum(-1, keepdim=True) + e2).sqrt()
            kk = kk / (kk.square().sum(-1, keepdim=True) + e2).sqrt()
        score = -(q.unsqueeze(-2) - kk).square().sum(-1)
        score = score / (self.query_dim * self.theta)                # theta는 d_q 스케일로 고정 (O3)
        if mode == 'sparse' or mode == 'mass_matched':
            p = sparsemax(score)                                     # 정확한 0을 만든다 (gate G8b)
        elif mode == 'dense':
            p = torch.softmax(score, dim=-1)
        elif mode == 'recent':
            p = torch.zeros_like(score)
            p[..., -1] = 1.                                          # 직전 patch만 읽는 사소한 정책
        elif mode == 'oracle':
            if oracle_p is None:
                raise ValueError('oracle mode needs oracle_p')
            p = oracle_p.to(score.dtype)
        else:
            raise ValueError

        mass = b_hist.sum()                                          # B_n = sum_{j<n} b(n-j)
        denom = (b_hist * p).sum(-1, keepdim=True).clamp_min(1e-12)
        rho = (1. - self.eta) + self.eta * mass * p / denom           # 볼록 결합: eta=0이면 rho == 1
        raw = b_hist * rho
        would_cap = (raw > cap).to(raw.dtype).mean(-1)                # 정책의 잠재 초과율
        c = torch.minimum(raw, cap) if self.cap else raw             # R3 (--no-cap은 탐색 전용)
        if mode == 'mass_matched':
            # 상한을 적용한 뒤의 총질량을 맞춘다. 상한 이전에 맞추면 항상 B라서 full이 된다.
            # kappa <= 1 이고 b_j <= b_0 이므로 kappa*b_j <= b_0, 즉 이 대조군은 상한도 만족한다.
            c = (c.sum(-1, keepdim=True) / mass) * b_hist.expand_as(c)

        aux = {'p': p.detach(), 'rho': rho.detach(), 'score': score.detach(),
               'eta': self.eta.detach().expand(*c.shape[:-1]),
               'kappa': (c.sum(-1) / mass).detach(),                 # 상한이 물리면 1보다 작아진다
               # cap_rate는 반환된 계수가 실제로 잘린 비율이다. cap=False면 0이고,
               # mass_matched는 kappa*b <= b_0이라 구조적으로 0이다. 정책 자체의
               # 잠재 초과율은 would_cap_rate로 따로 본다 (audit 추적 §9.3).
               'cap_rate': ((c < raw - 1e-12).to(c.dtype).mean(-1).detach()
                            if mode != 'mass_matched' else torch.zeros_like(would_cap)),
               'would_cap_rate': would_cap.detach(),
               # D-K: 네 층위의 support는 서로 다르다. 각각 따로 보고한다 (gate G13).
               'support_p': (p > 0).to(c.dtype).mean(-1).detach(),
               'support_rho': (rho > 0).to(c.dtype).mean(-1).detach(),
               'support_braw': (raw > 0).to(c.dtype).mean(-1).detach(),
               'support_c': (c > 0).to(c.dtype).mean(-1).detach(),
               # singleton support에서는 sparsemax의 score gradient가 0이다. 비율을 본다.
               'support_size': (p > 0).sum(-1).to(c.dtype).detach(),
               # 과거가 1칸인 시점(n=1)에서 표본표준편차는 NaN이다. 모표준편차를 쓴다.
               'score_std': score.std(-1, unbiased=False).detach()}

        return c, aux


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
    def __init__(self, embed_dim, num_population=4, tau_s=2., threshold=1., surrogate_scale=5.):
        super().__init__()

        if tau_s <= 1. or threshold <= 0.:
            raise ValueError('Require tau_s > 1 and threshold > 0')
        self.weight = nn.Parameter(torch.full((embed_dim, num_population), 1. / num_population))
        self.tau_s = float(tau_s)
        self.threshold = float(threshold)
        self.surrogate_scale = float(surrogate_scale)                # O2: pinned 코드가 넘기는 5.0

    def forward(self, u, v):                                         # u: [B, D, K]   v: [B, D]
        a = (self.weight * u).sum(-1)                                # 구성원 혼합 -> 소마 입력 전류
        charge = v + (a - v) / self.tau_s
        s = arctan_spike(charge - self.threshold, self.surrogate_scale)

        return charge - self.threshold * s.detach(), s, a            # detach_reset


class PopulationNeuron(nn.Module):
    """One logical neuron = K non-resetting fractional branches + one soma (prereg v3-A).

    Same input current, same output spike; the population only differs inside. A branch does
    not carry its previous value forward -- every step re-sums the whole increment history
    with power-law weights, which is what makes the memory non-Markovian:

        f(n,k)   = (I_n - u(n,k)) / tau_k
        xi_n     = [u(n,1..K) ; I_n]                 query uses the PRE-update state (causal)
        c(n,j)   = min(b(n-j) * rho(n,j), b_0)       R3 cap, shared across k
        u(n+1,k) = b_0 f(n,k) + sum_{j<n} c(n,j) f(j,k)
        s_n      = Soma(u(n+1))                      D-D

    tau = [4, 8, 16, 32] (F1). The one-octave shift off [2, 4, 8, 16] is a stability fix, not a
    taste call: with the cap on, tau=2 still grew with sequence length (max|u| 10.86 at T=42 ->
    83.24 at T=84 under the worst policy) while tau>=4 stayed at 0.5503 for every eta and T.
    Cost is O(T^2) and that cost belongs entirely to the selection (gate G3 gives the closed
    form for the eta=0 branch).
    """
    def __init__(self, embed_dim, num_population=4, alpha=.7, tau=(4., 8., 16., 32.),
                 heterogeneous=True, max_length=64, theta=1., eta_init=-4.,
                 eta_fixed=None, cap=True, key_norm='none', qk_norm=False, qk_eps=1e-6, tau_s=2.,
                 threshold=1., surrogate_scale=5., query_dim=None, dtype=torch.float32,
                 hard_axis='shared', hard_stat='pearson', hard_q=1., hard_seed=0):
        super().__init__()

        # 계수표는 생성 시점의 dtype으로 만든다. float32로 만든 뒤 .double()로 올리면
        # 정밀도가 이미 소실되어 외부 float64 기준(G4)과 3e-8 어긋난다.
        tau = torch.as_tensor(tau, dtype=dtype)
        if tau.numel() != num_population or (tau <= 1.).any():
            raise ValueError('Require one tau per constituent, each > 1')
        if not heterogeneous:
            # 동질 대조군: 조화평균으로 총 누설을 맞춘다. tau=[4,8,16,32]이면 8.533
            tau = torch.full_like(tau, float(num_population / (1. / tau).sum()))
        self.register_buffer('tau', tau)
        self.register_buffer('b', fractional_coefficients(alpha, max_length, dtype))

        self.num_population = num_population
        self.alpha = float(alpha)
        self.max_length = int(max_length)
        self.selector = Selector(num_population, query_dim, theta, eta_init, eta_fixed, cap,
                                 key_norm, qk_norm, qk_eps, hard_axis, hard_stat, hard_q, hard_seed)
        self.soma = Soma(embed_dim, num_population, tau_s, threshold, surrogate_scale)

    def forward(self, x, mode='sparse', oracle_p=None, return_aux=False, analog=False):
        #  x: [T, B, D] input current (B = batch * channel flattened)  ->  spikes: [T, B, D]
        """
        oracle_p[n] is [B, D, n] when mode == 'oracle'; every other mode ignores it.

        `analog=True` additionally returns aux['analog'], the membrane sequence WITH its
        graph attached, so a readout placed on it trains the neuron end to end. aux['voltage']
        stays detached and remains a diagnostic. Keeping the two apart is deliberate: a probe
        on the detached value measures whether the information is present, while the analog
        readout measures whether the spike path is the bottleneck, and those are different
        claims (audit A08).
        """
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError('Expected nonempty [T, B, D] current')
        T, B, D = x.shape
        if T > self.max_length:
            raise ValueError(f'T={T} exceeds the coefficient table ({self.max_length})')

        b0 = self.b[0].item()
        u = x.new_zeros(B, D, self.num_population)                   # u_0 = 0, 리셋 없음
        v = x.new_zeros(B, D)
        incs, keys, spikes, live = [], [], [], []                    # f(j), xi_j, s_n, v_n(graph)
        keep = ('kappa', 'cap_rate', 'would_cap_rate', 'eta', 'support_p', 'support_rho',
                'support_braw', 'support_c', 'support_size', 'score_std')
        logs = {k: [] for k in ('state', 'voltage', 'coeff') + keep}

        for t in range(T):
            f = (x[t].unsqueeze(-1) - u) / self.tau                  # [B, D, K] 같은 전류, 다른 시간척도
            xi = torch.cat([u, x[t].unsqueeze(-1)], dim=-1)          # 질의는 갱신 전 상태로 (gate G6)
            c = None
            nxt = b0 * f                                             # 현재 스텝 항
            if t:
                past = torch.stack(incs, dim=-2)                     # [B, D, t, K]
                b_hist = self.b[1:t + 1].flip(0)                     # 오래된 slot이 앞 (b(n-j))
                c, aux = self.selector(xi, torch.stack(keys, dim=-2), b_hist, b0, mode,
                                       None if oracle_p is None else oracle_p[t])
                nxt = nxt + torch.einsum('bdj,bdjk->bdk', c, past)   # 과거 증분 재합산
            elif return_aux:
                # t=0은 history가 없다. 평균에서 빼도록 별도 mask로 표시한다 (audit A08).
                aux = {k: x.new_zeros(B, D) for k in keep}
                aux['eta'] = self.selector.eta.detach().expand(B, D)

            incs.append(f)
            keys.append(xi)
            u = nxt                                                  # u(t+1)
            v, s, drive = self.soma(u, v)                            # D-D: 갱신된 상태를 읽는다
            spikes.append(s)
            if analog:
                # 'analog'는 리셋 이후 막전위, 'drive'는 소마 이전 가지 혼합이다.
                # 전자는 스파이크 경로가 병목인지, 후자는 정보가 가지에 있는지를 묻는다.
                live.append(drive if analog == 'drive' else v)

            if return_aux:
                logs['state'].append(u.detach())
                logs['voltage'].append(v.detach())
                logs['coeff'].append(None if c is None else c.detach())
                for key in keep:
                    logs[key].append(aux[key])

        out = torch.stack(spikes)                                    # [T, B, D]
        if not return_aux:
            return (out, {'analog': torch.stack(live)}) if analog else out

        has_history = torch.zeros(T, dtype=torch.bool, device=x.device)
        has_history[1:] = True                                       # t=0은 집계에서 뺀다
        aux = {'spikes': out.detach(), 'coeff': logs['coeff'], 'has_history': has_history}
        if analog:
            aux['analog'] = torch.stack(live)                        # graph 유지
        aux.update({k: torch.stack(logs[k]) for k in ('state', 'voltage') + keep})

        return out, aux

    @torch.no_grad()
    def score_scale(self, x, mode='full'):
        """mean ||W_Q xi_n - W_K xi_j||^2 / d_q over all (n, j), at the current weights.

        This is the quantity theta has to match. With theta=1 and states of order 20 the raw
        scores span hundreds, sparsemax becomes a hard argmax, and the backward pass through
        T recurrent steps blows up (3.8e3 at eta=1, T=42, against 6e-3 at eta<=0.2). Setting
        theta to this scale puts the score spread at order 1.
        """
        aux = self(x, mode=mode, return_aux=True)[1]
        state, sel, total = aux['state'], self.selector, []
        for n in range(1, x.shape[0]):
            xi = torch.cat([state[n - 1], x[n].unsqueeze(-1)], dim=-1)
            hist = torch.stack([torch.cat([state[j - 1] if j else torch.zeros_like(state[0]),
                                           x[j].unsqueeze(-1)], dim=-1) for j in range(n)], dim=-2)
            if sel.key_norm == 'frozen':
                xi = (xi - sel.key_mean) / sel.key_std
                hist = (hist - sel.key_mean) / sel.key_std
            # 선택자가 실제로 쓰는 변환과 같아야 한다. qk_norm을 빠뜨리면
            # 정규화된 점수에 정규화 안 된 theta를 물리게 된다.
            q, kk = sel.query(xi), sel.key(hist)
            if sel.qk_norm:
                e2 = sel.qk_eps ** 2
                q = q / (q.square().sum(-1, keepdim=True) + e2).sqrt()
                kk = kk / (kk.square().sum(-1, keepdim=True) + e2).sqrt()
            total.append((q.unsqueeze(-2) - kk).square().sum(-1).mean().item())

        return float(sum(total) / len(total) / sel.query_dim)

    @torch.no_grad()
    def fit_key_norm(self, x, mode='full'):
        """Freeze the query/key standardisation from train-split states. Idempotent."""
        if self.selector.key_norm != 'frozen':
            return None
        aux = self(x, mode=mode, return_aux=True)[1]
        xi = torch.cat([torch.cat([torch.zeros_like(aux['state'][:1]), aux['state'][:-1]]),
                        x.unsqueeze(-1)], dim=-1)
        self.selector.key_mean.copy_(xi.mean(dim=(0, 1, 2)))
        self.selector.key_std.copy_(xi.std(dim=(0, 1, 2)).clamp_min(1e-6))

        return True

    @torch.no_grad()
    def fast_path(self, x):                                          # x: [T, B, D] -> [T, B, D, K]
        """Closed form of the neutral limit (eta=0). Gate G3 only, never the training path.

        u(n+1) = (B f)_n with f = (I - u)/tau and u = J(B f), so the branch operator is
        (Id + B J / tau)^-1 B / tau. Multiplying by B alone drops the leak feedback and is
        simply wrong (|err| = 0.945 where the correct operator gives 5e-16).
        """
        T = x.shape[0]
        b = self.b[:T].double()
        lag = torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)
        kernel = torch.where(lag >= 0, b[lag.clamp_min(0)], torch.zeros(1, dtype=b.dtype))
        shift = torch.eye(T, dtype=b.dtype).roll(1, 0)               # J: 한 칸 미루기
        shift[0] = 0.
        signal = x.double().reshape(T, -1)

        out = []
        for tau in self.tau.double():
            system = torch.eye(T, dtype=b.dtype) + kernel @ shift / tau
            out.append(torch.linalg.solve(system, kernel @ signal / tau))

        return torch.stack(out, dim=-1).reshape(T, *x.shape[1:], self.num_population)


def to_patches(x, patch_size):
    #  x: [B, L, C]  ->  [T, B*C, patch_size], T = L // patch_size
    """Chronological non-overlapping patches, channel-independent (v2 idiom, unchanged)."""
    B, L, C = x.shape
    x = x.transpose(1, 2).reshape(B * C, L)

    return x.unfold(-1, patch_size, patch_size).permute(1, 0, 2).contiguous()


class Embedding(nn.Module):
    """Patch -> input current -> population neuron. D11 fixes input_scale before training.

    ``input_norm='frozen'`` divides out a per-unit mean and std estimated ONCE on the train
    split and then frozen into buffers. It exists because input_scale is a gain and a gain
    cannot revive a unit whose drive is negative for every input -- on the recall task the
    cue is one-hot, so a unit's drive takes only n_keys discrete levels and a unit whose
    levels are all negative stays silent at any scale. Unlike BatchNorm this never reads
    batch statistics, so a window's output does not depend on who it is batched with
    (gate G9).
    """
    def __init__(self, patch_size, embed_dim, input_scale=2., input_norm='none', **neuron_args):
        super().__init__()

        if input_norm not in ('none', 'frozen'):
            raise ValueError("input_norm must be 'none' or 'frozen'")
        self.emb_linear = nn.Linear(patch_size, embed_dim, bias=True)
        self.input_scale = float(input_scale)
        self.input_norm = input_norm
        self.register_buffer('norm_mean', torch.zeros(embed_dim))
        self.register_buffer('norm_std', torch.ones(embed_dim))
        self.neuron = PopulationNeuron(embed_dim, **neuron_args)

    def current(self, x):
        #  x: [T, B, patch_size]  ->  [T, B, D] input current
        z = self.emb_linear(x)
        if self.input_norm == 'frozen':
            z = (z - self.norm_mean) / self.norm_std

        return z * self.input_scale

    @torch.no_grad()
    def fit_norm(self, patches):
        """Freeze the per-unit standardisation from train-split patches. Idempotent."""
        z = self.emb_linear(patches)
        self.norm_mean.copy_(z.mean(dim=(0, 1)))
        self.norm_std.copy_(z.std(dim=(0, 1)).clamp_min(1e-6))

    def forward(self, x, mode='sparse', oracle_p=None, return_aux=False, analog=False):
        #  x: [T, B, patch_size]  ->  spikes: [T, B, D]
        return self.neuron(self.current(x), mode, oracle_p, return_aux, analog)
