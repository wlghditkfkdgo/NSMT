import math

import torch
import torch.nn as nn


__all__ = ['PopulationNeuron', 'Embedding']

SELECTOR_MODES = ('full', 'dense', 'sparse', 'recent', 'mass_matched', 'oracle')


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
    """
    def __init__(self, num_population=4, query_dim=None, theta=1., eta_init=-4.):
        super().__init__()

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

    @property
    def eta(self):
        return torch.sigmoid(self.eta_hat)

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
            return c, {'p': c / c.sum(-1, keepdim=True), 'rho': torch.ones_like(c),
                       'score': torch.zeros_like(c), 'eta': zeros,
                       'kappa': ones, 'cap_rate': zeros}

        score = -(self.query(xi).unsqueeze(-2) - self.key(xi_hist)).square().sum(-1)
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
        if mode == 'mass_matched':
            # D4 대조군: 정책의 총질량만 남기고 내용 의존성을 지운다 (감쇠와 구분하기 위해)
            raw = (raw.sum(-1, keepdim=True) / mass) * b_hist.expand_as(raw)
        c = torch.minimum(raw, cap)                                  # R3

        aux = {'p': p.detach(), 'rho': rho.detach(), 'score': score.detach(),
               'eta': self.eta.detach().expand(*c.shape[:-1]),
               'kappa': (c.sum(-1) / mass).detach(),                 # 상한이 물리면 1보다 작아진다
               'cap_rate': (raw > cap).to(c.dtype).mean(-1).detach()}

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
        v = v + (a - v) / self.tau_s
        s = arctan_spike(v - self.threshold, self.surrogate_scale)

        return v - self.threshold * s.detach(), s                    # detach_reset


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
                 tau_s=2., threshold=1., surrogate_scale=5., query_dim=None):
        super().__init__()

        tau = torch.as_tensor(tau, dtype=torch.float32)
        if tau.numel() != num_population or (tau <= 1.).any():
            raise ValueError('Require one tau per constituent, each > 1')
        if not heterogeneous:
            # 동질 대조군: 조화평균으로 총 누설을 맞춘다. tau=[4,8,16,32]이면 8.533
            tau = torch.full_like(tau, float(num_population / (1. / tau).sum()))
        self.register_buffer('tau', tau)
        self.register_buffer('b', fractional_coefficients(alpha, max_length))

        self.num_population = num_population
        self.alpha = float(alpha)
        self.max_length = int(max_length)
        self.selector = Selector(num_population, query_dim, theta, eta_init)
        self.soma = Soma(embed_dim, num_population, tau_s, threshold, surrogate_scale)

    def forward(self, x, mode='sparse', oracle_p=None, return_aux=False):
        #  x: [T, B, D] input current (B = batch * channel flattened)  ->  spikes: [T, B, D]
        """
        oracle_p[n] is [B, D, n] when mode == 'oracle'; every other mode ignores it.
        """
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError('Expected nonempty [T, B, D] current')
        T, B, D = x.shape
        if T > self.max_length:
            raise ValueError(f'T={T} exceeds the coefficient table ({self.max_length})')

        b0 = self.b[0].item()
        u = x.new_zeros(B, D, self.num_population)                   # u_0 = 0, 리셋 없음
        v = x.new_zeros(B, D)
        incs, keys, spikes = [], [], []                              # f(j), xi_j, s_n
        logs = {k: [] for k in ('state', 'voltage', 'coeff', 'kappa', 'cap_rate', 'eta')}

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
                aux = {'kappa': x.new_zeros(B, D), 'cap_rate': x.new_zeros(B, D),
                       'eta': self.selector.eta.detach().expand(B, D)}

            incs.append(f)
            keys.append(xi)
            u = nxt                                                  # u(t+1)
            v, s = self.soma(u, v)                                   # D-D: 갱신된 상태를 읽는다
            spikes.append(s)

            if return_aux:
                logs['state'].append(u.detach())
                logs['voltage'].append(v.detach())
                logs['coeff'].append(None if c is None else c.detach())
                for key in ('kappa', 'cap_rate', 'eta'):
                    logs[key].append(aux[key])

        out = torch.stack(spikes)                                    # [T, B, D]
        if not return_aux:
            return out

        aux = {'spikes': out.detach(), 'coeff': logs['coeff']}
        aux.update({k: torch.stack(logs[k]) for k in ('state', 'voltage', 'kappa', 'cap_rate', 'eta')})

        return out, aux

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


class Embedding(nn.Module):
    """Patch -> input current -> population neuron. D11 fixes input_scale before training."""
    def __init__(self, patch_size, embed_dim, input_scale=2., **neuron_args):
        super().__init__()

        self.emb_linear = nn.Linear(patch_size, embed_dim, bias=True)
        self.input_scale = float(input_scale)
        self.neuron = PopulationNeuron(embed_dim, **neuron_args)

    def forward(self, x, mode='sparse', oracle_p=None, return_aux=False):
        #  x: [T, B, patch_size]  ->  spikes: [T, B, D]
        return self.neuron(self.emb_linear(x) * self.input_scale, mode, oracle_p, return_aux)
