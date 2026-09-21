"""Population f-LIF v3-A: non-resetting fractional branches + one spiking soma.

Pre-registered specification: NSMT/docs/Population_fLIF_v3_prereg_KO.md (rev.1)
and NSMT/docs/IDEA_LOG.md (rev.1).  The rev.1 amendments that this file encodes:

  R3   coefficient cap   c_{n,j} = min(b_{n-j} * rho_{n,j}, b_0)
  F1   branch constants  tau = [4, 8, 16, 32]
  D-A  no tempered prior (g = 0, pi == 1)
  D-D  the soma reads u_{n+1}, the state that already contains I_n
  D-K  "sparse" applies to p, not to the final coefficients unless eta == 1
"""
import math

import torch
import torch.nn as nn


class Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score):
        # Martins & Astudillo (2016), Alg.1: project onto the probability simplex.
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
        # Eq.14 avoids sort/gather's nondeterministic CUDA scatter backward in torch1.12.
        support, = ctx.saved_tensors
        selected = gradient * support
        mean = selected.sum(dim=-1, keepdim=True) / support.sum(dim=-1, keepdim=True)
        return support * (gradient - mean)


def sparsemax(score):
    return Sparsemax.apply(score)


class ArcTanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward.

    O2 pins the derivative to the form the f-SNN reference code evaluates,
    `(s/2) / (1 + (pi/2 * s * x)^2)` with s = 5.0 -- the value `LIFNeuron`
    passes, not the function default 2.0.
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = scale
        return (x >= 0.).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        x, = ctx.saved_tensors
        s = ctx.scale
        return gradient * (s / 2.) / (1. + (math.pi / 2. * s * x).square()), None


def arctan_spike(x, scale=5.):
    return ArcTanSpike.apply(x, scale)


def fractional_coefficients(alpha, length, dtype=torch.float32):
    """b_d = [(d+1)^alpha - d^alpha] / Gamma(alpha+1) for d = 0..length-1."""
    if not 0. < alpha <= 1.:
        raise ValueError('Require 0 < alpha <= 1')
    d = torch.arange(length, dtype=torch.float64)
    b = ((d + 1.).pow(alpha) - d.pow(alpha)) / math.exp(math.lgamma(alpha + 1.))
    return b.to(dtype)


SELECTOR_MODES = ('full', 'dense', 'sparse', 'recent', 'mass_matched', 'oracle')


class Selector(nn.Module):
    """Content score over past patches -> mass-preserving rescale -> capped coefficients.

    Every mode leaves through the same path  p -> rho_tilde -> rho -> c = min(b*rho, b0),
    so an intervention only replaces `p`.  `uniform` is deliberately absent: a uniform
    `p` gives rho_tilde == 1, which is exactly `full` (checked by gate G7b).
    """

    def __init__(self, num_population=4, query_dim=None, theta=1., eta_init=-4.):
        super().__init__()
        self.num_population = num_population
        self.query_dim = num_population if query_dim is None else query_dim
        self.theta = float(theta)
        if self.theta <= 0:
            raise ValueError('Require theta > 0')
        # xi = [u_1..u_K ; I] in R^{K+1};  D5 initialises W_Q = W_K = [I 0].
        self.query = nn.Linear(num_population + 1, self.query_dim, bias=False)
        self.key = nn.Linear(num_population + 1, self.query_dim, bias=False)
        with torch.no_grad():
            self.query.weight.zero_()
            self.key.weight.zero_()
            eye = min(self.query_dim, num_population)
            self.query.weight[:eye, :eye] = torch.eye(eye)
            self.key.weight[:eye, :eye] = torch.eye(eye)
        self.eta_hat = nn.Parameter(torch.tensor(float(eta_init)))

    @property
    def eta(self):
        return torch.sigmoid(self.eta_hat)

    def coefficients(self, xi_now, xi_hist, b_hist, b0, mode='sparse', oracle_p=None):
        """[N,D,K+1], [N,D,J,K+1], [J] -> c [N,D,J] and diagnostics.

        `b_hist[i]` is b_{n-j} for the i-th stored past slot (oldest first), so
        b_hist = flip(b[1..J]).  `b0` is the cap level.  Returns capped coefficients
        for j < n only; the current-step term b_0 * f_n is added by the caller.
        """
        if mode not in SELECTOR_MODES:
            raise ValueError(f'Unknown selector mode {mode!r}')
        past = xi_hist.shape[-2]
        if past == 0:
            raise ValueError('Empty history is handled by the caller')
        b0 = b_hist.new_tensor(float(b0))
        if mode == 'full':
            c = b_hist.expand(*xi_now.shape[:-1], past).clone()
            zero = c.new_zeros(*c.shape[:-1], 1)
            return c, {'p': c / c.sum(-1, keepdim=True), 'rho': torch.ones_like(c),
                       'eta': zero.squeeze(-1), 'kappa': torch.ones_like(zero.squeeze(-1)),
                       'cap_rate': torch.zeros_like(zero.squeeze(-1)), 'score': torch.zeros_like(c)}

        score = -(self.query(xi_now).unsqueeze(-2) - self.key(xi_hist)).square().sum(-1)
        score = score / (self.query_dim * self.theta)
        if mode == 'sparse':
            p = sparsemax(score)
        elif mode == 'dense':
            p = torch.softmax(score, dim=-1)
        elif mode == 'recent':
            p = torch.zeros_like(score)
            p[..., -1] = 1.
        elif mode == 'oracle':
            if oracle_p is None:
                raise ValueError('oracle mode needs oracle_p')
            p = oracle_p.to(score.dtype)
        else:                                   # mass_matched: policy first, content removed
            p = sparsemax(score)

        mass = b_hist.sum()                                       # B_n = sum_{j<n} b_{n-j}
        denom = (b_hist * p).sum(-1, keepdim=True).clamp_min(1e-12)
        rho_tilde = mass * p / denom
        eta = self.eta
        rho = (1. - eta) + eta * rho_tilde
        raw = b_hist * rho
        if mode == 'mass_matched':
            # D4 control: keep the policy's total mass, drop every content dependence.
            kappa_raw = raw.sum(-1, keepdim=True) / mass
            raw = kappa_raw * b_hist.expand_as(raw)
        c = torch.minimum(raw, b0)
        aux = {'p': p.detach(), 'rho': rho.detach(), 'score': score.detach(),
               'eta': eta.detach().expand(*c.shape[:-1]),
               'kappa': (c.sum(-1) / mass).detach(),
               'cap_rate': (raw > b0).to(c.dtype).mean(-1).detach()}
        return c, aux


class Soma(nn.Module):
    """Mix the branch states, integrate once, spike, subtract.

    D-D: `step` is called with u_{n+1}, the state that already contains I_n.
    """

    def __init__(self, embed_dim, num_population=4, tau_s=2., threshold=1., surrogate_scale=5.):
        super().__init__()
        if tau_s <= 1. or threshold <= 0.:
            raise ValueError('Require tau_s > 1 and threshold > 0')
        self.weight = nn.Parameter(torch.full((embed_dim, num_population), 1. / num_population))
        self.tau_s, self.threshold, self.surrogate_scale = float(tau_s), float(threshold), float(surrogate_scale)

    def step(self, state, voltage):
        """[N,D,K], [N,D] -> voltage [N,D], spike [N,D]."""
        drive = (self.weight * state).sum(-1)
        voltage = voltage + (drive - voltage) / self.tau_s
        spike = arctan_spike(voltage - self.threshold, self.surrogate_scale)
        return voltage - self.threshold * spike.detach(), spike


class PopulationNeuron(nn.Module):
    """One logical neuron = K non-resetting fractional branches + one soma.

        f_{n,k}   = (I_n - u_{n,k}) / tau_k
        xi_n      = [u_{n,1..K} ; I_n]                    query uses the pre-update state
        c_{n,j}   = min(b_{n-j} * rho_{n,j}, b_0)
        u_{n+1,k} = b_0 f_{n,k} + sum_{j<n} c_{n,j} f_{j,k}
        a_n       = sum_k w_k u_{n+1,k}                   D-D: the updated state
        s_n       = H(v_n - theta),  v_n <- v_n - theta s_n
    """

    def __init__(self, embed_dim, num_population=4, alpha=.7, tau=(4., 8., 16., 32.),
                 heterogeneous=True, max_length=64, theta=1., eta_init=-4.,
                 tau_s=2., threshold=1., surrogate_scale=5., query_dim=None):
        super().__init__()
        tau = torch.as_tensor(tau, dtype=torch.float32)
        if tau.numel() != num_population or (tau <= 1.).any():
            raise ValueError('Require one tau per constituent, each > 1')
        if not heterogeneous:                       # harmonic mean keeps total leak equal
            tau = torch.full_like(tau, float(num_population / (1. / tau).sum()))
        self.register_buffer('tau', tau)
        self.register_buffer('b', fractional_coefficients(alpha, max_length))
        self.num_population, self.alpha, self.max_length = num_population, float(alpha), int(max_length)
        self.selector = Selector(num_population, query_dim, theta, eta_init)
        self.soma = Soma(embed_dim, num_population, tau_s, threshold, surrogate_scale)

    def forward(self, current, mode='sparse', oracle_p=None, return_aux=False):
        """[T,N,D] -> spikes [T,N,D] (+ aux).  `oracle_p[n]` is [N,D,n] when mode='oracle'."""
        if current.ndim != 3 or current.shape[0] < 1:
            raise ValueError('Expected nonempty [T,N,D] current')
        steps = current.shape[0]
        if steps > self.max_length:
            raise ValueError(f'T={steps} exceeds the coefficient table ({self.max_length})')
        b0 = self.b[0].item()
        state = current.new_zeros(*current.shape[1:], self.num_population)   # u_n, u_0 = 0
        voltage = current.new_zeros(*current.shape[1:])
        increments, histories, spikes = [], [], []
        log = {k: [] for k in ('kappa', 'cap_rate', 'eta', 'state', 'drive', 'coeff')}
        for n in range(steps):
            increment = (current[n].unsqueeze(-1) - state) / self.tau          # f_n  [N,D,K]
            update = b0 * increment
            coeff = None
            if n:
                past = torch.stack(increments, dim=-2)                         # [N,D,n,K]
                b_hist = self.b[1:n + 1].flip(0)                               # oldest first
                xi_now = torch.cat([state, current[n].unsqueeze(-1)], dim=-1)
                xi_hist = torch.stack(histories, dim=-2)
                coeff, aux = self.selector.coefficients(
                    xi_now, xi_hist, b_hist, b0, mode,
                    None if oracle_p is None else oracle_p[n])
                update = update + torch.einsum('ndj,ndjk->ndk', coeff, past)
                if return_aux:
                    for key in ('kappa', 'cap_rate', 'eta'):
                        log[key].append(aux[key])
            elif return_aux:
                zero = current.new_zeros(*current.shape[1:])
                log['kappa'].append(zero); log['cap_rate'].append(zero)
                log['eta'].append(self.selector.eta.detach().expand_as(zero))
            histories.append(torch.cat([state, current[n].unsqueeze(-1)], dim=-1))
            increments.append(increment)
            state = update                                                     # u_{n+1}
            voltage, spike = self.soma.step(state, voltage)                    # D-D
            spikes.append(spike)
            if return_aux:
                log['state'].append(state.detach())
                log['drive'].append(voltage.detach())
                log['coeff'].append(None if coeff is None else coeff.detach())
        output = torch.stack(spikes)
        if not return_aux:
            return output
        aux = {'spikes': output.detach(), 'state': torch.stack(log['state']),
               'voltage': torch.stack(log['drive']), 'coeff': log['coeff'],
               'kappa': torch.stack(log['kappa']), 'cap_rate': torch.stack(log['cap_rate']),
               'eta': torch.stack(log['eta'])}
        return output, aux

    @torch.no_grad()
    def fast_path(self, current):
        """Closed form for the neutral limit (eta = 0), used only by gate G3.

        u_{n+1} = (B f)_n with f = (I - u)/tau and u = J(Bf), so the branch operator
        is  (Id + B J / tau)^{-1} B / tau  -- multiplying by B alone drops the leak
        feedback and is wrong.  Returns the u_{n+1} sequence, i.e. what the soma reads.
        """
        steps = current.shape[0]
        b = self.b[:steps].double()
        rows = torch.arange(steps).unsqueeze(1) - torch.arange(steps).unsqueeze(0)
        kernel = torch.where(rows >= 0, b[rows.clamp_min(0)], torch.zeros(1, dtype=b.dtype))
        shift = torch.eye(steps, dtype=b.dtype).roll(1, 0)
        shift[0] = 0.
        signal = current.double().reshape(steps, -1)
        out = []
        for tau in self.tau.double():
            system = torch.eye(steps, dtype=b.dtype) + kernel @ shift / tau
            out.append(torch.linalg.solve(system, kernel @ signal / tau))
        return torch.stack(out, dim=-1).reshape(steps, *current.shape[1:], self.num_population)


class Embedding(nn.Module):
    def __init__(self, patch_size, embed_dim, input_scale=2., **neuron_args):
        super().__init__()
        self.proj = nn.Linear(patch_size, embed_dim, bias=True)
        self.input_scale = float(input_scale)
        self.neuron = PopulationNeuron(embed_dim, **neuron_args)

    def forward(self, x, mode='sparse', oracle_p=None, return_aux=False):
        return self.neuron(self.proj(x) * self.input_scale, mode, oracle_p, return_aux)
