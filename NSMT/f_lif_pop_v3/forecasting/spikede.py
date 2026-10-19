"""Verbatim copy of the spikeDE code used by neurons.py. DO NOT EDIT -- regenerate with
`python spikede_copy.py build <spikeDE root>` and verify with `python spikede_copy.py check <root>`.

Source: https://github.com/PhysAGI/spikeDE, commit fcd743befe504b1a471fa81887e6af7d6789da2e
        ("Fractional-order Spiking Neural Network", Ge et al., ICLR 2026). MIT licence, reproduced below.
Every definition below is copied whole and unchanged from the upstream file named in its banner;
only the imports at the top are written here (the union of what the copied code uses).
"""

# MIT License
# 
# Copyright (c) 2026 PhysAGI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ---- spikeDE/surrogate.py:8-69 (file sha256 5a81fa928ae1d525) --------------------
class SigmoidSurrogate(torch.autograd.Function):
    r"""
    Sigmoid-based surrogate gradient function for Spiking Neural Networks (SNNs).

    This class implements a custom autograd function where the forward pass uses a hard
    Heaviside step function to generate discrete spikes, while the backward pass approximates
    the undefined gradient using the derivative of a scaled sigmoid function.

    Forward Pass:

    $$ S(x) = H(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases} $$

    Backward Pass (Surrogate Gradient):

    $$ \sigma'(x) = \kappa \cdot \text{sigmoid}(\kappa x) \cdot (1 - \text{sigmoid}(\kappa x)) $$

    Where: $x$ is the input (membrane potential minus threshold, $U - \theta$), $\kappa$ (scale) controls the sharpness of the approximation.

    Attributes:
        scale (float): The scaling factor $\kappa$. Larger values approximate the true
                       step function more closely but may lead to vanishing gradients.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, scale: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold (Heaviside step function).

        Args:
            ctx: Context object to save tensors for the backward pass.
            input: Input tensor representing the membrane potential minus threshold ($U - \theta$).
            scale: Scaling factor ($\kappa$) controlling the sharpness of the surrogate gradient.

        Returns:
            A binary tensor of spikes (0.0 or 1.0).
        """
        ctx.save_for_backward(input)
        ctx.scale = scale
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using the sigmoid derivative as a surrogate.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            grad_output: Gradient of the loss with respect to the output of the forward pass.

        Returns:
            A tuple containing the gradient with respect to the input
                                       and None for the non-differentiable scale parameter.
        """
        (input,) = ctx.saved_tensors
        scale = ctx.scale
        sigmoid = torch.sigmoid(scale * input)
        surrogate_grad = scale * sigmoid * (1 - sigmoid)
        return grad_output * surrogate_grad, None


# ---- spikeDE/surrogate.py:72-86 (file sha256 5a81fa928ae1d525) --------------------
def sigmoid_surrogate(input: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Sigmoid surrogate gradient.

    Allows gradients to flow through the non-differentiable spiking operation during
    backpropagation by replacing the step function's derivative with a smooth sigmoid derivative.

    Args:
        input: Input tensor representing membrane potential minus threshold ($U - \theta$).
        scale: Scaling factor ($\kappa$). Higher values make the surrogate sharper.

    Returns:
        A tensor of binary spikes (0.0 or 1.0) with custom gradient flow.
    """
    return SigmoidSurrogate.apply(input, scale)


# ---- spikeDE/surrogate.py:90-151 (file sha256 5a81fa928ae1d525) --------------------
class ArctanSurrogate(torch.autograd.Function):
    r"""
    Arctangent-based surrogate gradient function for SNNs.

    This method uses the derivative of the arctangent function as the surrogate gradient.
    It features heavier tails compared to the sigmoid, allowing gradients to propagate
    even when the membrane potential is far from the threshold.

    Forward Pass:

    $$ S(x) = H(x) $$

    Backward Pass (Surrogate Gradient):
        
    $$ \sigma'(x) = \frac{\kappa}{1 + (\frac{\pi}{2} \kappa x)^2} $$

    Note: 
        The implementation includes a normalization factor involving $\pi/2$ to ensure stable gradient magnitudes, slightly modifying the standard arctan derivative.

    Attributes:
        scale (float): The scaling factor $\kappa$.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, scale: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold.

        Args:
            ctx: Context object to save tensors for the backward pass.
            input: Input tensor ($U - \theta$).
            scale: Scaling factor ($\kappa$).

        Returns:
            Binary spike tensor.
        """
        ctx.save_for_backward(input)
        ctx.scale = scale
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using the normalized arctangent derivative.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Upstream gradient from the loss function.

        Returns:
            Gradient w.r.t input and None for scale.
        """
        (input,) = ctx.saved_tensors
        scale = ctx.scale
        return (
            scale / 2 / (1 + (math.pi / 2 * scale * input).pow_(2)) * grad_output,
            None,
        )


# ---- spikeDE/surrogate.py:154-168 (file sha256 5a81fa928ae1d525) --------------------
def arctan_surrogate(input: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Arctan surrogate gradient.

    Ideal for deep networks where gradient vanishing is a concern due to its heavy-tailed
    gradient distribution.

    Args:
        input: Input tensor ($U - \theta$).
        scale: Scaling factor ($\kappa$).

    Returns:
        Binary spike tensor with arctan-based gradient flow.
    """
    return ArctanSurrogate.apply(input, scale)


# ---- spikeDE/surrogate.py:172-229 (file sha256 5a81fa928ae1d525) --------------------
class PiecewiseLinearSurrogate(torch.autograd.Function):
    r"""
    Piecewise Linear (Triangular) surrogate gradient function.

    A computationally efficient approximation that defines a triangular window around
    the threshold. Gradients are constant within the window and zero outside.

    Forward Pass:

    $$ S(x) = H(x) $$

    Backward Pass (Surrogate Gradient):

    $$ \sigma'(x) = \begin{cases} \frac{1}{2\gamma} & \text{if } |x| \leq \gamma \\ 0 & \text{otherwise} \end{cases} $$

    Where $\gamma$ (gamma) defines the width of the active region.

    Attributes:
        gamma (float): Half-width of the linear region.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold.

        Args:
            ctx: Context object to save tensors.
            input: Input tensor ($U - \theta$).
            gamma: Width parameter ($\gamma$).

        Returns:
            Binary spike tensor.
        """
        ctx.save_for_backward(input)
        ctx.gamma = gamma
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using a rectangular window function.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Upstream gradient.

        Returns:
            Gradient w.r.t input and None for gamma.
        """
        (input,) = ctx.saved_tensors
        gamma = ctx.gamma
        surrogate_grad = ((input >= -gamma) & (input <= gamma)).float() / (2 * gamma)
        return grad_output * surrogate_grad, None


# ---- spikeDE/surrogate.py:232-246 (file sha256 5a81fa928ae1d525) --------------------
def piecewise_linear_surrogate(input: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Piecewise Linear surrogate gradient.

    Best for high-speed training on resource-constrained hardware or models requiring
    sparse gradient updates.

    Args:
        input: Input tensor ($U - \theta$).
        gamma: Width of the active region ($\gamma$).

    Returns:
        Binary spike tensor with linear-based gradient flow.
    """
    return PiecewiseLinearSurrogate.apply(input, gamma)


# ---- spikeDE/surrogate.py:250-309 (file sha256 5a81fa928ae1d525) --------------------
class GaussianSurrogate(torch.autograd.Function):
    r"""
    Gaussian-based surrogate gradient function.

    Uses a normalized Gaussian function to approximate the derivative. It offers the
    smoothest profile with exponential decay, providing very localized gradient updates.

    Forward Pass:

    $$ S(x) = H(x) $$

    Backward Pass (Surrogate Gradient):

    $$ \sigma'(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}} $$

    Where $\sigma$ (sigma) controls the spread (standard deviation) of the gradient.

    Attributes:
        sigma (float): Standard deviation of the Gaussian.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold.

        Args:
            ctx: Context object to save tensors.
            input: Input tensor ($U - \theta$).
            sigma: Standard deviation parameter ($\sigma$).

        Returns:
            Binary spike tensor.
        """
        ctx.save_for_backward(input)
        ctx.sigma = sigma
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using the Gaussian PDF.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Upstream gradient.

        Returns:
            Gradient w.r.t input and None for sigma.
        """
        (input,) = ctx.saved_tensors
        sigma = ctx.sigma
        surrogate_grad = torch.exp(-(input**2) / (2 * sigma**2)) / (
            sigma * torch.sqrt(torch.tensor(2 * torch.pi))
        )
        return grad_output * surrogate_grad, None


# ---- spikeDE/surrogate.py:312-325 (file sha256 5a81fa928ae1d525) --------------------
def gaussian_surrogate(input: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Gaussian surrogate gradient.

    Best for precision tasks where only neurons very close to firing should receive updates.

    Args:
        input: Input tensor ($U - \theta$).
        sigma: Spread of the gradient ($\sigma$).

    Returns:
        Binary spike tensor with Gaussian-based gradient flow.
    """
    return GaussianSurrogate.apply(input, sigma)


# ---- spikeDE/neuron.py:12-17 (file sha256 5644600634da22a3) --------------------
surrogate_f = {
    "sigmoid_surrogate": sigmoid_surrogate,
    "arctan_surrogate": arctan_surrogate,
    "piecewise_linear_surrogate": piecewise_linear_surrogate,
    "gaussian_surrogate": gaussian_surrogate,
}


# ---- spikeDE/neuron.py:20-120 (file sha256 5644600634da22a3) --------------------
class BaseNeuron(nn.Module):
    r"""Base class for spiking neuron models with configurable membrane time constant and surrogate gradients.

    This abstract class provides the foundational structure for spiking neurons.
    It supports learnable or fixed membrane time constant ($\tau$) and
    customizable surrogate gradient functions for backpropagation through non-differentiable spikes.

    The effective membrane time constant is computed as:
    
    $$
    \tau =
    \begin{cases}
        \tau_0 \cdot (1 + e^{\theta}) & \text{if } \texttt{tau_learnable=True} \\
        \tau_0 & \text{otherwise}
    \end{cases}
    $$

    where $\tau_0$ is the initial value and $\theta$ is a learnable parameter.

    Subclasses must implement the `forward` method to define specific dynamics.

    Attributes:
        initial_tau (float): Initial value of the membrane time constant $\tau_0$.
        tau_param (Optional[torch.nn.parameter.Parameter]): Learnable parameter $\theta$ if `tau_learnable=True`;
            otherwise `None`.
        tau (float): Fixed $\tau$ used when `tau_learnable=False`.
        threshold (float): Firing threshold $V_{\text{th}}$.
        surrogate_grad_scale (float): Scaling factor for surrogate gradient steepness.
        surrogate_f (typing.Callable): Surrogate gradient function (e.g., arctan-based).
        tau_learnable (bool): Whether $\tau$ is trainable.
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate_grad_scale: float = 5.0,
        surrogate_opt: str = "arctan_surrogate",
        tau_learnable: bool = False,
    ) -> None:
        r"""Initializes the BaseNeuron module.

        Args:
            tau: The base membrane time constant $\tau$. Used directly if `tau_learnable=False`,
                or as a scaling factor if `tau_learnable=True`.
            threshold: The membrane potential threshold at which the neuron fires a spike.
            surrogate_grad_scale: Scaling factor applied inside the surrogate gradient function
                to control gradient magnitude during backpropagation.
            surrogate_opt: Name of the surrogate gradient function to use.
                Must be a key in the global `surrogate_f` dictionary (e.g., `"arctan_surrogate"`).
            tau_learnable: If `True`, $\tau$ becomes a learnable parameter.
                If `False`, $\tau$ remains fixed.
        """
        super(BaseNeuron, self).__init__()
        self.initial_tau = tau

        if tau_learnable:
            self.tau_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.tau_param = None
            self.tau = tau

        self.threshold = threshold
        self.surrogate_grad_scale = surrogate_grad_scale
        self.surrogate_f = surrogate_f[surrogate_opt]
        self.tau_learnable = tau_learnable

    def get_tau(self) -> float:
        r"""Returns the effective membrane time constant $\tau$.

        Ensures positivity via exponential reparameterization when learnable.

        Returns:
            Scalar tensor representing $\tau$.
        """
        if self.tau_learnable:
            return self.initial_tau * (1 + torch.exp(self.tau_param))
        else:
            return self.tau

    def forward(
        self, v_mem: torch.Tensor, current_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Performs one step of neuron state update.

        Must be overridden by subclasses to implement specific spiking dynamics.

        Args:
            v_mem: Membrane potential tensor of shape `(batch_size, ...)`.
            current_input: Input current tensor, same shape as `v_mem`.

        Returns:
            A tuple `(dv_dt, spike)` where:

                - `dv_dt`: Effective derivative of membrane potential.
                - `spike`: Continuous spike approximation in [0, 1].

        Raises:
            NotImplementedError: Always raised here; subclass must implement.
        """
        raise NotImplementedError("Neuron forward method must be overridden.")


# ---- spikeDE/neuron.py:165-200 (file sha256 5644600634da22a3) --------------------
class LIFNeuron(BaseNeuron):
    r"""Leaky Integrate-and-Fire (LIF) spiking neuron model with surrogate gradients.

    Implements classic leaky dynamics governed by:

    $$
        \tau \frac{\text{d}v}{\text{d}t} = -v + I(t), \quad
        \text{spike} = \sigma(v - V_{\text{th}})
    $$

    where $\sigma$ is a differentiable surrogate.
    """

    def forward(
        self, v_mem: torch.Tensor, current_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for LIF neuron dynamics (discrete-time, `dt=1.0`).

        Args:
            v_mem: Current membrane potential.
            current_input: Input current (same shape as `v_mem`).

        Returns:
            Tuple `(dv_dt, spike)` representing effective derivative and spike output.
        """
        if current_input is None:
            return v_mem
        tau = self.get_tau()
        dt = 1.0
        dv_no_reset = (-v_mem + current_input) / tau
        v_post_charge = v_mem + dt * dv_no_reset
        spike = self.surrogate_f(
            v_post_charge - self.threshold, self.surrogate_grad_scale
        )
        dv_dt = dv_no_reset - (spike.detach() * self.threshold) / tau
        return dv_dt, spike


# ---- spikeDE/solver.py:9-80 (file sha256 ed69e3920e1d473f) --------------------
def euler_integrate_tuple(
    ode_func: Callable[
        [torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]
    ],
    y0_tuple: Tuple[torch.Tensor, ...],
    t_grid: torch.Tensor,
    neuron_count: int,
) -> List[List[torch.Tensor]]:
    r"""
    Performs standard explicit Euler integration for integer-order ODEs ($D^1 y = f(t, y)$).

    This function distinguishes between dynamic state variables (neurons) which are integrated,
    and boundary outputs (e.g., spike outputs) which are treated as pass-through values computed
    directly from the derivative without accumulation.

    The update rule for integrated components is:

    $$y_{k+1} = y_k + \Delta t \cdot f(t_k, y_k)$$

    Args:
        ode_func: A callable `f(t, y_tuple)` returning a tuple of derivatives.
                  Expected format: `(dy_1, ..., dy_N, boundary_1, ...)`.
        y0_tuple: A tuple of initial state tensors `(y_1, ..., y_N, boundary_1, ...)`.
        t_grid: A 1D tensor of time points `[t_0, t_1, ..., t_N]`. Step sizes can be non-uniform.
        neuron_count: The number of components in the state tuple representing dynamic neurons
                      to be integrated. Components beyond this index are treated as pass-through boundaries.

    Returns:
        A list of lists, where `history[i][k]` is the state of component `i` at time step `k+1`.
        The length of each inner list is `len(t_grid) - 1`.

    Raises:
        AssertionError: If `y0_tuple` is not a tuple or `t_grid` has fewer than 2 points.
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    n_integrate = neuron_count
    # n_integrate is the number of neurons,
    # i.e., length of (dv1/dt, dv2/dt, ..., dvN/dt)
    n_components = len(y0_tuple)
    # length of (dv1/dt, dv2/dt, ..., dvN/dt, boundary_1, boundary_2, ...)

    # Initialize history lists for each component
    y_current = list(y0_tuple)
    y_history = [[] for _ in y0_tuple]

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(N - 1):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(
            tk, tuple(y_current)
        )  # Expect tuple return, consistent with y structure
        # assert isinstance(dy, tuple) and len(dy) == len(y)

        # Update all integrated components except the last one
        for i in range(n_integrate):
            y_current[i] = y_current[i] + dt * dy[i]
            y_history[i].append(y_current[i])

        # Pass-through boundary output e.g. final spike output
        # See odefunc_fx.md
        for i in range(n_integrate, n_components):
            y_current[i] = dy[i]
            y_history[i].append(y_current[i])

    return y_history


# ---- spikeDE/solver.py:102-131 (file sha256 ed69e3920e1d473f) --------------------
@dataclass
class PerLayerAlphaInfo:
    r"""
    Metadata container for the fractional order ($\alpha$) configuration of a single layer.

    Stores the fractional order(s) and precomputed constants required for numerical integration.
    To ensure gradient flow during backpropagation when $\alpha$ is learnable, all values are
    stored as `torch.Tensor` objects rather than Python floats.

    Attributes:
        alpha: A tensor containing the fractional order(s).
               Shape `(1,)` for single-term, shape `(M,)` for multi-term with $M$ terms.
        is_multi_term: Boolean flag indicating if the layer has multiple fractional terms ($M > 1$).
        coefficient: Optional tensor of coefficients $[c_1, ..., c_M]$ for multi-term equations.
                     Defaults to ones if not provided.
        h_alpha: Precomputed $h^\alpha$ (Single-term only).
        h_alpha_gamma: Precomputed $h^\alpha \cdot \Gamma(2-\alpha)$ (Single-term only).
        h_alpha_over_alpha_gamma: Precomputed $h^\alpha / (\alpha \cdot \Gamma(\alpha))$ (Single-term only).
    """

    alpha: (
        torch.Tensor
    )  # Always tensor: 1-element for single-term, n-element for multi-term
    is_multi_term: bool
    coefficient: Optional[torch.Tensor] = None  # Coefficients (defaults to ones)

    # Precomputed constants for single-term (populated during config creation)
    h_alpha: Optional[torch.Tensor] = None
    h_alpha_gamma: Optional[torch.Tensor] = None
    h_alpha_over_alpha_gamma: Optional[torch.Tensor] = None


# ---- spikeDE/solver.py:134-278 (file sha256 ed69e3920e1d473f) --------------------
@dataclass
class SNNSolverConfig:
    r"""
    Central configuration object for SNN fractional solvers.

    Aggregates simulation parameters, device information, and per-layer fractional metadata
    to streamline the solver execution loop.

    Attributes:
        N: Number of time points in the grid.
        h: Step size tensor (scalar), assumed uniform $h = t_{k+1} - t_k$.
        device: Torch device for computation.
        dtype: Torch data type for computation.
        n_components: Total number of state components (neurons + boundaries).
        n_integrate: Number of components to integrate (excludes boundary outputs).
        per_layer_info: List of `PerLayerAlphaInfo` objects, one per integrated layer.
    """

    N: int  # Number of time points
    h: torch.Tensor  # Step size
    device: torch.device
    dtype: torch.dtype
    n_components: int  # Total number of state components
    n_integrate: int  # Number of components to integrate (excludes spike)

    # Per-layer alpha information
    per_layer_info: List[PerLayerAlphaInfo] = field(default_factory=list)

    @classmethod
    def from_inputs(
        cls,
        y0_tuple: Tuple[torch.Tensor, ...],
        per_layer_alpha: List[Any],
        t_grid: torch.Tensor,
        per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> "SNNSolverConfig":
        r"""
        Constructs a solver configuration from user inputs.

        Processes raw alpha inputs (scalars, lists, or tensors) into standardized
        `PerLayerAlphaInfo` objects. Precomputes constants involving the Gamma function
        for single-term solvers to optimize the main integration loop.

        Args:
            y0_tuple: Tuple of initial state tensors. Used to infer device and dtype.
            per_layer_alpha: List of alpha values. Each element can be:

                - `float`: Single-term scalar.
                - `torch.Tensor`: 1-element (single-term) or M-element (multi-term).
                - `list`: Converted to tensor.
            t_grid: Time grid tensor.
            per_layer_coefficient: Optional list of coefficient tensors for multi-term layers.

        Returns:
            A configured `SNNSolverConfig` instance.

        Raises:
            AssertionError: If `t_grid` has fewer than 2 points or coefficient dimensions mismatch alpha.
        """
        device = y0_tuple[0].device
        dtype = y0_tuple[0].dtype

        N = len(t_grid)
        assert N >= 2, "t_grid must have at least 2 points"

        t_grid = t_grid.to(device=device, dtype=dtype)
        h = t_grid[-1] - t_grid[-2]

        n_components = len(y0_tuple)
        # length of (dv1/dt, dv2/dt, ..., dvN/dt, boundary_1, boundary_2, ...)

        n_integrate = len(per_layer_alpha)
        # n_integrate is the number of neurons,
        # i.e., length of (dv1/dt, dv2/dt, ..., dvN/dt)

        # Build per-layer info
        per_layer_info = []
        for i, alpha in enumerate(per_layer_alpha):
            # Determine if this layer is multi-term based on number of elements
            if isinstance(alpha, torch.Tensor):
                is_multi_term = alpha.numel() > 1
                alpha_tensor = alpha.to(device=device, dtype=dtype)
            elif isinstance(alpha, (list, tuple)):
                is_multi_term = len(alpha) > 1
                alpha_tensor = torch.tensor(alpha, dtype=dtype, device=device)
            else:
                # Scalar float
                is_multi_term = False
                alpha_tensor = torch.tensor([alpha], dtype=dtype, device=device)

            # Get coefficient for this layer
            coeff = None
            if per_layer_coefficient is not None and i < len(per_layer_coefficient):
                coeff = per_layer_coefficient[i]
                if coeff is not None and not isinstance(coeff, torch.Tensor):
                    coeff = torch.tensor(coeff, dtype=dtype, device=device)
                elif coeff is not None:
                    coeff = coeff.to(device=device, dtype=dtype)

                    # Now compare dimensions when coeff exists
                assert (
                    coeff.numel() == alpha_tensor.numel()
                ), f"Coefficient tensor size mismatch: {coeff.numel()} vs {alpha_tensor.numel()}"

            # Default coefficient to ones if not provided (for both single and multi-term)
            if coeff is None:
                coeff = torch.ones(alpha_tensor.numel(), dtype=dtype, device=device)

            # For single-term, precompute constants (used by non-multiterm solvers)
            if not is_multi_term:
                alpha_val = (
                    alpha_tensor.squeeze()
                )  # Keep as 0-dim tensor for gradient flow
                h_alpha = torch.pow(h, alpha_val)
                gamma_2_minus_alpha = math.gamma(
                    2 - alpha_val.item()
                )  # gamma needs float
                gamma_alpha = math.gamma(alpha_val.item())

                info = PerLayerAlphaInfo(
                    alpha=alpha_tensor,  # Keep as tensor for gradient flow
                    is_multi_term=False,
                    coefficient=coeff,
                    h_alpha=h_alpha,
                    h_alpha_gamma=h_alpha * gamma_2_minus_alpha,
                    h_alpha_over_alpha_gamma=h_alpha / (alpha_val * gamma_alpha),
                )
            else:
                info = PerLayerAlphaInfo(
                    alpha=alpha_tensor,
                    is_multi_term=True,
                    coefficient=coeff,
                )

            per_layer_info.append(info)

        return cls(
            N=N,
            h=h,
            device=device,
            dtype=dtype,
            n_components=n_components,
            n_integrate=n_integrate,
            per_layer_info=per_layer_info,
        )


# ---- spikeDE/solver.py:281-305 (file sha256 ed69e3920e1d473f) --------------------
def get_memory_bounds(k: int, memory: Optional[int]) -> Tuple[int, int]:
    r"""
    Calculates the range of history indices to include in the convolution sum.

    Supports memory truncation for long sequences to reduce computational complexity
    from $O(N^2)$ to $O(N \cdot M)$, where $M$ is the memory length.

    Args:
        k: Current time step index.
        memory: Maximum number of history steps to retain. If `None` or `-1`, uses full history.

    Returns:
        A tuple `(start_idx, memory_length)` defining the slice of history to use.

            - `start_idx`: The starting index in the history list;
            - `memory_length`: The number of elements to include.
    """
    if memory is None or memory == -1:
        memory_length = k + 1
    else:
        memory_length = min(memory, k + 1)
        assert memory_length > 0, "memory must be greater than 0"

    start_idx = max(0, k + 1 - memory_length)
    return start_idx, memory_length


# ---- spikeDE/solver.py:308-422 (file sha256 ed69e3920e1d473f) --------------------
class SNNFractionalMethod(ABC):
    r"""
    Abstract Base Class (ABC) for SNN fractional differential equation solvers.

    Defines the interface for various numerical methods (GL, L1, Trapezoidal, etc.).
    Implementations must define how weights are computed, how convolutions are performed,
    and how the state update is calculated.

    Subclasses distinguish themselves by:

    1. The formulation used (Riemann-Liouville vs. Caputo).
    2. The type of history stored ($y$ values vs. $f(t,y)$ values).
    3. Support for single-term vs. multi-term equations.
    """

    @property
    @abstractmethod
    def stores_f_history(self) -> bool:
        r"""
        Indicates whether the method stores function evaluations $f(t, y)$ or state values $y$ in history.

        Returns:
            `True` if the method (e.g., Adams-Bashforth) relies on $f$-history, `False` if the method (e.g., GL, L1) relies on $y$-history.
        """
        pass

    @abstractmethod
    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> Any:
        r"""
        Computes the convolution weights for a specific layer at time step $k$.

        Args:
            k: Current time step index.
            start_idx: Start index of the history window.
            config: Solver configuration containing layer metadata.
            layer_idx: Index of the layer being processed.

        Returns:
            A tensor or structure containing the weights $w_j$ for the convolution sum.
        """
        pass

    @abstractmethod
    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        r"""
        Computes the next state $y_{k+1}$ for a specific layer.

        Combines the current derivative $f_k$ and the convolution sum according to the method's formula.
        Note: The method name in the original code was slightly misleading; this function computes the update,
        while `compute_convolution` computes the sum. Based on usage in `snn_solve`, this function
        applies the final formula. However, looking at the implementation in subclasses,
        `compute_convolution` actually performs the summation loop, and this function applies the scaling.

        Correction based on code analysis:
        `compute_convolution` returns the sum $\sum w_j h_j$.
        `compute_update_for_layer` takes that sum and $f_k$ to return $y_{k+1}$.

        Args:
            f_k_i: The derivative/value $f(t_k, y_k)$ for this layer.
            convolution_sum: The result of the history convolution.
            config: Solver configuration.
            layer_idx: Index of the layer.

        Returns:
            The updated state tensor $y_{k+1}$.
        """
        pass

    @abstractmethod
    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: Any,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        r"""
        Computes the weighted sum (convolution) of history values.

        Calculates $\sum w_j \cdot h_j$, where $h_j$ is either $y_j$ or $f_j$ depending on `stores_f_history`.

        Args:
            k: Current time step index.
            start_idx: Start index of the history window.
            weights: Weights computed by `compute_weights_for_layer`.
            history_i: List of historical tensors for the current component.
            config: Solver configuration.
            layer_idx: Index of the layer.

        Returns:
            The result of the convolution sum (tensor).
        """
        pass

    def initialize(self, config: SNNSolverConfig) -> None:
        r"""
        Optional hook for method-specific precomputation before the time loop.

        Used to precompute static coefficients (e.g., GL binomial coefficients) that depend
        on $\alpha$ and $N$ but not on the state $y$.

        Args:
            config: Solver configuration.
        """
        pass


# ---- spikeDE/solver.py:425-512 (file sha256 ed69e3920e1d473f) --------------------
class GrunwaldLetnikovSNN(SNNFractionalMethod):
    r"""
    Grünwald-Letnikov (GL) solver for single-term Riemann-Liouville Fractional Differential Equations (FDEs).

    This class implements the standard GL discretization scheme, which approximates the
    Riemann-Liouville fractional derivative $D^\alpha y(t)$ using a finite difference convolution.

    Mathematical Formulation:
    The update rule for the state $y$ at step $k+1$ is given by:

    $$y_{k+1} = h^\alpha f(t_k, y_k) - \sum_{j=0}^{k} c_{k-j}^{(\alpha)} y_j$$

    where $h$ is the step size, $f(t, y)$ is the ODE function, and $c_j^{(\alpha)}$ are the
    Grünwald-Letnikov coefficients generated recursively:

    $$c_0^{(\alpha)} = 1, \quad c_j^{(\alpha)} = \left(1 - \frac{1+\alpha}{j}\right)c_{j-1}^{(\alpha)} \quad \text{for } j \ge 1$$

    Key Characteristics:

    - **Formulation**: Riemann-Liouville.
    - **Accuracy**: First-order $O(h)$.
    - **Memory**: Requires full history of states $y$ unless truncated.
    - **Constraint**: Strictly supports single-term fractional orders ($\alpha$ is a scalar per layer).
      Attempting to use multi-term $\alpha$ will raise a `ValueError`. For multi-term support,
      use `GrunwaldLetnikovMultitermSNN`.
    """

    def __init__(self):
        """Initializes the solver with empty coefficient storage."""
        self._c_per_layer: Optional[List[torch.Tensor]] = None

    @property
    def stores_f_history(self) -> bool:
        return False

    def initialize(self, config: SNNSolverConfig) -> None:
        self._c_per_layer = []

        for i, layer_info in enumerate(config.per_layer_info):
            if layer_info.is_multi_term:
                raise ValueError(
                    f"Layer {i} has multi-term alpha but GrunwaldLetnikovSNN only supports "
                    f"single-term. Use GrunwaldLetnikovMultitermSNN instead."
                )

            # Single-term: alpha is a 1-element tensor
            alpha = layer_info.alpha.squeeze()  # Get scalar tensor

            # Avoid in-place operations to preserve gradient flow when alpha is learnable
            # Build coefficients using list and stack instead of in-place assignment
            c_list = [torch.ones(1, dtype=config.dtype, device=config.device)]
            for j in range(1, config.N + 1):
                c_next = (1 - (1 + alpha) / j) * c_list[j - 1]
                c_list.append(c_next.unsqueeze(0) if c_next.dim() == 0 else c_next)
            c = torch.cat(c_list, dim=0)
            self._c_per_layer.append(c)

    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> torch.Tensor:
        return self._c_per_layer[layer_idx]

    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: torch.Tensor,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        if k > 0:
            convolution_sum = 0
            for j in range(start_idx, k):
                convolution_sum = convolution_sum + weights[k - j] * history_i[j]
        else:
            convolution_sum = 0
        return convolution_sum

    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        return layer_info.h_alpha * f_k_i - convolution_sum


# ---- spikeDE/solver.py:719-790 (file sha256 ed69e3920e1d473f) --------------------
class AdamsBashforthSNN(SNNFractionalMethod):
    r"""
    Adams-Bashforth predictor method for single-term Caputo FDEs.

    This method serves as a predictor step in predictor-corrector schemes (like PECE).
    Unlike the other methods which convolve state history $y_j$, Adams-Bashforth convolves
    the history of function evaluations $f(t_j, y_j)$.

    The update rule is:

    $$y_{k+1} = \sum_{j=0}^{k} b_{j,k+1} f(t_j, y_j)$$

    where the weights are:

    $$b_{j,k+1} = \frac{h^\alpha}{\alpha \Gamma(\alpha)} \left[ (k+1-j)^\alpha - (k-j)^\alpha \right]$$

    Key Characteristics:

    - **Formulation**: Caputo (Predictor).
    - **History Type**: Stores $f(t, y)$ instead of $y$.
    - **Constraint**: Single-term $\alpha$ only.
    """

    @property
    def stores_f_history(self) -> bool:
        return True

    def initialize(self, config: SNNSolverConfig) -> None:
        for i, layer_info in enumerate(config.per_layer_info):
            if layer_info.is_multi_term:
                raise ValueError(
                    f"Layer {i} has multi-term alpha but AdamsBashforthSNN only supports "
                    f"single-term. Use GrunwaldLetnikovMultitermSNN instead."
                )

    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        alpha = layer_info.alpha.squeeze()  # Get scalar tensor for single-term

        j_vals = torch.arange(
            start_idx, k + 1, dtype=config.dtype, device=config.device
        )
        b_j_kp1 = layer_info.h_alpha_over_alpha_gamma * (
            torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )
        return b_j_kp1

    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: torch.Tensor,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        convolution_sum = 0
        for j in range(start_idx, k + 1):
            local_idx = j - start_idx
            convolution_sum = convolution_sum + weights[local_idx] * history_i[j]
        return convolution_sum

    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        return convolution_sum


# ---- spikeDE/solver.py:895-995 (file sha256 ed69e3920e1d473f) --------------------
def snn_solve(
    ode_func: Callable[[torch.Tensor, Tuple], Tuple],
    y0_tuple: Tuple[torch.Tensor, ...],
    per_layer_alpha: List[Any],
    t_grid: torch.Tensor,
    method: SNNFractionalMethod,
    memory: Optional[int] = None,
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Unified driver function for solving SNN fractional differential equations.

    Orchestrates the time-stepping loop, managing state history, memory truncation,
    and dispatching to the specific numerical method provided.

    Args:
        ode_func: Function `f(t, y_tuple)` returning derivatives.
        y0_tuple: Initial state tuple.
        per_layer_alpha: List of fractional orders per layer.
        t_grid: Time points tensor.
        method: Instance of `SNNFractionalMethod` (e.g., `GrunwaldLetnikovSNN`).
        memory: Optional integer to limit history length for convolution.
        per_layer_coefficient: Coefficients for multi-term layers.

    Returns:
        List of lists containing the trajectory of each state component.
    """
    assert isinstance(y0_tuple, tuple), "y0_tuple must be a tuple"

    # Create configuration with per-layer alpha
    config = SNNSolverConfig.from_inputs(
        y0_tuple, per_layer_alpha, t_grid, per_layer_coefficient
    )

    # Move t_grid to correct device/dtype
    t_grid = t_grid.to(device=config.device, dtype=config.dtype)

    # Initialize method
    method.initialize(config)

    # Initialize state
    y_current = list(y0_tuple)

    # History
    y_history = [[] for _ in y0_tuple]

    # For predictor method, we need f-history
    if method.stores_f_history:
        fhistory = [[] for _ in range(config.n_integrate)]
    else:
        fhistory = None

    # Main loop
    for k in range(config.N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Store function evaluations if needed
        if method.stores_f_history:
            for i in range(config.n_integrate):
                fhistory[i].append(f_k[i])

        # Determine memory range
        start_idx, _ = get_memory_bounds(k, memory)

        # Update each integrated component with its own alpha
        for i in range(config.n_integrate):
            # Compute weights for this specific layer
            weights = method.compute_weights_for_layer(
                k, start_idx, config, layer_idx=i
            )

            # Get appropriate history
            history_i = fhistory[i] if method.stores_f_history else y_history[i]

            # Compute convolution sum
            convolution_sum = method.compute_convolution(
                k, start_idx, weights, history_i, config, layer_idx=i
            )

            # Compute update for this layer
            y_current[i] = method.compute_update_for_layer(
                f_k[i], convolution_sum, config, layer_idx=i
            )

            # Store in history
            if not method.stores_f_history:
                y_history[i].append(y_current[i])

        # Pass-through boundary output e.g. the final spike output
        for i in range(config.n_integrate, config.n_components):
            y_current[i] = f_k[i]
            y_history[i].append(y_current[i])

    # Cleanup
    if fhistory is not None:
        del fhistory

    return y_history
