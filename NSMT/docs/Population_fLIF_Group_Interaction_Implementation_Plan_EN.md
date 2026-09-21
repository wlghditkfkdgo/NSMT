# Population f-LIF with Shared History Selection and Internal Interaction
## Research specification, upstream code snapshots, extension plan, and ETT forecasting protocol

**Document date:** 17 September 2026  
**Status:** Proposed first experimental contract; not a validated model or a claim of completed reproduction.  
**Audience:** A research/coding agent developing the corrected PopulationLIF idea.  
**Upstream reference:** `PhysAGI/spikeDE`, commit `fcd743befe504b1a471fa81887e6af7d6789da2e`.  
**Scope:** Mathematical behavior, source correspondence, integration responsibilities, acceptance criteria, and experimental design. Existing upstream code is reproduced where requested. This document does not prescribe new source code, pseudocode, class signatures, directory layouts, coding style, installation commands, or launch scripts.

## Navigation

| Topic | Location |
|---|---|
| Corrected idea and source basis | Sections 0–2 |
| Important original code excerpts | Section 3 |
| Source discrepancies and proposed defaults | Sections 4–5 |
| Complete mathematical behavior and neutral limits | Sections 6–7 |
| Extension milestones and correctness gates | Sections 8–9 |
| ETT data and MLP-first architecture progression | Sections 10–12 |
| Comparison groups and ablations | Sections 13–14 |
| Training, evaluation, and interpretation | Sections 15–17 |
| Reproducibility, deferred choices, and handoff summary | Sections 18–20 |
| Source links, code license, and decision register | References and Appendices A–B |

---

## 0. Read this first: the model identity must not change again

The intended model is **not an ordinary LIF neuron with an additional retrieved-memory residual**. Its membrane state must be produced by **fractional history integration itself**.

One logical neuron is a **population of interacting f-LIF constituents**. The population jointly decides which historical population events to use. Constituents share the temporal selection, while their states interact through the neuronal dynamics. Each constituent retains its own membrane state and emits its own spike.

The defining structure is

$$
\boxed{
\text{coupled population dynamics}
\;\longrightarrow\;
\text{shared selection of historical population dynamics}
\;\longrightarrow\;
\text{fractional state integration}
}
$$

It is **not**

$$
\text{ordinary LIF update}
+
\gamma\times\text{retrieved membrane vector}.
$$

The earlier v1/v2 prototypes used the latter structure. Their completed experiments remain useful exploratory evidence about population representations and retrieval operators, but **they neither implement nor validate the corrected fractional-population model**. Their checkpoints and numerical results must not be relabeled as results for this model. The supplied project handoff explicitly identified that earlier implementation as direct membrane retrieval, or Branch B. [P1]

### 0.1 Requirements versus proposals

The document uses the following evidence categories:

| Category | Meaning |
|---|---|
| **User requirement** | The corrected research intention or an explicit constraint in the present request. |
| **Upstream observation** | A statement supported by the paper or the inspected commit-pinned source. |
| **Proposed contract** | A concrete first design selected here to make the experiment well-defined. It is not an established result or a previously confirmed user decision. |
| **Deferred alternative** | A scientifically plausible extension that is not part of the first model. |

**User requirements:** actual f-LIF-based state integration; population-level shared temporal selection; internal population interaction; chronological time-series forecasting; a Spiking MLP first and increased architectural complexity afterward; ETT prediction lengths of **96 and 720**.

**Proposed, not user-mandated:** the particular coupling graph, a common fixed fractional order, the selector formula, its bounded scaling, population size, exact training budget, and the staged dataset expansion below.

The initial contract deliberately chooses one option for each necessary component. Changing a proposed contract is permissible as a separately identified experiment; silently mixing alternatives is not.

### 0.2 One-sentence scientific claim to investigate

> A coupled population of fractional spiking constituents may use its collective temporal state to select useful historical dynamics, allowing current membrane states to be formed through context-dependent fractional integration rather than indiscriminate history accumulation.

“May” is essential: the mechanism, its stability, and its forecasting benefit require separate validation.

---

## 1. Why this is an extension of f-LIF

### 1.1 The original starting point

The paper defines a fractional LIF subthreshold model using the Caputo derivative:

$$
\tau D^\alpha U(t)=-U(t)+R I_{\mathrm{in}}(t),
\qquad 0<\alpha\le1.
$$

For $0<\alpha<1$, the derivative is nonlocal in time. In the predictor discretization used in the paper's main presentation, historical **dynamics evaluations** contribute to the present state. This is the relevant starting point for the present design. [P0, R1]

Two meanings of memory must remain distinct:

- A membrane state $U_j$ summarizes the system at a past time.
- A predictor-history item $F_j$ is the evaluated dynamics at that time, including the leak/input terms and, in the inspected software route, a reset-related term.

The new model may inspect population states to decide relevance, while integrating the corresponding **population dynamics vectors**. This is intentional, not an inconsistency.

### 1.2 Why population coding is relevant

The population represents one logical feature through a pattern across $K$ constituents. The population dimension is neither an additional time axis nor a set of unrelated input channels.

Different constituent response scales can distinguish input histories that a single scalar membrane state compresses to similar values. Prior work on neuronal heterogeneity supports the usefulness of diverse temporal dynamics in some SNN tasks; it does not establish that the proposed selection mechanism works. [R13]

Here, the population is a **persistent internal state representation**, not merely a Gaussian encoding of the current input value. Gaussian receptive-field encoding is not required by the first contract.

### 1.3 Why interaction is a separate requirement

Shared selection alone is insufficient to establish direct constituent interaction. A population could choose the same lag for every constituent while each constituent still evolves independently.

The model therefore has two distinct relationships:

$$
\underbrace{\rho_{n,j}^{(g)}}_{\text{which historical population event is used}}
\qquad\text{and}\qquad
\underbrace{A_{k\ell}}_{\text{how constituent }\ell\text{ affects constituent }k}.
$$

Here $g$ indexes the logical population. The temporal coefficient has no constituent index $k$; the interaction matrix does.

### 1.4 What the model does not automatically inherit

A content-dependent modification of the fractional history kernel is not the original fixed-kernel Caputo f-LIF. The proposed model is a **selective extension of a Caputo-predictor-based spiking model**, with a recoverable unselected reference.

The original paper's relaxation, robustness, and expressivity results do not automatically carry over to the selected, coupled, reset-bearing model. In particular:

- Using fractional coefficients does not guarantee that distant history remains influential when it is masked out.
- A finite population is not an exact replacement for the continuum-of-timescales interpretation discussed in the paper.
- The asymptotic predictor coefficient $b_d\propto d^{\alpha-1}$ and the subthreshold relaxation tail proportional to $t^{-\alpha}$ describe different quantities. A simple universal ordering of “memory length” by $\alpha$ should not be asserted.
- Coupling and sparsity are not evidence of biological equivalence or energy savings.

These are limits on interpretation, not reasons to avoid testing the design. [P0]

---

## 2. Verified upstream reference and scope of inspection

### 2.1 Repository and pin

The repository identifies itself as the official implementation of the f-SNN paper. [R2]

- Repository: https://github.com/PhysAGI/spikeDE
- Inspected commit: https://github.com/PhysAGI/spikeDE/commit/fcd743befe504b1a471fa81887e6af7d6789da2e
- Commit message: `Bug fix: Correcting Example Code and Code Typing.`
- License: MIT, copyright (c) 2026 PhysAGI. [R3]

All **spikeDE source snapshots** in this document refer to that immutable commit, not to a moving `main` branch. The document was prepared by online static source inspection. No claim is made that the repository was installed, its examples executed, or its experimental scores reproduced while preparing this document.

### 2.2 Source responsibilities

| Upstream source | Observed responsibility | Relevance to the new model |
|---|---|---|
| `spikeDE/neuron.py` | Local neuronal dynamics and spike/reset-related calculations | Establish the scalar reference and add population interaction to the local vector field. |
| `spikeDE/solver.py` | Fractional coefficients, history accumulation, solver driver | Apply shared temporal selection inside the predictor history sum. |
| `spikeDE/snn.py` | State initialization, solver dispatch, time-grid handling, output boundaries | Preserve chronological state/output semantics and verify the actual execution path. |
| `spikeDE/odefunc.py` | Network-to-vector-field transformation and input interpolation | Keep population state boundaries distinct from spike outputs and avoid unintended temporal leakage. |
| `spikeDE/surrogate.py` | Binary forward spikes with surrogate derivatives | Preserve spike behavior and distinguish surrogate-gradient tests from ordinary finite differences. |
| `examples/ICLR_Release/Neuromorphic/model.py` | Public Spikformer-based network wrapped by `SNNWrapper` | An example of combining a backbone with the solver, not an ETT forecasting implementation. |

The files are linked individually in the references. [R4–R9]

### 2.3 Why the first route is `LIFNeuron` plus `pred`

The repository includes multiple routes. The `pred` route uses Adams–Bashforth predictor coefficients and stores function evaluations. The GL route and `LIFNeuronFDE` use different state-history formulations. They must not be combined under an unspecified label such as “the original f-LIF code.” [R4, R5]

The proposed reference is **`LIFNeuron` + the single-term `pred` path**, because it corresponds directly to the main-paper predictor-history discussion. A public Citeseer example explicitly selects `fdeint` and `pred`, although its training setup and learnable time constant are not adopted as our forecasting defaults. [R9b]

The direct `SNNWrapper` branch for `fdeint` dispatches to local solver functions. The presence of `torchfde` imports elsewhere does not establish which solver actually runs. [R6]

---

## 3. Important upstream code snapshots

The following are existing upstream excerpts, not newly proposed code. Executable statements are retained; excerpts omit surrounding file content and are not standalone programs. Line references use the source file's one-based numbering. The MIT notice is reproduced in Appendix A.

### Snapshot A — Local LIF dynamics and reset contribution

**Source:** [`spikeDE/neuron.py`, lines 175–185](https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/neuron.py#L175-L185), inside `LIFNeuron.forward`. [R4]

```python
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
```

**Interpretation.** The function returns a dynamics value and a binary spike. It does not itself assign the final fractional state. Its local trial charge uses `dt=1.0`; the returned derivative-like value contains a detached reset contribution divided by `tau`.

This is not interchangeable with applying a standard hard or subtractive reset after a completed fractional convolution. The first reference-compatible contract below preserves the observed rule and names it explicitly.

### Snapshot B — Predictor coefficient scale

**Source:** [`spikeDE/solver.py`, lines 219–234](https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/solver.py#L219-L234), configuration construction. [R5]

```python
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
```

**Interpretation.** For the predictor, the prefactor is $h^\alpha/[\alpha\Gamma(\alpha)]=h^\alpha/\Gamma(\alpha+1)$. The snippet also shows why fixed $\alpha$ is the first contract: some gamma-function dependence passes through Python scalar extraction and is not an ordinary differentiable tensor expression.

### Snapshot C — The actual predictor history sum

**Source:** [`spikeDE/solver.py`, lines 685–718](https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/solver.py#L685-L718), inside `AdamsBashforthSNN`. [R5]

```python
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
```

**Interpretation.** This is the principal extension point. Historical function vectors enter the state update through fractional weights. Our group selection will modulate these historical contributions; it will not construct an unrelated membrane residual after an ordinary LIF update.

The return value in this route does not explicitly add a nonzero initial state. Default wrapper states are zero. General nonzero-initial-state Caputo tests and zero-initial-state upstream parity must therefore be distinguished.

### Snapshot D — Function history is distinct from returned state history

**Source:** [`spikeDE/solver.py`, lines 855–870](https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/solver.py#L855-L870). [R5]

```python
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
```

**Source continuation:** [`spikeDE/solver.py`, lines 881–899](https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/solver.py#L881-L899). [R5]

```python
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
```

**Interpretation.** In the predictor branch, returned integrated-state history is not populated in the same way as function history, while boundary outputs are appended. A population selector requires an explicitly identified historical population context in addition to $F_j$. It must not mistake boundary spikes for membrane history.

### Snapshot E — Actual direct fractional solver dispatch

**Source:** [`spikeDE/snn.py`, lines 914–925](https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/snn.py#L914-L925), within the `fdeint` branch. [R6]

```python
            else:
                # Case A (per-layer single-term): use requested solver
                integrate_method = SOLVERS[method]
                v_mem_all_time_and_final_spike = integrate_method(
                    self.ode_func,
                    initial_state,
                    per_layer_alpha,
                    output_time,
                    memory=memory,
                    per_layer_coefficient=per_layer_coefficient,
                )
            return process_boundaries(v_mem_all_time_and_final_spike)
```

**Interpretation.** The first extension should have a traceable relationship to this direct predictor route. Replacing it by a state-returning GL neuron, a multi-term solver, or an adjoint route is a distinct change, not an implementation detail.

### Snapshot F — Spikes remain spikes in the network graph

**Source:** [`spikeDE/odefunc.py`, lines 513–523](https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/odefunc.py#L513-L523). [R7]

```python
                    dv_dt = new_graph.call_function(
                        operator.getitem, args=(out_node, 0)
                    )
                    spike_output = new_graph.call_function(
                        operator.getitem, args=(out_node, 1)
                    )

                    dv_dt_list.append(dv_dt)
                    # For the graph flow, the neuron output is the spike
                    node_map[node] = spike_output
                    nodes_in_ode_graph.append(node)
```

**Interpretation.** Solver states and layer outputs have different roles. The forecasting design keeps constituent spikes as the communicated representation. Continuous membrane states are used internally by the dynamics and selector, not silently substituted for spike outputs in the main comparison.

---

## 4. Source issues that must remain visible in the extension plan

The following are static observations and scope limitations, not claims that every upstream example is invalid.

| Issue | Consequence for this project |
|---|---|
| Local spike calculation uses a unit-time trial before the fractional state update. | Preserve and test this timing in the reference-compatible variant; do not call it post-convolution thresholding. |
| Reset-related decrement is included in the returned $F_n$ and divided by $\tau$. | Do not apply a second post-integration subtractive reset. Selection of full $F_j$ also changes historical reset contributions. |
| Predictor update omits an explicit nonzero $U_0$ term. | Zero-initial-state parity is the direct source comparison. A general Caputo adapter with explicit $U_0$ is a documented extension. |
| Predictor stores function evaluations, not a complete returned membrane trace. | Population context/state history has to be defined separately for selection and diagnostics. |
| Wrapper generates an extra terminal grid point from the input grid, even when `output_time` was supplied. | Verify the number and alignment of input evaluations, state updates, and output spikes rather than relying on argument names. |
| Alpha lists denote per-layer or multi-term configurations. | They do not automatically mean one distinct order per population constituent. |
| Some alpha-dependent calculations use scalar extraction. | The first model fixes alpha. Learnable alpha is deferred until its full derivative path is verified. |
| Source modules call `torch.compile`. | Existing v1/v2 environment compatibility is not established. Runtime changes require separate provenance, not silent environment replacement. |
| Multiple solver families are available. | All principal comparisons use the same selected predictor convention. |

These points are supported by the inspected neuron, solver, wrapper, and graph code. [R4–R8]

The mathematically intended subthreshold equation, the paper's discrete presentation, and the published software's spike/reset convention are related but not identical descriptions. This document does not silently reconcile them. It chooses a **source-compatible discrete spiking contract** below and reserves alternative reset formulations for separately named studies.

---

## 5. Proposed first experimental contract

The defaults below make the first candidate fully specified at the model-behavior level. They are starting values for an auditable experiment, not claimed optima.

| Component | Proposed first contract |
|---|---|
| Fractional integration | Single-term Caputo Adams–Bashforth predictor, full available history |
| Fractional order | Common fixed $\alpha=0.7$ in all constituents and layers |
| Integer-order control | $\alpha=1$ with the same source-compatible vector field and discretization |
| Population size | $K=4$ |
| Heterogeneous scales | Fixed $\tau=[2,4,8,16]$ |
| Homogeneous scale | Match mean inverse scale: $\tau_{\mathrm{hom}}=K/\sum_k\tau_k^{-1}=64/15$ |
| State initialization | Zero at each independent input window |
| Time convention | One chronological patch is one model-time step, $h=1$ |
| Spike threshold | $\theta=1$ for every constituent |
| Spike/reset convention | Source-compatible rule in Section 6; no second reset |
| Surrogate | Upstream arctangent surrogate, fixed scale 5; verify the actual expression in the pinned code |
| Coupling graph | Fixed undirected nearest-neighbor chain along the ordered scale axis |
| Coupling strength | $\lambda=0.1$ when enabled; $\lambda=0$ for the matched control |
| Selection unit | One time-coefficient vector per logical population, shared across its $K$ constituents |
| Selection context | Current/historical integrated population state and the associated current |
| Score | Learned projected squared distance, no L2 normalization, temperature $0.25$ |
| Selection variants | Full history; dense content modulation; sparse content modulation |
| Coefficient scaling | Probability divided by its maximum, not a probability-weighted history average |
| Current dynamics term | Always retained with its original fractional coefficient |
| Null-only read | Not in the first candidate; deferred because deleting all past integration is not an innocuous memory-off operation |
| Trainable new components | Query and key projections; graph, coupling strength, alpha, and tau fixed initially |
| History gradients | Direct differentiation through the unrolled history, except the source-compatible detached spike reset |

The homogeneous control uses inverse-scale matching because $1/\tau_k$ appears in the vector field. It does **not** copy the old exponential-beta mean-matching rule. The two conventions correspond to different discretizations.

Deferred changes include learned coupling, per-constituent orders $\alpha_k$, learned orders or time constants, null-read policies, fractional adjoints, streaming across windows, and acceleration. None should appear implicitly in the initial experiment.

---

## 6. Complete mathematical behavior of the first candidate

### 6.1 Notation and ownership of state

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $C$ | Original time-series variables; 7 in the ETT multivariate task |
| $D$ | Logical populations per layer |
| $K$ | Constituents per logical population |
| $T$ | Number of chronological input patches |
| $n$ | Current patch index, $0,\ldots,T-1$ |
| $j$ | Historical patch index, strictly $j<n$ for selection |
| $k,\ell$ | Constituent indices, never time indices |
| $g$ | A logical population within a sample, variable, and layer |
| $\mathbf U_n^{(g)}\in\mathbb R^K$ | Integrated population state available before processing current $I_n^{(g)}$ |
| $I_n^{(g)}\in\mathbb R$ | Current shared by the population constituents |
| $\mathbf D_n^{(g)}$ | Subthreshold coupled vector field |
| $\mathbf S_n^{(g)}\in\{0,1\}^K$ | Constituent spike output under the source-compatible rule |
| $\mathbf F_n^{(g)}$ | Complete vector field including reset-related contribution |
| $b_d^{(\alpha)}$ | Predictor coefficient at lag $d$ |
| $\rho_{n,j}^{(g)}$ | Shared population coefficient modulating historical dynamics |

State ownership is local to one sample/window, one original variable, one layer, and one logical population. Sharing model parameters across populations does not mean sharing their states or memory banks.

For clarity, the following equations suppress the logical-population superscript $g$.

### 6.2 Input and population diversity

Every constituent receives the same $I_n$. Diversity comes from the fixed scale vector $\boldsymbol\tau$, not from giving each constituent an unrelated input projection.

Let

$$
\mathsf T_\tau=\operatorname{diag}(\tau_1,\ldots,\tau_K).
$$

With fixed common $\alpha$, $\tau_k$ is a model-time scale parameter under the chosen fractional equation. It must not be interpreted as a physical time constant in hours without a consistent unit conversion. Model time is normalized to the patch index in the first study.

### 6.3 Internal population interaction

Define a chain adjacency by $A_{k\ell}=1$ when $|k-\ell|=1$, and $A_{k\ell}=0$ otherwise. The diagonal is zero. Let

$$
\mathsf L_{\mathrm{pop}}=\operatorname{diag}(A\mathbf 1)-A.
$$

For $K=1$, define $\mathsf L_{\mathrm{pop}}=0$. The population is ordered by its scale parameters so the chain has an interpretable temporal-response neighborhood.

The coupled subthreshold field is

$$
\boxed{
\mathbf D_n
=
\mathsf T_\tau^{-1}
\left[
-\mathbf U_n+\mathbf 1 I_n
-\lambda\mathsf L_{\mathrm{pop}}\mathbf U_n
\right].
}
$$

Equivalently,

$$
D_{n,k}
=
\frac{
-U_{n,k}+I_n
+\lambda\sum_{\ell\ne k}A_{k\ell}(U_{n,\ell}-U_{n,k})
}{\tau_k}.
$$

An off-diagonal interaction is explicit:

$$
\frac{\partial D_{n,k}}{\partial U_{n,\ell}}
=
\frac{\lambda A_{k\ell}}{\tau_k},
\qquad k\ne\ell.
$$

This is an algebraic property of the proposed subthreshold field, not an empirical claim. It distinguishes direct coupling from merely sharing an attention score.

The coupling exchanges information while tending to reduce neighboring state differences. Excessive coupling may erase diversity. Fixed weak coupling is selected first to separate interaction from the extra flexibility of learning an arbitrary mixing matrix.

### 6.4 Source-compatible spike and reset convention

To preserve the inspected local-neuron behavior, define

$$
\mathbf Z_n=\mathbf U_n+\mathbf D_n,
$$

$$
\boxed{\mathbf S_n=H(\mathbf Z_n-\theta\mathbf 1),}
$$

$$
\mathbf R_n
=
\theta\mathsf T_\tau^{-1}\operatorname{stopgrad}(\mathbf S_n),
\qquad
\boxed{\mathbf F_n=\mathbf D_n-\mathbf R_n.}
$$

The unit coefficient in $\mathbf Z_n$ reproduces the source's local `dt=1.0`. This document's main experiment fixes the solver step to $h=1$ as well.

Important consequences:

1. $\mathbf U_n$ is the solver's integrated, reset-influenced state. It is **not** the old v2 post-subtraction membrane variable and need not lie below threshold.
2. $\mathbf Z_n$ is only the local spike-test value. The final state is determined by the fractional sum in Section 6.8.
3. No additional hard reset or threshold subtraction is applied after that sum.
4. Reset spikes are detached only in $\mathbf R_n$. Emitted spikes retain the chosen surrogate path for network learning.
5. Historical $\mathbf F_j$ contains reset effects. The first candidate modulates the entire historical vector consistently with the source-based extension. Its consequences require the reset tests in Section 9.

This convention is selected for source correspondence, not asserted to be the unique physically correct hybrid f-LIF reset. A post-convolution event/reset formulation is a separate model and must be named and compared separately.

The actual upstream arctangent backward expression contains a factor of $1/2$ that is easy to miss when reading only explanatory formulas. Matching the pinned executable expression, rather than a rephrased derivative, is the appropriate reference rule. [R8]

### 6.5 Historical items: key context versus integrated value

For each completed evaluation $j$, retain the conceptual record

$$
\mathcal H_j=(\mathbf U_j,I_j,\mathbf F_j).
$$

The key context and integrated value are different parts of the same event:

$$
\boldsymbol\xi_j=[\mathbf U_j;I_j]\in\mathbb R^{K+1},
\qquad
\text{integration value}=\mathbf F_j\in\mathbb R^K.
$$

The selector examines a group state/current context. Once selected, the corresponding **whole group dynamics vector** participates in integration. It does not retrieve one constituent's scalar history independently.

Historical $\mathbf F_j$ means the vector evaluated at event $j$ along the actual evolving trajectory. It is not recomputed later using the current state's coupling term. Within a forward trajectory, trainable parameters are fixed; across training iterations the whole trajectory is recomputed.

Skipping a historical item at step $n$ is a read decision, not permanent deletion. The full bank remains available later in the same window.

### 6.6 Shared relevance score

Use $d_q=K$ and two learned projections

$$
W_Q,W_K\in\mathbb R^{K\times(K+1)}.
$$

They are shared across logical populations in a layer, but different layers may own different projection parameters. The initial projections are $[I_K\;0]$, so the starting score compares population states; the input-context column remains learnable. No bias or L2 normalization is used in the first candidate.

$$
\mathbf q_n=W_Q\boldsymbol\xi_n,
\qquad
\mathbf k_j=W_K\boldsymbol\xi_j,
$$

$$
\boxed{
 e_{n,j}
 =-\frac{1}{d_q\,\vartheta}
 \|\mathbf q_n-\mathbf k_j\|_2^2,
 \qquad \vartheta=0.25,\quad j<n.
}
$$

The $1/d_q$ factor corresponds to a mean squared distance, not an unnormalized squared norm. Amplitude differences are retained. Learned projections allow a task-dependent comparison, but do not establish that high scores identify genuinely useful memories.

The score is computed from $(\mathbf U_n,I_n)$, both known before $\mathbf U_{n+1}$ exists. It therefore avoids circular use of the state being computed.

There is one $e_{n,j}$ per logical population and historical event. There is no $e_{n,j,k}$ in the principal model.

### 6.7 Three selection modes with a defined neutral limit

For $n>0$, the historical candidate set is exactly $j=0,\ldots,n-1$. The current dynamics item $j=n$ is outside this selector.

**Full history:**

$$
\rho_{n,j}=1\quad\text{for every }j<n.
$$

**Dense content modulation:**

$$
p_{n,:}=\operatorname{softmax}(e_{n,:}),
\qquad
\rho_{n,j}=\frac{p_{n,j}}{\max_{\ell<n}p_{n,\ell}}.
$$

**Sparse content modulation:**

$$
p_{n,:}=\operatorname{sparsemax}(e_{n,:}),
\qquad
\rho_{n,j}=\frac{p_{n,j}}{\max_{\ell<n}p_{n,\ell}}.
$$

Sparsemax is the Euclidean projection onto the probability simplex and can assign exact zeros. [R12]

This **relative-to-maximum scaling is a new proposed design**, not an upstream f-SNN rule. It has useful, testable properties:

$$
0\le\rho_{n,j}\le1,
\qquad
\max_j\rho_{n,j}=1.
$$

Equal scores yield equal probabilities and therefore $\rho_{n,j}=1$, recovering the full-history coefficients. Exact sparsemax zeros remain exact zeros. For example, probabilities $[0.7,0.3,0]$ become modulation coefficients $[1,3/7,0]$.

The probabilities serve only to construct relative coefficients. **The final fractional kernel is not renormalized to sum to one.** A direct multiplication by a uniform probability $1/n$ would change the original integral's scale and would not provide the same neutral limit.

For $n=0$ there is no past selector. For $n=1$ there is only one historical candidate, so both non-null modes give its coefficient as 1. These are expected boundary cases, not failures of sparsity.

The first design cannot reject every real historical item at once. Adding a null atom would change the model and requires a separate definition of the empty-history dynamics. The old v2 null/gate cannot simply be transferred.

### 6.8 Fractional integration is the state update

Define the single-term predictor coefficient

$$
\boxed{
 b_d^{(\alpha)}
 =\frac{h^\alpha}{\Gamma(\alpha+1)}
 \big[(d+1)^\alpha-d^\alpha\big],
 \qquad d\ge0.
}
$$

The time-scale factor $1/\tau_k$ is already inside $\mathbf F_n$. It must not be inserted a second time into $b_d$.

The candidate state equation is

$$
\boxed{
\mathbf U_{n+1}
=
\mathbf U_0
+
 b_0^{(\alpha)}\mathbf F_n
+
 \sum_{j=0}^{n-1}
 b_{n-j}^{(\alpha)}\rho_{n,j}\mathbf F_j.
}
$$

The initial experiment has $\mathbf U_0=0$. Writing the term explicitly defines the mathematical extension for nonzero-initial-condition numerical tests; it is not a claim that the pinned predictor already includes that term.

There is no extra $\beta\mathbf U_n$ outside this equation, no retrieved-state residual, and no inherited $\gamma g_n\mathbf M_n$ pathway.

For constituent $k$, one historical contribution includes

$$
 b_{n-j}^{(\alpha)}\rho_{n,j}
 \frac{
 -U_{j,k}+I_j
 +\lambda\sum_{\ell\ne k}A_{k\ell}(U_{j,\ell}-U_{j,k})
 -\theta\operatorname{stopgrad}(S_{j,k})
 }{\tau_k}.
$$

Thus a historical state of constituent $\ell$ can influence the current state of constituent $k$, while the entire population shares the same choice of historical event $j$.

### 6.9 Source-compatible timing and the final-step issue

Under this first contract, $\mathbf S_n$ is emitted during evaluation of $\mathbf F_n$, before the selected fractional sum produces $\mathbf U_{n+1}$.

Therefore:

- The newly computed $\rho_{n,:}$ changes $\mathbf U_{n+1}$.
- Its first opportunity to change a subsequent emitted spike is ordinarily $\mathbf S_{n+1}$.
- For a sequence whose prediction head consumes only $\mathbf S_0,\ldots,\mathbf S_{T-1}$, the selection performed at $n=T-1$ affects the terminal state $\mathbf U_T$ but not an additional emitted spike in that forward pass.

This timing is an important consequence of Snapshot A plus the driver, not a new simulation trick. It must appear in tests and diagnostics. A final-step selector with no direct prediction gradient in this setup is not automatically broken.

A synthetic query must allow at least one subsequent response step when testing this contract. A different model that emits its spike after the selected fractional convolution must be treated as a separate timing/reset variant, not silently substituted to obtain an immediate response.

### 6.10 What is shared and what remains distinct

| Shared within a population | Distinct within a population |
|---|---|
| Current input $I_n$ | Membrane components $U_{n,k}$ |
| Chosen historical event indices | Scale parameters $\tau_k$ |
| Temporal coefficients $\rho_{n,j}$ | Component values of $\mathbf F_j$ |
| Common alpha in the initial contract | Spike decisions $S_{n,k}$ |
| Group-level selection decision | Position in the coupling graph |

Across logical populations, query/key parameters may be shared, but states, scores, selections, and trajectories remain sample- and population-specific.

---

## 7. Limit cases that define whether the idea was implemented correctly

### 7.1 Scalar source correspondence

With $K=1$, $\lambda=0$, full history, $U_0=0$, matching threshold/surrogate, matching tau, and matching time grid, the candidate must reproduce the pinned scalar `LIFNeuron + pred` arithmetic and emitted spikes within documented numerical tolerance.

This is a zero-initial-state source parity test. A mismatch cannot be excused by citing the shared term “fractional LIF.”

### 7.2 Independent fractional population

With $K>1$, $\lambda=0$, and full history, each constituent follows the scalar fractional reference at its own tau. The vectorized result must match the stack of independent scalar trajectories under the same current.

### 7.3 Coupled fractional population without selection

Setting every $\rho_{n,j}=1$ recovers the full-history coupled system. Turning off selection must not turn off fractional history.

### 7.4 Integer-order limit

For $\alpha=1$,

$$
b_d^{(1)}=h.
$$

With full history,

$$
\mathbf U_{n+1}=\mathbf U_0+h\sum_{j=0}^{n}\mathbf F_j,
$$

and first differences give

$$
\mathbf U_{n+1}=\mathbf U_n+h\mathbf F_n.
$$

This is the Euler limit of the **same source-compatible vector field**, not the old prototype's exponential-beta update. Because the inspected reset contribution is scaled inside $F_n$, it is also not automatically identical to every library's conventional post-charge-reset LIF.

When selection remains content-dependent, setting $\alpha=1$ does **not** generally reduce the model to a Markovian LIF: the new history modulation still depends on past records. Integer-order selected controls are useful precisely because they isolate the fractional kernel from the explicit history-selection mechanism.

### 7.5 Homogeneous invariant population

With equal taus, equal initial states, common current, equal thresholds, and shared temporal coefficients, all constituents remain identical. Their graph Laplacian contribution is zero. Under this specific diffusive coupling, a homogeneous population with coupling enabled is mathematically redundant with coupling disabled, apart from numerical error.

Consequences:

- Homogeneous populations are a redundancy control, not an effective-capacity-matched rich population.
- There is no need to train duplicate homogeneous coupling cells as independent evidence.
- This invariance is a required model test.

### 7.6 No legacy “memory-off” meaning

Three operations must not share one scientific label:

| Operation | Meaning |
|---|---|
| $\rho_{n,j}=1$ | Disable selection and restore all fractional history. |
| $\lambda=0$ | Disable direct internal coupling; retain fractional history and any selection. |
| Remove every past integral term | Replace the original dynamics by a different history-truncated system. |

The last operation is a stress intervention, not an ordinary LIF baseline or the natural “off” condition of the new model.

---

## 8. Source-to-model extension plan

This is a plan of mathematical and integration responsibilities, not a prescription for software organization.

### Phase A — Establish the reference contract

**Starting evidence:** Snapshots A–E.

**Outcome:** A traceable scalar reference with known coefficients, state/output timing, reset behavior, and active solver route.

**Acceptance:** Reproduction of small source trajectories and independent no-spike numerical checks. Record discrepancies such as the nonzero-initial-state behavior separately. An unmodified source snapshot and any explicitly corrected adapter are different references.

### Phase B — Introduce the population state axis

**Extension:** One scalar f-LIF state becomes a vector with a persistent constituent axis. Same-current broadcasting and fixed tau diversity are the only changes.

**Acceptance:** The independent-vector model agrees with separate scalar reference evaluations; $K=1$ and homogeneous invariants hold.

### Phase C — Introduce internal coupling

**Extension:** The subthreshold vector field gains $-\lambda\mathsf L_{\mathrm{pop}}\mathbf U_n$ inside the rate expression. The fractional integrator remains full-history.

**Acceptance:** The off-diagonal dependence and zero-coupling recovery are verified. Spike and reset calculations use the same coupled field and the same declared reference convention.

### Phase D — Introduce population-wide history selection

**Extension:** Current and historical population contexts determine a shared coefficient for each historical $\mathbf F_j$. The coefficient acts inside the predictor sum.

**Acceptance:** Full-mode and equal-score recovery, exact sparse support, group-shared selection, causal access, and unchanged retention of unselected records are verified.

### Phase E — Integrate into a chronological Spiking MLP

**Extension:** A patch produces a current, the fractional population produces constituent spikes, and a fixed forecast head reads the sequence of spike representations.

**Acceptance:** Input/output alignment, per-window state isolation, source-compatible timing, finite training loss, nontrivial spike activity, and expected gradients are checked before interpreting task metrics.

### Phase F — Increase architecture complexity and execute paired forecasting studies

**Extension:** Add depth, then causal temporal convolution, then a separately declared attention-based comparison. The neuron definition and experimental protocol remain fixed within each comparison.

**Acceptance:** Full matched-condition matrices, checkpoint provenance, data-split checks, independent reload, and paired analyses are complete. A few favorable runs or a partially finished matrix are not sufficient.

### Information that must be available to the integration layer

The conceptual integrator needs current population state, complete historical dynamics, historical population contexts, the fixed fractional coefficients, and shared temporal modulation. It also needs an unambiguous boundary between solver state and communicated spikes.

A neuron-local vector field alone cannot choose among all historical events unless the appropriate history is explicitly made available. Conversely, a history selector does not itself define the correct fractional state update. These responsibilities must remain distinguishable even if they are realized in one software component.

---

## 9. Verification before forecasting conclusions

### 9.1 Numerical and source-level tests

| Test | Expected result | Important qualification |
|---|---|---|
| Constant forcing, no spikes: $D^\alpha U=c$ | $U(t)=U_0+c t^\alpha/\Gamma(\alpha+1)$ at the numerical scheme's expected accuracy | This isolates the integrator, not the reset model. |
| Zero forcing with nonzero initial condition | Constant state for $D^\alpha U=0$ | A documented Caputo extension test; the pinned zero-initial-state path is not claimed to pass unchanged. |
| Scalar reference trajectory | Candidate equals pinned scalar source under matched zero-initial conditions | Include inputs that do and do not produce spikes. |
| $\alpha=1$, full history | Equality to Euler integration of the same $F_n$ | Not equality to the old exponential-beta prototype. |
| Independent population | Equality to independent scalar evaluations | Test heterogeneous and homogeneous scales. |
| Zero coupling | Equality to the uncoupled population | Same weights, same currents, same initial state. |
| Coupling derivative | Nonzero off-diagonal subthreshold dependence on connected constituents | Ordinary derivative test on smooth dynamics, not hard spikes. |
| Equal selector scores | All $\rho=1$ and recovery of full history | Includes the one-candidate boundary case. |
| Sparse support | An excluded item has zero direct contribution to the selected integral | Its information can still survive indirectly through later states. |
| Group sharing | One historical coefficient is identical across all constituents of a logical neuron | Distinct populations may select different histories. |
| Window/batch separation | Separate windows produce the same results whether processed alone or batched | No cross-sample or cross-channel memory reuse. |
| Causality | Changing later input patches does not change earlier states, scores, or spikes | Patches are timestamped when the entire patch has been observed. |
| State/output timing | $T$ input evaluations, $T$ emitted spike vectors, and a clearly identified terminal state | Respect the one-step effect of new selection under the source convention. |

Tolerances must be reported with dtype and device. A reasonable initial numerical target is around $10^{-8}$ absolute and $10^{-6}$ relative for small smooth float64 tests, and a separately documented float32 tolerance. These are proposed checking targets, not measured achievements.

### 9.2 Gradient tests

Check smooth vector-field, coefficient-modulation, and population-projection gradients independently. Sparsemax derivatives should be tested away from support boundaries; ties and support changes are nondifferentiable points with a defined generalized-gradient convention. [R12]

The hard spike's forward numerical derivative is not the surrogate derivative. Whole-network finite differences through a hard threshold are therefore not a valid test that surrogate backpropagation is “wrong.”

Verify that reset detachment is localized to the reset term, historical dynamics remain differentiable along their actual trajectories, and selector parameters receive a task-loss gradient at decisions capable of affecting observed outputs. A zero direct gradient for the final unused selection is expected under Section 6.9.

### 9.3 Reset-specific stress tests

Use controlled isolated pulses, bursts, constant current, pulse–silence sequences, and negative/positive currents. Examine membrane trajectories, emitted spike timing, and the separate contributions of input/leak/coupling/reset history.

Because $\rho_{n,j}$ modulates complete historical $F_j$, it may suppress an earlier reset-related contribution. This can alter later states even if the selected physical evidence is unhelpful. The experiment must distinguish that effect from successful contextual recall.

There is no universal rule that a fractional neuron must stop spiking immediately after a pulse. The correct reference is the declared dynamics and matched full-history trajectory, not an invented spike-pattern expectation.

### 9.4 Stability scope

For the uncensored no-spike field, a positive scale matrix and a nonnegative symmetric graph produce a dissipative-looking coupled linear structure. This motivates the first coupling choice. It does not prove numerical stability of the selected history equation, stability under repeated resets, bounded surrogate gradients, or noise robustness.

The old condition $\gamma<1-\max\beta$ belongs to the discarded additive-memory prototype and is not a stability certificate for this model.

Record state magnitude, vector-field magnitude, firing rate, support size, and gradients over increasing sequence length. Failed trajectories are results to diagnose, not runs to silently discard.

---

## 10. Forecasting task and data protocol

### 10.1 Dataset scope

**Required horizons:** $H\in\{96,720\}$.

**Proposed staged coverage:**

- Development matrix: ETTh1 and ETTh2, preserving continuity with the previous project while using new model identifiers and fresh training.
- Confirmatory ETT coverage: ETTh1, ETTh2, ETTm1, and ETTm2, with the same two horizons.

The four-series ETT benchmark and the standard chronological hourly/minute loader conventions are documented in the official data repository and Informer data loader. [R10, R11]

The task is multivariate forecasting of all seven numeric variables, excluding the timestamp from prediction targets. The first architecture is channel-independent: parameters are shared across original variables, but their state/history banks are separate. No claim about learned cross-variable interactions is made for this setting.

### 10.2 Exact chronological boundaries

Use half-open row intervals for target regions:

| Dataset family | Training rows | Validation target region | Test target region |
|---|---:|---:|---:|
| ETTh1 / ETTh2 | $[0,8640)$ | $[8640,11520)$ | $[11520,14400)$ |
| ETTm1 / ETTm2 | $[0,34560)$ | $[34560,46080)$ | $[46080,57600)$ |

These boundaries follow the $12/4/4$ blocks represented as $30$-day months in the referenced ETT loader, with the minute variant using four observations per hour. They are row-index conventions, not a new calendar-based split. [R11]

Fit per-variable mean and standard deviation on the training region only. Use those statistics for validation and test. No test-period fitting, retrospective imputation using future values, or selection of preprocessing from test performance is allowed.

For forecast origin $o$, the input is $[o-L,o)$ and the target is $[o,o+H)$. The complete target must lie within its designated region. Validation/test inputs may use already observed preceding context, including context before the target-region boundary. This is not target leakage.

### 10.3 Length, patching, and time units

The initial context length is $L=336$ raw observations. Use chronological nonoverlapping patches of size $P=8$, producing $T=L/P=42$ model steps. Each patch is considered available at its endpoint. No inference is claimed at an interior time before all eight observations have arrived.

One model step is one patch, with $h=1$ in all first-stage neuron comparisons. A second simulation-time axis formed by repeatedly presenting the entire window is not used.

At these settings, increasing $H$ from 96 to 720 changes the future forecast horizon, **not** the number of historical memory items. Both tasks have the same 42-patch context. Hourly and quarter-hourly series also span different physical durations for the same number of rows; report those sampling conventions rather than calling them identical physical-memory tests.

Changing patch size changes both $T$ and the physical duration of a model-time unit. A later patch-size study must distinguish that change from a pure change in the number of memory items.

### 10.4 Window counts as data-pipeline checks

For stride one and the preceding-context policy above, the expected counts are:

| Family | Horizon | Train windows | Validation windows | Test windows |
|---|---:|---:|---:|---:|
| ETTh | 96 | 8,209 | 2,785 | 2,785 |
| ETTh | 720 | 7,585 | 2,161 | 2,161 |
| ETTm | 96 | 34,129 | 11,425 | 11,425 |
| ETTm | 720 | 33,505 | 10,801 | 10,801 |

These are calculated checks from the stated boundaries, $L$, $H$, and stride, not observed new dataset-loading results. Training windows remain entirely within training rows. No final validation/test windows are dropped for batch-size convenience.

### 10.5 Forecasting interface

$$
X\in\mathbb R^{B\times336\times7}
\longrightarrow
\widehat Y\in\mathbb R^{B\times H\times7}.
$$

No future target values enter currents, group-selection contexts, normalization, or neuronal histories. The initial study uses neither future decoder inputs nor teacher forcing. Calendar features and reversible instance normalization are deferred; introducing them later requires the same treatment across matched comparisons.

---

## 11. Architecture progression: start with a Spiking MLP

The neuron must be validated before backbone complexity is increased. More complex architecture is not automatically stronger scientific evidence about memory.

### Stage M1 — Shallow chronological Spiking MLP

The initial backbone contains **one population-spiking hidden stage**, with no attention and no temporal convolution.

Each original variable's length-8 patch is mapped by a learned affine projection into $D=32$ currents. A fixed input-current scale of 2 is the initial contract. Every current drives one $K=4$ population with the same neuron definition.

$$
\text{patch}
\rightarrow
\text{affine current embedding}
\rightarrow
\text{population f-LIF}
\rightarrow
\text{constituent spikes}
\rightarrow
\text{forecast head}.
$$

The projection and forecast head are ordinary real-valued interfaces; the hidden nonlinear/stateful processing is spiking. This does not imply every arithmetic operation is an event-driven addition.

The internal tensor semantics are

$$
[T,BC,P]\rightarrow[T,BC,D]\rightarrow[T,BC,D,K].
$$

Flattening $D\times K$ is allowed at the subsequent synaptic/readout interface. It must not cause the internal history selector to reinterpret constituents as separate logical populations.

### Stage M3 — Deeper Spiking MLP

Use three population-spiking hidden stages at the same $D$ and $K$. Each additional stage maps the same-step $DK$ constituent spikes through an affine projection to $D=32$ new currents, using the same fixed current scale of 2, then applies the same fractional-population neuron. There is no additional nonspiking activation between that affine current map and the population stage.

There is no explicit temporal mixing operator between these hidden stages. Temporal processing still arises from the neuronal dynamics. No analog input-to-head skip path is added in the principal mechanism study.

The same neuron alternatives are compared within M1 and M3. A gain from adding layers is not attributed to the proposed neuron.

### Stage TCN — Causal spiking convolutional extension

After the MLP mechanism checks, add two temporal stages with kernel size 3 and dilations 1 and 2, using only left context at the patch level. Each stage produces currents for the same population neuron.

The convolution-only receptive field is seven patches for the two-stage sequence. The total model's temporal dependence is not limited to seven patches because the neurons already possess history-dependent state integration.

This stage asks whether the neuron adds value beyond local temporal feature extraction. The precise synaptic and residual arrangement must remain identical across its neuron comparisons. It need not reproduce a published Spike-TCN architecture exactly and should not be labeled as such without a source-matched reproduction.

### Stage ATT — Higher-complexity attention comparison

This stage is confirmatory and follows MLP/TCN validation. A causal attention path can be added to test whether selective fractional dynamics provide practical benefit when another component can already access long-range context.

A declared hybrid comparison may use a single causal attention block with feature width 32 and four heads, surrounded by population-spiking stages. It is a **hybrid attention–SNN**, not automatically an all-spiking Transformer. If a genuinely spike-driven attention backbone is studied, its exact attention operation requires a separate frozen specification, with Spikformer as a primary-source reference. [R14]

No attention-based result substitutes for the MLP mechanism experiment. Attention introduces an alternative route to history and a different compute/parameter budget.

### Progression rule

The order is **M1 → M3 → TCN → ATT**. Basic operator correctness is a prerequisite for any training stage. Once an architectural comparison is launched, complete its declared matched conditions regardless of which early run looks favorable. Expansion decisions should use validity and a prespecified research question, not opportunistic selection from test scores.

---

## 12. Readout design: practical forecasting versus memory stress

### 12.1 Practical forecasting head

Project each final-layer $DK$ spike vector to a common readout width of 32, then use the whole chronological sequence for a direct length-$H$ forecast. The same head dimensions and initialization protocol apply to every neuron condition within a backbone.

This is the primary practical ETT comparison. It lets the predictor access all temporal spike outputs. Therefore it tests the incremental value of neuronal memory when historical representations are already exposed to the head.

### 12.2 Bottlenecked spike readout

As a separate mechanism study, restrict the head to the final spike representation, or to a prespecified short terminal segment. The first exact comparison uses the final vector, after the same width-32 spike projection, followed by a length-$H$ forecast mapping.

The head then cannot directly inspect every earlier spike output. However, it sees the source-compatible spike timing of Section 6.9, not the terminal continuous state by accident.

The full-history and final-only heads have different numbers of parameters. Compare neuron variants **within each head setting**; do not attribute the entire difference between head types to memory. A parameter-matched readout sensitivity study is a separate control.

### 12.3 Interpretation boundaries

A final-only head can create an artificially severe binary information bottleneck, especially for $H=720$. Failure there does not by itself disprove fractional memory. Conversely, success under a flatten head does not prove correct historical retrieval.

A continuous terminal-membrane readout can be examined as an explicit diagnostic, but it changes the output representation and cannot silently replace the spike-based head in the main study.

---

## 13. Comparison groups and what each comparison establishes

Use **Full**, **Dense**, and **Sparse** as selection labels. Avoid calling Full “no memory.” Every f-LIF condition still has fractional history.

### 13.1 Primary factorial comparison within heterogeneous populations

All six conditions use the same common alpha, tau vector, population size, spike/reset convention, backbone, and forecast head.

| ID | Internal coupling | History modulation | Main question |
|---|---|---|---|
| H-U-F | Off | Full | Independent heterogeneous f-LIF population reference |
| H-C-F | On | Full | Does coupling help without selection? |
| H-U-D | Off | Dense | Does content modulation help without direct coupling? |
| H-C-D | On | Dense | Does coupling complement dense modulation? |
| H-U-S | Off | Sparse | Does sparse group selection help without direct coupling? |
| **H-C-S** | **On** | **Sparse** | **The proposed interacting population with shared sparse history selection** |

The Full conditions may retain unused selector parameters for paired initialization, but inactive parameters must be reported as inactive. Nominal parameter equality is not equal active capacity.

### 13.2 Homogeneous population controls

Add three homogeneous conditions, all at $\tau_{\mathrm{hom}}=64/15$:

| ID | Coupling | Modulation | Purpose |
|---|---|---|---|
| O-F | Off | Full | Redundant-state reference |
| O-D | Off | Dense | Does dense selection need population diversity? |
| O-S | Off | Sparse | Does sparse selection need population diversity? |

Under the specified shared-current/diffusive-coupling contract, homogeneous coupling is identically inactive. Its equivalence is tested algebraically and numerically rather than counted as three additional independent training conditions.

These nine conditions form the core matrix. The homogeneous control cannot establish that the heterogeneous representation beats every equally expressive alternative; it deliberately collapses state diversity.

### 13.3 Essential external anchors

| Anchor | Role and fairness condition |
|---|---|
| Scalar f-LIF, $K=1$ | Source correspondence and performance anchor. Include a scalar model with comparable total constituent count/width, not only a narrower scalar model. |
| Conventional LIF SNN | Establish an ordinary SNN baseline. State its discretization/reset convention explicitly; do not assert it is identical to the upstream Euler limit. |
| Source-compatible $\alpha=1$ controls | Isolate fractional-order behavior while retaining the same vector field and selection machinery. |
| Equal-constituent scalar f-LIF bank | Test whether gains arise just from more physical neurons rather than population grouping/shared-current structure. |
| Simple linear or ANN MLP forecaster | A task sanity anchor under the same ETT split/preprocessing, not a neuron-mechanism control. |
| Original additive-memory v2 | Optional historical alternative retrained under the new common protocol. Old scores are not directly comparable. |

The equal-constituent scalar forecasting anchors use $DK=128$ scalar units where the population model uses $D=32$, $K=4$, with fixed $\tau_{\mathrm{hom}}=64/15$ as their initial scale. The separate scalar source-parity test can use $\tau=2$ to match the individual reference invocation. For the conventional-LIF forecasting anchor, use $\beta=e^{-1/\tau_{\mathrm{hom}}}$, charging $V_n=\beta U_n+(1-\beta)I_n$, binary output $S_n=H(V_n-\theta)$, and a single subtractive reset $U_{n+1}=V_n-\theta\operatorname{stopgrad}(S_n)$. This is intentionally a separately declared conventional rule, not the source-compatible Euler reset. Their synaptic parameter counts will generally differ because the population shares input current across constituents. Report physical units, active parameters, and compute separately. A further parameter-budget-matched comparison is needed for strong efficiency or expressivity claims.

### 13.4 Tests of group selection specifically

A secondary constituent-specific selector can serve as a contrast, but it is not the proposed model. Compare it under an explicit parameter/search budget and ask whether shared selection improves consistency, accuracy, or cost. It may legitimately be more expressive; no outcome is predetermined.

Selection sharing and internal coupling are separate axes. Demonstrating one does not demonstrate the other.

---

## 14. Ablations that prevent incorrect attribution

### 14.1 Fractional kernel versus explicit history access

At minimum, compare the full proposed model and its full-history coupled reference at $\alpha=1$ and at the fixed fractional alpha. A later common grid such as $\{0.5,0.7,0.9,1.0\}$ can assess sensitivity.

The same validation-only selection policy and tuning budget must apply to compared conditions. Never choose a different best alpha from the test set for each method.

### 14.2 Coupling and population diversity

Study $K\in\{1,2,4,8\}$, fixed versus heterogeneous tau, and coupling strength separately. For $K=1$, coupling is zero by definition; it is not a four-component model with the population axis accidentally collapsed.

Population-size changes require fixed-logical-width and fixed-total-constituent-budget views. Learned tau, per-constituent alpha, and learned coupling are separate later ablations rather than additions to the main candidate midway through evaluation.

### 14.3 Content selection versus simple policies

Compare learned selection with recent-history, fixed-lag, and random-support alternatives, using the same eligible historical set and current-term treatment.

Separate **retrained policy controls** from **same-checkpoint interventions**. They answer different questions. A learned model damaged by an inference-time intervention may simply have co-adapted to that pathway.

### 14.4 Selection versus integral attenuation

The bounded selector can lower the total historical coefficient mass. A benefit could therefore arise from reducing the strength of historical dynamics rather than selecting relevant content.

Define, for $n>0$,

$$
\kappa_n
=
\frac{\sum_{j<n}b_{n-j}^{(\alpha)}\rho_{n,j}}
{\sum_{j<n}b_{n-j}^{(\alpha)}}.
$$

A useful intervention replaces all historical coefficients by the uniform modulation $\rho^{\mathrm{mass}}_{n,j}=\kappa_n$. This matches total kernel coefficient mass while removing its temporal allocation. The statistic must be computed causally from the declared reference trajectory; frozen-bank local interventions and recursively regenerated trajectories are different experiments.

Also compare raw sparse support masking against the full sparse amplitude modulation. Sparsemax versus softmax changes both support and relative amplitude; it is not a pure test of “zeros versus no zeros.”

### 14.5 Selection of reset contributions

The first candidate modulates $\mathbf F_j=\mathbf D_j-\mathbf R_j$. A diagnostic alternative modulates only the historical subthreshold field while retaining the full reset history:

$$
\mathbf U_{n+1}^{\mathrm{reset\text{-}protected}}
=
\mathbf U_0+b_0\mathbf F_n
+\sum_{j<n}b_{n-j}\big(\rho_{n,j}\mathbf D_j-\mathbf R_j\big).
$$

This is a **different reset-history policy**, not a correction silently applied to the principal model. If it changes conclusions materially, the paper must distinguish evidence selection from reset manipulation.

Likewise, a post-convolution jump/reset formulation needs its own self-consistent history definition and reference. It is not specified as the first candidate here.

### 14.6 State representation and selection context

Compare the population state/current context with scalar state, state-only, and current-only contexts. Keep the integrated value type fixed when assessing key quality.

Directly replacing $\mathbf F_j$ by $\mathbf U_j$ changes the neuronal model into a different memory integration scheme. It can be a comparative model, but it is not a neutral key ablation.

### 14.7 History span versus forecasting horizon

Vary context length or maximum accessible lag at a fixed prediction length to test historical-memory capability. Vary prediction length at fixed context to test forecast extrapolation. The two questions must not be conflated.

Any short-memory truncation uses actual chronological lags in $b_d$; retained events must not be reindexed as if the discarded time never occurred.

---

## 15. Training, model selection, and repeated-run protocol

### 15.1 Proposed common training budget

| Setting | Proposed initial value |
|---|---|
| Objective | Mean squared error over all forecast positions and all seven variables |
| Optimizer | AdamW |
| Initial learning rate | $10^{-3}$ |
| Weight decay | $10^{-2}$ on trainable affine weight matrices; zero on biases |
| Batch size | 128, held identical within a matched comparison |
| Gradient norm limit | 1 |
| Maximum epochs | 50 |
| Early stopping | 10 epochs without strictly lower validation MSE |
| Learning-rate schedule | Reduce on validation plateau, factor 0.5, patience 5 |
| Development seeds | 7, 13, 21 |
| Confirmatory seeds | 7, 13, 21, 42, 123 |
| Checkpoint | Strict minimum validation MSE; an exact tie keeps the earlier checkpoint |
| Final reporting | Fresh evaluation after restoring the selected checkpoint |

This budget is a proposed new common protocol. It differs from some v1/v2 budgets and therefore does not support direct comparisons with their archived scores. [P1]

No new optimum is claimed for these settings. A prespecified validation-only budget sensitivity check can compare 50 with 100 maximum epochs using the same stopping rule. Report best epoch and actual trained epochs; a nominal maximum alone does not establish undertraining or convergence.

### 15.2 Initialization and fair treatment

Shared backbone/head components begin from paired initial values across neuron variants for each dataset, horizon, and seed. Selector projections have the explicit initialization from Section 6.6. The initial coupling graph is fixed, so coupling does not add an uncontrolled trainable capacity advantage.

All methods use the same data windows, order/randomization policy, metric definition, and checkpoint selection rule. Device/dtype/determinism settings and any numerical incompatibilities are part of the experiment provenance.

A source-compatible scalar benchmark and a library-native conventional LIF benchmark may require different internal integration code paths. That difference is documented rather than hidden under nominally matching tau values.

### 15.3 Staged matrix size

For the nine-condition core matrix, one architecture over ETTh1/ETTh2, two horizons, and three seeds requires

$$
9\times2\times2\times3=108
$$

training runs. Adding scalar/ordinary anchors, readout variants, or reset alternatives adds separate runs.

The four-dataset, five-seed confirmatory core requires

$$
9\times4\times2\times5=360
$$

runs per architecture. This is a planning count, not a demand to launch every architecture immediately. Numerical and mechanism gates precede large grids.

### 15.4 Validation and test discipline

Use validation MSE for training decisions. Do not select alpha, coupling, score temperature, readout, or reset policy based on test performance. Repeated design changes after inspecting ETT test outcomes make subsequent scores exploratory; additional held-out confirmation should be used before strong claims.

A planned architecture stage is not selectively completed only for promising variants. Failures, numerical divergences, missing runs, and retries must remain visible. An incomplete matrix is not used for a balanced method average.

---

## 16. Evaluation criteria: what would count as evidence?

### 16.1 Forecast metrics

For $N_w$ evaluation windows,

$$
\mathrm{MSE}
=\frac{1}{N_wHC}
\sum_{i=1}^{N_w}\sum_{h=1}^{H}\sum_{c=1}^{C}
(\widehat y_{i,h,c}-y_{i,h,c})^2,
$$

$$
\mathrm{MAE}
=\frac{1}{N_wHC}
\sum_{i,h,c}|\widehat y_{i,h,c}-y_{i,h,c}|.
$$

Primary metrics use training-standardized data. Raw-unit metrics may be supplementary and must be labeled. Each window–horizon–channel forecast is an evaluation item; overlapping target timestamps are intentionally counted across their distinct forecast origins.

Report each dataset and each horizon separately, then any declared macro average. Do not average 96 and 720 into a single headline that hides opposite trends.

### 16.2 Paired contrasts

For every dataset, horizon, backbone, and seed, calculate at least:

$$
\Delta_{\mathrm{selection}}
=E_{\mathrm{H-C-S}}-E_{\mathrm{H-C-F}},
$$

$$
\Delta_{\mathrm{coupling}}
=E_{\mathrm{H-C-S}}-E_{\mathrm{H-U-S}},
$$

and the interaction contrast

$$
\boxed{
\Delta_{\mathrm{interaction}}
=
\big[E_{\mathrm{H-C-S}}-E_{\mathrm{H-C-F}}\big]
-
\big[E_{\mathrm{H-U-S}}-E_{\mathrm{H-U-F}}\big].
}
$$

Here $E$ is a lower-is-better metric. A negative interaction contrast indicates a larger sparse-selection improvement when coupling is present, on that metric and experimental condition. It is not automatically statistically significant.

Use both absolute differences and clearly defined relative changes. The variance of a paired difference is not obtained by casually comparing the two models' separate standard deviations.

### 16.3 Population and selector diagnostics

Record more than support density:

- Off-diagonal state dependence and response to constituent perturbations.
- Constituent membrane correlation, spike correlation, state variance, and effective representation rank.
- Selection support size, its dependence on eligible history length, and its lag distribution.
- Variation of selected lags with current context.
- Raw score scale, raw probability, modulation coefficient $\rho$, and the final coefficient $b_d\rho$ as distinct quantities.
- Kernel-mass ratio $\kappa_n$, coupling contribution magnitude, and reset-history contribution magnitude.
- Per-layer and per-dataset behavior, not only the final layer.

Use all evaluation windows when feasible, or a predeclared temporally distributed sample. The first few highly overlapping windows are not representative of the entire test period. The earlier handoff's first-eight-window statistics must not be generalized to this new model. [P1]

### 16.4 Retrieval validity versus sparse behavior

Exact zeros prove selective support, not correct memory selection. High weights are not causal explanations. A useful diagnostic removes/replaces selected historical **content** while specifying whether the bank and later trajectory are frozen or regenerated.

A same-checkpoint intervention and a separately trained control answer different questions. Both are valuable, but they must not be pooled into one “ablation score.”

### 16.5 Controlled temporal-recall study

Before interpreting ETT as evidence of contextual recall, include a synthetic task with known relevant events, distractors, and context-dependent lag changes.

Because the first ETT model is channel-independent, a synthetic cue and its associated value must be accessible within the same tested input stream/population pathway. Placing the cue in another independent channel would make the desired dependency unavailable by construction.

The response must occur after the query has had time to influence the emitted spike under the source-compatible timing. Use an explicit query–response separation rather than calling a same-step output a test of newly selected history.

A membrane/dynamics record may summarize multiple earlier observations. A relevant raw lag is not always a unique correct memory slot. Define relevant event intervals or eligible slot sets in advance and report:

- Target prediction error.
- Mass or rank assigned to the known relevant event set.
- Support-hit rate relative to support size and a random-policy expectation.
- Sensitivity to distractor count and delay.
- Changes under causal content replacement.

A large support-hit rate alone is weak evidence when the support contains most of the bank. Failure to recover one arbitrarily designated slot is also not conclusive when later records contain the same information.

### 16.6 Statistical scope

Three seeds provide an initial paired variability check, not broad evidence of significance. Confirmatory runs should report seed-level paired differences and confidence intervals, with the method and assumptions declared.

Overlapping forecasting windows are temporally dependent. Treating thousands of overlapping windows as independent replications would overstate precision. Any resampling over data should respect temporal blocks; datasets and seeds remain separate sources of variation.

No claim of universal improvement should follow from a macro mean that conceals dataset/horizon reversals.

---

## 17. How to interpret possible outcomes

| Result pattern | Supported interpretation | Unsupported shortcut |
|---|---|---|
| H-C-S beats H-C-F and H-U-S, with useful controlled recall | Evidence for a coupled selective fractional population under the tested conditions | “Universally superior neuron” |
| H-C-F helps, but H-C-S does not | Internal fractional-population interaction may help; selection is not yet justified | Claiming the selector is the source of the gain |
| H-U-S helps as much as H-C-S | Shared fractional-history selection may help; direct coupling is unnecessary in that setting | Claiming interaction is essential |
| Homogeneous sparse performs as well as heterogeneous sparse | State diversity is not established as the reason retrieval works | Calling redundancy-control results proof of population coding |
| Sparse beats dense, but mass-matched uniform is equally good | Attenuation or regularization may explain the improvement | “Correct memories were found” |
| Gains vanish when reset history is protected | Reset modulation is a serious alternative explanation | Treating the effect as semantic recall without further evidence |
| $\alpha=1$ selected control performs equally well | The explicit selection may matter more than fractional dynamics | Claiming the power-law kernel is essential |
| Only the bottleneck head benefits | A memory-dependent task regime has been identified | Assuming broad ETT forecasting superiority |
| Only a complex backbone benefits | An architecture interaction is possible | Claiming the standalone neuron was validated |
| Controlled recall works but ETT gains are small | Mechanism validity and practical task benefit differ | Calling either result an automatic failure of the other |

The original paper's motivation remains a starting point, not a substitute for these comparisons.

---

## 18. Reproducibility and deliverable requirements for the eventual experiment

The research record should make it possible to identify, for each result:

- The pinned upstream revision and the exact new-model revision.
- The numerical route, initial-state convention, time grid, spike/reset convention, and surrogate.
- Dataset file identity, split boundaries, scaler fit region, input/target indexing, and window counts.
- Population size, scale assignment, coupling graph/strength, selector mode, and all coefficient normalizations.
- Backbone and head definitions, total constituents, nominal/active trainable parameters, and any analog temporal paths.
- Seed, training budget, actual epochs, best validation epoch, selected checkpoint, and fresh-reload result.
- State/spike/selection diagnostics, numerical failures, and the distinction between intervention and retraining.

Do not reuse old v2 `memory_strength` or null/gate settings under unchanged names with different meanings. Preserve prior experiments as a separate model family.

No latency or energy claim follows from sparse support alone. Dense pairwise scoring over a length-$T$ history remains potentially quadratic in temporal comparisons, and sparsemax has its own selection cost. Coupling, stored states, stored dynamics, query/key evaluation, and memory traffic all count. Stationary-convolution acceleration from the unselected fractional model does not automatically apply to content-dependent coefficients.

Shared-GPU wall-clock logs are provenance, not a controlled efficiency benchmark. No new GPU timings, energy figures, or runtime success are asserted in this document.

---

## 19. Deferred research directions

These are not part of the first experiment contract:

| Direction | Why deferred |
|---|---|
| Per-constituent $\alpha_k$ | Requires explicit multi-order state semantics; the upstream alpha-list interface is not sufficient evidence of support. |
| Learned fractional order | Full coefficient gradients and discretization sensitivity must first be validated. |
| Learned tau or coupling graph | Adds optimization/capacity changes before the fixed coupled population is understood. |
| Null-only history selection | Removing all historical integration has a different meaning from disabling an additive-memory branch. |
| Post-convolution reset | Needs a separately specified impulsive/history convention and source comparison. |
| Fractional kernel plus extra temporal prior | The model already contains a fractional kernel; an additional prior is not automatically necessary. |
| Cross-window streaming state | Changes the statistical protocol and memory ownership beyond the independent-window study. |
| Sparse search acceleration | Requires reducing search work, not just zeroing final contributions. |
| All-spiking Transformer | Needs an explicit attention definition and fair operation-level comparison. |
| Anomaly detection | Requires additional treatment of abnormal-history retention and task-specific causality. |

A deferred feature should not be introduced to rescue an unfavorable test result without a new, documented research comparison.

---

## 20. Final handoff summary

The agent's first goal is not to recreate the old ordinary-LIF-plus-memory prototype under a new name. It is to establish a source-traceable fractional population model with these properties:

1. **Fractional state formation:** The membrane state comes from the declared predictor history equation.
2. **Whole-population temporal choice:** The same historical coefficient is shared by all constituents of one logical neuron.
3. **Direct internal interaction:** Other constituents' states enter a constituent's vector field through an explicit coupling law.
4. **Recoverable reference cases:** Disabling coupling and selection, or reducing to $K=1$, yields the specified reference rather than a different undocumented neuron.
5. **Inspectable timing and reset:** The upstream spike/reset convention is retained transparently in the first contract, including its one-step selection effect.
6. **Evidence beyond sparsity:** Forecasting comparisons separate coupling, diversity, selection, coefficient attenuation, and fractional order.
7. **Chronological evaluation:** Spiking MLP experiments precede more complex backbones, with ETT horizons 96 and 720 evaluated under fixed train/validation/test rules.

The compact model statement is

$$
\boxed{
\begin{aligned}
\mathbf D_n
&=\mathsf T_\tau^{-1}
[-\mathbf U_n+\mathbf1 I_n-\lambda\mathsf L_{\mathrm{pop}}\mathbf U_n],\\
\mathbf S_n
&=H(\mathbf U_n+\mathbf D_n-\theta\mathbf1),\\
\mathbf F_n
&=\mathbf D_n-\theta\mathsf T_\tau^{-1}\operatorname{stopgrad}(\mathbf S_n),\\
\mathbf U_{n+1}
&=\mathbf U_0+b_0^{(\alpha)}\mathbf F_n
+\sum_{j<n}b_{n-j}^{(\alpha)}\rho_{n,j}\mathbf F_j.
\end{aligned}
}
$$

The coupling law and selector above are a **proposed first realization of the user's requirements**. Their empirical effectiveness and theoretical properties remain to be established.

---

## References and source provenance

### User-provided project sources

**[P0] Uploaded f-SNN paper.** `ICLR-2026-fractional-order-spiking-neural-network-Paper-Conference.pdf`. Relevant locations: Sections 2–3, especially equations (11)–(13); Appendix C; Appendix E. The uploaded version is the reference for manuscript claims. A public paper landing page is provided in [R1].

**[P1] Uploaded PopulationLIF forecasting handoff.** `붙여넣은 마크다운(1).md`, snapshot timestamp **2026-09-15 13:59:24 KST**, old branch `exp/f-lif-pop-v2`, recorded HEAD `8c3d13673ce628bff51e7ab858a67b5bc72584da`. Relevant sections: 2–6 for the old Branch B model/protocol; 8 for completed patch results and diagnostic limitations; 11 for not-yet-run follow-up ideas; 15 for the old actual code. This document does not claim access to its linked live server logs or to a newer project state.

### Public primary sources

**[R1] Ge, C., Peng, Y., Li, Z., Kang, Q., Fu, X., Li, X., Zhang, Q., Ren, J., and Zha, Z.-J.** *Fractional-order Spiking Neural Network.* ICLR 2026 manuscript supplied by the user; public arXiv record **2507.16937**.  
https://arxiv.org/abs/2507.16937

**[R2] PhysAGI.** *spikeDE: official f-SNN repository.*  
Repository: https://github.com/PhysAGI/spikeDE  
Pinned revision: https://github.com/PhysAGI/spikeDE/commit/fcd743befe504b1a471fa81887e6af7d6789da2e  
Documentation: https://physagi.github.io/spikeDE/

**[R3] PhysAGI.** *MIT License at the inspected commit.*  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/LICENSE

**[R4] PhysAGI.** *`spikeDE/neuron.py`, pinned source.* Relevant identifiers: `BaseNeuron`, `LIFNeuron`, `LIFNeuron_OrderOne`, `LIFNeuronFDE`.  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/neuron.py  
Raw: https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/neuron.py

**[R5] PhysAGI.** *`spikeDE/solver.py`, pinned source.* Relevant identifiers: `SNNSolverConfig`, `AdamsBashforthSNN`, `snn_solve`, `pred_integrate_tuple`, `SOLVERS`.  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/solver.py  
Raw: https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/solver.py

**[R6] PhysAGI.** *`spikeDE/snn.py`, pinned source.* Relevant identifiers: `PerLayerAlphaConfig`, `SNNWrapper`, direct `fdeint` dispatch, state initialization, boundary processing.  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/snn.py  
Raw: https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/snn.py

**[R7] PhysAGI.** *`spikeDE/odefunc.py`, pinned source.* Relevant identifiers: `ODEFuncFromFX`, `interpolate`, spike-output graph mapping.  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/odefunc.py  
Raw: https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/odefunc.py

**[R8] PhysAGI.** *`spikeDE/surrogate.py`, pinned source.* Relevant identifiers: `SigmoidSurrogate`, `ArctanSurrogate`.  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/surrogate.py

**[R9] PhysAGI.** *Public neuromorphic model example, pinned source.*  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/examples/ICLR_Release/Neuromorphic/model.py

**[R9b] PhysAGI.** *Public SGCN/Citeseer predictor example, pinned source.* Used only to verify that a released example selects the `pred` path. Its command is not a forecast implementation prescription.  
https://github.com/PhysAGI/spikeDE/blob/fcd743befe504b1a471fa81887e6af7d6789da2e/examples/ICLR_Release/Graph/scripts/run_sgcn_citeseer.sh

**[R10] Zhou, H., et al.** *ETDataset: Electricity Transformer dataset repository.*  
https://github.com/zhouhaoyi/ETDataset

**[R11] Informer2020 authors.** *Official ETT hourly/minute data-loader conventions.* The split and scaling facts were inspected online; the experiment should record its own pinned loader/data revisions.  
https://github.com/zhouhaoyi/Informer2020/blob/main/data/data_loader.py  
https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/data/data_loader.py

**[R12] Martins, A. F. T., and Astudillo, R. F. (2016).** *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification.* Proceedings of ICML, PMLR 48:1614–1623.  
https://proceedings.mlr.press/v48/martins16.html

**[R13] Perez-Nieves, N., Leung, V. C. H., Dragotti, P. L., and Goodman, D. F. M. (2021).** *Neural heterogeneity promotes robust learning.* Nature Communications 12:5791. DOI: 10.1038/s41467-021-26022-3. Author-maintained paper page used for verification:  
https://neural-reckoning.org/pub_heterogeneity.html  
https://doi.org/10.1038/s41467-021-26022-3

**[R14] Zhou, Z., Zhu, Y., He, C., Wang, Y., Yan, S., Tian, Y., and Yuan, L.** *Spikformer: When Spiking Neural Network Meets Transformer.* ICLR 2023; arXiv:2209.15425. Used as a later attention-backbone reference, not a claim that the proposed forecast architecture reproduces it.  
https://arxiv.org/abs/2209.15425

**[R15] Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J.** *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023; arXiv:2211.14730. A reference for patching/channel-independent forecasting context, not the definition of the initial Spiking MLP.  
https://arxiv.org/abs/2211.14730

Public sources were inspected on **17 September 2026**. The commit-pinned spikeDE links are the authoritative source locations for the excerpts. Moving external repository links are reference material, not a claim that their HEAD matches any archived local project.

---

## Appendix A. Upstream license notice for reproduced code

The upstream code excerpts in Section 3 are reproduced under the following license. [R3]

```text
MIT License

Copyright (c) 2026 PhysAGI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Appendix B. Decision register

| Decision | Status in this document | Revision rule |
|---|---|---|
| f-LIF-based integration rather than ordinary-LIF additive memory | User requirement | Do not change the model family silently. |
| Group-shared temporal selection | User requirement | Constituent-specific retrieval is a comparator, not the principal model. |
| Internal population interaction | User requirement | A shared selector alone does not satisfy this requirement. |
| MLP-first forecasting, H96 and H720 | User requirement | Both horizons remain visible in reporting. |
| Common fixed alpha, heterogeneous tau | Proposed first contract | Multi-order/learnable dynamics are separate variants. |
| Fixed chain coupling | Proposed first contract | Learned or directed coupling is a separately named extension. |
| Source-compatible spike/reset timing | Proposed first contract | Post-integration spike/reset is a different scientific contract. |
| Relative-to-maximum selector scaling | Proposed first contract | Alternative scaling must preserve or explicitly change the neutral limit. |
| No null-only read initially | Proposed first contract | An empty-history policy needs an explicit dynamical interpretation. |
| ETTh development, all four ETT series confirmation | Proposed staged scope | Changes must be declared before using test outcomes. |
| Training and seed budgets | Proposed experimental protocol | Common within matched comparisons; do not mix old budgets into new evidence. |

**End of document.**
