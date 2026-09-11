"""Temporary model placeholder for population-selective membrane memory.

Concept: ../docs/population_selective_membrane_memory_snn_concept.md
Status: design discussion only; no model or training implementation yet.

Intended direction (concept Branch B):
    heterogeneous LIF population -> population membrane-state history
    -> current-conditioned temporal retrieval -> current spiking dynamics.

One logical neuron contains K constituent LIF neurons with diverse temporal
responses. Retrieval selects historical time slots jointly for the population;
its constituents share the temporal weights. Time follows observations or
ordered patches, and retrieval reads only earlier states of the same sequence.

User-confirmed memory use (2026-09-11): evidence reinforcement through an
additive contribution to the current membrane state before spike generation.
A candidate gated formulation is v_t = u_bar_t + gamma * g_t * memory_t,
where u_bar_t is the charged state before retrieval and memory_t is the
retrieved population vector. The current-state coefficient remains one.
Setting the memory contribution to zero recovers the underlying population
update. Gate parameterization and memory strength are still open choices.
The intended order is charge -> retrieve past slots -> add memory evidence
-> spike/reset -> store the chosen state for future time steps.

Open design choices include reset/storage semantics, relevance scoring,
soft versus hard selection, memory-use gating, population spike readout,
and an optional fractional temporal prior. These remain unimplemented.

The requested filename is provisional: direct membrane-state retrieval does
not by itself implement a fractional differential equation or inherit f-LIF
theoretical guarantees. No forecasting performance has been measured.
"""
