#!/usr/bin/env python3
"""Build or check spikede.py, the verbatim copy of the spikeDE code the neurons need.

    python spikede_copy.py build <spikeDE root>     write spikede.py from the upstream files
    python spikede_copy.py check <spikeDE root>     every copied block must equal upstream, character
                                                    for character, and nothing else may be defined

spikeDE: https://github.com/PhysAGI/spikeDE, commit fcd743befe504b1a471fa81887e6af7d6789da2e (MIT).
Only top-level definitions are copied, each one whole (decorators and docstrings included), in the
upstream order. The public wrappers pred_integrate_tuple / gl_integrate_tuple are not copied: for a
single-term alpha they are exactly snn_solve(..., AdamsBashforthSNN() | GrunwaldLetnikovSNN())
(spikeDE solver.py, _get_solver_with_multiterm_fallback), which is what neurons.py calls, and
copying them would drag in the multi-term solvers the neurons never use.
"""
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

COMMIT = 'fcd743befe504b1a471fa81887e6af7d6789da2e'
BLOCKS = [                                   # (upstream file, top-level names in upstream order)
    ('spikeDE/surrogate.py', ['SigmoidSurrogate', 'sigmoid_surrogate', 'ArctanSurrogate', 'arctan_surrogate',
                              'PiecewiseLinearSurrogate', 'piecewise_linear_surrogate',
                              'GaussianSurrogate', 'gaussian_surrogate']),
    ('spikeDE/neuron.py', ['surrogate_f', 'BaseNeuron', 'LIFNeuron']),
    ('spikeDE/solver.py', ['euler_integrate_tuple', 'PerLayerAlphaInfo', 'SNNSolverConfig', 'get_memory_bounds',
                           'SNNFractionalMethod', 'GrunwaldLetnikovSNN', 'AdamsBashforthSNN', 'snn_solve']),
]
OUT = Path(__file__).resolve().with_name('spikede.py')
HEADER = f'''"""Verbatim copy of the spikeDE code used by neurons.py. DO NOT EDIT -- regenerate with
`python spikede_copy.py build <spikeDE root>` and verify with `python spikede_copy.py check <root>`.

Source: https://github.com/PhysAGI/spikeDE, commit {COMMIT}
        ("Fractional-order Spiking Neural Network", Ge et al., ICLR 2026). MIT licence, reproduced below.
Every definition below is copied whole and unchanged from the upstream file named in its banner;
only the imports at the top are written here (the union of what the copied code uses).
"""
'''
IMPORTS = '''import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
'''


def segments(path):
    """Top-level name -> (first line, last line, exact source text) of an upstream or local file."""
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        for name in names:
            out[name] = (start, node.end_lineno, ''.join(lines[start - 1:node.end_lineno]))
    return out


def upstream_commit(root):
    r = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else 'unknown'


def build(root):
    if upstream_commit(root) != COMMIT:
        raise SystemExit(f'{root} is not at {COMMIT}')
    parts = [HEADER, '\n# ' + '\n# '.join((root / 'LICENSE').read_text().strip().splitlines()) + '\n\n', IMPORTS]
    for rel, names in BLOCKS:
        seg = segments(root / rel)
        sha = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for name in names:
            first, last, src = seg[name]
            parts.append(f'\n\n# ---- {rel}:{first}-{last} (file sha256 {sha[:16]}) ' + '-' * 20 + '\n')
            parts.append(src)
    OUT.write_text(''.join(parts))
    print(f'[spikede] wrote {OUT.name}: {sum(len(n) for _, n in BLOCKS)} definitions from {COMMIT[:9]}')


def check(root):
    """Fail closed: missing, extra or changed definitions all fail."""
    if upstream_commit(root) != COMMIT:
        print(f'[spikede] FAIL | upstream root is not at {COMMIT}')
        return 1
    mine = segments(OUT)
    wanted = [n for _, names in BLOCKS for n in names]
    problems = [f'extra definition {n}' for n in mine if n not in wanted]
    for rel, names in BLOCKS:
        seg = segments(root / rel)
        for name in names:
            if name not in mine:
                problems.append(f'missing {name}')
            elif mine[name][2] != seg[name][2]:
                problems.append(f'{name} differs from {rel}:{seg[name][0]}-{seg[name][1]}')
    imports_ok = OUT.read_text().count(IMPORTS) == 1
    ok = not problems and imports_ok
    print(f"[spikede] {'PASS' if ok else 'FAIL'} | {len(wanted)} definitions identical to spikeDE {COMMIT[:9]}: "
          f"{len(wanted) - len([p for p in problems if 'differs' in p or 'missing' in p])}/{len(wanted)}; "
          f"extra definitions: {len([p for p in problems if p.startswith('extra')])}; imports block intact: {imports_ok}")
    for p in problems:
        print(f'[spikede]        {p}')
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('build', 'check'):
        raise SystemExit(__doc__)
    root = Path(sys.argv[2]).resolve()
    raise SystemExit(build(root) if sys.argv[1] == 'build' else check(root))
