"""Regenerate the pinned spikeDE scalar trajectories used by gate G4.

The golden file checked in at reference/golden/scalar_trajectories.csv was produced from
PhysAGI/spikeDE at commit fcd743befe504b1a471fa81887e6af7d6789da2e, run on CPU with
torch 2.11 (the `snn_jelly` environment; `snn_recall` is torch 1.12 and cannot import it).
Its SHA256 is pinned in check_model.py, so a silent change to the reference fails the gate
rather than passing quietly.

This script does not vendor the upstream source. Point it at a checkout of that commit:

    /home/yschoi/.conda/envs/snn_jelly/bin/python reference/make_golden.py \
        --source /path/to/spikeDE --out reference/golden/scalar_trajectories.csv

What G4 compares, and what it does not
--------------------------------------
v3-A's branches never reset (prereg D1) while the upstream neuron subtracts at threshold,
so the two are the same recurrence only up to the step before the first spike. 7 of the 24
recorded conditions never spike and are compared in full; the rest are compared on their
pre-spike prefix. Passing G4 therefore means source parity for the fractional INTEGRATOR,
not for the whole neuron, and the reference grade is reported with that qualifier.
"""

import csv
import argparse
import hashlib
from pathlib import Path

ALPHAS = (0.3, 0.5, 0.7, 1.0)
TAUS = (4.0, 8.0)
INPUTS = ('constant', 'pulse', 'random')
STEPS = 40
SEED = 7


def main():
    parser = argparse.ArgumentParser(description='regenerate the G4 golden trajectories')
    parser.add_argument('--source', required=True, help='checkout of spikeDE at fcd743b')
    parser.add_argument('--out', default=str(Path(__file__).parent / 'golden'
                                             / 'scalar_trajectories.csv'))
    args = parser.parse_args()

    import sys
    sys.path.insert(0, args.source)
    import torch                                             # noqa: F401  (torch 2.x required)

    raise SystemExit(
        'Not implemented here on purpose. The trajectories currently checked in were produced '
        'by the audit session and verified twice from an independent port, with state errors '
        'at 1e-16 (results/assessment/20260921-145954-utc/reference/reference_results.json). '
        'Rather than re-derive the upstream call sequence from memory and risk a silently '
        'different reference, fill this in against the actual module layout of the checkout '
        f'at {args.source}, covering alpha in {ALPHAS}, tau in {TAUS}, inputs {INPUTS}, '
        f'{STEPS} steps, seed {SEED}, float64, writing '
        'input,alpha,tau,n,current,upstream_U_next,upstream_spike to ' + args.out)


if __name__ == '__main__':
    main()
