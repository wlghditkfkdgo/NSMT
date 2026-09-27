#!/usr/bin/env python3
"""Post-hoc descriptive check for reading M7 (added after the diagnostic output): how often do the
pearson masks chosen at alpha=0.7 and at alpha=1 (fresh seed-7 embeddings, same patches) agree?
On ETTh2 the M7 replay peaks equalled the dynamic peaks exactly; identical masks would make that
replay uninformative. Reports the share of (sequence, step, slot) mask entries that differ, overall
and in the sequence that holds each run's peak |u|."""
import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_stability as S                                 # noqa: E402


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    out = {}
    for data in S.A.DATASETS:
        patches = S.batches_of(data).to(S.DT)
        runs = {}
        for a in (.7, 1.):
            emb = S.init_embedding(data, a, patches.float()).double()
            st, masks, _ = S.run(emb.current(patches), emb.neuron.b.to(S.DT), emb.neuron.tau.to(S.DT), 'pearson', .5)
            runs[a] = (st, masks)
        diff, total = 0, 0
        for t in range(1, len(runs[.7][1])):
            m7, m1 = runs[.7][1][t], runs[1.][1][t]
            diff += int((m7 != m1).sum()); total += m7.numel()
        peak_seq = {a: int(runs[a][0].abs().amax(dim=(0, 2, 3)).argmax()) for a in runs}
        seq_diff = {str(a): sum(int((runs[.7][1][t][s] != runs[1.][1][t][s]).sum()) for t in range(1, 42))
                    for a, s in peak_seq.items()}
        out[data] = {'differing_share': diff / total, 'peak_sequence': {str(a): s for a, s in peak_seq.items()},
                     'differing_entries_in_peak_sequence': seq_diff}
        print(f"[masks] {data}: {100 * diff / total:.2f}% of mask entries differ between alpha 0.7 and 1; "
              f"peak sequences {peak_seq}; differing entries there {seq_diff}", flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
