"""Open confirm4 once: O7 for the hard design and the GRU comparison (prereg 2M).

Conditions per pre-registered seed (D-AX): pearson (2J checkpoint), q1 (2J checkpoint),
oracle (mode=hard, hard_stat=oracle, trained for 2M), gru (GRUBaseline, trained for 2M), and
tto -- the pearson checkpoint with the oracle mask injected at evaluation only (O7-4 diagnostic).

Procedure (D-BA; 2L D-AV as actually implemented, audit 41):
  1. every run must have finished its registered 12 epochs;
  2. BEFORE the split is claimed, all 32 runs are checked against the registered manifest and
     the reused checkpoints against the 2J record's hashes -- any mismatch refuses the opening;
  3. the torch version must be the one the models were trained and evaluated with, because
     top-k ties are broken by that implementation of torch.topk (the declared tie rule);
  4. the split is claimed in the global registry, then the record is created with 'x';
  5. each model is passed over the WHOLE split once; each batch is ONE forward that yields
     the error sums, every state, and the coefficients (for M_eff and ties);
  6. safe = every state finite AND max|u| < the frozen bound (myModel); output finite (GRU);
     an unsafe model withholds every verdict it enters.

Verdicts (D-AZ): O7-1 M_eff(pearson) >= 0.5; G14 q1 - oracle CI lower > MDE 0.005; O7-2
G = (E_q1 - E_pearson) / (E_q1 - E_oracle) >= 0.5, ratio of 8-seed means; O7-3 pearson - q1
<= -20% with CI upper < 0; O7-4 |E_tto - E_q1| / E_q1 as a diagnostic; GRU two-sided.

Usage:
    python hard_benchmark.py --twoj hardsel-014614 --suite hardbench-XXXXXX --out hardbench-XXXXXX
"""
import sys
import glob
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK                                        # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from test import EVENT_KINDS, kind_mask                        # noqa: E402
import layers                                                  # noqa: E402
from hard_selection import load, sha256, SOURCES               # noqa: E402
from split_registry import open_once                           # noqa: E402

SEEDS = (7, 13, 21, 42, 123, 256, 512, 1024)
SPLIT = 'confirm4'
TORCH = '1.12.0+cu113'
BOUND = 305.0375175476074
MDE = .005
COMMON = {'task': 'recall', 'n_train': 2048, 'n_val': 256, 'batch_size': 64, 'epoch': 12,
          'data_seed': 20260921, 'n_confirm': 1000}
ANALYSIS = ('hard_benchmark.py', 'split_registry.py', 'hard_selection.py', 'hard_safety.py')


def locate(suite, seed, pattern, result):
    runs = glob.glob(str(TASK / 'log' / suite / 'recall' / '*' / '*' / f'seed{seed}_{pattern}'))
    res = glob.glob(str(TASK / 'results' / suite / f'{result}_seed{seed}.json'))
    if len(runs) != 1 or len(res) != 1:
        raise SystemExit(f'[bench] {suite} seed {seed} {pattern}: {len(runs)} dirs, {len(res)} results')
    return runs[0], res[0]


def targets(twoj_suite, suite, twoj_hashes):
    """(seed, condition) -> dict(run, result, manifest, stored hash or None)."""
    out = {}
    for seed in SEEDS:
        for cond, q in (('pearson', .5), ('q1', 1.)):
            run, res = locate(twoj_suite, seed, f'*hard-shared-pearson-q{q:g}_k3',
                              f'*hard-shared-pearson-q{q:g}_k3')
            out[(seed, cond)] = {'run': run, 'result': res, 'hash': twoj_hashes[(seed, q)],
                                 'want': {'model': 'myModel', 'mode': 'hard', 'hard_axis': 'shared',
                                          'hard_stat': 'pearson', 'hard_q': q}}
        run, res = locate(suite, seed, '*hard-shared-oracle-q1_k3', '*hard-shared-oracle-q1_k3')
        out[(seed, 'oracle')] = {'run': run, 'result': res, 'hash': None,
                                 'want': {'model': 'myModel', 'mode': 'hard', 'hard_axis': 'shared',
                                          'hard_stat': 'oracle'}}
        run, res = locate(suite, seed, '*heterogeneous_full_k3', 'GRU_*heterogeneous_full_k3')
        out[(seed, 'gru')] = {'run': run, 'result': res, 'hash': None, 'want': {'model': 'GRU'}}
    return out


def manifest(config, t, seed):
    want = {**COMMON, **t['want'], 'seed': seed}
    bad = [f'{k}: {getattr(config, k, None)!r} != {v!r}' for k, v in want.items() if getattr(config, k, None) != v]
    if t['want']['model'] == 'myModel' and getattr(config, 'g11_bound', None) != BOUND:
        bad.append(f'g11_bound {getattr(config, "g11_bound", None)!r} != {BOUND!r}')
    epochs = json.load(open(t['result'])).get('train', {}).get('epochs_run')
    if epochs != COMMON['epoch']:
        bad.append(f'epochs_run {epochs!r} != {COMMON["epoch"]}')
    digest = sha256(Path(t['run']) / 'model_state' / 'best+model.pt')
    if t['hash'] is not None and digest != t['hash']:
        bad.append('checkpoint sha256 differs from the 2J record')
    return bad, digest


@torch.no_grad()
def one_pass(model, loader, config, spiking, oracle, count_ties):
    """Whole split, one forward per batch: error sums, safety, M_eff (D-Q), kept mass, ties."""
    model.eval()
    sums = {name: torch.zeros(3, dtype=torch.float64) for name in EVENT_KINDS}
    peak, finite, nonfinite = 0., True, 0
    meff_seq, kept_seq, seqs = [], [], 0
    ties = tie_n = mismatch = 0
    for x, y, truth, kind in loader:
        x, y = x.float(), y.float()
        output, aux = model(x, mode=config.mode if spiking else 'full', truth=truth,
                            kind=kind if oracle else None, return_aux=True)
        error = (output - y).double()                          # test.evaluate의 합산 순서 그대로
        for name, want in EVENT_KINDS.items():
            m = kind_mask(kind, want)
            sums[name][0] += (error * m).square().sum()
            sums[name][1] += (error.abs() * m).sum()
            sums[name][2] += m.sum()
        seqs += x.shape[0]
        if not spiking:
            if not torch.isfinite(output).all():
                finite, nonfinite = False, nonfinite + 1
            continue
        state = aux['state']
        if not (torch.isfinite(state).all() and torch.isfinite(output).all()):
            finite, nonfinite = False, nonfinite + 1           # 최대값에 섞지 않는다
        else:
            peak = max(peak, state.abs().max().item())
        B = x.shape[0]
        b = model.embedding.neuron.b
        tot, cnt = torch.zeros(B, dtype=torch.float64), torch.zeros(B, dtype=torch.float64)
        kept = torch.zeros(B, dtype=torch.float64)
        for n in range(1, len(aux['coeff'])):
            c = aux['coeff'][n].double()                       # [B, D, n] = m * b
            bh = b[1:n + 1].flip(0).double()
            kept += (c.sum(-1) / bh.sum()).mean(-1)
            answer = truth[:, n, :n]
            valid = (kind[:, n] > 0) & answer.any(-1)
            share = ((c * answer.unsqueeze(1)).sum(-1) / c.sum(-1).clamp_min(1e-12)).mean(-1)
            tot += torch.where(valid, share, torch.zeros_like(share))
            cnt += valid.double()
        kept_seq.extend((kept / (len(aux['coeff']) - 1)).tolist())
        meff_seq.extend((tot[cnt > 0] / cnt[cnt > 0]).tolist())
        if count_ties:                                         # 학습된 pearson 모델의 실제 top-k 경계 동점
            sel = model.embedding.neuron.selector
            current = model.embedding.current(layers.to_patches(x, config.patch_size))
            zero = torch.zeros_like(state[0])
            for n in range(1, current.shape[0]):
                xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
                hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                                    for j in range(n)], dim=-2)
                m, sim = sel.hard_mask(xi, hist)
                mismatch += int(not torch.equal(m, (aux['coeff'][n] > 0).to(m.dtype)))
                k = max(1, int(round(sel.hard_q * n)))
                if k < n:
                    srt = sim[:, 0, :].sort(dim=-1, descending=True).values
                    ties += int((srt[:, k - 1] == srt[:, k]).sum())
                tie_n += B
    res = {name: (t[0] / t[2]).item() for name, t in sums.items() if t[2] > 0}
    safe = finite and (not spiking or peak < BOUND)
    return {'recall_mse': res['recall'], 'copy_mse': res['copy'], 'recall_first_mse': res['recall_first'],
            'all_mse': res['all'], 'sequences': seqs, 'finite': finite, 'nonfinite_batches': nonfinite,
            'max_abs_state': peak if spiking else None, 'safe': safe,
            'm_eff': float(np.mean(meff_seq)) if meff_seq else None,
            'kept_mass_frac': float(np.mean(kept_seq)) if kept_seq else None,     # 시퀀스 가중 평균
            'ties': ties if count_ties else None, 'tie_decisions': tie_n if count_ties else None,
            'mask_mismatch': mismatch if count_ties else None}


def paired(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)
    return {'deltas': d.tolist(), 'mean': float(mean), 'sd': float(sd), 'ci95': [float(mean - half), float(mean + half)],
            'relative': float(mean / np.mean(b)), 'a_lower': int((d < 0).sum()), 'n': n}


def main():
    parser = argparse.ArgumentParser(description='prereg 2M: O7 for the hard design and the GRU comparison on confirm4')
    parser.add_argument('--twoj', required=True, help='2J selection suite (its record lists the confirmation checkpoints)')
    parser.add_argument('--suite', required=True, help='2M training suite (oracle, gru)')
    parser.add_argument('--out', required=True)
    cli = parser.parse_args()

    if torch.__version__ != TORCH:                             # D-BA 5: 선언한 동점 규칙은 이 구현이다
        raise SystemExit(f'[bench] torch {torch.__version__} != {TORCH}; the declared tie rule is that torch.topk')
    twoj = json.load(open(TASK / 'results' / cli.twoj / 'hard_selection_record.json'))['confirm']
    hashes = {(r['seed'], float(r['q'])): r['checkpoint_sha256'] for r in twoj['rows']}
    runs = targets(twoj['suite'], cli.suite, hashes)

    # ---- 1-2. 개방 전 전체 대조: 하나라도 어긋나면 confirm4를 열지 않는다 --------------------
    loaded, errors = {}, {}
    for (seed, cond), t in runs.items():
        model, config = load(t['run'])
        bad, digest = manifest(config, t, seed)
        loaded[(seed, cond)] = (model, config, digest)
        if bad:
            errors[f'{seed}/{cond}'] = bad
    if errors:
        for k, v in errors.items():
            print(f'[bench] manifest {k}: {v}')
        raise SystemExit(f'[bench] {len(errors)} run(s) fail the manifest; {SPLIT} was NOT opened')
    print(f'[bench] manifest OK for all {len(runs)} runs (8 seeds x pearson/q1/oracle/gru); torch {torch.__version__}')

    # ---- 3-4. 전역 등록부 잠금 -> 기록 생성 -> 분할 개방 ----------------------------------------
    out = TASK / 'results' / cli.out
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'hard_benchmark_record.json'
    open_once(SPLIT, 'prereg 2M: O7 for the hard design and the GRU comparison', record_path)
    head = {'prereg': '2M', 'split': SPLIT, 'seeds': list(SEEDS), 'twoj': cli.twoj, 'suite': cli.suite,
            'torch': torch.__version__, 'bound': BOUND, 'mde': MDE,
            'checkpoint_sha256': {f'{s}/{c}': v[2] for (s, c), v in loaded.items()},
            'source_sha256': {f: sha256(TASK / f) for f in SOURCES},
            'analysis_sha256': {f: sha256(HERE / f) for f in ANALYSIS}}
    with open(record_path, 'x') as handle:
        json.dump({**head, 'status': f'opening {SPLIT}'}, handle, indent=2)

    rows, mse, unsafe = [], {c: [] for c in ('q1', 'pearson', 'oracle', 'gru', 'tto')}, {}
    print(f"{'seed':>5} {'q1':>9} {'pearson':>9} {'oracle':>9} {'gru':>9} {'tto':>9}  "
          f"{'Meff p/q1/o':>17}  {'max|u| p/q1/o/tto':>23}  ties")
    for seed in SEEDS:
        line = {}
        for cond in ('q1', 'pearson', 'oracle', 'gru', 'tto'):
            if cond == 'tto':                                  # pearson checkpoint, oracle mask at evaluation only
                model, config = load(runs[(seed, 'pearson')]['run'])
                model.embedding.neuron.selector.hard_stat = 'oracle'
            else:
                model, config, _ = loaded[(seed, cond)]
            spiking = cond != 'gru'
            oracle = cond in ('oracle', 'tto')
            _, loader = data_provider(config, SPLIT)
            r = one_pass(model, loader, config, spiking, oracle, count_ties=cond == 'pearson')
            r.update(seed=seed, condition=cond)
            rows.append(r)
            mse[cond].append(r['recall_mse'])
            if not r['safe']:
                unsafe.setdefault(cond, []).append(seed)
            line[cond] = r
        p, q, o, t = line['pearson'], line['q1'], line['oracle'], line['tto']
        print(f"{seed:>5} {q['recall_mse']:>9.6f} {p['recall_mse']:>9.6f} {o['recall_mse']:>9.6f} "
              f"{line['gru']['recall_mse']:>9.6f} {t['recall_mse']:>9.6f}  "
              f"{p['m_eff']:.3f}/{q['m_eff']:.3f}/{o['m_eff']:.3f}  "
              f"{p['max_abs_state']:.1f}/{q['max_abs_state']:.1f}/{o['max_abs_state']:.1f}/{t['max_abs_state']:.1f}  "
              f"{p['ties']}/{p['tie_decisions']}")

    def blocked(*conds):
        return sorted({(c, s) for c in conds for s in unsafe.get(c, [])})

    E = {c: float(np.mean(v)) for c, v in mse.items()}
    meff = float(np.mean([r['m_eff'] for r in rows if r['condition'] == 'pearson']))
    g14 = paired(mse['q1'], mse['oracle'])
    G = (E['q1'] - E['pearson']) / (E['q1'] - E['oracle'])
    G_s = [(a - b) / (a - c) for a, b, c in zip(mse['q1'], mse['pearson'], mse['oracle'])]
    half = stats.t.ppf(.975, len(G_s) - 1) * np.std(G_s, ddof=1) / math.sqrt(len(G_s))
    o3 = paired(mse['pearson'], mse['q1'])
    delta_o = abs(E['tto'] - E['q1']) / E['q1']
    gru = paired(mse['pearson'], mse['gru'])
    gru_q1 = paired(mse['gru'], mse['q1'])

    verdict = {
        'O7-1': {'m_eff_pearson': meff, 'pass': meff >= .5, 'blocked': blocked('pearson')},
        'G14': {**g14, 'pass': g14['ci95'][0] > MDE, 'blocked': blocked('q1', 'oracle')},
        'O7-2': {'G': G, 'G_seed': G_s, 'G_seed_mean': float(np.mean(G_s)),
                 'G_seed_ci95': [float(np.mean(G_s) - half), float(np.mean(G_s) + half)],
                 'pass': G >= .5, 'blocked': blocked('q1', 'pearson', 'oracle')},
        'O7-3': {**o3, 'pass': o3['ci95'][1] < 0. and o3['relative'] <= -.2, 'blocked': blocked('pearson', 'q1')},
        'O7-4': {'delta_O': delta_o, 'within_5pct': delta_o <= .05, 'diagnostic_only': True},
        'GRU': {'pearson_minus_gru': gru, 'gru_minus_q1': gru_q1, 'blocked': blocked('pearson', 'gru', 'q1')},
    }
    print(f"\n[bench] mean confirm4 recall MSE: " + ', '.join(f'{c} {v:.6f}' for c, v in E.items()))
    print(f"[bench] O7-1  M_eff(pearson) = {meff:.4f}  (q1 {np.mean([r['m_eff'] for r in rows if r['condition'] == 'q1']):.4f}, "
          f"oracle {np.mean([r['m_eff'] for r in rows if r['condition'] == 'oracle']):.4f})  -> {'PASS' if meff >= .5 else 'FAIL'}")
    print(f"[bench] G14   q1 - oracle mean {g14['mean']:+.6f}, 95% CI [{g14['ci95'][0]:+.6f}, {g14['ci95'][1]:+.6f}] "
          f"vs MDE {MDE}  -> {'PASS' if verdict['G14']['pass'] else 'FAIL (O7-2 excluded)'}")
    print(f"[bench] O7-2  G = {G:.4f} (per-seed mean {np.mean(G_s):.4f}, 95% CI [{verdict['O7-2']['G_seed_ci95'][0]:.4f}, "
          f"{verdict['O7-2']['G_seed_ci95'][1]:.4f}])  -> {'PASS' if G >= .5 else 'FAIL'}"
          + ('' if verdict['G14']['pass'] else '  [excluded: G14 failed]'))
    print(f"[bench] O7-3  pearson - q1 mean {o3['mean']:+.6f}, 95% CI [{o3['ci95'][0]:+.6f}, {o3['ci95'][1]:+.6f}], "
          f"rel {100 * o3['relative']:+.2f}%  -> {'PASS' if verdict['O7-3']['pass'] else 'FAIL'}")
    print(f"[bench] O7-4  |E_tto - E_q1| / E_q1 = {100 * delta_o:.2f}%  (diagnostic; <= 5% triggers the table's diagnostic row)")
    print(f"[bench] GRU   pearson - gru mean {gru['mean']:+.6f}, 95% CI [{gru['ci95'][0]:+.6f}, {gru['ci95'][1]:+.6f}], "
          f"rel {100 * gru['relative']:+.2f}%, pearson lower in {gru['a_lower']}/8; gru - q1 {gru_q1['mean']:+.6f} "
          f"[{gru_q1['ci95'][0]:+.6f}, {gru_q1['ci95'][1]:+.6f}]  (not capacity-matched, A10-PARAM)")
    if unsafe:
        print(f'[bench] UNSAFE models (their verdicts are withheld): {unsafe}')
    ties = sum(r['ties'] for r in rows if r['ties'] is not None)
    dec = sum(r['tie_decisions'] for r in rows if r['tie_decisions'] is not None)
    mism = sum(r['mask_mismatch'] for r in rows if r['mask_mismatch'] is not None)
    print(f'[bench] pearson top-k boundary ties {ties}/{dec}; mask reconstruction mismatches {mism}')

    json.dump({**head, 'status': 'done', 'rows': rows, 'recall_mse': mse, 'means': E, 'unsafe': unsafe,
               'verdict': verdict, 'ties': [ties, dec], 'mask_mismatch': mism}, open(record_path, 'w'), indent=2)
    print(f'[bench] recorded to {record_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
