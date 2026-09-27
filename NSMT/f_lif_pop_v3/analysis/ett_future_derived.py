#!/usr/bin/env python3
"""Derived 2O records after audits 52/53 (A25). The original future records are NOT modified.

A25-INTERACTION-RELATIVE  I = S_on - S_off is compared with zero, so its 'relative' is mean/0
                          (+Infinity on ETTh1, -Infinity on ETTh2) and strict JSON rejects it.
                          Here the relative change of I is left undefined (null).
A25-UNSAFE-SECONDARY      every comparison now carries the unsafe (cond, seed) cells it uses and
                          a status; prereg 2O D-BN withholds a comparison that contains one.
A25-CHANNEL               D-BN asked for per-channel errors; the 2O evaluator only accumulated the
                          overall MSE/MAE and saved no predictions. They cannot be rebuilt from the
                          record, and the future period is not passed again to get them.
Also stored: the relative change of every non-I contrast from the stored means, for the corrected
wording in PROJECT_LOG (audit 53 A26-INTERPRETATION point 1).
Only stored values are used; no model, data split or registry is touched.
"""
import json
import hashlib
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'forecasting' / 'results'
ORIGINAL_SHA = {'ETTh1': '187f376b2227d066ec77dfeea32a403501f4bcf5eb531d01c26955adb09fe236',   # 감사 53이 기록한 값
                'ETTh2': '972f6ee0f03be00056615bd5de4712fb152310f1a7860135aef6a45e15430f8a'}
USES = {'D_q = q1_R - q1': ('q1_R', 'q1'), 'S_off = pearson - q1': ('pearson', 'q1'),
        'S_on = pearson_R - q1_R': ('pearson_R', 'q1_R'),
        'I = S_on - S_off': ('pearson_R', 'q1_R', 'pearson', 'q1'),
        'gru_R - gru': ('gru_R', 'gru'), 'linear_R - linear': ('linear_R', 'linear')}


def main():
    for data, want in ORIGINAL_SHA.items():
        src = RESULTS / f'etthard-revin-20260927-future-{data}' / 'ett_future_record.json'
        raw = src.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        assert got == want, f'{data}: original record changed ({got})'
        try:
            json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
            strict = 'accepted'
        except ValueError as e:
            strict = f'rejected ({e})'
        record = json.loads(raw)
        unsafe = [tuple(u) for u in record['unsafe']]
        mean = {c: sum(v) / len(v) for c, v in record['mse'].items()}
        secondary = {}
        for name, t in record['secondary'].items():
            conds = USES[name]
            block = [list(u) for u in unsafe if u[0] in conds]
            row = {k: t[k] for k in ('mean', 'sd', 'level', 'ci', 'a_lower', 'n', 'deltas')}
            if name.startswith('I '):
                row['relative'] = None                               # 0에 대한 상대 변화는 정의되지 않는다
                row['relative_note'] = 'undefined: I is a difference of differences compared with 0'
            else:
                row['relative'] = t['mean'] / mean[conds[1]]
            row.update(uses=list(conds), blocked=block,
                       status='withheld: unsafe cell' if block else 'reported (secondary, 95% CI, no significance claim)')
            secondary[name] = row
        derived = {'derived_from': {'path': str(src.relative_to(RESULTS.parent)), 'sha256': got,
                                    'strict_json_of_original': strict},
                   'audits': ['52 (2026-09-27 18:15)', '53 (2026-09-27 18:26)'],
                   'data': data, 'primary': {k: record['primary'][k] for k in
                                             ('mean', 'ci', 'relative', 'a_lower', 'n', 'level', 'verdict', 'blocked')},
                   'secondary': secondary, 'unsafe': [list(u) for u in unsafe],
                   'mean_mse': mean,
                   'channel_errors': None,
                   'channel_errors_note': ('not accumulated by ett_future.test (overall MSE/MAE only) and no '
                                           'predictions were saved; unrecoverable without passing the opened '
                                           'future period again, which is not done (audit 52 (c)2, 53 A26-NEXT)')}
        out = src.with_name('ett_future_record_derived.json')
        with open(out, 'x') as handle:
            json.dump(derived, handle, indent=2, allow_nan=False)            # 엄격 JSON
        json.loads(out.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        print(f"[derived] {data}: original strict JSON {strict}; derived written (strict JSON OK) -> {out.name}")
        for name, row in secondary.items():
            rel = 'undefined' if row['relative'] is None else f"{100 * row['relative']:+.4f}%"
            print(f"[derived] {data} {name:<24} mean {row['mean']:+.9f}  rel {rel:<10}  {row['status']}")


if __name__ == '__main__':
    main()
