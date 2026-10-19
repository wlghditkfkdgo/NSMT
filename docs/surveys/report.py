"""Stage 8 survey: build the protocol tables and frequency summaries from the extraction JSONL files.

Inputs:  raw/extract/<task>_extract_*.jsonl (agent/manual extraction, one JSON per paper), <task>_screen.csv, <task>_prisma.json
Outputs: <task>_protocols_2023_2026.csv (one row per included paper, raw extracted values + normalised buckets)
         <task>_protocols_2023_2026.md  (PRISMA counts, frequency tables with tier-A-only and A+B columns, paper list)
usage: python report.py --task shd|ecg
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_FULL = {'shd': 'shd_ssc', 'ecg': 'ecg_rhythm'}


def _ns(v):
    return (not v) or re.match(r'\s*(not stated|not applicable|n/?a|unknown|null|none stated|not printed|not reported)', v, flags=re.I) is not None


def bucket(value, rules, default='other'):
    """Field-aware normalisation of the free-text extraction values into the buckets used by the frequency tables."""
    v = (str(value) if value is not None else '').strip()
    name = rules[0][0] if rules else ''
    if _ns(v) and not v.lower().startswith('none ('):
        return 'not stated' if not v.lower().startswith('none') else 'none stated'
    if rules is BIN:
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ms|\()', v) or re.match(r'\s*(\d+(?:\.\d+)?)', v) or re.search(r'dt\s*=\s*(\d+(?:\.\d+)?)', v)
        if v.lower().startswith('none') or 'event-by-event' in v.lower():
            return 'event-driven (no bin)'
        if not m:
            return 'not stated'
        x = float(m.group(1).replace(',', ''))
        return {1: '1 ms', 2: '2 ms', 4: '4 ms', 5: '5 ms', 10: '10 ms', 20: '20 ms'}.get(int(x) if x == int(x) else -1, 'other')
    if rules is STEPS:
        m = re.search(r'\b(\d{2,4})\b', v)
        if not m:
            return 'not stated'
        x = int(m.group(1))
        return ('≤50' if x <= 50 else '100' if x == 100 else '140' if x == 140 else '150' if x == 150 else '200' if x == 200 else '250' if x == 250
                else '>250' if x > 250 else 'other')
    if rules is CHAN:
        m = re.search(r'\b(700|140|70|256|350|120)\b', v)
        return ('not stated' if not m else {'700': '700', '140': '140', '70': '70'}.get(m.group(1), 'other'))
    if rules is VAL:
        lv = v.lower()
        if re.search(r'test set (?:used )?as (?:the )?validation|test set used|selection on test|peak test|test accuracy at', lv):
            return 'test set used for selection'
        if re.search(r'leave-one-speaker|cross-validation|k-fold', lv):
            return 'cross-validation on train'
        if re.search(r'no (?:separate )?validation set', lv):
            return 'no validation set (procedure not stated)'
        if re.search(r'held|split|%|/\d|validation set used|deterministic split|official', lv):
            return 'held-out part of train / official val'
        return 'not stated'
    if rules is SEEDS:
        m = re.search(r'(\d+)', v)
        if not m:
            return 'not stated'
        x = int(m.group(1))
        return '1 / single' if x == 1 else '3' if x == 3 else '5' if x == 5 else '10' if x == 10 else f'other ({x})' if x < 100 else 'not stated'
    if rules is REPORT:
        lv = v.lower()
        if re.search(r'±|std|s\.d\.|standard dev|confidence|error bar', lv):
            return 'mean±std (table or figure)'
        if lv.startswith('single') or 'single' in lv:
            return 'single value'
        if 'top' in lv or 'best' in lv:
            return 'best of N'
        if 'mean' in lv or 'average' in lv or 'median' in lv:
            return 'mean only'
        return 'other'
    if rules is AUG:
        lv = v.lower()
        if lv.startswith('none') or re.search(r'no data augmentation|no augmentation', lv):
            return 'none stated'
        if re.search(r'shift|drop|mix|jitter|noise|augment|blend|subsampl|cutmix|specaug|warp|mask|dropout', lv):
            return 'used'
        return 'other'
    for nm, pat in rules:
        if re.search(pat, v, flags=re.I):
            return nm
    return default


SEEDS = [('1 / single', r'^1\b|single'), ('3', r'\b3\b'), ('5', r'\b5\b'), ('10', r'\b10\b'), ('other n', r'\d')]
# ---- SHD / SSC buckets
BIN = [('1 ms', r'^1(\.0)?\b'), ('2 ms', r'^2\b'), ('4 ms', r'^4\b'), ('5 ms', r'^5\b'), ('10 ms', r'^10\b'), ('20 ms', r'^20\b'), ('other', r'\d')]
STEPS = [('≤50', r'^\s*(?:[1-4]?\d|50)\b'), ('100', r'^100\b'), ('140', r'^140\b'), ('150', r'^150\b'), ('200', r'^200\b'), ('250', r'^250\b'), ('>250', r'^\s*(?:[3-9]\d\d|\d{4,})\b'), ('other', r'\d')]
CHAN = [('700', r'^700\b'), ('140', r'^140\b'), ('70', r'^70\b'), ('other', r'\d')]
VAL = [('held-out part of train', r'held|split|%|validation (?:set|split) (?:from|of|carved)|portion of (?:the )?train'),
       ('test set used for selection', r'test set|test accuracy|best test|selected on test'), ('official validation (SSC)', r'official|provided'),
       ('cross-validation', r'cross|fold')]
AUG = [('none stated', r'^none'), ('used', r'shift|drop|mix|jitter|noise|augment|blend|cutmix|specaug|warp|mask')]
FAM = [('spiking transformer', r'transformer|attention'), ('state-space', r'state.space|ssm|s4|mamba'), ('delays', r'delay'),
       ('resonate-and-fire', r'resonat'), ('recurrent SNN', r'recurrent|rsnn|lstm|gru'), ('feedforward SNN', r'feed.?forward|ffsnn|fsnn|mlp'),
       ('hardware/other', r'hardware|fpga|chip|other')]
NEURON = [('multi-compartment / dendritic / branches', r'compart|dendr|branch|two-compartment|multi-timescale neuron'),
          ('adaptive (ALIF)', r'alif|adaptive'), ('heterogeneous time constants', r'heterogen|learnable (?:time|tau)|trainable (?:time|tau)'),
          ('resonate-and-fire', r'resonat'), ('plain LIF', r'\blif\b|leaky integrate')]
REPORT = [('mean±std', r'±|std|standard dev|mean ?\+/-'), ('mean only', r'mean'), ('best of N', r'best'), ('single', r'single')]
# ---- ECG buckets
DATASETS = [('CinC 2017 + AFDB', r'(cinc|2017|physionet).*(afdb|mit-bih atrial|af database)|(afdb|mit-bih atrial|af database).*(cinc|2017)'),
            ('CinC 2017', r'cinc|2017|physionet challenge|alivecor'), ('AFDB', r'afdb|mit-bih atrial|af database')]
SPLIT = [('official hidden test', r'hidden|official test|challenge test|phase'), ('k-fold CV', r'fold|cross'), ('fixed hold-out', r'hold|split|%|train/test|8:2|7:3|80/20|70/30')]
UNIT = [('patient/record-wise', r'patient|subject|record|inter'), ('random (sample-wise)', r'random|sample|intra|stratif')]
FS = [('300 Hz', r'300'), ('250 Hz', r'250'), ('125/128 Hz', r'12[58]'), ('100 Hz', r'\b100\b'), ('500 Hz', r'500'), ('other', r'\d')]
LEN = [('30 s', r'\b30\b'), ('10 s', r'\b10\b'), ('60 s', r'\b60\b'), ('9 s', r'\b9\b'), ('variable / full', r'variab|full|whole|entire'), ('5 s', r'\b5\b'), ('other', r'\d')]
METRIC = [('F1 (mean of classes)', r'f1'), ('accuracy', r'acc'), ('sensitivity/specificity', r'sens|spec'), ('AUROC/AUC', r'au[cr]')]
LOSS = [('class-weighted / focal', r'weighted|focal|cost-sensitive'),
        ('resampling / balanced subset', r'over.?sampl|under.?sampl|smote|randomly chosen|random selection|balanced to|adding .* samples|equal numbers'),
        ('no imbalance handling stated (plain loss)', r'no (?:class-)?imbalance handling|no imbalance|no other imbalance|imbalance acknowledged|cross.?entropy|hinge|square loss|binary classification loss'),
        ('not applicable (no training)', r'not applicable')]


def load_extract(task):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'raw', 'extract', f'{task}_extract_*.jsonl'))):
        for line in open(f, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'bad line in {f}: {e}: {line[:80]}')
    return rows


def freq_table(rows, field, rules, title, default='other'):
    a = [r for r in rows if r.get('tier') == 'A']
    ca = collections.Counter(bucket(r.get(field), rules, default) for r in a)
    cab = collections.Counter(bucket(r.get(field), rules, default) for r in rows)
    order = [k for k, _ in rules] + [default, 'not stated']
    keys = sorted(set(ca) | set(cab), key=lambda k: (order.index(k) if k in order else len(order) - 2, k))
    keys = [k for k in keys if k != 'not stated'] + (['not stated'] if ('not stated' in ca or 'not stated' in cab) else [])
    L = [f'**{title}** (A군 n={len(a)} / A+B군 n={len(rows)})', '', '| 값 | A군 | A+B군 |', '|---|---|---|']
    for k in keys:
        L.append(f'| {k} | {ca[k]} ({100 * ca[k] / max(1, len(a)):.0f}%) | {cab[k]} ({100 * cab[k] / max(1, len(rows)):.0f}%) |')
    return L + ['']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['shd', 'ecg'], required=True)
    a = ap.parse_args()
    ex = load_extract(a.task)
    screen = {r['key']: r for r in csv.DictReader(open(os.path.join(HERE, f'{a.task}_screen.csv'), encoding='utf-8'))}
    ex = [r for r in ex if r.get('key') in screen]                      # drop papers no longer in the A/B screen (venue re-tiered)
    ov_path = os.path.join(HERE, f'{a.task}_overrides.json')             # manual exclusions after reading (reason recorded)
    overrides = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
    for r in ex:
        if r.get('key') in overrides:
            r['manual_exclude'] = overrides[r['key']]
    for r in ex:
        s = screen.get(r.get('key'), {})
        r['tier'] = s.get('tier', r.get('tier', '?')); r['doi'] = s.get('doi'); r['arxiv'] = s.get('arxiv'); r['fulltext_source'] = s.get('fulltext_source')
    if a.task == 'shd':
        inc = [r for r in ex if (str(r.get('uses_shd')).lower() == 'true' or str(r.get('uses_ssc')).lower() == 'true') and not r.get('manual_exclude')]
    else:
        inc = [r for r in ex if (str(r.get('uses_cinc2017')).lower() == 'true' or str(r.get('uses_afdb')).lower() == 'true') and not r.get('manual_exclude')]
    prisma = json.load(open(os.path.join(HERE, f'{a.task}_prisma.json')))
    sc = list(screen.values())
    n_ab = {t: sum(1 for r in sc if r['tier'] == t) for t in ('A', 'B')}
    n_ft = {t: sum(1 for r in sc if r['tier'] == t and r['fulltext_source']) for t in ('A', 'B')}
    n_use = {t: sum(1 for r in sc if r['tier'] == t and r['uses_dataset'] == 'yes') for t in ('A', 'B')}
    n_inc = {t: sum(1 for r in inc if r['tier'] == t) for t in ('A', 'B')}

    L = [f'# {"SHD / SSC" if a.task == "shd" else "ECG 리듬 분류 (CinC 2017 · MIT-BIH AFDB)"} — 2023–2026 선행 연구 프로토콜 조사', '',
         '생성: `docs/surveys/report.py` (수집 `collect.py`, 선별 `screen.py`, 추출 `raw/extract/*.jsonl`). 수치는 추출 표의 값을 그대로 집계한 것이며, '
         '"not stated"는 본문에 명시되지 않은 항목이다.', '',
         '## 1. 모집단과 선별 (PRISMA 식 건수)', '',
         f"- 수집(2023–2026, 경로별 원건수): {', '.join(f'{k} {v}' for k, v in prisma['per_route_raw'].items())} → 중복 제거 후 {prisma['unique_in_years']}편",
         f"- 장소(venue) 분류 · 전체: A군 {prisma['by_tier']['A']}, B군 {prisma['by_tier']['B']}, 제외군(MDPI·Frontiers·Hindawi·CinC 프로시딩·워크숍) {prisma['by_tier']['EXCL']}, "
         f"프리프린트 전용 {prisma['by_tier']['PREPRINT']}, 출처 불명/기타 {prisma['by_tier']['UNKNOWN']}",
         f"- 제목·초록 주제 일치 · A군 {prisma['topical_by_tier']['A']}, B군 {prisma['topical_by_tier']['B']}",
         f"- 공개 전문 확보 · A군 {n_ft['A']}/{n_ab['A']}, B군 {n_ft['B']}/{n_ab['B']} (나머지는 초록만)",
         f"- 전문에서 대상 데이터셋을 실제로 사용(언급 ≥ 2) · A군 {n_use['A']}, B군 {n_use['B']}",
         f"- 추출 후 포함(자기 모델의 결과를 보고) · **A군 {n_inc['A']}, B군 {n_inc['B']}**", '',
         '## 2. 항목별 빈도', '']
    if a.task == 'shd':
        L += freq_table(inc, 'bin_ms', BIN, '시간 구간 폭')
        L += freq_table(inc, 'n_steps', STEPS, '스텝 수')
        L += freq_table(inc, 'channels_used', CHAN, '입력 채널 수')
        L += freq_table(inc, 'validation_split', VAL, '모델 선택(검증) 방식')
        L += freq_table(inc, 'augmentation', AUG, '증강', default='other')
        L += freq_table(inc, 'model_family', FAM, '모델 계열')
        L += freq_table(inc, 'neuron_model', NEURON, '뉴런 모형')
        L += freq_table(inc, 'seeds_n', SEEDS, 'seed/run 수')
        L += freq_table(inc, 'report_form', REPORT, '보고 형식')
        cols = ['key', 'title', 'year', 'venue', 'tier', 'doi', 'arxiv', 'uses_shd', 'uses_ssc', 'uses_gsc', 'paper_role', 'bin_ms', 'n_steps', 'channels_used',
                'length_handling', 'validation_split', 'augmentation', 'model_family', 'neuron_model', 'n_layers', 'n_neurons', 'loss_regularisation',
                'epochs', 'optimizer', 'lr', 'seeds_n', 'report_form', 'metric', 'reported_shd_acc', 'reported_ssc_acc', 'params_reported',
                'energy_reported', 'code_public', 'extraction_confidence', 'fulltext_source']
    else:
        L += freq_table(inc, 'datasets', DATASETS, '데이터셋')
        L += freq_table(inc, 'split', SPLIT, '분할 방식')
        L += freq_table(inc, 'cv_k', [('5', r'\b5\b'), ('10', r'\b10\b'), ('other k', r'\d')], 'k-fold의 k')
        L += freq_table(inc, 'split_unit', UNIT, '분할 단위')
        L += freq_table(inc, 'fs_hz', FS, '표본화율')
        L += freq_table(inc, 'input_len_s', LEN, '입력 길이')
        L += freq_table(inc, 'loss_imbalance', LOSS, '불균형 처리')
        L += freq_table(inc, 'metric_primary', METRIC, '주 지표')
        L += freq_table(inc, 'metric_noise_included', [('included', r'^(true|yes|includ)'), ('excluded', r'^(false|no|exclud)')], 'F1 평균에 잡음 클래스 포함 여부')
        L += freq_table(inc, 'augmentation', AUG, '증강')
        L += freq_table(inc, 'seeds_n', SEEDS, 'seed/run 수')
        L += freq_table(inc, 'report_form', REPORT, '보고 형식')
        cols = ['key', 'title', 'year', 'venue', 'tier', 'doi', 'arxiv', 'uses_cinc2017', 'uses_afdb', 'other_datasets', 'paper_role', 'datasets', 'label_version',
                'split', 'cv_k', 'split_unit', 'fs_hz', 'filter', 'normalisation', 'input_len_s', 'segmenting', 'padding', 'augmentation', 'loss_imbalance',
                'model_family', 'epochs', 'optimizer', 'lr', 'metric_primary', 'metric_noise_included', 'metrics_other', 'inference_aggregation',
                'reported_f1', 'reported_acc', 'reported_other', 'seeds_n', 'report_form', 'energy_params_reported', 'code_public', 'extraction_confidence', 'fulltext_source']
    L += ['## 3. 포함 논문', '', '| # | 연도 | 장소 | 군 | 제목 | DOI/arXiv | 추출 신뢰도 |', '|---|---|---|---|---|---|---|']
    for i, r in enumerate(sorted(inc, key=lambda r: (r['tier'], -int(r.get('year') or 0), str(r.get('title')))), 1):
        L.append(f"| {i} | {r.get('year')} | {str(r.get('venue'))[:45]} | {r['tier']} | {str(r.get('title'))[:90]} | {r.get('doi') or r.get('arxiv') or ''} | {r.get('extraction_confidence')} |")
    exc = [r for r in ex if r not in inc]
    L += ['', f'## 4. 추출했으나 제외 (대상 데이터셋에서 자기 모델 결과를 보고하지 않음): {len(exc)}편', '']
    for r in sorted(exc, key=lambda r: str(r.get('title'))):
        L.append(f"- {r.get('year')} | {str(r.get('venue'))[:40]} | {str(r.get('title'))[:90]}" + (f" — 수동 제외: {r['manual_exclude']}" if r.get('manual_exclude') else ''))
    # appendix A: abstract-only candidates that name the dataset (paywalled; not extracted)
    ao_path = os.path.join(HERE, 'raw', 'extract', f'{a.task}_abstract_only_hits.csv')
    if os.path.exists(ao_path):
        ao = list(csv.DictReader(open(ao_path, encoding='utf-8')))
        L += ['', f'## 부록 A. 초록에서 데이터셋을 언급하지만 공개 전문이 없어 추출하지 못한 논문: {len(ao)}편', '',
              '| 연도 | 장소 | 군 | 제목 | DOI |', '|---|---|---|---|---|']
        for r in sorted(ao, key=lambda r: (r['tier'], -int(r['year'] or 0))):
            L.append(f"| {r['year']} | {r['venue'][:40]} | {r['tier']} | {r['title'][:90]} | {r.get('doi') or ''} |")
    # appendix B: excluded-venue tallies among topical candidates
    cands = list(csv.DictReader(open(os.path.join(HERE, f'{a.task}_candidates.csv'), encoding='utf-8')))
    exv = collections.Counter(r['venue'] for r in cands if r['tier'] == 'EXCL' and r['topical'] == 'True')
    unv = collections.Counter(r['venue'] for r in cands if r['tier'] == 'UNKNOWN' and r['topical'] == 'True')
    L += ['', f'## 부록 B. 제외군(MDPI·Frontiers·Hindawi·CinC·워크숍) 주제 일치 후보: {sum(exv.values())}편 (장소별 상위)', '']
    L += [f'- {v or "(없음)"}: {n}' for v, n in exv.most_common(15)]
    L += ['', f'## 부록 C. A/B군이 아닌 그 밖의 장소(출처 불명 포함) 주제 일치 후보: {sum(unv.values())}편 (장소별 상위 25)', '']
    L += [f'- {v or "(없음)"}: {n}' for v, n in unv.most_common(25)]
    # adopted-protocol section written by hand, appended when present
    hand = os.path.join(HERE, f'{TASK_FULL[a.task]}_adopted.md')
    if os.path.exists(hand):
        L += ['', open(hand, encoding='utf-8').read()]
    open(os.path.join(HERE, f'{TASK_FULL[a.task]}_protocols_2023_2026.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    with open(os.path.join(HERE, f'{TASK_FULL[a.task]}_protocols_2023_2026.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in inc:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    print('\n'.join(L[:40]))
    print(f'... -> {TASK_FULL[a.task]}_protocols_2023_2026.{{md,csv}} ({len(inc)} included, {len(exc)} extracted-but-excluded)')


if __name__ == '__main__':
    main()
