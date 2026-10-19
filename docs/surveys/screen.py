"""Stage 8 survey: full-text screening of tier A/B candidates (MODI_PROJECT_LOG 2026-10-19).

For every candidate of tier A or B (from <task>_candidates.csv) the script
  1. finds an open-access full text: arXiv (by id, else by exact-title lookup on the arXiv API), else the pdf_url carried by
     the collector (Semantic Scholar openAccessPdf / OpenAlex best_oa_location), else PubMed Central (by PMID);
  2. downloads the PDF to raw/pdf/<key>.pdf and converts it with pdftotext (raw/txt/<key>.txt);
  3. counts dataset mentions and cuts the sentences around them into raw/snippets/<key>.txt, together with regex hints for the
     protocol fields (bins, channels, seeds, folds, segment length, metrics ...);
  4. writes <task>_screen.csv: one row per candidate with fulltext_source, mention counts, 'uses_dataset' (>= 2 mentions of
     the dataset in the body, or a results-table mention), and the hints -- the worklist for manual extraction.
Nothing is committed from raw/. usage: python screen.py --task ecg|shd [--only-new]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
PDF, TXT, SNIP = (os.path.join(RAW, d) for d in ('pdf', 'txt', 'snippets'))
UA = 'nsmt-survey/0.1 (research; eeailabseoultech@gmail.com)'

MENTIONS = {
    'shd': {'SHD': r'\bSHD\b|Spiking Heidelberg Digits|Heidelberg Digits', 'SSC': r'\bSSC\b|Spiking Speech Commands?',
            'GSC': r'\bGSC\b|Google Speech Commands'},
    'ecg': {'CINC2017': r'CinC\s*2017|PhysioNet/CinC Challenge 2017|2017 PhysioNet|AF Classification Challenge|Computing in Cardiology Challenge 2017',
            'AFDB': r'MIT-BIH Atrial Fibrillation|\bAFDB\b|Atrial Fibrillation Database', 'PTBXL': r'PTB-XL', 'CINC2020': r'CinC\s*20(?:20|21)|PhysioNet/CinC Challenge 20(?:20|21)',
            'MITBIH_ARR': r'MIT-BIH Arrhythmia'},
}
HINTS = {
    'shd': {'bin_ms': r'(\d+(?:\.\d+)?)\s*ms\s*(?:time\s*)?(?:bin|step|window|resolution)|bin(?:ned|ning)?\s*(?:size|width|of)?\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*ms',
            'n_steps': r'(\d{2,4})\s*(?:time\s*)?steps', 'channels': r'(\d{2,3})\s*(?:input\s*)?channels', 'seeds': r'(\d+)\s*(?:random\s*)?seeds|(\d+)\s*(?:independent\s*)?runs',
            'val_split': r'validation\s*(?:set|split)[^.]{0,80}', 'augment': r'(?:data\s*)?augmentation[^.]{0,80}', 'shd_acc': r'SHD[^.]{0,60}?(\d{2}\.\d+)\s*%',
            'ssc_acc': r'SSC[^.]{0,60}?(\d{2}\.\d+)\s*%'},
    'ecg': {'fs': r'(\d{2,4})\s*Hz', 'folds': r'(\d+)[- ]fold', 'seg_len': r'(\d+(?:\.\d+)?)[- ]?(?:s|sec|seconds?)\s*(?:segments?|windows?|length|long|recordings?)',
            'seeds': r'(\d+)\s*(?:random\s*)?seeds|(\d+)\s*(?:independent\s*)?runs', 'f1': r'F1[^.]{0,40}?(0\.\d{2,4}|\d{2}\.\d+\s*%)',
            'split_unit': r'(?:patient|subject|record)(?:-|\s)?(?:wise|level|independent|based)\s*(?:split|cross|validation|evaluation)[^.]{0,60}',
            'official_test': r'hidden test|official test|validation set of the challenge|challenge test set', 'noisy': r'nois[ye][^.]{0,60}(?:exclud|includ|omit)[^.]{0,40}',
            'augment': r'(?:data\s*)?augmentation[^.]{0,80}', 'filter': r'(?:band-?pass|high-?pass|low-?pass|Butterworth|wavelet)[^.]{0,60}'},
}


def fetch(url, out, binary=True, retries=3):
    hdr = {'User-Agent': UA, 'Accept': 'application/pdf,*/*'}
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
            if binary and not body.startswith(b'%PDF'):
                m = re.search(rb'https?://[^"\' >]+\.pdf[^"\' >]*', body)      # landing page -> first pdf link
                if m and i == 0:
                    url = m.group(0).decode(); continue
                return None
            open(out, 'wb').write(body)
            return out
        except Exception as e:
            time.sleep(3)
    return None


def arxiv_by_title(title):
    q = 'ti:"' + re.sub(r'[^\w\s-]', ' ', title).strip() + '"'
    url = 'https://export.arxiv.org/api/query?search_query=' + urllib.parse.quote(q) + '&max_results=3'
    try:
        body = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=60).read().decode()
    except Exception:
        return None
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    norm = lambda t: re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()
    for e in ET.fromstring(body).findall('a:entry', ns):
        t = ' '.join(e.find('a:title', ns).text.split())
        if norm(t) == norm(title):
            return re.sub(r'v\d+$', '', e.find('a:id', ns).text.rsplit('/', 1)[-1])
    time.sleep(3)
    return None


def get_fulltext(row, key):
    pdf = os.path.join(PDF, key + '.pdf'); txt = os.path.join(TXT, key + '.txt')
    if os.path.exists(txt):
        return open(txt, encoding='utf-8', errors='replace').read(), json.load(open(txt + '.src')) if os.path.exists(txt + '.src') else 'cached'
    src = None
    arx = row.get('arxiv') or ''
    if not arx and row.get('title'):
        arx = arxiv_by_title(row['title']) or ''
    if arx:
        if fetch(f'https://arxiv.org/pdf/{arx}', pdf):
            src = f'arxiv:{arx}'
        time.sleep(3)
    if src is None and row.get('pdf_url'):
        if fetch(row['pdf_url'], pdf):
            src = 'oa:' + row['pdf_url'][:80]
    if src is None and row.get('pmid'):
        u = f'https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{row["pmid"]}/pdf/'
        if fetch(u, pdf):
            src = 'pmc'
    if src is None:
        return None, None
    r = subprocess.run(['pdftotext', '-layout', pdf, txt], capture_output=True)
    if r.returncode != 0 or not os.path.exists(txt):
        return None, None
    json.dump(src, open(txt + '.src', 'w'))
    return open(txt, encoding='utf-8', errors='replace').read(), src


def snippets(text, pat, width=220, maxn=12):
    out = []
    for m in re.finditer(pat, text, flags=re.I):
        a, b = max(0, m.start() - width), min(len(text), m.end() + width)
        out.append(' '.join(text[a:b].split()))
        if len(out) >= maxn:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['ecg', 'shd'], required=True)
    ap.add_argument('--tiers', default='A,B')
    ap.add_argument('--only-topical', action='store_true', help='restrict to candidates whose title/abstract matched the topical regex')
    a = ap.parse_args()
    for d in (PDF, TXT, SNIP):
        os.makedirs(d, exist_ok=True)
    cands = list(csv.DictReader(open(os.path.join(HERE, f'{a.task}_candidates.csv'), encoding='utf-8')))
    tiers = set(a.tiers.split(','))
    sel = [c for c in cands if c['tier'] in tiers and (not a.only_topical or c['topical'] == 'True')]
    print(f'{len(sel)} candidates to screen (tiers {tiers})')
    out_rows = []
    for i, c in enumerate(sel):
        key = re.sub(r'[^A-Za-z0-9]+', '_', (c.get('doi') or c.get('arxiv') or c.get('s2') or c['title'])[:80]).strip('_')
        text, src = get_fulltext(c, key)
        row = dict(key=key, title=c['title'], year=c['year'], venue=c['venue'], tier=c['tier'], doi=c.get('doi'), arxiv=c.get('arxiv'),
                   routes=c['routes'], fulltext_source=src or '', n_chars=len(text) if text else 0)
        total = 0
        if text:
            lines = []
            for name, pat in MENTIONS[a.task].items():
                n = len(re.findall(pat, text, flags=re.I)); row['m_' + name] = n; total += n if name in ('SHD', 'SSC', 'CINC2017', 'AFDB') else 0
                sn = snippets(text, pat)
                if sn:
                    lines.append(f'## {name} ({n} mentions)'); lines += ['- ' + s for s in sn]
            for name, pat in HINTS[a.task].items():
                sn = snippets(text, pat, width=120, maxn=6)
                if sn:
                    lines.append(f'## hint {name}'); lines += ['- ' + s for s in sn]
            open(os.path.join(SNIP, key + '.txt'), 'w', encoding='utf-8').write(f"# {c['title']} ({c['year']}, {c['venue']})\n" + '\n'.join(lines))
        row['n_dataset_mentions'] = total
        row['uses_dataset'] = 'yes' if total >= 2 else ('abstract-only' if not text else 'no')
        out_rows.append(row)
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(sel)} screened', flush=True)
    cols = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ('key', 'title', 'year', 'venue', 'tier'), k))
    with open(os.path.join(HERE, f'{a.task}_screen.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    n_ft = sum(1 for r in out_rows if r['fulltext_source'])
    n_use = sum(1 for r in out_rows if r['uses_dataset'] == 'yes')
    print(f'full text found {n_ft}/{len(out_rows)}; uses dataset (>=2 mentions) {n_use}; -> {a.task}_screen.csv, snippets in raw/snippets/')


if __name__ == '__main__':
    main()
