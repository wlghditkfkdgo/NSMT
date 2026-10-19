"""Stage 8 survey (MODI_PROJECT_LOG 2026-10-19): candidate collection for two tasks, 2023-2026.

Routes (all read-only HTTP, raw responses cached under docs/surveys/raw/, git-ignored):
  S2CIT   Semantic Scholar citations of the dataset papers (main route, near-exhaustive)
  S2SRCH  Semantic Scholar keyword search (rate-limited; best effort)
  OACIT   OpenAlex works citing the dataset papers (second citation route; independent index)
  OASRCH  OpenAlex full-text search (title/abstract), venues from primary_location
  DBLP    dblp publication search -- BLOCKED by dblp's bot challenge from this server (kept for completeness; fails gracefully)
  ARXIV   arXiv API keyword search
  XREF    Crossref works query (publisher metadata, container titles)
  PUBMED  PubMed esearch + esummary (medical journals; ECG task)
Merged on DOI / arXiv id / normalised title; venue classified into tier A (top), tier B (user-approved border set),
EXCL (MDPI, Frontiers, Hindawi, CinC proceedings, workshops), PREPRINT (arXiv only), UNKNOWN.
Outputs docs/surveys/<task>_candidates.csv and <task>_prisma.json.
usage: python collect.py --task ecg|shd [--routes S2CIT,DBLP,ARXIV,XREF,PUBMED,S2SRCH] [--years 2023-2026]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
UA = 'nsmt-survey/0.1 (research; eeailabseoultech@gmail.com)'

TASKS = {
    'ecg': dict(
        seeds={'1896bab29091826644543ffc038d1633282eabcc': 'Clifford 2017 CinC challenge paper'},
        oa_seeds={'W2794550444': 'Clifford 2017 CinC challenge paper'},
        oa_queries=['"atrial fibrillation" "CinC 2017"', '"atrial fibrillation" "PhysioNet/CinC Challenge 2017"', '"atrial fibrillation" single-lead ECG deep learning',
                    '"MIT-BIH Atrial Fibrillation" deep learning', '"spiking neural network" ECG "atrial fibrillation"', '"spiking neural network" ECG arrhythmia'],
        s2_queries=['atrial fibrillation single-lead ECG CinC 2017', 'MIT-BIH atrial fibrillation database deep learning',
                    'spiking neural network ECG atrial fibrillation'],
        dblp_queries=['atrial fibrillation ECG', 'atrial fibrillation detection', 'AF detection ECG', 'ECG rhythm classification',
                      'spiking ECG arrhythmia', 'CinC 2017 ECG'],
        arxiv_queries=['abs:"atrial fibrillation" AND abs:ECG', 'abs:"CinC 2017"', 'abs:"PhysioNet/CinC Challenge 2017"',
                       'abs:"MIT-BIH Atrial Fibrillation"', 'abs:"spiking neural network" AND abs:ECG'],
        xref_queries=['atrial fibrillation detection single-lead ECG deep learning', 'PhysioNet CinC Challenge 2017 atrial fibrillation',
                      'MIT-BIH atrial fibrillation database deep learning', 'spiking neural network ECG arrhythmia'],
        pubmed_queries=['(atrial fibrillation[Title/Abstract]) AND (ECG OR electrocardiogram) AND (deep learning OR neural network) AND (single-lead OR "CinC 2017" OR PhysioNet OR "MIT-BIH")',
                        '("spiking neural network"[Title/Abstract]) AND (ECG OR electrocardiogram)'],
        keep_regex=r'atrial fibrillation|\bAF\b|arrhythm|rhythm|ECG|electrocardio|CinC|PhysioNet|single.lead|AFDB',
    ),
    'shd': dict(
        seeds={'90cecfb1c12530f1a4ecc83ab48a47e35ee60fab': 'Cramer 2020/2022 Heidelberg spiking data sets (TNNLS)'},
        oa_seeds={'W2980513133': 'Cramer 2020/2022 Heidelberg spiking data sets'},
        oa_queries=['"Spiking Heidelberg Digits"', '"Spiking Speech Commands"', '"Heidelberg Digits" spiking', 'SHD SSC "spiking neural network"'],
        s2_queries=['Spiking Heidelberg Digits', 'Spiking Speech Commands spiking neural network'],
        dblp_queries=['Spiking Heidelberg Digits', 'Spiking Speech Commands', 'spiking neural network speech', 'spiking recurrent neural network',
                      'spiking neural network delays', 'spiking neural network temporal'],
        arxiv_queries=['abs:"Spiking Heidelberg Digits"', 'abs:"Spiking Speech Commands"', 'abs:"Heidelberg Digits"',
                       'abs:SHD AND abs:"spiking neural"', 'abs:SSC AND abs:"spiking neural"'],
        xref_queries=['Spiking Heidelberg Digits', 'Spiking Speech Commands spiking neural network'],
        pubmed_queries=['"Spiking Heidelberg" OR "Spiking Speech Commands"'],
        keep_regex=r'Heidelberg|Spiking Speech|\bSHD\b|\bSSC\b|spiking|neuromorphic|SNN',
    ),
}

# ---- venue tiers (plan 1.3) --------------------------------------------------------------------------------------------
TIER_A = [
    r'neurips|neural information processing systems', r'\biclr\b|international conference on learning representations',
    r'\bicml\b|international conference on machine learning', r'\baaai\b', r'\bijcai\b', r'\bcvpr\b', r'\biccv\b', r'\beccv\b',
    r'pattern analysis and machine intelligence|tpami', r'neural networks and learning systems|tnnls', r'journal of machine learning research|\bjmlr\b',
    r'^neural networks$|neural networks \(|elsevier.*neural networks|^neural netw', r'neural computation', r'^patterns',
    r'^nature$', r'^science$', r'nature communications', r'nature machine intelligence', r'nature electronics', r'nature neuroscience',
    r'nature medicine', r'nature biomedical engineering', r'science advances', r'proceedings of the national academy|\bpnas\b',
    r'neuromorphic computing and engineering',
    r'transactions on biomedical engineering|\btbme\b', r'journal of biomedical and health informatics|\bjbhi\b',
    r'neural systems and rehabilitation|\btnsre\b', r'medical image analysis', r'npj digital medicine', r'lancet digital health',
    r'^circulation', r'journal of the american college of cardiology|\bjacc\b', r'european heart journal', r'artificial intelligence in medicine',
    r'journal of solid-state circuits|\bjssc\b', r'circuits and systems i\b|tcas-i|tcas i\b', r'\bisscc\b', r'design automation conference|\bdac\b',
    r'biomedical circuits and systems|tbiocas',
    # added during screening (flagged in the report for user review): field-top cardiology EP journals, TMLR, top EDA venues
    r'^heart rhythm$|^heart rhythm\b(?! o2)', r'europace', r'journal of the american heart association|\bjaha\b',
    r'transactions on machine learning research|trans\. mach\. learn\. res',
    r'computer-aided design of integrated circuits|\btcad\b', r'design, automation (?:&amp;|&|and) test in europe|\bdate\b',
]
TIER_B = [r'scientific reports|^sci\. rep', r'computers in biology and medicine|comput\. biol\. med', r'biomedical signal processing and control|biomed\. signal process',
          r'physiological measurement|physiol\. meas', r'expert systems with applications|expert syst\. appl', r'ieee access',
          # added during screening (flagged): top signal-processing conferences and TCAS-II, where SHD/SSC and ECG-SNN papers appear
          r'\bicassp\b|acoustics, speech,? and signal processing', r'interspeech', r'circuits and systems ii|tcas-ii|tcas ii\b']
EXCL = [r'\bmdpi\b', r'^sensors', r'^electronics', r'applied sciences', r'bioengineering', r'^mathematics$', r'^entropy$', r'^diagnostics',
        r'^information$', r'^algorithms$', r'frontiers', r'hindawi', r'computing in cardiology|\bcinc\b', r'workshop']


def classify_venue(v):
    s = (v or '').strip().lower()
    if not s:
        return 'UNKNOWN'
    if s in ('arxiv', 'arxiv.org', 'corr') or s.startswith('arxiv') or re.search(r'medrxiv|biorxiv|research square|ssrn|zenodo|\bhal\b|preprints?\.org|techrxiv|authorea', s):
        return 'PREPRINT'
    for pat in EXCL:
        if re.search(pat, s):
            return 'EXCL'
    for pat in TIER_A:
        if re.search(pat, s):
            return 'A'
    for pat in TIER_B:
        if re.search(pat, s):
            return 'B'
    return 'UNKNOWN'


# ---- HTTP with cache + backoff -----------------------------------------------------------------------------------------
def fetch(url, kind, data=None, headers=None, retries=6, sleep=2.0):
    os.makedirs(RAW, exist_ok=True)
    key = hashlib.sha1((url + (data or '')).encode()).hexdigest()[:16]
    path = os.path.join(RAW, f'{kind}_{key}.txt')
    if os.path.exists(path):
        return open(path, encoding='utf-8').read()
    hdr = {'User-Agent': UA}
    hdr.update(headers or {})
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=data.encode() if data else None, headers=hdr)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode('utf-8', 'replace')
                if r.status == 202:                                   # dblp "come back later"
                    raise urllib.error.HTTPError(url, 202, 'accepted', hdr, None)
            open(path, 'w', encoding='utf-8').write(body)
            time.sleep(sleep)
            return body
        except urllib.error.HTTPError as e:
            wait = min(90, 8 * (2 ** i)) if e.code in (429, 202, 503) else 3
            print(f'  [{kind}] HTTP {e.code}, retry {i + 1}/{retries} in {wait}s', file=sys.stderr)
            time.sleep(wait)
        except Exception as e:                                        # network hiccup
            print(f'  [{kind}] {type(e).__name__}: {e}; retry {i + 1}/{retries}', file=sys.stderr)
            time.sleep(5)
    print(f'  [{kind}] GAVE UP: {url[:100]}', file=sys.stderr)
    return None


def norm_title(t):
    return re.sub(r'[^a-z0-9]+', ' ', (t or '').lower()).strip()


class Pool:
    def __init__(self):
        self.items = {}                                               # key -> record
        self.counts = {}

    def add(self, rec, route):
        self.counts[route] = self.counts.get(route, 0) + 1
        doi = (rec.get('doi') or '').lower().strip()
        arx = (rec.get('arxiv') or '').strip()
        key = ('doi:' + doi) if doi else (('arxiv:' + arx) if arx else ('title:' + norm_title(rec.get('title'))))
        # merge preprints with their published version by title
        tkey = 'title:' + norm_title(rec.get('title'))
        if key not in self.items and tkey in self.items:
            key = tkey
        cur = self.items.get(key)
        if cur is None:
            rec = dict(rec); rec['routes'] = {route}
            self.items[key] = rec
            if tkey != key:
                self.items[tkey] = rec                                # alias
        else:
            cur['routes'].add(route)
            for k, v in rec.items():
                if v and not cur.get(k):
                    cur[k] = v
            if rec.get('venue') and classify_venue(rec['venue']) in ('A', 'B') and classify_venue(cur.get('venue')) not in ('A', 'B'):
                cur['venue'] = rec['venue']                            # prefer a published venue over 'arXiv'

    def records(self):
        seen, out = set(), []
        for r in self.items.values():
            if id(r) in seen:
                continue
            seen.add(id(r)); out.append(r)
        return out


# ---- routes -----------------------------------------------------------------------------------------------------------
S2F = 'title,year,venue,externalIds,citationCount,publicationVenue,journal,abstract,openAccessPdf'


def s2_rec(p):
    ext = p.get('externalIds') or {}
    venue = p.get('venue') or ''
    pv = p.get('publicationVenue') or {}
    if pv and pv.get('name'):
        venue = pv['name']
    j = p.get('journal') or {}
    if not venue and j.get('name'):
        venue = j['name']
    return dict(title=p.get('title'), year=p.get('year'), venue=venue, doi=ext.get('DOI'), arxiv=ext.get('ArXiv'),
                pmid=ext.get('PubMed'), s2=p.get('paperId'), cites=p.get('citationCount'), abstract=p.get('abstract'),
                pdf_url=((p.get('openAccessPdf') or {}).get('url')))


def route_s2cit(task, pool, years):
    for pid, name in TASKS[task]['seeds'].items():
        off, n = 0, 0
        while True:
            url = f'https://api.semanticscholar.org/graph/v1/paper/{pid}/citations?fields={S2F}&limit=500&offset={off}'
            body = fetch(url, 's2cit', sleep=4.0)
            if body is None:
                break
            d = json.loads(body)
            for c in d.get('data', []):
                p = c.get('citingPaper') or {}
                if p.get('year') and years[0] <= p['year'] <= years[1]:
                    pool.add(s2_rec(p), 'S2CIT'); n += 1
            if d.get('next') is None:
                break
            off = d['next']
        print(f'S2CIT {name}: {n} citing papers in {years}')


def route_s2srch(task, pool, years):
    for q in TASKS[task]['s2_queries']:
        url = ('https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=' + urllib.parse.quote(q)
               + f'&year={years[0]}-{years[1]}&fields={S2F}')
        body = fetch(url, 's2srch', sleep=4.0)
        if body is None:
            continue
        d = json.loads(body); n = 0
        for p in d.get('data', []):
            pool.add(s2_rec(p), 'S2SRCH'); n += 1
        print(f'S2SRCH "{q}": {n} (total reported {d.get("total")})')


def route_dblp(task, pool, years):
    for q in TASKS[task]['dblp_queries']:
        for y in range(years[0], years[1] + 1):
            url = f'https://dblp.org/search/publ/api?q={urllib.parse.quote(q + " year:" + str(y))}&format=json&h=1000'
            body = fetch(url, 'dblp', sleep=1.5)
            if body is None:
                continue
            try:
                hits = json.loads(body)['result']['hits']
            except Exception:
                continue
            n = 0
            for h in hits.get('hit', []) or []:
                i = h['info']
                pool.add(dict(title=i.get('title', '').rstrip('.'), year=int(i.get('year', 0) or 0), venue=i.get('venue') if isinstance(i.get('venue'), str) else ' / '.join(i.get('venue') or []),
                              doi=i.get('doi'), dblp=i.get('key'), dblp_type=i.get('type')), 'DBLP'); n += 1
            print(f'DBLP "{q}" {y}: {n}')


def route_arxiv(task, pool, years):
    for q in TASKS[task]['arxiv_queries']:
        start, n = 0, 0
        while True:
            url = ('https://export.arxiv.org/api/query?search_query=' + urllib.parse.quote(q)
                   + f'&start={start}&max_results=200&sortBy=submittedDate&sortOrder=descending')
            body = fetch(url, 'arxiv', sleep=3.5)
            if body is None:
                break
            root = ET.fromstring(body)
            ns = {'a': 'http://www.w3.org/2005/Atom', 'ar': 'http://arxiv.org/schemas/atom'}
            entries = root.findall('a:entry', ns)
            if not entries:
                break
            stop = False
            for e in entries:
                y = int(e.find('a:published', ns).text[:4])
                if y < years[0]:
                    stop = True; continue
                if y > years[1]:
                    continue
                aid = e.find('a:id', ns).text.rsplit('/', 1)[-1]
                aid = re.sub(r'v\d+$', '', aid)
                doi_el = e.find('ar:doi', ns); jr = e.find('ar:journal_ref', ns)
                pool.add(dict(title=' '.join(e.find('a:title', ns).text.split()), year=y, venue=(jr.text if jr is not None else 'arXiv'),
                              doi=(doi_el.text if doi_el is not None else None), arxiv=aid,
                              abstract=' '.join(e.find('a:summary', ns).text.split())), 'ARXIV'); n += 1
            if stop or len(entries) < 200:
                break
            start += 200
        print(f'ARXIV "{q}": {n}')


def route_xref(task, pool, years):
    for q in TASKS[task]['xref_queries']:
        cursor, n = '*', 0
        while True:
            url = ('https://api.crossref.org/works?query=' + urllib.parse.quote(q)
                   + f'&filter=from-pub-date:{years[0]}-01-01,until-pub-date:{years[1]}-12-31,type:journal-article,type:proceedings-article'
                   + f'&rows=200&cursor={urllib.parse.quote(cursor)}&select=DOI,title,container-title,issued,publisher,abstract,type&mailto=eeailabseoultech@gmail.com')
            body = fetch(url, 'xref', sleep=1.5)
            if body is None:
                break
            d = json.loads(body)['message']
            items = d.get('items', [])
            for it in items:
                y = (it.get('issued', {}).get('date-parts') or [[None]])[0][0]
                pool.add(dict(title=' '.join((it.get('title') or [''])[0].split()), year=y, venue=(it.get('container-title') or [''])[0],
                              doi=it.get('DOI'), publisher=it.get('publisher'), abstract=re.sub(r'<[^>]+>', ' ', it.get('abstract') or '')), 'XREF'); n += 1
            if len(items) < 200 or n >= 1000:                         # relevance-ranked; 1000 per query is plenty
                break
            cursor = d.get('next-cursor', '*')
        print(f'XREF "{q}": {n}')


def route_pubmed(task, pool, years):
    for q in TASKS[task]['pubmed_queries']:
        term = f'({q}) AND ({years[0]}:{years[1]}[dp])'
        body = fetch('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax=5000&retmode=json&term=' + urllib.parse.quote(term), 'pubmed', sleep=0.5)
        if body is None:
            continue
        ids = json.loads(body)['esearchresult'].get('idlist', [])
        n = 0
        for i in range(0, len(ids), 200):
            chunk = ids[i:i + 200]
            b2 = fetch('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=' + ','.join(chunk), 'pubmed', sleep=0.5)
            if b2 is None:
                continue
            res = json.loads(b2)['result']
            for pid in chunk:
                r = res.get(pid) or {}
                doi = next((x['value'] for x in r.get('articleids', []) if x.get('idtype') == 'doi'), None)
                pool.add(dict(title=r.get('title', '').rstrip('.'), year=int((r.get('pubdate') or '0')[:4] or 0), venue=r.get('fulljournalname') or r.get('source'),
                              doi=doi, pmid=pid), 'PUBMED'); n += 1
        print(f'PUBMED "{q[:50]}": {n}')


OASEL = 'id,doi,display_name,publication_year,primary_location,locations,type,ids,abstract_inverted_index,cited_by_count,open_access,best_oa_location'


def oa_rec(w):
    loc = w.get('primary_location') or {}
    src = (loc.get('source') or {}) if loc else {}
    venue = src.get('display_name') or ''
    if not venue:                                                     # fall back to any location with a source
        for l in w.get('locations') or []:
            if l.get('source') and l['source'].get('display_name'):
                venue = l['source']['display_name']; break
    inv = w.get('abstract_inverted_index') or {}
    abstract = ''
    if inv:
        pos = sorted((i, t) for t, idx in inv.items() for i in idx)
        abstract = ' '.join(t for _, t in pos)
    ids = w.get('ids') or {}
    arx = None
    for l in w.get('locations') or []:
        u = (l.get('landing_page_url') or '') + ' ' + (l.get('pdf_url') or '')
        m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', u)
        if m:
            arx = m.group(1); break
    doi = (w.get('doi') or '').replace('https://doi.org/', '') or None
    pmid = (ids.get('pmid') or '').rsplit('/', 1)[-1] or None
    boa = w.get('best_oa_location') or {}
    pdf = boa.get('pdf_url') or (w.get('open_access') or {}).get('oa_url')
    return dict(title=w.get('display_name'), year=w.get('publication_year'), venue=venue, doi=doi, arxiv=arx, pmid=pmid,
                openalex=w.get('id'), cites=w.get('cited_by_count'), abstract=abstract, pdf_url=pdf)


def oa_pages(url_base, kind, max_pages=50):
    """OpenAlex basic paging (page=1..; cursor paging needs an API key since 2026). 200 x 50 = 10,000 results max."""
    for page in range(1, max_pages + 1):
        body = fetch(url_base + f'&per-page=200&page={page}&mailto=eeailabseoultech@gmail.com', kind, sleep=1.0)
        if body is None:
            return
        d = json.loads(body)
        res = d.get('results', [])
        for w in res:
            yield w
        if len(res) < 200:
            return


def route_oacit(task, pool, years):
    for wid, name in TASKS[task]['oa_seeds'].items():
        n = 0
        for w in oa_pages(f'https://api.openalex.org/works?filter=cites:{wid},publication_year:{years[0]}-{years[1]}&select={OASEL}', 'oacit'):
            pool.add(oa_rec(w), 'OACIT'); n += 1
        print(f'OACIT {name}: {n}')


def route_oasrch(task, pool, years):
    for q in TASKS[task]['oa_queries']:
        n = 0
        for w in oa_pages(f'https://api.openalex.org/works?search={urllib.parse.quote(q)}&filter=publication_year:{years[0]}-{years[1]}&select={OASEL}', 'oasrch'):
            pool.add(oa_rec(w), 'OASRCH'); n += 1
            if n >= 2000:
                break
        print(f'OASRCH "{q}": {n}')


ROUTES = {'OACIT': route_oacit, 'OASRCH': route_oasrch, 'S2CIT': route_s2cit, 'S2SRCH': route_s2srch, 'DBLP': route_dblp, 'ARXIV': route_arxiv, 'XREF': route_xref, 'PUBMED': route_pubmed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=list(TASKS), required=True)
    ap.add_argument('--routes', default='S2CIT,OACIT,OASRCH,ARXIV,XREF,PUBMED,S2SRCH')
    ap.add_argument('--years', default='2023-2026')
    a = ap.parse_args()
    years = tuple(int(x) for x in a.years.split('-'))
    pool = Pool()
    for r in a.routes.split(','):
        print(f'== route {r}')
        try:
            ROUTES[r](a.task, pool, years)
        except Exception as e:
            print(f'route {r} failed: {type(e).__name__}: {e}', file=sys.stderr)
    recs = pool.records()
    keep = re.compile(TASKS[a.task]['keep_regex'], re.I)
    # venue resolution by DOI (Crossref) for topical records whose venue is empty or a book-series container
    vague = re.compile(r'^$|^pubmed$|lecture notes in computer science|communications in computer and information science|^springer|^elsevier bv$|^ieee$', re.I)
    n_res = 0
    for r in recs:
        if r.get('doi') and vague.search((r.get('venue') or '').strip()) and keep.search(f"{r.get('title', '')} {r.get('abstract', '') or ''}"):
            body = fetch('https://api.crossref.org/works/' + urllib.parse.quote(r['doi']) + '?mailto=eeailabseoultech@gmail.com', 'xrefdoi', sleep=0.7)
            if not body:
                continue
            try:
                m = json.loads(body)['message']
            except Exception:
                continue
            ev = (m.get('event') or {}).get('name') or ''
            ct = (m.get('container-title') or [''])
            cand = ev or (ct[0] if ct else '') or (m.get('publisher') or '')
            if cand and (vague.search(cand) is None or ev):
                r['venue'] = cand; r['venue_resolved'] = True; n_res += 1
    print(f'venue resolved by DOI for {n_res} vague-venue topical records')
    rows = []
    for r in recs:
        y = r.get('year')
        if not y or not (years[0] <= int(y) <= years[1]):
            continue
        text = f"{r.get('title', '')} {r.get('abstract', '') or ''}"
        topical = bool(keep.search(text))
        rows.append(dict(title=r.get('title'), year=y, venue=r.get('venue'), tier=classify_venue(r.get('venue')), doi=r.get('doi'),
                         arxiv=r.get('arxiv'), pmid=r.get('pmid'), s2=r.get('s2'), openalex=r.get('openalex'), cites=r.get('cites'),
                         routes='|'.join(sorted(r['routes'])), topical=topical, has_abstract=bool(r.get('abstract')), pdf_url=r.get('pdf_url'),
                         _abstract=(r.get('abstract') or '')[:4000]))
    rows.sort(key=lambda x: (x['tier'], -(x['year'] or 0), x['title'] or ''))
    out = os.path.join(HERE, f'{a.task}_candidates.csv')
    pub = [k for k in (rows[0].keys() if rows else ['title']) if not k.startswith('_')]
    with open(out, 'w', newline='', encoding='utf-8') as f:                       # committed: metadata only
        w = csv.DictWriter(f, fieldnames=pub, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RAW, f'{a.task}_candidates_abstracts.csv'), 'w', newline='', encoding='utf-8') as f:   # local: with abstracts
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['title'])
        w.writeheader(); w.writerows(rows)
    unknown = sorted({(r['venue'] or '') for r in rows if r['tier'] == 'UNKNOWN' and r['topical']})
    open(os.path.join(HERE, f'{a.task}_unknown_venues.txt'), 'w', encoding='utf-8').write('\n'.join(unknown) + '\n')
    print(f'{len(unknown)} distinct UNKNOWN venues among topical candidates -> {a.task}_unknown_venues.txt (review for tier assignment)')
    prisma = dict(task=a.task, years=years, per_route_raw=pool.counts, unique_in_years=len(rows),
                  by_tier={t: sum(1 for r in rows if r['tier'] == t) for t in ('A', 'B', 'EXCL', 'PREPRINT', 'UNKNOWN')},
                  topical_by_tier={t: sum(1 for r in rows if r['tier'] == t and r['topical']) for t in ('A', 'B', 'EXCL', 'PREPRINT', 'UNKNOWN')})
    json.dump(prisma, open(os.path.join(HERE, f'{a.task}_prisma.json'), 'w'), indent=1)
    print(json.dumps(prisma, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
