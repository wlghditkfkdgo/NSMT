"""Check that extraction evidence quotes occur in the source text (whitespace/hyphenation-insensitive). usage: python verify_quotes.py <task>"""
import json, re, sys, glob
task = sys.argv[1]
norm = lambda s: re.sub(r'[^a-z0-9]+', '', s.lower())
wl = {}
for f in glob.glob(f'raw/extract/{task}_worklist_*.json'):
    wl.update({w['key']: w for w in json.load(open(f))})
tot = ok = placeholder = 0; missing = []
for f in sorted(glob.glob(f'raw/extract/{task}_extract_*.jsonl')):
    for l in open(f, encoding='utf-8'):
        if not l.strip(): continue
        r = json.loads(l); w = wl.get(r['key'])
        if not w: continue
        t = norm(open(w['txt'], encoding='utf-8', errors='replace').read())
        for fld, q in (r.get('evidence') or {}).items():
            if not isinstance(q, str): continue
            if q.startswith('(') or len(q) < 15: placeholder += 1; continue
            tot += 1
            frag = norm(re.split(r'\.\.\.|…', q)[0])[:45]
            if frag and frag in t: ok += 1
            else: missing.append((r['title'][:40], fld, q[:60]))
print(f'{task}: quotes verified {ok}/{tot} ({placeholder} placeholders skipped)')
for m in missing: print('  MISSING', m)
