"""One opening per held-out split, whatever the output directory (prereg 2M D-AY, audit 41).

Every evaluator used to guard its held-out split with an exclusive-create record inside its OWN
output directory, so a second run under a different --out name could open the same split again.
The registry moves that guard to one place: results/split_openings/<split>.json, created with
'x' before the split is read. A split that is already registered cannot be opened by anything.

The splits opened before the registry existed are entered as historical openings, with the
records that opened them, so they are closed too.
"""
import json
import getpass
import datetime
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / 'forecasting' / 'results' / 'split_openings'

HISTORICAL = {
    'confirm':  {'prereg': '2I', 'records': ['results/seeds-173737/eta_selection_record.json']},
    'confirm2': {'prereg': '2J (+ 2L safety supplement)',
                 'records': ['results/hardsel-014614/hard_selection_record.json',
                             'results/hardsafety-170640/hard_safety_record.json']},
    'confirm3': {'prereg': '2K (+ 2L safety supplement)',
                 'records': ['results/hardctrl-152631/hard_control_record.json',
                             'results/hardsafety-170640/hard_safety_record.json']},
}


def register_history():
    """Enter the pre-registry openings; existing entries are left alone."""
    REGISTRY.mkdir(parents=True, exist_ok=True)
    for split, entry in HISTORICAL.items():
        path = REGISTRY / f'{split}.json'
        if not path.exists():
            path.write_text(json.dumps({'split': split, 'historical': True, **entry}, indent=2))


def open_once(split, purpose, record):
    """Claim `split` for one evaluation or refuse. Call BEFORE the split is read."""
    register_history()
    path = REGISTRY / f'{split}.json'
    entry = {'split': split, 'historical': False, 'purpose': purpose, 'record': str(record),
             'opened_at': datetime.datetime.now().astimezone().isoformat(), 'user': getpass.getuser()}
    try:
        with open(path, 'x') as handle:                        # 전역 잠금: 출력 이름과 무관하다
            json.dump(entry, handle, indent=2)
    except FileExistsError:
        raise SystemExit(f'[registry] {split} is already opened ({path}); '
                         f'a new question needs a new appendix and a new split.') from None
    return path
