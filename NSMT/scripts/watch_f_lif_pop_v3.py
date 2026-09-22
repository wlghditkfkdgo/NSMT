"""Change-triggered audit queue for the user's existing Codex thread.

Cron invokes tick every ten minutes. No model/training code is executed here.
Runtime state and the single-flight lock stay local under scripts/queues/.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / 'scripts/queues/assessment_watch'
PROMPT = ROOT / 'docs/ASSESSMENT_WATCH_PROMPT.md'
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.csv', '.json', '.yaml', '.yml', '.toml', '.sh'}
EXCLUDED_PARTS = {'__pycache__', '.git', '.pytest_cache', 'assessment', 'assessment_watch',
                  'dataset', 'datasets', 'cache', 'wandb', 'model_state'}
EXCLUDED_DOCS = {'ASSESMENT.md', 'DOCS_REVIEW_MEMORY.md', 'PROJECT_LOG.md',
                 'ASSESSMENT_WATCH_PROMPT.md', 'ASSESSMENT_WATCH.md'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def event(action, **fields):
    row = {'utc': now(), 'action': action, **fields}
    with (STATE_DIR / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def manifest():
    files = {}
    for base in (ROOT / 'docs', ROOT / 'f_lif_pop_v3'):
        for path in sorted(base.rglob('*')):
            rel = path.relative_to(ROOT)
            if path.is_symlink() or not path.is_file() or set(rel.parts) & EXCLUDED_PARTS:
                continue
            if base.name == 'docs' and path.name in EXCLUDED_DOCS:
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('File changed during fingerprint: ' + str(rel))
            files[str(rel)] = digest.hexdigest()
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    return {'head': head, 'files': files}


def changes(previous, current):
    before = previous.get('files', {})
    after = current['files']
    paths = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
    if previous.get('head') != current['head']:
        paths.insert(0, 'GIT_HEAD')
    return paths


def codex_path(config):
    path = Path(config['codex'])
    if path.is_file():
        return str(path)
    found = shutil.which('codex')
    if found:
        return found
    # IDE upgrades can replace the extension's versioned executable path.
    candidates = list((Path.home() / '.vscode-server/extensions').glob(
        'openai.chatgpt-*/bin/linux-x86_64/codex'))
    if candidates:
        return str(max(candidates, key=lambda p: p.stat().st_mtime_ns))
    raise RuntimeError('Codex executable unavailable; pending audit not acknowledged')


def tick(config, state, force=False):
    state['last_check_utc'] = now()
    if not config['enabled']:
        event('paused')
        return
    if state.get('pending'):
        event('audit_already_pending', run_id=state['pending']['run_id'])
        return
    current = manifest()
    changed = changes(state.get('last_completed_manifest', {}), current)
    if not changed and not force:
        event('no_change')
        return
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run_dir = STATE_DIR / 'runs' / run_id
    run_dir.mkdir(parents=True)
    record = {'run_id': run_id, 'queued_utc': now(), 'manifest': current,
              'changed_paths': changed, 'reason': 'manual_force' if force else 'changed'}
    save_json(run_dir / 'trigger.json', record)
    prompt = PROMPT.read_text().replace('{{RUN_ID}}', run_id).replace('{{ROOT}}', str(ROOT))
    prompt += '\n변경 감지 목록(JSON):\n' + json.dumps(changed, ensure_ascii=False)
    prompt += '\n감지 시점 manifest: ' + str(run_dir / 'trigger.json') + '\n'
    state['pending'] = {'run_id': run_id, 'queued_utc': record['queued_utc']}
    # Persist before delivery, so the receiving thread can acknowledge immediately.
    save_json(STATE_DIR / 'state.json', state)
    try:
        reply = subprocess.run([codex_path(config), 'queue', '--thread', config['thread_id'],
                                '--message', prompt], cwd=str(ROOT), capture_output=True,
                               text=True, timeout=45)
        (run_dir / 'delivery.stdout').write_text(reply.stdout + reply.stderr)
        if reply.returncode:
            raise RuntimeError('codex queue exit ' + str(reply.returncode) + ': ' + reply.stderr[-500:])
        state['last_delivery_utc'] = now()
        event('queued', run_id=run_id, changed_count=len(changed), response=reply.stdout.strip())
    except subprocess.TimeoutExpired:
        # Delivery might have succeeded before the timeout: retain the pending marker.
        state['last_error'] = 'Delivery timed out; status unknown; inspect before retrying'
        event('delivery_uncertain', run_id=run_id)
        raise
    except Exception:
        state.pop('pending', None)
        raise


def acknowledge(state, run_id):
    if state.get('pending', {}).get('run_id') != run_id:
        raise RuntimeError('Acknowledgment does not match the pending run')
    marker = '<!-- assessment-watch:' + run_id + ' -->'
    if marker not in (ROOT / 'docs/ASSESMENT.md').read_text():
        raise RuntimeError('Append the audit and its exact completion marker before acknowledging')
    record = read_json(STATE_DIR / 'runs' / run_id / 'trigger.json')
    # Acknowledge only the trigger snapshot; changes during the audit remain detectable.
    state['last_completed_manifest'] = record['manifest']
    state['last_completed_utc'] = now()
    state['last_completed_run_id'] = run_id
    state.pop('pending')
    state.pop('last_error', None)
    event('acknowledged', run_id=run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'tick', 'status', 'pause', 'resume', 'ack'])
    parser.add_argument('--thread')
    parser.add_argument('--run-id')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another watcher operation is running; skip this tick.')
            return 0
        config_path, state_path = STATE_DIR / 'config.json', STATE_DIR / 'state.json'
        if args.action == 'init':
            if config_path.exists():
                raise RuntimeError('Already initialized; existing queue preserved')
            if not args.thread or not shutil.which('codex'):
                raise RuntimeError('init requires --thread and a working Codex CLI')
            save_json(config_path, {'enabled': True, 'thread_id': args.thread,
                                   'codex': shutil.which('codex'), 'created_utc': now(),
                                   'schedule': '*/10 * * * *', 'root': str(ROOT)})
            save_json(state_path, {'last_completed_manifest': manifest(),
                                   'baseline_utc': now(), 'baseline_is_audit': False})
            event('initialized', thread_id=args.thread)
            return 0
        config, state = read_json(config_path), read_json(state_path)
        if config is None or state is None:
            raise RuntimeError('Watcher has not been initialized')
        if args.action == 'status':
            print(json.dumps({'config': config, 'state': {k:v for k,v in state.items()
                             if k != 'last_completed_manifest'}}, ensure_ascii=False, indent=2))
            return 0
        if args.action in ('pause', 'resume'):
            config['enabled'] = args.action == 'resume'
            save_json(config_path, config)
            event(args.action)
            return 0
        try:
            if args.action == 'ack':
                acknowledge(state, args.run_id)
            else:
                tick(config, state, args.force)
        except Exception as exc:
            state['last_error'] = str(exc)
            event('error', detail=str(exc))
            raise
        finally:
            save_json(state_path, state)
    return 0


if __name__ == '__main__':
    sys.exit(main())
