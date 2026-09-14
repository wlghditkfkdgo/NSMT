"""One training process per GPU; adapted from simple_test_model_v1 launcher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
from itertools import product
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
from utils import write_json


def main(args):
    queue_root = TASK / 'scripts/queues'
    queue_root.mkdir(parents=True, exist_ok=True)
    lock = (queue_root / (args.suite + '.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = TASK / 'results' / args.suite / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Suite exists; use a new suite name')
    jobs = []
    # Primary only: two datasets, three seeds, four population/memory conditions.
    for head in ['flatten']:
        for seed, data, heterogeneous, retrieval in product([7, 13, 21] if head == 'flatten' else [7],
                                                            ['ETTh1', 'ETTh2'], [False, True], [False, True]):
            variant = ('heterogeneous' if heterogeneous else 'homogeneous') + ('_retrieval' if retrieval else '_no_memory')
            command = [sys.executable, str(TASK / 'train.py'), '--suite', args.suite,
                       '--data', data, '--seed', str(seed), '--head_mode', head,
                       '--heterogeneous' if heterogeneous else '--no-heterogeneous',
                       '--retrieval' if retrieval else '--no-retrieval',
                       '--epoch', str(args.epochs), '--batch_size', str(args.batch_size),
                       '--pred_len', str(args.pred_len)]
            jobs.append({'id': f'{data}_p{args.pred_len}_{head}_{variant}_seed{seed}', 'seed': seed,
                         'command': command, 'status': 'pending'})
    manifest = {'suite': args.suite, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'launcher_command': shlex.join([sys.executable, *sys.argv]), 'gpus': args.gpus,
                'pred_len': args.pred_len, 'jobs': jobs, 'primary_runs': 24, 'exploratory_last_head_runs': 0}
    write_json(manifest_path, manifest)
    status_path = queue_root / (args.suite + '.json')
    write_json(status_path, manifest)
    mutex = threading.Lock()

    def worker(gpu):
        while True:
            with mutex:
                job = next((j for j in jobs if j['status'] == 'pending'), None)
                if job is None:
                    return
                job.update(status='running', gpu=gpu, started_utc=datetime.now(timezone.utc).isoformat())
                write_json(status_path, manifest)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(job['seed']), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            env['LD_LIBRARY_PATH'] = str(Path(sys.executable).resolve().parents[1] / 'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            stdout = TASK / 'log' / args.suite / (job['id'] + '.stdout')
            stdout.parent.mkdir(parents=True, exist_ok=True)
            print(f'START gpu={gpu} {job["id"]}', flush=True)
            with stdout.open('x') as handle:
                process = subprocess.Popen(job['command'], cwd=TASK, env=env, stdout=handle, stderr=subprocess.STDOUT)
                with mutex:
                    job.update(pid=process.pid, stdout=str(stdout))
                    write_json(status_path, manifest)
                code = process.wait()
            with mutex:
                job.update(status='complete' if code == 0 else 'failed', returncode=code,
                           finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(status_path, manifest)
            print(f'END gpu={gpu} code={code} {job["id"]}', flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
            future.result()
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(TASK / 'results' / args.suite / 'completion.json', manifest)
    if any(j['status'] != 'complete' for j in jobs):
        raise SystemExit('Some jobs failed; inspect scripts/queues and local stdout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='ett-tcn-h96-20260914')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pred_len', type=int, choices=[96, 720], default=96)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError('GPU IDs must be unique')
    main(args)
