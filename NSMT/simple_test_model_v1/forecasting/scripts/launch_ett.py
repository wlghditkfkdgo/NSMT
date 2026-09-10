"""Run the predeclared 4 ETT x 2 horizon x 5 variant matrix across GPUs.

One subprocess per GPU. Failed jobs are recorded; remaining jobs still run.
Queue status/lock/PID are in scripts/queues. Runtime console output is local.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK))
from run_ett import NSMT, write_json


def main(args):
    queue_root = TASK / "scripts/queues"
    queue_root.mkdir(parents=True,exist_ok=True)
    lock = (queue_root/(args.suite+".lock")).open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path = queue_root/(args.suite+".json")
    manifest_path = TASK / "results" / args.suite / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Suite already exists; use a new suite name")
    jobs = []
    for variant in ("temporal","temporal_embedding","population","no_attention","linear"):
        for horizon in (720,96):
            for dataset in ("ETTm1","ETTm2","ETTh1","ETTh2"):
                command = [sys.executable,str(TASK/"run_ett.py"),"--dataset",dataset,"--pred-len",str(horizon),
                           "--variant",variant,"--suite",args.suite,"--epochs",str(args.epochs),
                           "--patience",str(args.patience),"--batch-size",str(args.batch_size),
                           "--seed",str(args.seed),"--device","cuda:0","--backend","cupy"]
                jobs.append({"id":f"{dataset}_p{horizon}_{variant}_seed{args.seed}","command":command,"status":"pending"})
    manifest = {"suite":args.suite,"created_utc":datetime.now(timezone.utc).isoformat(),
                "launcher_command":shlex.join([sys.executable,*sys.argv]),"pid":os.getpid(),
                "gpus":args.gpus,"jobs":jobs}
    write_json(manifest_path,manifest)
    write_json(status_path,manifest)
    mutex = threading.Lock()
    def worker(gpu):
        while True:
            with mutex:
                pending = next((job for job in jobs if job["status"]=="pending"),None)
                if pending is None:
                    return
                pending.update(status="running",gpu=gpu,started_utc=datetime.now(timezone.utc).isoformat())
                write_json(status_path,manifest)
            stdout = TASK/"log"/args.suite/(pending["id"]+".stdout")
            stdout.parent.mkdir(parents=True,exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONHASHSEED"] = str(args.seed)
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["LD_LIBRARY_PATH"] = str(Path(sys.executable).resolve().parents[1]/"lib") + (
                ":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            print(f"START gpu={gpu} {pending['id']}",flush=True)
            with stdout.open("x") as handle:
                process = subprocess.Popen(pending["command"],cwd=NSMT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                with mutex:
                    pending["pid"] = process.pid
                    write_json(status_path,manifest)
                code = process.wait()
            with mutex:
                pending.update(status="complete" if code==0 else "failed",returncode=code,
                               finished_utc=datetime.now(timezone.utc).isoformat(),stdout=str(stdout))
                write_json(status_path,manifest)
            print(f"END gpu={gpu} code={code} {pending['id']}",flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker,gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path,manifest)
    write_json(TASK/"results"/args.suite/"completion.json",manifest)
    if any(job["status"]!="complete" for job in jobs):
        raise SystemExit("Some jobs failed; inspect queue status and stdout")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--gpus",nargs="+",type=int,default=[0,1,2,3])
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--patience",type=int,default=3)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--seed",type=int,default=7)
    args=parser.parse_args()
    if len(args.gpus)!=len(set(args.gpus)):
        raise ValueError("GPU IDs must be unique")
    main(args)
