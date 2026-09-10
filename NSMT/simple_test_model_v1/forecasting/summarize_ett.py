"""Validate and summarize the predeclared ETT comparison without selecting runs."""
import argparse
import csv
import json
from pathlib import Path
import statistics

TASK = Path(__file__).resolve().parent
VARIANTS = ("population", "temporal", "temporal_embedding", "no_attention", "linear")


def summarize(suite, allow_partial=False, plot=False):
    root = TASK / "results" / suite
    manifest = json.loads((root/"manifest.json").read_text())
    runs, rows = {}, []
    for job in manifest["jobs"]:
        path = root/(job["id"]+".json")
        if not path.exists():
            if allow_partial:
                continue
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        config = result["config"]
        assert result["status"] == "complete"
        assert not result["protocol"]["quick_smoke_only"]
        for split in ("validation", "test"):
            name = "val" if split == "validation" else split
            expected = result["data"]["splits"][name]["windows"] * config["pred_len"] * 7
            assert result[split]["elements"] == expected, (job["id"],split)
        minimum = min(epoch["validation"]["mse"] for epoch in result["history"])
        assert abs(minimum-result["validation"]["mse"]) < 1e-7
        assert result["best_epoch"] <= result["epochs_run"] <= config["epochs"]
        key = (config["dataset"],config["pred_len"],config["variant"])
        assert key not in runs
        runs[key] = result
        rates = result["test_first_batch_spikes"]
        attention_rates = [value for name,value in rates.items() if ".attn.proj.lif" in name]
        row = {"dataset":key[0],"pred_len":key[1],"variant":key[2],"seed":config["seed"],
               "test_mse":result["test"]["mse"],"test_mae":result["test"]["mae"],
               "val_mse":result["validation"]["mse"],"best_epoch":result["best_epoch"],
               "epochs_run":result["epochs_run"],"epoch_cap_reached":result["epochs_run"]==config["epochs"],
               "parameters":result["parameters"],"seconds":result["seconds"],
               "peak_gpu_gib":result["peak_gpu_memory_bytes"]/2**30,
               "test_first_batch_embedding_rate":rates.get("embedding.lif"),
               "test_first_batch_mean_ssa_rate":statistics.mean(attention_rates) if attention_rates else None,
               "persistence_mse":result["test"]["persistence"]["mse"],
               "window_mean_mse":result["test"]["window_mean"]["mse"],
               "checkpoint":result["checkpoint"],"source_commit":result["git_commit"]}
        rows.append(row)
    if not rows:
        return {"complete":0,"expected":len(manifest["jobs"])}
    rows.sort(key=lambda r:(r["dataset"],r["pred_len"],VARIANTS.index(r["variant"])))
    with (root/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    comparisons=[]
    for dataset in ("ETTh1","ETTh2","ETTm1","ETTm2"):
        for horizon in (96,720):
            for tested,reference in (("temporal","population"),("temporal_embedding","temporal"),
                                     ("temporal","no_attention"),("temporal","linear")):
                a,b=runs.get((dataset,horizon,tested)),runs.get((dataset,horizon,reference))
                if a is None or b is None:
                    continue
                assert a["data"]==b["data"]
                assert a["source_sha256"]==b["source_sha256"]
                difference=a["test"]["mse"]-b["test"]["mse"]
                comparisons.append({"dataset":dataset,"pred_len":horizon,"tested":tested,"reference":reference,
                                    "mse_difference":difference,"relative_mse_percent":100*difference/b["test"]["mse"]})
    with (root/"comparisons.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=("dataset","pred_len","tested","reference","mse_difference","relative_mse_percent"),lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    summary={"complete":len(rows),"expected":len(manifest["jobs"]),"suite":suite,"variants":{},"comparisons":{}}
    for variant in VARIANTS:
        subset=[r for r in rows if r["variant"]==variant]
        if subset:
            summary["variants"][variant]={"runs":len(subset),"macro_mse":statistics.mean(r["test_mse"] for r in subset),
                                           "macro_mae":statistics.mean(r["test_mae"] for r in subset),
                                           "total_seconds":sum(r["seconds"] for r in subset)}
    for tested,reference in sorted(set((r["tested"],r["reference"]) for r in comparisons)):
        subset=[r for r in comparisons if (r["tested"],r["reference"])==(tested,reference)]
        summary["comparisons"][tested+" vs "+reference]={"pairs":len(subset),
            "mse_wins":sum(r["mse_difference"]<0 for r in subset),
            "mean_relative_mse_percent":statistics.mean(r["relative_mse_percent"] for r in subset)}
    (root/"aggregate.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=[f"# ETT quick validation — {suite}","",f"Completed {len(rows)}/{len(manifest['jobs'])} predefined runs.","",
           "Seed 7; full canonical ETT splits; input 96; horizon 96/720; patch/stride 8; "
           "maximum 10 epochs, patience 3; best validation MSE checkpoint. "
           "Metrics are on the train-standardized scale, averaged over all windows/horizon points/channels.","",
           "All SNN variants use fixed Gaussian coding, direct input currents and the same two-stage head. "
           "Temporal embedding adds a trainable population identity. No-attention keeps embedding and IAND MLP. "
           "Linear is a shared per-channel Linear(96,H) with the same input-window normalization.","",
           "| Dataset | Horizon | K-axis MSE/MAE | N-axis MSE/MAE | N-axis + identity MSE/MAE | No SSA MSE/MAE | Linear MSE/MAE |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("ETTh1","ETTh2","ETTm1","ETTm2"):
        for horizon in (96,720):
            cells=[]
            for variant in VARIANTS:
                result=runs.get((dataset,horizon,variant))
                cells.append(f"{result['test']['mse']:.4f} / {result['test']['mae']:.4f}" if result else "pending")
            lines.append(f"| {dataset} | {horizon} | "+" | ".join(cells)+" |")
    lines.extend(["","Macro averages weight the eight dataset/horizon tasks equally; they are not pooled errors.","",
                  "| Variant | Tasks | Macro MSE | Macro MAE |","|---|---:|---:|---:|"])
    for name,value in summary["variants"].items():
        lines.append(f"| {name} | {value['runs']} | {value['macro_mse']:.4f} | {value['macro_mae']:.4f} |")
    lines.extend(["","## Paired comparisons","","Negative relative MSE means the tested variant improved over the reference.","",
                  "| Tested vs reference | MSE wins/tasks | Mean relative MSE |","|---|---:|---:|"])
    for name,value in summary["comparisons"].items():
        lines.append(f"| {name} | {value['mse_wins']}/{value['pairs']} | {value['mean_relative_mse_percent']:+.2f}% |")
    lines.extend(["","## Limits","",
                  "One seed and a short epoch budget provide screening evidence, not statistical significance or convergence. "
                  "No hyperparameters were selected using test scores. The two-stage head is shared by all SNN conditions; "
                  "this matrix does not separately establish its advantage over the old flatten head. "
                  "No-population SNN and multiple-seed comparisons were not run. "
                  "IAND only suppresses spikes; population and temporal axes use the same fixed scale=1. "
                  "Their individually optimal scales may differ. Spike statistics cover the first evaluation batch only.","",
                  "See summary.csv for epochs, validation scores, checkpoints and parameters; comparisons.csv for paired differences; "
                  "each run JSON contains the exact command, code/data hashes, environment and epoch history.",""])
    (root/"REPORT.md").write_text("\n".join(lines))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        pairs=(("temporal","population"),("temporal_embedding","temporal"),
               ("temporal","no_attention"),("temporal","linear"))
        tasks=[(dataset,horizon) for dataset in ("ETTh1","ETTh2","ETTm1","ETTm2") for horizon in (96,720)]
        lookup={(r["dataset"],r["pred_len"],r["tested"],r["reference"]):r["relative_mse_percent"] for r in comparisons}
        matrix=np.array([[lookup[(*task,*pair)] for pair in pairs] for task in tasks])
        limit=max(5,float(np.abs(matrix).max()))
        fig,ax=plt.subplots(figsize=(9,6))
        mesh=ax.imshow(matrix,cmap="RdYlGn_r",vmin=-limit,vmax=limit,aspect="auto")
        ax.set_xticks(range(4),["N-axis vs K-axis","Identity vs no identity","N-axis vs no SSA","N-axis vs linear"])
        ax.set_yticks(range(8),[f"{dataset} / {horizon}" for dataset,horizon in tasks])
        for row in range(8):
            for column in range(4):
                ax.text(column,row,f"{matrix[row,column]:+.1f}%",ha="center",va="center",fontsize=11)
        ax.set_title("Relative test MSE: negative means improvement",pad=16)
        fig.colorbar(mesh,ax=ax,label="Relative MSE change (%)",shrink=0.8)
        fig.text(0.5,0.025,"Seed 7 | input 96 | full ETT splits | max 10 epochs, patience 3 | validation-selected checkpoints",
                 ha="center",fontsize=9)
        fig.tight_layout(rect=(0,0.06,1,1))
        fig.savefig(root/"comparison.png",dpi=180)
        fig.savefig(root/"comparison.pdf")
        plt.close(fig)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    print(json.dumps(summarize(args.suite,args.allow_partial,args.plot),indent=2))
