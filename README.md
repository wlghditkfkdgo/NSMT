# Fast Spikes, Slow Trends: Neuro-Inspired Spiking Memory Transformer for Time-Series Analysis

## ⚙️ Environment Setup

This repository contains the official implementation for the paper: **"Fast Spikes, Slow Trends: Neuro-Inspired Spiking Memory Transformer for Time-Series Analysis"**.

To ensure perfect reproducibility, we use the exact same Python environment across all three tasks: `forecasting`, `anomaly_detection`, and `classification`. We provide an Anaconda environment file (`environment.yml`) to help you easily replicate our setup.

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### Installation Steps

**1. Create the conda environment**
Run the following command at the root of the repository to create the environment named `snn_jelly` using the provided configuration file:
```bash
conda env create -f environment.yml
```

**2. Activate the environment**
Once the installation is complete, activate the environment before running any scripts:
```bash
conda activate snn_jelly
```

*(Optional) If you prefer using `pip`, you can also install the dependencies via the requirements file (if provided):*
`pip install -r requirements.txt`


```
-Python: 3.10.18
-PyTorch: 1.12.0 (with CUDA 11.3.1)
-SpikingJelly: 0.0.0.0.14
-NumPy: 1.26.4
-Scikit-learn: 1.7.1
-timm: 1.0.19
```


## Experiment workflow

Keep `main` as the reference branch. Develop new experiments on `exp/<experiment-name>` and record changes, reproduction commands, results, and annotated tags by appending to [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md). The current local snapshot is published on `exp/local-snapshot-20260910`; it is not a new validated baseline on `main`.

The repository excludes local datasets, checkpoints, caches, archives, and raw runtime logs. See the project log for artifact locations and setup requirements. NSMT task layout and runners are described in [NSMT/docs/project_layout.md](NSMT/docs/project_layout.md).
