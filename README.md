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
