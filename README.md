<div align="center">

# TADR

### Threat-Aligned Data Refinement against Label-Flipping Attacks in AI-powered Intrusion Detection Systems

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#-installation)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Datasets](https://img.shields.io/badge/Datasets-NSL--KDD%20%7C%20CIC--IDS--2018%20%7C%20Edge--IIoTset-2ea44f.svg)](#-supported-datasets)
[![License](https://img.shields.io/badge/License-Academic%20Use--Only-orange.svg)](#-license)

An ablation framework for studying **asymmetric targeted label-flipping attacks** and the **TADR** defense pipeline (DAE + KMeans benign-core + weighted KNN filtering, followed by DNN-based recovery) on three widely used network-intrusion-detection datasets.

[Overview](#-overview) · [Supported Datasets](#-supported-datasets) · [Data Acquisition](#-data-acquisition) · [Installation](#-installation) · [Running the Ablation](#-running-the-ablation-study) · [Outputs](#-outputs) · [Per-Dataset Guides](#-per-dataset-guides) · [Repository Layout](#-repository-layout) · [Citation](#-citation)

</div>

---

## Overview

This repository hosts the implementation used in the paper:

> **TADR: Threat-Aligned Data Refinement against Label-Flipping Attacks in AI-powered Intrusion Detection Systems.**

It provides an end-to-end ablation pipeline that:

1. **Injects an asymmetric targeted label-flipping attack** — non-benign labels are flipped to `Benign`/`Normal` at a configurable rate (symmetric per-class noise).
2. **Trains a pool of supervised detectors** — XGBoost, CatBoost, Bagging, LightGBM, Random Forest, and a Deep Neural Network (DNN).
3. **Applies the TADR defense** — a Denoising Autoencoder (DAE) produces latent embeddings, KMeans selects a representative benign core, and a weighted KNN flags suspicious samples as poison.
4. **Recovers high-confidence samples** with a DNN using fixed per-class thresholds and a conservative recovery cap.
5. **Re-trains the detector pool** at every stage and emits a consolidated metrics summary.

---

## Supported Datasets

| Dataset | Folder |
|---|---|
| **NSL-KDD** | `src/apelid/NSL_KDD/` | 
| **CIC-IDS-2018** | `src/apelid/IDS18/` | 
| **Edge-IIoTset** | `src/apelid/EdgeIIoTset/` |

> All three pipelines share the same six-stage structure; per-dataset CLI flags, encoders, and preprocessing are isolated inside each folder. See the [Per-Dataset Guides](#-per-dataset-guides) for full details.

---

## Data Acquisition

You can either use the preprocessed bundle or download the raw datasets yourself:

- **Option A** — Use the preprocessed CSV bundle from Google Drive: `<PASTE_GOOGLE_DRIVE_LINK_HERE>` (recommended, fastest).
- **Option B** — Download the raw datasets from the original sources:
  - **NSL-KDD**: <https://www.unb.ca/cic/datasets/nsl.html>
  - **CIC-IDS-2018**: <https://www.unb.ca/cic/datasets/ids-2018.html>
  - **Edge-IIoTset**: <https://github.com/Edge-IIoTset/Edge-IIoTset_Dataset>

---

## Installation

The project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

### Requirements

- **Python** ≥ 3.9
- Works on **Windows**, **Linux**, and **macOS**
- Optional GPU acceleration for the DNN stages (set `--device auto`)

### 1. Install `uv`

```bash
# Option A — pip
pip install uv

# Option B — pipx (recommended for CLI tools)
pipx install uv
```

### 2. Sync the project environment

From the project root (where `pyproject.toml` is located):

```bash
uv sync
```

This creates a local `.venv/` and installs all dependencies declared in `pyproject.toml`.

### 3. Activate the virtual environment

**Windows — CMD:**

```cmd
.\.venv\Scripts\activate.bat
```

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

> Prefer not to use `uv`? Create a venv with `python -m venv .venv`, activate it, and run `pip install -r requirements.txt` (or `pip install -e .` if the project is packaged).

---

## Running the Ablation Study

Every dataset ships with a single `ablation_pipeline.py` runner that executes all six stages per noise rate and emits a consolidated metrics CSV.

### NSL-KDD

```bash
python src/apelid/NSL_KDD/ablation_pipeline.py \
  --noise-rates 30,40,50 \
  --kmeans-budget 15000 \
  --k-neighbors 15 \
  --log-dir src/apelid/NSL_KDD/reports/ablation_logs
```

### CIC-IDS-2018

```bash
python src/apelid/IDS18/ablation_pipeline.py \
  --noise-rates 30,40,50 \
  --kmeans-budget 6000 \
  --k-neighbors 10 \
  --device auto \
  --log-dir src/apelid/IDS18/reports/ablation_logs
```

### Edge-IIoTset

```bash
python src/apelid/EdgeIIoTset/ablation_pipeline.py \
  --noise-rates 30,40,50 \
  --kmeans-budget 15000 \
  --k-neighbors 15 \
  --log-dir src/apelid/EdgeIIoTset/reports/ablation_logs
```

### Common CLI flags

| Flag | Description |
|---|---|
| `--noise-rates` | Comma-separated percentages, e.g. `30,40,50,60,70`. |
| `--kmeans-budget` | Number of representative benign-core samples used in the DAE+KMeans+KNN defense. |
| `--k-neighbors` | KNN neighborhood size for the defense. |
| `--device` | `auto` or `CPU` (only on the CIC-IDS-2018 / Edge-IIoTset runners). |
| `--summary-csv` | Optional explicit path for the consolidated summary CSV. |
| `--log-dir` | Root output directory for the timestamped run. |
| `--clean-input` | (Edge-IIoTset only) Path to the clean preprocessed training CSV. |

---

## Outputs

Each pipeline run creates a **timestamped folder** under the dataset's `reports/ablation_logs/`:

```text
src/apelid/<DATASET>/reports/ablation_logs/
└── {YYYYMMDD_HHMMSS}/
    ├── ablation_f1_macro_summary.csv
    └── noise_{pct}/
        ├── 01_symmetric_label_noise/
        ├── 02_model_training_after_noise/
        ├── 03_dae_kmeans_knn/
        ├── 04_model_training_after_dae/
        ├── 05_dnn_recover/
        └── 06_model_training_after_recover/
```

### Stage-by-stage artifacts

| Stage | Folder | Key outputs |
|---|---|---|
| **01 — Noise injection** | `01_symmetric_label_noise/` | `noise_{pct}_data.csv`, `noise_{pct}_symmetric_label_noise.log` |
| **02 — Train on poisoned data** | `02_model_training_after_noise/` | `<model>.pkl` / `.pth`, `<model>_metrics.json`, `<model>_cm.png` |
| **03 — DAE + KMeans + KNN defense** | `03_dae_kmeans_knn/` | `noise_{pct}_clean.csv`, `noise_{pct}_noise.csv`, `dae_model.pth`, `noise_detection_report.json`|
| **04 — Train after defense** | `04_model_training_after_dae/` | same as stage 02 |
| **05 — DNN-based recovery** | `05_dnn_recover/` | `noise_{pct}_recovered_final.csv`, `noise_{pct}_remaining_after_recover.csv`, `recovery_summary.json`, `recovery_per_class.csv` |
| **06 — Train after recovery** | `06_model_training_after_recover/` | same as stage 02 |

### The consolidated summary CSV

`ablation_f1_macro_summary.csv` has the columns:

| Column | Meaning |
|---|---|
| `noise_pct` | Poisoning rate, e.g. `30`, `40`, `50`. |
| `stage` | `01_after_noise`, `02_after_dae`, or `03_after_recover`. |
| `model` | `xgb`, `catb`, `bagging`, `lgbm`, `rf`, `dnn` |
| `accuracy` | Test-set accuracy. |
| `f1_macro` | Test-set macro-averaged F1. |
| `train_file` | CSV used to fit the model. |

---

## Per-Dataset Guides

Detailed I/O, CLI, and file-layout documentation for each dataset lives in its own README:

- NSL-KDD → [`src/apelid/NSL_KDD/README.md`](src/apelid/NSL_KDD/README.md)
- CIC-IDS-2018 → [`src/apelid/IDS18/README.md`](src/apelid/IDS18/README.md)
- Edge-IIoTset → [`src/apelid/EdgeIIoTset/README.md`](src/apelid/EdgeIIoTset/README.md)

---

## Repository Layout

```text
.
├── pyproject.toml                 # Project metadata & dependencies (managed by uv)
├── README.md                      # This file
└── src/
    └── apelid/
        ├── NSL_KDD/               # NSL-KDD ablation pipeline
        │   ├── ablation_pipeline.py
        │   ├── symmetric_label_noise.py
        │   ├── dae_kmeans_knn_benign_filter.py
        │   ├── dnn_recover_grid.py
        │   ├── training/
        │   ├── preprocessing.py
        │   └── README.md
        ├── IDS18/                 # CIC-IDS-2018 ablation pipeline
        │   ├── ablation_pipeline.py
        │   ├── symmetric_label_noise.py
        │   ├── dae_kmeans_knn_benign_filter.py
        │   ├── dnn_recover_grid.py
        │   ├── model_training.py
        │   ├── preprocessing/
        │   └── README.md
        └── EdgeIIoTset/           # Edge-IIoTset ablation pipeline
            ├── ablation_pipeline.py
            ├── preprocessing.py
            ├── symmetric_label_noise.py
            ├── dae_kmeans_knn_benign_filter.py
            ├── dnn_recover_grid.py
            ├── model_training.py
            └── README.md
```