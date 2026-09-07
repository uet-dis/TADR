# Edge-IIoTset Ablation Study Guide

This document describes the **Edge-IIoTset** ablation pipeline inside `src/tadr/EdgeIIoTset/`.

It is part of the larger TADR ablation framework. For the top-level description, see the [project README](../../../../README.md).

---

## Table of contents

1. [Data Preprocessing](#1-data-preprocessing)
2. [Pipeline overview](#2-pipeline-overview)
3. [How to run the full pipeline](#3-how-to-run-the-full-pipeline)
4. [Stage-by-stage I/O details](#4-stage-by-stage-io-details)
5. [Baseline Defenses](#5-baseline-defenses)

---

## 1. Data Preprocessing

### Option A — Use the preprocessed CSV bundle (recommended, fastest)
Download the preprocessed CSV bundle from the project Google Drive:
**[https://drive.google.com/drive/folders/1Kxowaim0NSRChkR8NtBkiQ51kpvxWJlO?usp=sharing](https://drive.google.com/drive/folders/1Kxowaim0NSRChkR8NtBkiQ51kpvxWJlO?usp=sharing)**

### Option B — Preprocess from raw sources
Run the single preprocessing script to produce the cleaned training CSV and fitted encoders:

```bash
python src/tadr/EdgeIIoTset/preprocessing.py
```

**Outputs:**
| File | Description |
|---|---|
| `edgeiot_clean_merged.csv` | Full cleaned dataset |
| `edgeiot_train_clean_merged.csv` | 70 % training split — used by all baselines |
| `edgeiot_test_clean_merged.csv` | 30 % test split |

---

## 2. Pipeline overview

**Main runner:** `ablation_pipeline.py`

**Execution order for each noise rate:**

1. `symmetric_label_noise.py` — inject asymmetric targeted label-flipping attack
2. `model_training.py` on the noisy data
3. `dae_kmeans_knn_benign_filter.py` — TADR defense (DAE + KMeans + weighted KNN)
4. `model_training.py` on the DAE-cleaned data
5. `dnn_recover_grid.py` — DNN-based recovery of high-confidence samples
6. `model_training.py` on the recovered-clean data

---

## 3. How to run the full pipeline

```bash
python src/tadr/EdgeIIoTset/ablation_pipeline.py \
  --noise-rates 30,40,50,60,70 \
  --kmeans-budget 15000 \
  --k-neighbors 15 \
  --log-dir src/tadr/EdgeIIoTset/reports/ablation_logs
```

The pipeline trains six models: **xgb, catb, bagging, lgbm, rf, dnn**.

---

## 4. Stage-by-stage I/O details

### 4.1 `symmetric_label_noise.py`

**What it is:** Inject targeted noise by flipping non-`Normal` labels to `Normal`.

**Input:** Clean CSV (pipeline passes `--clean-input`).

**Output:**
- `noise_{pct}_data.csv` (with `original_label` and `is_noisy` columns when `--keep-tracking` is on)
- `noise_{pct}_symmetric_label_noise.log`

**Run:**

```bash
python src/tadr/EdgeIIoTset/symmetric_label_noise.py \
  --input src/tadr/EdgeIIoTset/resources/edgeiot/clean_merged/edgeiot_train_clean_merged.csv \
  --noise-rate 0.30 \
  --keep-tracking \
  --seed 42 \
  --output src/tadr/EdgeIIoTset/reports/noise_30_data.csv
```

---

### 4.2 `dae_kmeans_knn_benign_filter.py`

**What it is:** TADR defense stage (DAE embeddings + KMeans benign-core + weighted KNN detection).

**Input:**
- `--input <noisy.csv>`
- `--kmeans-budget <n>`
- `--k-neighbors <k>`

**Output:**
- `<name>_clean.csv`
- `<name>_noise.csv`
- `noise_detection_report.json`
- `noise_detection_per_label.csv`
- `*_dae_kmeans_knn.log`

**Run:**

```bash
python src/tadr/EdgeIIoTset/dae_kmeans_knn_benign_filter.py \
  --input path/to/noisy.csv \
  --kmeans-budget 15000 \
  --k-neighbors 15 \
  --output-dir path/to/dae_outputs \
  --name noise_30
```

---

### 4.3 `dnn_recover_grid.py`

**What it is:** Recover high-confidence samples from the DAE-noise set using a fixed DNN configuration (attack/benign threshold = 0.9, per-class recovery cap ratio = 0.5).

**Input:**
- `--clean-in <dae_clean.csv>`
- `--noise-in <dae_noise.csv>`

**Output:**
- `<name>_recovered_final.csv`
- `<name>_remaining_after_recover.csv`
- `recovery_summary.json`
- `recovery_per_class.csv`
- `*_dnn_recover.log`

**Run:**

```bash
python src/tadr/EdgeIIoTset/dnn_recover_grid.py \
  --clean-in path/to/dae_clean.csv \
  --noise-in path/to/dae_noise.csv \
  --output-dir path/to/recover_outputs \
  --name noise_30
```

---

### 4.4 `model_training.py`

**What it is:** Train all ablation models and export model artifacts + metrics.

**Input:**
- `--train-in <csv>`
- `--model all`, `--resource edgeiot`, `--device {auto|CPU}`

**Output (per training phase):**
- `*_model_training.log`
- For each of `xgb, catb, bagging, lgbm, rf, dnn`:
  - `<model>.pkl` (or `<model>.pth` for DNN)
  - `<model>_metrics.json`
  - `<model>_cm.png`

**Run:**

```bash
python src/tadr/EdgeIIoTset/model_training.py \
  --model all \
  --resource edgeiot \
  --train-in path/to/train.csv \
  --output-dir path/to/train_outputs \
  --device auto
```

---

## 5. Baseline Defenses

### KNN (`benchmark_sota.py`)

| Stage | Folder | Reads | Writes |
|---|---|---|---|
| Noise injection | `noise_{pct}/00_symmetric_label_noise/` | `--clean-input` (clean CSV) | `noise_{pct}_data.csv` + `.log` |
| Sanitization | `noise_{pct}/01_knn_sanitization/` | `noise_{pct}_data.csv` | `noise_{pct}_knn_sanitized.csv` + `_report.json` + `.log` |
| Training | `noise_{pct}/02_model_training/` | `noise_{pct}_knn_sanitized.csv` | per-model `.pkl` / `.pth`, `_metrics.json`, `_cm.png` |
| Aggregation | `<reports>/sota-benign-only/{timestamp}/` | per-noise-rate metrics | `sota_knn_summary.csv` |

---

### Cleanlab (`benchmark_cleanlab_sota.py`)

| Stage | Folder | Reads | Writes |
|---|---|---|---|
| Noise injection | `noise_{pct}/00_symmetric_label_noise/` | `--clean-input` (clean CSV) | `noise_{pct}_data.csv` + `.log` |
| Sanitization | `noise_{pct}/01_cleanlab_sanitization/` | `noise_{pct}_data.csv` | `noise_{pct}_cleanlab_sanitized.csv` + `_report.json` + `.log` |
| Training | `noise_{pct}/02_model_training/` | `noise_{pct}_cleanlab_sanitized.csv` | per-model `.pkl` / `.pth`, `_metrics.json`, `_cm.png` |
| Aggregation | `<reports>/sota_cleanlab_benign_only/{timestamp}/` | per-noise-rate metrics | `sota_cleanlab_summary.csv` |

---

### UQ-LED (`benchmark_uqled_sota.py`)

UQ-LED produces both global and benign-only CSVs in a single run.

| Stage | Folder | Reads | Writes |
|---|---|---|---|
| Noise injection | `noise_{pct}/00_targeted_label_noise/` | `--clean-input` (clean CSV) | `noise_{pct}_data.csv` + `.log` |
| UQ-LED defense | `noise_{pct}/01_uqled_cl_mcd_e/` | `noise_{pct}_data.csv` | `noise_{pct}_uqled_global.csv`, `noise_{pct}_uqled_benign_only.csv`, `_report.json`, `_oof_mcd.npz`, `.log` |
| Global training | `noise_{pct}/02_global_model_training/` | `noise_{pct}_uqled_global.csv` | per-model `.pkl` / `.pth`, `_metrics.json`, `_cm.png` |
| Benign-only training | `noise_{pct}/03_benign_only_model_training/` | `noise_{pct}_uqled_benign_only.csv` | per-model `.pkl` / `.pth`, `_metrics.json`, `_cm.png` |
| Aggregation | `<reports>/sota_uqled/{timestamp}/` | per-noise-rate & per-scope metrics | `sota_uqled_summary.csv` |
