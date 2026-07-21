# Edge-IIoTset Ablation Study Guide

This document describes the **Edge-IIoTset** ablation pipeline inside `src/apelid/EdgeIIoTset/`.

It is part of the larger TADR ablation framework. For the top-level description, see the [project README](../../../../README.md).

---

## Table of contents

1. [Pipeline overview](#1-pipeline-overview)
2. [How to run the full pipeline](#2-how-to-run-the-full-pipeline)
3. [Stage-by-stage I/O details](#3-stage-by-stage-io-details)
   - 3.1 [`preprocessing.py`](#31-preprocessingpy)
   - 3.2 [`symmetric_label_noise.py`](#32-symmetric_label_noisepy)
   - 3.3 [`dae_kmeans_knn_benign_filter.py`](#33-dae_kmeans_knn_benign_filterpy)
   - 3.4 [`dnn_recover_grid.py`](#34-dnn_recover_gridpy)
   - 3.5 [`model_training.py`](#35-model_trainingpy)

---

## 1) Pipeline overview

**Main runner:** `ablation_pipeline.py`

**Execution order for each noise rate:**

1. `symmetric_label_noise.py` — inject asymmetric targeted label-flipping attack
2. `model_training.py` on the noisy data
3. `dae_kmeans_knn_benign_filter.py` — TADR defense (DAE + KMeans + weighted KNN)
4. `model_training.py` on the DAE-cleaned data
5. `dnn_recover_grid.py` — DNN-based recovery of high-confidence samples
6. `model_training.py` on the recovered-clean data

---

## 2) How to run the full pipeline

```bash
python src/apelid/EdgeIIoTset/ablation_pipeline.py \
  --noise-rates 30,40,50,60,70 \
  --kmeans-budget 15000 \
  --k-neighbors 15 \
  --log-dir src/apelid/EdgeIIoTset/reports/ablation_logs
```

The pipeline trains six models: **xgb, catb, bagging, lgbm, rf, dnn**.

---

## 3) Stage-by-stage I/O details

### 3.1 `symmetric_label_noise.py`

**What it is:** Inject targeted noise by flipping non-`Normal` labels to `Normal`.

**Input:** Clean CSV (pipeline passes `--clean-input`).

**Output:**
- `noise_{pct}_data.csv` (with `original_label` and `is_noisy` columns when `--keep-tracking` is on)
- `noise_{pct}_symmetric_label_noise.log`

**Run:**

```bash
python src/apelid/EdgeIIoTset/symmetric_label_noise.py \
  --input src/apelid/EdgeIIoTset/resources/edgeiot/clean_merged/edgeiot_train_clean_merged.csv \
  --noise-rate 0.30 \
  --keep-tracking \
  --seed 42 \
  --output src/apelid/EdgeIIoTset/reports/noise_30_data.csv
```

---

### 3.2 `dae_kmeans_knn_benign_filter.py`

**What it is:** TADR defense stage (DAE embeddings + KMeans benign-core + weighted KNN detection).

**Input:**
- `--input <noisy.csv>`
- `--kmeans-budget <n>`
- `--k-neighbors <k>`

**Output:**
- `<name>_clean.csv`
- `<name>_noise.csv`
- `noise_detection_report.json`
- `noise_detection_per_label.csv` (when original labels available)
- `*_dae_kmeans_knn.log`

**Run:**

```bash
python src/apelid/EdgeIIoTset/dae_kmeans_knn_benign_filter.py \
  --input path/to/noisy.csv \
  --kmeans-budget 15000 \
  --k-neighbors 15 \
  --output-dir path/to/dae_outputs \
  --name noise_30
```

---

### 3.3 `dnn_recover_grid.py`

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
python src/apelid/EdgeIIoTset/dnn_recover_grid.py \
  --clean-in path/to/dae_clean.csv \
  --noise-in path/to/dae_noise.csv \
  --output-dir path/to/recover_outputs \
  --name noise_30
```

---

### 3.4 `model_training.py`

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
python src/apelid/EdgeIIoTset/model_training.py \
  --model all \
  --resource edgeiot \
  --train-in path/to/train.csv \
  --output-dir path/to/train_outputs \
  --device auto
```

> `--output-dir` is required. Per-model hyperparameters live in `params/training/*.json`.