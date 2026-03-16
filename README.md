# 🚀 Spacecraft Telemetry Anomaly Detection
**ISRO Vikram Sarabhai Space Centre (VSSC) — Internship Project**

> An end-to-end deep learning pipeline for **unsupervised anomaly detection** in spacecraft telemetry using LSTM Autoencoders, validated on NASA's SMAP & MSL benchmark dataset.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Stage-by-Stage Breakdown](#-stage-by-stage-breakdown)
- [LSTM Autoencoder Architecture](#-lstm-autoencoder-architecture)
- [Training](#-training)
- [Threshold Detection](#-threshold-detection)
- [Results](#-results)
- [Discussion & Limitations](#-discussion--limitations)
- [Setup & Installation](#-setup--installation)
- [Project Structure](#-project-structure)
- [Key Design Decisions](#-key-design-decisions)
- [References](#-references)

---

## 🛰 Project Overview

Spacecraft generate continuous streams of telemetry — sensor readings, command states, engineering measurements — that must be monitored for anomalies in real time. Manual inspection at scale is infeasible. This project builds a fully **unsupervised anomaly detection system** that trains only on nominal (healthy) operation data, then flags windows where reconstruction error exceeds a learned threshold.

**Why unsupervised?** In real missions, labeled anomaly data is rare, expensive, and often unavailable until after post-incident analysis. An autoencoder learns the normal manifold from healthy data alone — no anomaly labels required during training.

**Core mechanism:**
```
Train LSTM Autoencoder on normal sequences only
         ↓
At inference: reconstruct each test window
         ↓
Anomaly score = MSE(input, reconstruction)
         ↓
Flag window if score > threshold
```

**Project highlights:**
- 7-stage modular notebook pipeline, fully reproducible on Google Colab
- Strict no-leakage guarantee — scaler fitted on training data only, verified by assertion
- Data-driven window size via Autocorrelation Function (ACF) analysis
- Dual thresholding strategies compared: 99th Percentile vs Dynamic EWM
- Full evaluation suite: F1, ROC-AUC, PR-AUC, Confusion Matrix

---

## 📡 Dataset

**NASA Anomaly Detection Dataset — SMAP & MSL**
Source: [Kaggle / Patrick Fleith](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl) · Downloaded via KaggleHub

| Property | Value |
|---|---|
| Spacecraft | SMAP (Soil Moisture Active Passive) + MSL (Mars Science Laboratory) |
| Total channels | **82** (train) / **82** (test) |
| Format | `.npy` multivariate time series arrays |
| Ground truth | `labeled_anomalies.csv` — anomaly intervals as `[start, end]` index pairs |
| Primary channel used | **P-1** (SMAP) |

**Channel P-1 dimensions:**

| Split | Shape | Meaning |
|---|---|---|
| Train | (2,872 × 25) | 2,872 timesteps, 25 raw features |
| Test | (8,505 × 25) | 8,505 timesteps, 25 raw features |

The test set contains **3 labeled anomaly intervals** across ~8,500 timesteps — a realistic, heavily imbalanced scenario (~10% anomaly rate by window count).

### Test Signal with Ground Truth Anomaly Regions

![P-1 Test Telemetry with Anomaly Intervals](P1_data_check.png)

> The top panel shows Feature 0 of the normalized test signal. The 3 red-shaded regions are the labeled anomaly intervals — notably, the anomalies manifest as **suppression of signal amplitude and variance** rather than large spikes. The bottom panel shows the corresponding binary ground-truth label array.

---

## 🏗 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   RAW TELEMETRY (.npy)                      │
│              Train: (2872, 25)  Test: (8505, 25)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   01  DATA LOADER       │
              │  Load · Validate · Label │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   02  PREPROCESSING     │  Drop 9 const cols → Normalize
              │  (2872,25)→(2872,96)    │  → Delta → Rolling features
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   03  FEATURE ENG.      │  ACF → window=41
              │  Sequences (N,41,96)    │  Sliding window, step=1
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   04  MODEL DEV.        │  LSTM Autoencoder
              │  Encoder→Latent→Decoder │  ~620K parameters
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   05  TRAINING          │  100 epochs · Best @ ep.98
              │  val_loss: 0.004983     │  ReduceLROnPlateau
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   06  THRESHOLDING      │  Dynamic EWM
              │  threshold = 0.01540    │  μ_ewm + 3σ_ewm
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   07  EVALUATION        │  F1=0.095 · ROC-AUC=0.467
              │  Confusion · ROC · PR   │  PR-AUC=0.097
              └─────────────────────────┘
```

---

## 📓 Stage-by-Stage Breakdown

### Stage 01 — Data Loader

Sets up project structure and provides reusable data access functions:

| Function | Purpose |
|---|---|
| `load_channel(channel_id)` | Load train/test `.npy` arrays with shape + NaN assertions |
| `load_anomaly_labels(channel_id)` | Parse `labeled_anomalies.csv` → `[start, end]` interval list |
| `build_label_array(intervals, n)` | Binary timestep-level ground truth array |
| `build_window_label_array(intervals, n, w)` | Window-level ground truth for evaluation |
| `list_channels(spacecraft)` | All channel IDs, optionally filtered by spacecraft |
| `get_dataset_summary()` | Aggregate statistics grouped by spacecraft |

---

### Stage 02 — Preprocessing

A strict **no-leakage pipeline** — all statistics derived exclusively from training data.

| Step | Transform | Train shape | Test shape |
|---|---|---|---|
| Raw input | — | (2872, 25) | (8505, 25) |
| Drop constant cols | Remove 9 zero-variance features (`std < 1e-6`) | (2872, 16) | (8505, 16) |
| MinMax Normalize | Scale to `[0, 1]` — **fit on train only** | (2872, 16) | (8505, 16) |
| Delta features | Append first-order differences | (2872, 32) | (8505, 32) |
| Rolling features | Append 10-step rolling mean + std | **(2872, 96)** | **(8505, 96)** |

`Train min: 0.0 · Train max: 1.0` — normalization verified by assertion.

**Leakage guard:** A dedicated cell asserts `np.allclose(train_min, scaler.data_min_)` — confirms the scaler was never exposed to test data. ✅

![processing image](https://github.com/Monisha325/Spacecraft-Telemetry-Anomaly-Detection-using-LSTM-Autoencoders/blob/main/results/plots/P1_preprocessing.png?raw=true)

> Top: raw Feature 0 (train=blue, test=orange). Middle: after MinMax normalization. Bottom: delta feature on test set with anomaly regions highlighted — the suppressed-variance anomalies are visible as low-activity windows in the delta signal.

---

### Stage 03 — Feature Engineering

**Window size selection via ACF:**

The Autocorrelation Function is computed for lags 1–100 across all 96 features. The window size is set to the maximum lag at which any feature's ACF first crosses within the 95% confidence band `±1.96 / √n`.

![ACF — Window Size Selection](P1_acf.png)

> Feature 0 (blue) exhibits strong periodic autocorrelation that persists to high lags — evidence of a dominant oscillatory pattern in P-1's primary telemetry channel. Features 1 and 2 decorrelate by lag ~5. The window is set conservatively to capture the full temporal context of the slowest-decorrelating feature. **Selected window size: 41 timesteps.**

**Sequence construction:**

| Output | Shape | Memory |
|---|---|---|
| Train sequences | **(2547, 41, 96)** | 76.5 MB |
| Validation sequences | **(284, 41, 96)** | 8.5 MB |
| Test sequences | **(8464, 41, 96)** | 254.2 MB |

Sliding window with `step=1` — maximum overlap for dense anomaly scoring. A **chronological 90/10 train/val split** (no shuffling) preserves temporal ordering.

![Sequence Sample](P1_sequence_sample.png)

> Top: full normalized test signal (Feature 0). Middle: a single highlighted window at the onset of anomaly interval 1. Bottom: feature heatmap of that window across all 96 features — most features are near-zero (teal) with a few high-activity channels visible as bright columns.

**5-point integrity check — all PASS:**

| Check | Result |
|---|---|
| First window matches raw data | ✅ PASS |
| Train / val non-overlapping | ✅ PASS |
| No NaN in any split | ✅ PASS |
| All arrays are 3D | ✅ PASS |
| Test window matches raw test data | ✅ PASS |

---

### Stage 04 — Model Development (see [LSTM Autoencoder Architecture](#-lstm-autoencoder-architecture) below)

---

### Stage 05 — Training

| Parameter | Value |
|---|---|
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping | patience=10, restore best weights |
| LR reduction | ReduceLROnPlateau — factor=0.5, patience=5, min_lr=1e-6 |

![Training Curves](P-1_training_curves.png)

| Metric | Value |
|---|---|
| Best epoch | **98** |
| Best val_loss | **0.004983** |
| Final train_loss | **0.002249** |
| LR at convergence | ~1.95e-6 (reduced 9× from initial) |

The model trained for the full 100 epochs with consistent improvement. The learning rate was reduced 9 times by `ReduceLROnPlateau`, indicating the optimizer progressively refined the minimum. The train/val loss gap is stable and small — no significant overfitting. Both losses plateau smoothly from epoch ~75 onward.

---

## 🧠 LSTM Autoencoder Architecture

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INPUT          (batch, 41 timesteps, 96 features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ENCODER
  LSTM(128, tanh, return_sequences=True)
  Dropout(0.2)
  LSTM(64,  tanh, return_sequences=True)
  Dropout(0.2)
  LSTM(32,  tanh, return_sequences=False)   ← bottleneck
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BRIDGE
  RepeatVector(41)         ← expand latent → sequence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DECODER
  LSTM(32,  tanh, return_sequences=True)
  Dropout(0.2)
  LSTM(64,  tanh, return_sequences=True)
  Dropout(0.2)
  LSTM(128, tanh, return_sequences=True)
  TimeDistributed(Dense(96))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT         (batch, 41 timesteps, 96 features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Total parameters: ~620,000
 Output shape == Input shape  ✅ verified
```

**Three configurations evaluated:**

| Config | Latent Dim | Dropout | Parameters | Selected |
|---|---|---|---|---|
| Small | 16 | 0.1 | ~350K | |
| **Medium** | **32** | **0.2** | **~620K** | ✅ |
| Large | 64 | 0.3 | ~1.1M | |

Medium chosen as the optimal tradeoff between expressiveness and overfitting risk for a 2,547-sample training set.

---

## 🎯 Threshold Detection

Anomaly score per window = **mean squared error between input and reconstruction** across all timesteps and features.

Two strategies compared on training error distribution:

| Method | Formula | Threshold Value |
|---|---|---|
| Percentile (99th) | `np.percentile(train_errors, 99)` | ~0.01430 |
| **Dynamic EWM** ✅ | `μ_ewm + 3σ_ewm` (span=10) | **0.01540** |

Dynamic EWM selected — more robust to non-stationarity and slow signal drift over mission lifetime.

![Reconstruction Error Distribution](P-1_error_dist.png)

> The training error distribution is bimodal. The dominant cluster near `10⁻³` represents well-reconstructed normal windows. The secondary cluster near `10⁻²` reflects higher-complexity normal windows (e.g. high-variance bursts, pattern transitions). Both thresholds correctly sit beyond the right tail of this distribution.

---

## 📊 Results

### Evaluation Metrics — Channel P-1

| Metric | Value |
|---|---|
| **Precision** | 0.149 |
| **Recall** | 0.070 |
| **F1 Score** | 0.095 |
| **ROC-AUC** | 0.467 |
| **PR-AUC** | 0.097 |

### Confusion Matrix

![Confusion Matrix](P-1_cm.png)

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actual Normal** | 7,318 (TN) | 326 (FP) |
| **Actual Anomaly** | 763 (FN) | 57 (TP) |

### ROC Curve

![ROC Curve — AUC=0.467](P-1_roc.png)

### Precision-Recall Curve

![Precision-Recall Curve — AP=0.097](P-1_pr_curve.png)

---

## 🔎 Discussion & Limitations

### Honest Assessment

The results on channel P-1 indicate that the model **did not successfully learn to discriminate anomalous from normal windows on this specific channel**. ROC-AUC of 0.467 is marginally below random chance, and recall of 7% means 763 of 820 anomaly windows were missed. Understanding exactly why this happened is as valuable as a high-performing result.

### Root Cause Analysis

**1. Anomaly type mismatch with reconstruction-based detection.**
Looking at the data visualization, P-1's anomalies are **suppression-type** — the signal becomes quieter and loses variance during anomalous periods. An autoencoder trained on noisy, high-amplitude normal data reconstructs these quieter anomaly windows *more easily*, producing lower reconstruction errors — directly inverting the detection assumption. Reconstruction-based methods are well-suited for spike/deviation anomalies, but underperform on variance-suppression anomalies.

**2. Sequence reshape side-effect.**
In Stage 05, sequences required `expand_dims + repeat` to match the model's expected input shape. This introduced artificial temporal repetition that the model may have optimized around, reducing sensitivity to genuine temporal patterns.

**3. The model converges, but to a globally low MSE.**
Final train MSE of `0.002249` reflects a model that reconstructs nearly everything well — collapsing the reconstruction error margin between normal and anomalous classes below the detection threshold.

**4. Class imbalance at threshold selection.**
The dynamic threshold is derived purely from training error statistics, with no feedback from the anomaly class. On a heavily imbalanced test set, this heuristic produces a threshold that is not optimally placed for F1.

### Why PR-AUC is the Right Metric Here

P-1 has ~820 anomaly windows out of 8,464 total (~9.7% anomaly rate). Under this imbalance, a model predicting "normal" for everything achieves ROC-AUC ≈ 0.5, making ROC-AUC a poor discriminator of model quality. PR-AUC collapses toward the base rate (~0.097) when the model fails — which is exactly what is observed, confirming both metrics are self-consistent and the baseline has not been beaten on this channel.

### Proposed Improvements

| Improvement | Targets |
|---|---|
| **Prediction-based LSTM** — forecast the next timestep; score = forecast error | Directly sensitive to variance suppression anomalies |
| **Variance-aware anomaly score** — penalize windows where predicted variance exceeds actual | Explicitly designed for suppression-type anomalies |
| **Semi-supervised threshold tuning** on a small held-out labeled set | Directly optimize F1 instead of using a statistical heuristic |
| **Isolation Forest / One-Class SVM on latent vectors** | Learns a tighter normal boundary in compressed space |
| **Multi-channel evaluation** on M-1, D-14, P-15, etc. | P-1 is documented as a hard channel; pipeline-level metrics give a fairer picture |
| **Contrastive or energy-based training** | Explicitly pushes anomalous reconstructions to higher error |

---

## ⚙️ Setup & Installation

**Requirements:** Python 3.8+ · Google Colab (recommended) · Google Drive

```bash
pip install -r requirements.txt
```

**Dataset download** (handled automatically in `01_data_loader.ipynb`):
```bash
pip install kagglehub
```

---

## 🚀 Usage

Run notebooks in order from Google Colab:

```
01_data_loader.ipynb           Download & structure dataset (82 channels)
02_preprocessing.ipynb         Clean, normalize & engineer features
03_feature_engineering.ipynb   ACF window selection + sliding sequences
04_model_development.ipynb     Define LSTM Autoencoder (~620K params)
05_training.ipynb              Train · best epoch 98 · val_loss 0.004983
06_threshold_detection.ipynb   Score windows · threshold = 0.01540
07_evaluation.ipynb            Metrics · confusion matrix · ROC · PR curve
```

To run on a different channel:
```python
channel_id = "M-1"    # change at the top of each notebook
```

---

## 📁 Project Structure

```
rocket_telemetry_project/
│
├── data/
│   ├── train/                       # 82 raw .npy train channel files
│   ├── test/                        # 82 raw .npy test channel files
│   ├── labeled_anomalies.csv        # Ground truth anomaly intervals
│   ├── processed/
│   │   ├── P-1_train.npy            # (2872, 96) — after full preprocessing
│   │   └── P-1_test.npy             # (8505, 96)
│   └── sequences/
│       ├── P-1_train_seq.npy        # (2547, 41, 96)
│       ├── P-1_val_seq.npy          # (284,  41, 96)
│       └── P-1_test_seq.npy         # (8464, 41, 96)
│
├── models/
│   ├── architecture_P1.json         # Model architecture (JSON)
│   ├── lstm_ae_P-1.h5               # Best model weights (epoch 98)
│   └── scaler_P-1.pkl               # MinMaxScaler (train-only fit)
│
├── results/
│   ├── plots/
│   │   ├── P1_data_check.png        # Test signal + anomaly regions
│   │   ├── P1_preprocessing.png     # Raw → normalized → delta
│   │   ├── P1_acf.png               # ACF for window size selection
│   │   ├── P1_sequence_sample.png   # Window + feature heatmap
│   │   ├── P-1_training_curves.png  # Loss + LR decay over 100 epochs
│   │   ├── P-1_error_dist.png       # Reconstruction error distribution
│   │   ├── P-1_roc.png              # ROC curve (AUC=0.467)
│   │   ├── P-1_pr_curve.png         # PR curve (AP=0.097)
│   │   └── P-1_cm.png               # Confusion matrix (F1=0.095)
│   └── metrics/
│       ├── P-1_history.json         # Loss per epoch (100 epochs)
│       ├── P-1_threshold.json       # Threshold=0.01540, method=dynamic
│       └── P-1_eval.json            # TP=57, FP=326, TN=7318, FN=763
│
├── src/
│   ├── __init__.py
│   └── data_loader.py               # Reusable channel loading module
│
├── notebooks/                       # 7 ordered Colab notebooks
├── requirements.txt
└── README.md
```

---

## 🔍 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Unsupervised autoencoder** | No anomaly labels required — trains on normal data only, mirrors real mission constraints |
| **Scaler fit on train only** | Strict no-leakage guarantee — verified by assertion, not assumed |
| **ACF for window size** | Data-driven selection — captures actual temporal dependency length per channel |
| **Chronological val split** | Shuffling leaks future context — forbidden for time series evaluation |
| **Delta + rolling feature augmentation** | Exposes rate-of-change and local volatility to the model beyond raw values |
| **MSE reconstruction loss** | Penalizes large deviations from normal patterns; sensitive to structural outliers |
| **Dynamic EWM threshold** | Adapts to signal non-stationarity; more robust than a fixed percentile over mission lifetime |
| **PR-AUC as primary metric** | Correct choice under heavy class imbalance — ROC-AUC is misleading when anomalies are rare |

---

## 📚 References

1. Hundman, K., et al. (2018). **Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.** *KDD 2018.* [arXiv:1802.04893](https://arxiv.org/abs/1802.04893)
2. Malhotra, P., et al. (2016). **LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.** *ICML Anomaly Detection Workshop.*
3. Hochreiter, S. & Schmidhuber, J. (1997). **Long Short-Term Memory.** *Neural Computation, 9(8),* 1735–1780.
4. NASA SMAP/MSL Anomaly Dataset — [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

---

## 👨‍💻 About This Project

**Institution:** ISRO Vikram Sarabhai Space Centre (VSSC)
**Domain:** Deep Learning · Time Series · Anomaly Detection · Spacecraft Health Monitoring
**Dataset:** NASA SMAP & MSL — 82 telemetry channels, real mission data

---

*TensorFlow/Keras · Scikit-learn · NumPy · Pandas · Matplotlib · Seaborn · KaggleHub*
