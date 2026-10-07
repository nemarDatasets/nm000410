# SOZ Resting State SEEG Classification

Deep Learning Framework for Seizure Onset Zone (SOZ) Localization from Resting-State SEEG Signals

This repository contains a de-identified codebase for training and analyzing deep neural networks that classify seizure onset zones (SOZ) vs. non-SOZ brain regions using resting-state stereoelectroencephalography (SEEG) recordings.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation & Dependencies](#installation--dependencies)
4. [File Descriptions](#file-descriptions)
5. [Usage Guide](#usage-guide)
6. [Model Architectures](#model-architectures)
7. [Results & Model States](#results--model-states)
8. [Data Requirements](#data-requirements)
9. [Important Notes](#important-notes)
10. [Citation](#citation)

## Project Overview

### Purpose
This project implements a multi-scale deep learning approach to identify seizure onset zones in epilepsy patients using resting-state SEEG recordings. The models analyze temporal patterns at multiple scales (from milliseconds to seconds) to distinguish SOZ from non-SOZ brain regions.

### Key Features
- Multi-Scale Architecture: Parallel processing pathways capture features at temporal scales from 6ms to 2 seconds
- Patient-Level Cross-Validation: 5-fold cross-validation ensuring patient independence
- Ensemble Predictions: Combines predictions across multiple signal windows and cross-validation folds
- Interpretability: Comprehensive SHAP analysis for model explainability

### Inputs / Outputs
- Input: Resting-state SEEG recordings (500-512 Hz sampling rate)
- Output: Binary classification (SOZ vs. non-SOZ) at the electrode contact level

## Repository Structure

```
SOZ_RestingState/
├── README.md                          # This file
├── REFACTORING_SUMMARY.md            # Detailed refactoring documentation
├── seeg_classify_all.py              # Main training and evaluation script
│
├── models/                            # Neural network architectures
│   ├── basic_msresnet.py             # Basic multi-scale ResNet (3x3, 5x5, 7x7)
│   ├── multi_scale_ori.py            # Optimized multi-scale ResNet (production model)
│   └── dilated_enhanced_resnet.py    # Enhanced ResNet with dilations and attention
│
├── analysis/                          # Visualization and explainability tools
│   ├── plotting.py                   # Performance visualization and statistics
│   ├── plot_shap.py                  # Basic SHAP time-window analysis
│   ├── plot_shap_location.py         # Location-specific SHAP analysis
│   ├── plot_shap_morphology.py       # Morphology clustering SHAP analysis
│   ├── plot_shap_multimodel.py       # Multi-model SHAP comparison
│   └── plot_shap_onelocation.py      # Single-location deep-dive SHAP
│
└── results/                           # Saved model states and results
    ├── location_accuracy.csv         # Aggregate location-level performance
    ├── model_states/                 # Trained PyTorch model weights
    │   ├── fold_1_results_108_perm4_alllocs_v2_hist_model.pth
    │   ├── fold_2_results_108_perm4_alllocs_v2_hist_model.pth
    │   ├── fold_3_results_108_perm4_alllocs_v2_hist_model.pth
    │   ├── fold_4_results_108_perm4_alllocs_v2_hist_model.pth
    │   └── fold_5_results_108_perm4_alllocs_v2_hist_model.pth
    │
    └── results_json/                  # Detailed prediction results
        ├── complete_results_108_perm4_alllocs_v2_hist.json       # Main results
        ├── complete_results_108_perm4_alllocs_v2_hist_RAND.json  # Random control
        └── fold_[1-5]_results_108_perm4_alllocs_v2_hist*.json   # Per-fold results
```

## Installation & Dependencies

### Python Version
- Python 3.8+ recommended

### Core Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
pip install shap
pip install mne braindecode
pip install h5py mat73
pip install optuna ray[tune]
pip install pytorch-grad-cam torchcam
```

### Complete Requirements

```python
# Deep Learning
torch >= 1.10.0
torchvision >= 0.11.0
torchsummary

# Data Processing
numpy >= 1.21.0
pandas >= 1.3.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
h5py >= 3.0.0
mat73 >= 0.59

# Visualization
matplotlib >= 3.4.0
seaborn >= 0.11.0

# Explainability
shap >= 0.40.0
pytorch-grad-cam >= 1.3.0
torchcam >= 0.3.0

# EEG Analysis
mne >= 0.24.0
braindecode >= 0.6.0 # not strictly necessary

# Hyperparameter Optimization
optuna >= 2.10.0
ray[tune] >= 1.9.0
```

## File Descriptions

### Main Training Script

#### `seeg_classify_all.py` 
Primary entry point for model training and evaluation.

Key Components:
- Configuration Section (Lines ~90-120): Set data paths, model parameters, and experiment settings
- AdaptiveEEGAugmentation Class: Signal-adaptive data augmentation
- SPES_Comb Dataset Class: PyTorch dataset for loading SEEG signals
- PatientDataLoader: Custom dataloader ensuring patient-level batching
- Training Functions:
  - `train_cv_splits_earlystopping()`: Main 5-fold CV training loop
  - `evaluate_ensemble_predictions()`: Multi-window ensemble evaluation
  - `calculate_metrics()`: Comprehensive performance metrics
- Ensemble Methods: Combines predictions across time windows and folds

Before Running:
1. Configure data paths (marked with `TODO` comments)
2. Set up patient cross-validation splits
3. Prepare data in required format (see Data Requirements)

### Model Architectures

#### `models/multi_scale_ori.py` - PRIMARY MODEL
OptimizedMSResNet - Production model used in published results.

Architecture: (hypothesis for feature types)
- 5 Parallel Pathways with kernel sizes: 3, 11, 21, 65, 129
  - Path 3: ~6ms receptive field (high-frequency features)
  - Path 11: ~22ms receptive field (gamma/beta rhythms)
  - Path 21: ~42ms receptive field (alpha rhythms)
  - Path 65: ~130ms receptive field (slow oscillations)
  - Path 129: ~258ms receptive field (ultra-slow oscillations)
- Residual Blocks: Each pathway uses residual connections
- Multi-Scale Fusion: Concatenates features from all pathways
- Classification Head: 3-layer MLP with LayerNorm, GELU, Dropout

Key Classes:
- `BasicBlock3x3`, `BasicBlock11x11`, `BasicBlock21x21`, `BasicBlock65x65`, `BasicBlock129x129`
- `OptimizedMSResNet`: Main model class

Typical Parameters:
```python
model = OptimizedMSResNet(
    input_channel=1,          # Single-channel EEG (edit for channel dimension. We used input_channel == 4)
    num_classes=1,            # Binary classification
    initial_channels=32,      # Base channel count
    layers=[1, 1, 1, 1],     # Blocks per stage
    use_se=False              # Squeeze-and-Excitation
)
```

#### `models/basic_msresnet.py`
**MSResNet_Basic** - Simpler multi-scale model with 3x3, 5x5, 7x7 kernels.

Use Case: Baseline comparisons, faster training for prototyping.

#### `models/dilated_enhanced_resnet.py`
**EnhancedMSResNetDilated** - Advanced model with attention mechanisms.

**Features:**
- Dilated convolutions with increasing dilation rates (1, 2, 4, 8)
- Squeeze-and-Excitation (SE) attention blocks
- Bidirectional LSTM for temporal modeling
- Attention-weighted feature aggregation

Use Case: Experimental architecture, not used in main results.

### Analysis & Visualization

#### `analysis/plotting.py`
Comprehensive visualization suite for model performance analysis.

**Main Functions:**
- `plot_violin_comparison()`: Violin plots comparing model performance
- `calculate_confidence_intervals()`: Bootstrap confidence intervals
- `plot_location_accuracy()`: Per-brain-region performance visualization
- `plot_roc_curves()`: ROC curves with AUC metrics
- `statistical_tests()`: Paired t-tests, Wilcoxon tests

#### `analysis/plot_shap.py`
Basic SHAP analysis for time-window attribution.

**Key Class:** `SHAPTimeWindowAnalyzer`
- Identifies high-confidence prediction windows
- Extracts temporal patterns driving predictions
- Visualizes SHAP values over time

#### `analysis/plot_shap_location.py`
Location-specific SHAP analysis.

**Key Class:** `EnhancedSHAPTimeWindowAnalyzer`
- Analyzes SHAP patterns for specific brain regions (e.g., hippocampus, amygdala)
- Compares SOZ vs. non-SOZ patterns within locations
- Regional feature importance

#### `analysis/plot_shap_morphology.py`
Morphology-based SHAP analysis.

**Key Class:** `MorphologySHAPAnalyzer`
- Clusters EEG waveform morphologies using K-means
- Associates morphology clusters with SOZ predictions
- Identifies characteristic waveform patterns in SOZ regions

#### `analysis/plot_shap_multimodel.py`
Multi-model ensemble SHAP comparison.

**Features:**
- Compares SHAP explanations across different model architectures
- Analyzes consistency of feature importance across models
- Regional SHAP analysis for specific brain structures

#### `analysis/plot_shap_onelocation.py`
Single-location deep-dive SHAP analysis.

Use Case: Detailed investigation of model decisions for individual brain regions.


##  Usage Guide

### 1. Data Preparation

Required Data Format:

Your data should be organized as follows:

```
<DATA_DIR>/
├── patient_001/
│   ├── pt_file.mat/
└── patient_002/
    └── ...
```

Signal Format:
- Type: 1D time-series SEEG recordings
- Sampling Rate: 500-512 Hz
- Window Length: 2048 samples (~4 seconds at 512 Hz)
- Normalization: Histogram equalization or z-score normalization
- Label: Binary (0 = non-SOZ, 1 = SOZ)

### 2. Configuration

Edit `seeg_classify_all.py`:

```python
# TODO: Configure these paths for your system
DATA_DIR = "/path/to/your/seeg/data"
PICKLE_DIR = "/path/to/patient/lists"
TS_DATA_DIR = "/path/to/timeseries/data"

# Experiment parameters
EXPERIMENT_NAME = "results_108_perm4_alllocs_v2_hist"
NUM_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
```

### 3. Running Training

```bash
# Full 5-fold cross-validation training
python seeg_classify_all.py

# The script will:
# 1. Load patient cross-validation splits
# 2. Train OptimizedMSResNet for each fold
# 3. Save model checkpoints to results/model_states/
# 4. Save detailed predictions to results/results_json/
# 5. Compute ensemble metrics across folds
```

Expected Output:
- Model checkpoints (`.pth` files) for each fold
- JSON files with detailed per-patient, per-location predictions
- Console output showing training progress and metrics


**Training Configuration:**
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5)
- **Loss:** BCEWithLogitsLoss (class-weighted for imbalanced data)
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Early Stopping:** Patience=15 epochs
- **Augmentation:** Adaptive augmentation (time shift, noise, scaling, filtering)


##  Results & Model States

### Results Files

**Complete Results:** `complete_results_108_perm4_alllocs_v2_hist.json` (74 MB)

**Structure:**
```json
{
  "metrics": {
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1_score": float,
    "auroc": float,
    "auprc": float
  },
  "location_metrics": {
    "hippocampus": {...},
    "amygdala": {...},
    ...
  },
  "patient_predictions": {
    "patient_001": {
      "location_hippocampus": {
        "predictions": [...],
        "labels": [...],
        "probabilities": [...]
      }
    }
  },
  "confusion_matrix": [[TN, FP], [FN, TP]]
}
```

**Random Control Results:** `complete_results_108_perm4_alllocs_v2_hist_RAND.json`
- Same structure as main results
- Labels randomly shuffled to verify model is learning real patterns
- Should show ~50% accuracy if model is working correctly

### Expected Performance Metrics

Based on the most recent training run:

| Metric | Value Range | Description |
|--------|-------------|-------------|
| **Accuracy** | 70-85% | Overall classification accuracy |
| **AUROC** | 0.75-0.90 | Area under ROC curve |
| **AUPRC** | 0.65-0.85 | Area under precision-recall curve |
| **Sensitivity** | 65-80% | True positive rate (recall) |
| **Specificity** | 70-85% | True negative rate |


##  Data Requirements

### Cross-Validation Splits

Required File: Patient cross-validation JSON

```json
[
  {
    "train": ["patient_001", "patient_002", ...],
    "val": ["patient_015", "patient_016", ...],
    "test": ["patient_030", "patient_031", ...]
  },
  {
    "train": [...],
    "val": [...],
    "test": [...]
  },
  ...  # 5 folds total
]
```

Important: Splits must be patient-independent (no patient appears in multiple splits).

##  Contact

For questions, issues, or collaboration:

- **Email:** sameer.sundrani@vanderbilt.edu



**Last Updated:** October 6, 2025
**Version:** 1.0.0
