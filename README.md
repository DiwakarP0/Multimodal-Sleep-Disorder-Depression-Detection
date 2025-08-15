# Multimodal Sleep Disorder & Depression Prediction

This project leverages **multimodal machine learning** to predict sleep disorders and depression using data from speech (audio), text (transcripts), and facial video features. It utilizes a subset of the EDAIC dataset, aiming to develop interpretable and robust models for mental health screening.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Text-Based Modeling](#text-based-modeling)
  - [Speech-Based Modeling](#speech-based-modeling)
  - [Fusion & Ensemble](#fusion--ensemble)
- [Results](#results)
- [Key Scripts & Notebooks](#key-scripts--notebooks)
- [Team](#team)


## Overview

This repository implements a pipeline for **sleep disorder and depression prediction** using various modalities. It demonstrates:
- Feature extraction (audio, text, face)
- Sequence modeling using BiLSTM, CNN-BiLSTM, TCN, and attention mechanisms
- Text modeling via decision trees, random forests, XGBoost, and boosting
- Fusion strategies (late fusion/ensemble methods)
- Evaluation with confusion matrices, accuracy curves, F1-score, and class-balanced metrics


## Problem Statement

Predict the presence and severity of **sleep disorders** (binary and multi-class) and **depression** (binary) based on:
1. **Speech Features** (audio-derived temporal features)
2. **Transcript Features** (linguistic & timing cues from interviews)
3. **Facial Video Features**
4. **Fusion** of modalities for improved robustness

## Dataset

- **Source:** EDAIC (Emotionally Diverse Audio-Visual Interactions Corpus)
- **Participants:** 275
- **Modalities:** Speech (.csv features), Text (transcripts), Facial features (OpenFace)
- **Labels:** Derived from PHQ-8 and PCL-C for sleep, depression scales
- **File Types:** `.csv`, `.xlsx`, `.docx`, `.pdf`, Jupyter notebooks

## Project Structure

```plaintext
Report_Sleep_Disorder_Prediction.docx     # Summary of sleep audio DL models
Sleep-CNN_Bi_LSTM_Attention_with_wighted_loss.ipynb  # Audio seq model notebook
Sleep-BEST_Bi_LSTM_with-wighted_loss.ipynb           # Audio best BiLSTM notebook
Depression-CNN_BiLSTM_Attension_with_weighted_loss.ipynb  # Depression audio model
Depression-BEST_Bi_LSTM_with_wighted_loss.ipynb           # Depression best BiLSTM
TCNModel_on_CGS_server.py                   # TCN model script for sleep
L1BLSTM_Mlti.py                             # Multiclass BiLSTM code
text.py, text_final_modify_py.py             # Transcript feature engineering and modeling
Late_fusion_-depression.ipynb, Late_fusion.ipynb    # Fusion ensemble notebooks
audio_test_predictions_final_final.xlsx      # Audio test probabilities for fusion
Unseen-Predictions-with-Uncertainty.csv      # Face model predictions with uncertainty
text_probabilities.xlsx                      # Text model probabilities for fusion
prediction.py                               # Script for transcript-based prediction
audio_to_transcript.py                      # Whisper-based automatic audio transcription
Text.pdf, CGS616-A3-Report-1.pdf            # Project and text modality documentation
```
## Installation & Setup

1. **Clone the repository and place all files in the same working directory.**

2. **Environment Preparation:**
   - Python 3.8+
   - Recommended: [Anaconda](https://www.anaconda.com/) for package management
   - Required libraries (colab/CGS server):  
     - `torch`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`
     - For Whisper audio transcription: `git+https://github.com/openai/whisper.git`, `ffmpeg-python`

3. **Install dependencies:**
```bash
pip install torch pandas numpy scikit-learn xgboost matplotlib seaborn joblib
pip install git+https://github.com/openai/whisper.git
pip install ffmpeg-python
```

## Usage

### Preprocessing

- **Audio Features:**  
  Standardize, pad sequences, and extract MFCC/eGeMAPS features per participant
- **Text Features:**  
  Extract timing, lexical, and Bag-of-Words representations, then apply PCA
- **Label Handling:**  
  Use provided survey scores (PHQ-8, PCL-C) for binary/multi-class labels

### Text-Based Modeling

- Feature engineering: See `text_final_modify_py.py`
- Models tried: Decision Tree, Random Forest, XGBoost, HistGradientBoosting
- Cross-validation for robustness
- Top features plotted from PCA and XGBoost

### Speech-Based Modeling

- Sequence models tested on audio features:
  - **BiLSTM:** Best overall performance (highest test accuracy)
  - **CNN + BiLSTM**
  - **Weighted CNN + BiLSTM**
  - **CNN + BiLSTM + Attention**
  - **TCN (Temporal Convolutional Network)**
- Training tricks: Dropout, layer normalization, early stopping, gradient clipping

### Face-Based Modeling

- CSV files with pose/gaze features for each participant
- Analysis with MLP and uncertainty scoring

### Fusion & Ensemble

- **Late Fusion:** Average probabilities of text, audio, and face models
- **Scripts:**  
  - `Late_fusion_-depression.ipynb`, `Late_fusion.ipynb`
  - Merge all modality probabilities on `Participant_ID`
  - Final predictions: `avg_prob_1 > avg_prob_0`
  - Evaluation with accuracy, classification report, confusion matrix


## Results

| Task                        | Best Accuracy   | Model / Approach                        |
|-----------------------------|----------------|-----------------------------------------|
| Sleep Disorder (Binary)     | 0.6945         | Random Forest (text, cross-validation)  |
| Sleep Disorder (Multi-class)| 0.39           | XGBoost (text, multiclass)              |
| Depression (Binary)         | 0.64           | DecisionTree (text)                     |
| Audio (Single-layer BiLSTM) | ~0.79          | Best confusion matrix & F1-score        |
| Multi-modal Fusion          | 0.79 - 0.85    | Ensemble (audio+face+text fusion)       |

- See confusion matrices, loss/accuracy/F1 plots inside notebooks and attached images.
- Multimodal fusion improves robustness over unimodal models.

## Key Scripts & Notebooks

- **text.py, text_final_modify_py.py:** Text feature extraction & modeling
- **Sleep-BEST_Bi_LSTM_with-wighted_loss.ipynb:** Best audio model implementation
- **Late_fusion_-depression.ipynb, Late_fusion.ipynb:** Ensemble fusion for test predictions
- **audio_to_transcript.py:** Generate transcript from `.wav` using Whisper
- **prediction.py:** Example pipeline for test-time transcript prediction


## Team

- **Prem Kansagra:** Speech DL modeling, fusion, face MLP
- **Poojal Katiyar:** Text feature engineering, modeling, report writing
- **Ahmad Raza:** Audio model design, TCN implementation, pipeline integration
- **Nishita Gupta:** Dataset extraction, transcript scripting
- **Diwakar Prajapati:** Sequence model tuning, evaluation reporting


## References

For full documentation, see:
- `CGS616-A3-Report-1.pdf` – Main project report (methodology, results, applications)
- `Text.pdf` – Text modality analysis summary
