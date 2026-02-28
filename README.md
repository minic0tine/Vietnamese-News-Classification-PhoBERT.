# 📰 Robust News Topic Classification Using PhoBERT

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/vinai/phobert-base)

This repository contains the official implementation of the project: **"Robust News Topic Classification Using Deep Learning Models"**. The research focuses on evaluating the robustness of various architectures, with **PhoBERT** achieving a State-of-the-Art (SOTA) performance for Vietnamese news categorization.

---

## 🚀 Highlights

* **99.71% Accuracy** on the VNTC (Public Benchmark) dataset.
* **91.20% Accuracy** on Cross-domain evaluation (VnExpress), demonstrating high robustness against Domain Shift.
* Comprehensive comparison of 5 architectures: **KimCNN, BiLSTM+Attention, RCNN, Transformer, and PhoBERT**.
* Custom fine-tuning strategy with optimized Dropout (0.6) and AdamW optimizer.

---

## 🏗️ Model Architecture

The core of this project is the fine-tuned **PhoBERT-base** model. We leveraged the pre-trained weights from VinAI and added a custom classification head to capture deep semantic nuances in Vietnamese news headlines.

![PhoBERT Architecture](reports/figures/phobert_structure.png) *(Note: Replace with your image path)*

---

## 📊 Experimental Results

| Model | Public Benchmark (VNTC) | In-domain (VnExpress) | Cross-domain (Robustness) |
| :--- | :---: | :---: | :---: |
| KimCNN | 92.45% | 88.30% | 75.12% |
| BiLSTM + Attention | 94.12% | 90.15% | 79.45% |
| **PhoBERT (Ours)** | **99.71%** | **96.80%** | **91.20%** |

### Training Convergence
The PhoBERT model exhibits lightning-fast convergence compared to baseline CNN/RNN models, stabilizing within the first 500 training steps.

![Loss Curve](reports/figures/loss_curve.png) *(Note: Replace with your image path)*

---

## 📁 Project Structure

```text
.
├── data/               # Dataset links and samples
├── notebooks/          # Colab/Jupyter Notebooks
│   ├── Final_PhoBERT.ipynb
│   └── Baseline_Models.ipynb
├── reports/            # Project Report (PDF) and Figures
└── requirements.txt    # Dependencies
