# SPADE

**Scary Prompt Assessment and Dataset Evaluation**

A unified benchmark and empirical study of malicious prompt detection.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Author:** Sarthak Deore · NMIMS · sarthakdeore265@gmail.com

---

## Overview

SPADE unifies **38,814 prompts from 9 public datasets** into a single benchmark
for evaluating malicious prompt detection. It scores each source on a
**Dataset Quality Index (DQI)**, evaluates two baselines (TF-IDF + Logistic
Regression and fine-tuned DistilBERT) under random and leave-one-source-out
splits, and measures how classification metrics disagree on model ranking.

The repository includes the full pipeline, results, and manuscript.

---

## Key findings

1. **Accuracy and F1 rank models almost independently** (Spearman ρ = 0.086).
   MCC and AUPRC are perfectly rank-equivalent (ρ = 1.000).

2. **Cross-source performance collapses.** F1 drops from 0.698 (random split)
   to 0.069 on the hardest unseen source — a 0.63 collapse.

3. **DQI predicts AUPRC (ρ = 1.000) but not F1 (ρ = 0.700) or
   recall@1%FPR (ρ = 0.000).** Robust to weight perturbation (Kendall τ ≥ 0.833).

4. **A 60-word lexical classifier reproduces 59.5% of the full model's F1.**
   Two symmetric failure modes emerge: 46% detection on prompt injections,
   57.7% false-positive rate on safe-but-scary prompts.

5. **DistilBERT improves injection detection (0.460 → 0.560) but worsens
   over-refusal (0.577 → 0.615)** and loses to TF-IDF on XSTest
   (McNemar p = 0.043). The lexical shortcut persists under contextual encoding.

---

## Repository structure
