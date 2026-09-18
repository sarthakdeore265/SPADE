# SPADE - Scary Prompt Assessment and Dataset Evaluation

**A benchmark and empirical study of malicious prompt detection.**

Sarthak Deore - NMIMS - September 2026

## Overview

SPADE unifies 38,814 prompts from 9 public datasets into a single benchmark,
scores each source on a Dataset Quality Index (DQI), and evaluates two
baselines - TF-IDF + Logistic Regression and fine-tuned DistilBERT - under
random and leave-one-source-out splits.

## Key findings

1. Accuracy and F1 rank models almost independently (Spearman rho = 0.086).
2. Cross-source performance collapses - F1 drops from 0.698 to 0.069 on the
   hardest unseen source.
3. DQI predicts AUPRC (rho = 1.000) but not F1 or recall@1%FPR.
4. A 60-word lexical classifier reproduces 59.5% of the full model's F1,
   exposing symmetric failures: 46% detection on injections, 57.7% FPR on
   safe-but-scary prompts.
5. DistilBERT worsens over-refusal (57.7% -> 61.5%) and loses to TF-IDF on
   XSTest (McNemar p = 0.043).

## Repository structure

    SPADE/
    |-- README.md
    |-- LICENSE
    |-- CITATION.cff
    |-- requirements.txt
    |-- .gitignore
    |-- data/
    |-- docs/
    |-- src/
    |-- figures/
    |-- results/
    |-- notebooks/
    |-- paper/

## Quickstart

    git clone https://github.com/sarthakdeore265/SPADE.git
    cd SPADE
    pip install -r requirements.txt
    python src/01_download_data.py
    python src/02_build_benchmark.py
    python src/03_compute_dqi.py
    python src/04_train_tfidf.py
    python src/05_train_bert.py
    python src/06_metric_agreement.py
    python src/07_error_analysis.py
    python src/08_lexical_ablation.py
    python src/09_dqi_sensitivity.py
    python src/10_mcnemar.py

## Datasets

See docs/DATASETS.md for the 9 sources, download URLs, and licenses.

## Environment

- Python 3.10+
- See requirements.txt
- DistilBERT training requires a GPU (tested on Tesla T4 in Google Colab)

## Reproducibility

- All random splits use seed 42
- Deterministic training given the seed
- DQI component values in results/dqi.csv

## Citation

    @article{deore2026spade,
      title={SPADE: Evaluating Machine Learning Models for Malicious Prompt Detection},
      author={Deore, Sarthak},
      year={2026},
      url={https://github.com/sarthakdeore265/SPADE}
    }

## License

MIT License - see LICENSE.
