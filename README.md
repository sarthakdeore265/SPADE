# SPADE

**Scary Prompt Assessment and Dataset Evaluation**

A unified benchmark and empirical study of malicious prompt detection.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Author:** Sarthak Deore - NMIMS - sarthakdeore265@gmail.com

---

## Overview

SPADE unifies **38,814 prompts from 9 public datasets** into a single benchmark for evaluating malicious prompt detection. It scores each source on a **Dataset Quality Index (DQI)**, evaluates two baselines (TF-IDF + Logistic Regression and fine-tuned DistilBERT) under random and leave-one-source-out splits, and measures how classification metrics disagree on model ranking.

The repository includes the full pipeline, results, and manuscript.

---

## Key findings

1. **Accuracy and F1 rank models almost independently** (Spearman rho = 0.086). MCC and AUPRC are perfectly rank-equivalent (rho = 1.000).

2. **Cross-source performance collapses.** F1 drops from 0.698 (random split) to 0.069 on the hardest unseen source - a 0.63 collapse.

3. **DQI predicts AUPRC (rho = 1.000) but not F1 (rho = 0.700) or recall@1%FPR (rho = 0.000).** Robust to weight perturbation (Kendall tau >= 0.833).

4. **A 60-word lexical classifier reproduces 59.5% of the full model's F1.** Two symmetric failure modes emerge: 46% detection on prompt injections, 57.7% false-positive rate on safe-but-scary prompts.

5. **DistilBERT improves injection detection (0.460 to 0.560) but worsens over-refusal (0.577 to 0.615)** and loses to TF-IDF on XSTest (McNemar p = 0.043). The lexical shortcut persists under contextual encoding.

---

## Repository structure

    SPADE/
    |-- README.md
    |-- LICENSE
    |-- CITATION.cff
    |-- requirements.txt
    |-- .gitignore
    |-- src/
    |   |-- 01_download_data.py
    |   |-- 02_build_benchmark.py
    |   |-- 03_compute_dqi.py
    |   |-- 04_train_tfidf.py
    |   |-- 05_train_bert.py
    |   |-- 06_metric_agreement.py
    |   |-- 07_error_analysis.py
    |   |-- 08_lexical_ablation.py
    |   |-- 09_dqi_sensitivity.py
    |   |-- 10_mcnemar.py
    |-- docs/
    |   |-- DATASETS.md
    |-- data/                    (populated by 01_download_data.py)
    |-- figures/                 (Figures 1-9, PNG + PDF)
    |-- results/                 (CSV outputs + parquet splits)
    |-- notebooks/
    |-- paper/                   (manuscript)

---

## Installation

Requires **Python 3.10+**. DistilBERT training requires a GPU (tested on Tesla T4 in Google Colab).

    git clone https://github.com/sarthakdeore265/SPADE.git
    cd SPADE
    pip install -r requirements.txt

Or with conda:

    conda create -n spade python=3.10 -y
    conda activate spade
    pip install -r requirements.txt

---

## Quickstart

Run the pipeline in order:

    python src/01_download_data.py       # Download all 9 source datasets
    python src/02_build_benchmark.py     # Build unified 38,814-prompt parquet
    python src/03_compute_dqi.py         # Compute Dataset Quality Index
    python src/04_train_tfidf.py         # Train TF-IDF + LR (CPU, ~60 sec)
    python src/05_train_bert.py          # Fine-tune DistilBERT (GPU, ~10 min)
    python src/06_metric_agreement.py    # Spearman / Kendall across metrics
    python src/07_error_analysis.py      # Detection + FPR by attack type
    python src/08_lexical_ablation.py    # 60-word lexical-only model
    python src/09_dqi_sensitivity.py     # DQI weight sensitivity
    python src/10_mcnemar.py             # McNemar: TF-IDF vs DistilBERT

All outputs land in `results/`. Random splits use seed 42.

---

## Datasets

SPADE is built from **9 public datasets**. All are publicly available; none are redistributed in this repository. `src/01_download_data.py` fetches everything.

| # | Source | n | Malicious % | DQI | Link |
|---|--------|---|-------------|-----|------|
| 1 | JailbreakBench | 200 | 50.0 | **0.856** | [Hugging Face](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |
| 2 | XSTest | 450 | 44.4 | **0.847** | [Hugging Face](https://huggingface.co/datasets/Paul/XSTest) |
| 3 | BeaverTails | 7,764 | 57.7 | **0.831** | [Hugging Face](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) |
| 4 | deepset/prompt-injections | 662 | 39.7 | 0.793 | [Hugging Face](https://huggingface.co/datasets/deepset/prompt-injections) |
| 5 | ToxicChat | 9,740 | 1.9 | 0.707 | [Hugging Face](https://huggingface.co/datasets/lmsys/toxic-chat) |
| 6 | JailbreakHub | 18,527 | 8.4 | 0.658 | [GitHub](https://github.com/verazuo/jailbreak_llms) |
| 7 | JailbreakHub-Forbidden | 388 | 100.0 | 0.652 | [GitHub](https://github.com/verazuo/jailbreak_llms) |
| 8 | AdvBench-Strings | 574 | 100.0 | 0.575 | [GitHub](https://github.com/llm-attacks/llm-attacks) |
| 9 | AdvBench | 509 | 100.0 | 0.574 | [GitHub](https://github.com/llm-attacks/llm-attacks) |
| - | **Total** | **38,814** | **21.3** | - | - |

### Dataset licenses

| Dataset | License |
|---------|---------|
| deepset/prompt-injections | Apache-2.0 |
| JailbreakBench | MIT |
| ToxicChat | CC-BY-NC-4.0 |
| XSTest | CC-BY-4.0 |
| BeaverTails | CC-BY-NC-4.0 |
| JailbreakHub | MIT |
| AdvBench | MIT |

Please respect the original licenses when using these sources.

### Optional additional sources

- [HarmBench](https://github.com/centerforaisafety/HarmBench) - standardized harmful behavior benchmark
- [TensorTrust](https://github.com/HumanCompatibleAI/tensor-trust) - prompt injection attacks from an online game
- [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) - large-scale real + synthetic
- [DoNotAnswer](https://huggingface.co/datasets/Libr-AI/do-not-answer) - harm categories

See `docs/DATASETS.md` for details.

---

## Models

| Model | Description |
|-------|-------------|
| **TF-IDF + LR** | Unigrams + bigrams, 20,000 features, min_df=2, sublinear TF, balanced class weights, max_iter 2,000 |
| **DistilBERT** | distilbert-base-uncased, 2 epochs, LR 2e-5, batch 64, max seq length 128, class-weighted cross-entropy |
| **Lexical-only** | 60-word scary vocabulary as binary features, logistic regression, no context or n-grams |

---

## Results

### Random split

| Metric | TF-IDF | DistilBERT |
|--------|--------|------------|
| Accuracy | 0.848 | **0.867** |
| F1 | 0.698 | **0.743** |
| MCC | 0.613 | **0.676** |
| AUPRC | 0.758 | **0.822** |
| Recall@1%FPR | 0.265 | **0.333** |

### Cross-source generalization

| Test source | TF-IDF F1 | DistilBERT F1 |
|-------------|-----------|---------------|
| XSTest | 0.644 | **0.654** |
| JailbreakBench | 0.605 | **0.748** |
| deepset | **0.357** | 0.275 |
| ToxicChat | 0.069 | **0.140** |
| BeaverTails | 0.612 | **0.629** |

### Failure modes

| Failure mode | TF-IDF | DistilBERT | Delta |
|--------------|--------|------------|-------|
| Injection detection | 0.460 | **0.560** | +0.100 |
| Safe-but-scary FPR | 0.577 | 0.615 | +0.038 (worse) |
| Benign FPR | 0.142 | 0.139 | -0.003 |

### McNemar significance (exclusive correct predictions)

| Split | BERT only | TF-IDF only | chi2 | p | Sig. |
|-------|-----------|-------------|------|------|------|
| Random | 475 | 332 | 24.99 | < 0.0001 | *** |
| ToxicChat | 2,550 | 585 | 1,240 | < 0.0001 | *** |
| JailbreakBench | 33 | 15 | 6.02 | 0.014 | * |
| XSTest | 34 | 54 | 4.09 | 0.043 | * (TF-IDF wins) |
| deepset | 59 | 57 | 0.01 | 0.926 | ns |
| BeaverTails | 1,173 | 1,178 | 0.01 | 0.934 | ns |

Significance: * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant.

### Metric rank correlations

| Pair | Spearman rho | Interpretation |
|------|--------------|----------------|
| Accuracy vs F1 | 0.086 | Almost uncorrelated |
| MCC vs AUPRC | 1.000 | Perfectly rank-equivalent |
| ROC-AUC vs Recall@1%FPR | 0.943 | Tightly coupled |
| Precision vs ROC-AUC | -0.143 | Negatively correlated |

---

## Figures

All figures are in `figures/` as both PNG (300 DPI) and PDF (vector).

| # | Content |
|---|---------|
| 1 | DQI vs F1/AUPRC/R@1%FPR (3-panel scatter) |
| 2 | Cross-source generalization parity plot |
| 3 | Metric correlation heatmaps (Spearman, Kendall) |
| 4 | Metric rank stability across splits |
| 5 | Error by attack type (detection + FPR) |
| 6 | Lexical-shortcut F1 comparison |
| 7 | Model comparison (TF-IDF vs DistilBERT) |
| 8 | Failure-mode comparison |
| 9 | McNemar difference across splits |

---

## Reproducibility

- All random splits use **seed 42**
- TF-IDF + LR is deterministic given the seed
- DistilBERT is deterministic given the seed and CUDA configuration
- DQI component values are in `results/dqi.csv`
- No proprietary hardware or data was used

---

## Environment

- **Python:** 3.10+
- **Tested on:** Google Colab, Tesla T4 GPU
- **Key dependencies:** datasets, transformers, torch, scikit-learn, statsmodels

Full list in `requirements.txt`.

---

## Citation

If you use SPADE or the benchmark, please cite:

    @article{deore2026spade,
      title  = {SPADE: Evaluating Machine Learning Models for Malicious Prompt Detection},
      author = {Deore, Sarthak},
      year   = {2026},
      note   = {NMIMS},
      url    = {https://github.com/sarthakdeore265/SPADE}
    }

A machine-readable citation is available in `CITATION.cff`.

---

## License

Code is released under the **MIT License** - see [LICENSE](LICENSE).

Dataset licenses are governed by their original providers. See `docs/DATASETS.md`.

---

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

When contributing, please:

1. Add tests for new functionality where feasible
2. Update `README.md` and `docs/DATASETS.md` if relevant
3. Follow the existing code style (PEP 8)

---

## Contact

**Sarthak Deore**
NMIMS
sarthakdeore265@gmail.com

For dataset questions, contact the original providers.

---

## Acknowledgments

Thanks to the authors of the 9 source datasets for making their work publicly available. This study would not be possible without their contributions.
