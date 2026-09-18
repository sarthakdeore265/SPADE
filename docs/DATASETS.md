# Datasets

The unified benchmark combines 9 publicly available datasets. All are
downloaded by src/01_download_data.py.

| # | Dataset | Source | License |
|---|---------|--------|---------|
| 1 | deepset/prompt-injections | Hugging Face | Apache-2.0 |
| 2 | JailbreakBench (JBB-Behaviors) | Hugging Face | MIT |
| 3 | JailbreakHub | GitHub: verazuo/jailbreak_llms | MIT |
| 4 | TensorTrust | GitHub: HumanCompatibleAI/tensor-trust | MIT |
| 5 | ToxicChat | Hugging Face | CC-BY-NC-4.0 |
| 6 | XSTest | Hugging Face: Paul/XSTest | CC-BY-4.0 |
| 7 | BeaverTails | Hugging Face: PKU-Alignment/BeaverTails | CC-BY-NC-4.0 |
| 8 | AdvBench | GitHub: llm-attacks/llm-attacks | MIT |
| 9 | HarmBench (gated) | GitHub: centerforaisafety/HarmBench | MIT |

## Download

1. Create a Hugging Face account and token: https://huggingface.co/settings/tokens
2. Run: python src/01_download_data.py
3. For gated datasets (HarmBench), accept the terms on the HF page first.

## Unified benchmark statistics

- 38,814 unique prompts
- 8,251 malicious (21.3%), 30,563 benign (78.7%)
- 0 exact-string duplicates
- 9 sources with class balance ranging from 1.9% to 100% malicious
