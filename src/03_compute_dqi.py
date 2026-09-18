"""Compute Dataset Quality Index (DQI) per source."""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "results", "unified.parquet")
OUT = os.path.join(HERE, "..", "results", "dqi.csv")

df = pd.read_parquet(IN)
RELIABILITY = {
    "jailbreakbench": 0.95, "xstest": 0.95, "toxicchat": 0.80,
    "jailbreakhub": 0.55, "jailbreakhub_forbidden": 0.90,
    "deepset": 0.70, "beavertails": 0.75,
    "advbench": 0.60, "advbench_strings": 0.55,
}

rows = []
for src, g in df.groupby("source"):
    n = len(g)
    n_mal = int(g["label"].sum())
    n_ben = n - n_mal
    size = min(np.log10(max(n, 1)) / np.log10(10000), 1.0)
    if n_mal == 0 or n_ben == 0:
        balance = 0.0
    else:
        p = n_mal / n
        balance = 1 - abs(0.5 - p) * 2
    sample = g["prompt"].dropna().sample(min(2000, n), random_state=42).tolist()
    sample = [s for s in sample if isinstance(s, str) and len(s) > 5]
    if len(sample) >= 10:
        v = TfidfVectorizer(max_features=5000, stop_words="english").fit_transform(sample)
        sim = cosine_similarity(v[:500])
        iu = np.triu_indices_from(sim, k=1)
        diversity = float(1 - sim[iu].mean())
    else:
        diversity = 0.0
    lengths = g["prompt"].astype(str).str.len().clip(lower=1)
    length_div = min(np.log1p(lengths).std() / 2.0, 1.0)
    head = g["prompt"].astype(str).str[:100]
    no_leak = 1 - (1 - head.nunique() / max(len(head), 1))
    rows.append({
        "source": src, "n": n,
        "malicious_pct": round(n_mal / n, 3),
        "size": round(size, 3),
        "balance": round(balance, 3),
        "diversity": round(diversity, 3),
        "length_div": round(length_div, 3),
        "no_leakage": round(no_leak, 3),
        "reliability": RELIABILITY.get(src, 0.6),
    })

dqi = pd.DataFrame(rows)
w = {"size": 0.10, "balance": 0.20, "diversity": 0.20,
     "length_div": 0.10, "no_leakage": 0.15, "reliability": 0.25}
dqi["DQI"] = sum(dqi[k] * w for k, w in w.items()).round(3)
dqi = dqi.sort_values("DQI", ascending=False).reset_index(drop=True)
dqi.to_csv(OUT, index=False)
print(dqi.to_string(index=False))
