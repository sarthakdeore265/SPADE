"""DQI sensitivity to weight schemes."""
import os
import pandas as pd
from scipy.stats import kendalltau, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

dqi = pd.read_csv(RES + "/dqi.csv").set_index("source")
comp = dqi[["size", "balance", "diversity", "length_div", "no_leakage", "reliability"]]

schemes = {
    "baseline":          {"size": 0.10, "balance": 0.20, "diversity": 0.20,
                          "length_div": 0.10, "no_leakage": 0.15, "reliability": 0.25},
    "uniform":           {k: 1 / 6 for k in comp.columns},
    "balance_heavy":     {"size": 0.05, "balance": 0.40, "diversity": 0.15,
                          "length_div": 0.05, "no_leakage": 0.10, "reliability": 0.25},
    "reliability_heavy": {"size": 0.05, "balance": 0.15, "diversity": 0.15,
                          "length_div": 0.05, "no_leakage": 0.15, "reliability": 0.45},
    "diversity_heavy":   {"size": 0.05, "balance": 0.15, "diversity": 0.40,
                          "length_div": 0.15, "no_leakage": 0.10, "reliability": 0.15},
}

res = {}
for name, w in schemes.items():
    res[name] = (comp * pd.Series(w)).sum(axis=1)

mat = pd.DataFrame(res)
mat.to_csv(RES + "/dqi_sensitivity.csv")
base = mat["baseline"]
for name in schemes:
    t, _ = kendalltau(base, mat[name])
    s_, _ = spearmanr(base, mat[name])
    print(name, "Kendall tau=", round(t, 3), " Spearman=", round(s_, 3))
