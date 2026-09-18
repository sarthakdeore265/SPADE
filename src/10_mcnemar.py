"""McNemar test: TF-IDF vs DistilBERT."""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

rows = []
for s in ["random", "xstest", "jailbreakbench", "deepset", "toxicchat", "beavertails"]:
    test_path = RES + "/random_test.parquet" if s == "random" else RES + "/source_" + s + "_test.parquet"
    tf_path = RES + "/tfidf_" + s + "_preds.npy"
    be_path = RES + "/bert_" + s + "_preds.npy"
    if not (os.path.exists(tf_path) and os.path.exists(be_path)):
        continue
    test = pd.read_parquet(test_path)
    tf = np.load(tf_path)
    be = np.load(be_path)
    y = test["label"].values
    tf_ok = (tf == y)
    be_ok = (be == y)
    table = [[int((tf_ok & be_ok).sum()), int((tf_ok & ~be_ok).sum())],
             [int((~tf_ok & be_ok).sum()), int((~tf_ok & ~be_ok).sum())]]
    r = mcnemar(table, exact=False, correction=True)
    sig = "***" if r.pvalue < 0.001 else "**" if r.pvalue < 0.01 else "*" if r.pvalue < 0.05 else "ns"
    rows.append({
        "split": s,
        "only_bert_correct": table[1][0],
        "only_tfidf_correct": table[0][1],
        "chi2": round(r.statistic, 2),
        "p": round(r.pvalue, 6),
        "sig": sig,
    })

out = pd.DataFrame(rows)
out.to_csv(RES + "/mcnemar.csv", index=False)
print(out.to_string(index=False))
