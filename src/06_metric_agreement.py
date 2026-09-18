"""Metric rank correlations across splits."""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def rec(y, s, t):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.where(fpr <= t)[0]
    return float(tpr[i[-1]]) if len(i) else 0.0


def fm(y, p, s):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "mcc": matthews_corrcoef(y, p),
        "roc_auc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "recall@1%fpr": rec(y, s, 0.01),
        "recall@5%fpr": rec(y, s, 0.05),
    }


rows = []
for s in ["random", "xstest", "jailbreakbench", "deepset", "toxicchat", "beavertails"]:
    if s == "random":
        tr = pd.read_parquet(RES + "/random_train.parquet")
        te = pd.read_parquet(RES + "/random_test.parquet")
    else:
        tr = pd.read_parquet(RES + "/source_" + s + "_train.parquet")
        te = pd.read_parquet(RES + "/source_" + s + "_test.parquet")
    if te["label"].nunique() < 2:
        continue
    p = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    p.fit(tr["prompt"].astype(str), tr["label"])
    pr = p.predict(te["prompt"].astype(str))
    sc = p.predict_proba(te["prompt"].astype(str))[:, 1]
    r = fm(te["label"].values, pr, sc)
    r["split"] = s if s == "random" else "test=" + s
    rows.append(r)

mdf = pd.DataFrame(rows).set_index("split")
mdf.to_csv(RES + "/metric_matrix.csv")
cols = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "auprc", "recall@1%fpr"]
sp = pd.DataFrame(index=cols, columns=cols, dtype=float)
kn = pd.DataFrame(index=cols, columns=cols, dtype=float)
for a in cols:
    for b in cols:
        s_, _ = spearmanr(mdf[a], mdf[b])
        sp.loc[a, b] = round(s_, 3)
        k_, _ = kendalltau(mdf[a], mdf[b])
        kn.loc[a, b] = round(k_, 3)
sp.to_csv(RES + "/spearman.csv")
kn.to_csv(RES + "/kendall.csv")
print(sp)
