"""Train TF-IDF + Logistic Regression on random and cross-source splits."""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def rec(y, s, t):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.where(fpr <= t)[0]
    return float(tpr[i[-1]]) if len(i) else 0.0


def ev(y, p, s, name):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y, p), 4),
        "precision": round(precision_score(y, p, zero_division=0), 4),
        "recall": round(recall_score(y, p, zero_division=0), 4),
        "f1": round(f1_score(y, p, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y, p), 4),
        "roc_auc": round(roc_auc_score(y, s), 4),
        "auprc": round(average_precision_score(y, s), 4),
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        "recall@1%fpr": round(rec(y, s, 0.01), 4),
        "recall@5%fpr": round(rec(y, s, 0.05), 4),
    }


def pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


df = pd.read_parquet(RES + "/unified.parquet")
tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
tr, va = train_test_split(tr, test_size=0.1, random_state=42, stratify=tr["label"])
tr.to_parquet(RES + "/random_train.parquet", index=False)
va.to_parquet(RES + "/random_val.parquet", index=False)
te.to_parquet(RES + "/random_test.parquet", index=False)

results = []
for s in ["random", "xstest", "jailbreakbench", "deepset", "toxicchat", "beavertails"]:
    if s == "random":
        tr = pd.read_parquet(RES + "/random_train.parquet")
        te = pd.read_parquet(RES + "/random_test.parquet")
        tag = "TFIDF+LR (random)"
    else:
        tr = df[df["source"] != s]
        te = df[df["source"] == s]
        tr.to_parquet(RES + "/source_" + s + "_train.parquet", index=False)
        te.to_parquet(RES + "/source_" + s + "_test.parquet", index=False)
        tag = "TFIDF+LR (test=" + s + ")"
    if te["label"].nunique() < 2:
        continue
    p = pipe()
    p.fit(tr["prompt"].astype(str), tr["label"])
    pr = p.predict(te["prompt"].astype(str))
    sc = p.predict_proba(te["prompt"].astype(str))[:, 1]
    results.append(ev(te["label"], pr, sc, tag))
    np.save(RES + "/tfidf_" + s + "_preds.npy", pr)
    np.save(RES + "/tfidf_" + s + "_scores.npy", sc)

pd.DataFrame(results).to_csv(RES + "/baseline_results.csv", index=False)
print("Done.")
