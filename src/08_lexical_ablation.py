"""60-word scary vocabulary ablation."""
import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

SCARY = set([
    "kill", "killing", "killed", "murder", "murderer", "slay", "slaying", "slain",
    "destroy", "destroying", "destroyed", "destruction",
    "attack", "attacking", "attacked", "assault",
    "harm", "harming", "harmed", "hurt", "hurting",
    "abuse", "abusing", "abused", "violence", "violent",
    "weapon", "weapons", "gun", "guns", "bomb", "bombs",
    "explosive", "explosives", "hack", "hacking", "hacked",
    "malware", "virus", "ransomware", "steal", "stealing", "stolen",
    "theft", "rob", "robbing", "robbed", "choke", "choking",
    "strangle", "stab", "stabbing", "fraud", "scam",
    "launder", "laundering", "smuggle", "smuggling",
    "poison", "poisoned", "poisoning",
])


def feat(texts):
    vocab = sorted(SCARY)
    M = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = set(re.findall(r"\b[a-z]+\b", str(t).lower()))
        for j, w in enumerate(vocab):
            if w in toks:
                M[i, j] = 1.0
    return M


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
    Xtr = feat(tr["prompt"].tolist())
    Xte = feat(te["prompt"].tolist())
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, tr["label"])
    p = clf.predict(Xte)
    sc = clf.predict_proba(Xte)[:, 1]
    rows.append({
        "split": s,
        "scary_f1": round(f1_score(te["label"], p, zero_division=0), 3),
        "scary_auprc": round(average_precision_score(te["label"], sc), 3),
    })

pd.DataFrame(rows).to_csv(RES + "/lexical_ablation.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
