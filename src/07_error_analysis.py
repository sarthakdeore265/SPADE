"""Error analysis by attack type."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

test = pd.read_parquet(RES + "/random_test.parquet").copy()
test["pred"] = np.load(RES + "/tfidf_random_preds.npy")

rows = []
for atk, g in test.groupby("attack_type"):
    n = len(g)
    n_mal = int(g["label"].sum())
    n_ben = n - n_mal
    if n_mal > 0 and n_ben == 0:
        rows.append({"attack_type": atk, "n": n,
                     "detection_rate": round((g["pred"] == 1).mean(), 3)})
    elif n_mal == 0 and n_ben > 0:
        rows.append({"attack_type": atk, "n": n,
                     "false_positive_rate": round((g["pred"] == 1).mean(), 3)})
    else:
        rows.append({
            "attack_type": atk, "n": n,
            "detection_rate": round(recall_score(g["label"], g["pred"], zero_division=0), 3),
            "false_positive_rate": round(((g["pred"] == 1) & (g["label"] == 0)).sum() / n_ben, 3),
            "f1": round(f1_score(g["label"], g["pred"], zero_division=0), 3),
        })

err = pd.DataFrame(rows).sort_values("detection_rate", ascending=False, na_position="last")
err.to_csv(RES + "/error_by_attack_type.csv", index=False)
print(err.to_string(index=False))
