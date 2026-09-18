"""Fine-tune DistilBERT. Requires GPU."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN, BATCH, EPOCHS, LR, NW = 128, 64, 2, 2e-5, 2
torch.manual_seed(42)
np.random.seed(42)


class DS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(list(texts), truncation=True, padding=True,
                       max_length=MAX_LEN, return_tensors=None)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def rec(y, s, t):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.where(fpr <= t)[0]
    return float(tpr[i[-1]]) if len(i) else 0.0


def metr(y, p, s):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
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


def run(tr, te, name):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)
    counts = tr["label"].value_counts().to_dict()
    n = sum(counts.values())
    w = torch.tensor([n / (2 * counts.get(i, 1)) for i in (0, 1)],
                     dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    dl_tr = DataLoader(DS(tr["prompt"].astype(str), tr["label"], tok),
                       batch_size=BATCH, shuffle=True, num_workers=NW,
                       pin_memory=True, persistent_workers=(NW > 0))
    dl_te = DataLoader(DS(te["prompt"].astype(str), te["label"], tok),
                       batch_size=BATCH, num_workers=NW, pin_memory=True)
    opt = AdamW(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train()
        tot = 0
        for b in dl_tr:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            y = b.pop("labels")
            loss = loss_fn(model(**b).logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item()
        print("  ep", ep + 1, "loss", round(tot / len(dl_tr), 4))
    model.eval()
    preds, scores = [], []
    with torch.no_grad():
        for b in dl_te:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            b.pop("labels")
            p = torch.softmax(model(**b).logits, dim=-1)[:, 1].cpu().numpy()
            preds.extend((p > 0.5).astype(int))
            scores.extend(p)
    return (metr(te["label"].values, np.array(preds), np.array(scores)),
            np.array(preds), np.array(scores))


results = []
for s in ["random", "xstest", "jailbreakbench", "deepset", "toxicchat", "beavertails"]:
    if s == "random":
        tr = pd.read_parquet(RES + "/random_train.parquet")
        te = pd.read_parquet(RES + "/random_test.parquet")
        tag = "random"
    else:
        tr = pd.read_parquet(RES + "/source_" + s + "_train.parquet")
        te = pd.read_parquet(RES + "/source_" + s + "_test.parquet")
        tag = "test=" + s
    if te["label"].nunique() < 2:
        continue
    print("=== " + tag + " ===")
    m, pr, sc = run(tr, te, tag)
    m["split"] = tag
    results.append(m)
    np.save(RES + "/bert_" + s + "_preds.npy", pr)
    np.save(RES + "/bert_" + s + "_scores.npy", sc)
    print(m)

pd.DataFrame(results).to_csv(RES + "/bert_results.csv", index=False)
