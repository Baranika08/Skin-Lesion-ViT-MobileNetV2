import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix


def compute_metrics(y, p, k=7):
    labels = list(range(k))
    rec = recall_score(y, p, average=None, labels=labels, zero_division=0)
    return {
        "acc": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
        "bal_acc": float(np.mean(rec)),
        "recall": [float(v) for v in rec],
        "confusion": confusion_matrix(y, p, labels=labels).tolist(),
    }
