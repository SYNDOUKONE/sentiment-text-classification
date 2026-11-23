import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve)

RNG = np.random.default_rng(0)


def _thin(xs, ys, n=120):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [[round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in idx]


def bootstrap_macro_f1(y, pred, n=1000):
    y, pred = np.asarray(y), np.asarray(pred)
    scores = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        scores.append(f1_score(y[i], pred[i], average="macro"))
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def evaluate(y_true, proba, classes, binary_positive=None, ci=True):
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(1)]
    p, r, f, sup = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "weighted_f1": f1_score(y_true, pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
                      for i, c in enumerate(classes)},
        "confusion": confusion_matrix(y_true, pred, labels=classes).tolist(),
    }
    out = {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}
    if ci:
        out["macro_f1_ci95"] = bootstrap_macro_f1(y_true, pred)
    if binary_positive is not None:
        j = list(classes).index(binary_positive)
        yb, s = (y_true == binary_positive).astype(int), proba[:, j]
        fpr, tpr, _ = roc_curve(yb, s)
        prec, rec, _ = precision_recall_curve(yb, s)
        out["binary"] = {
            "positive_class": binary_positive,
            "roc_auc": round(float(roc_auc_score(yb, s)), 4),
            "ap": round(float(average_precision_score(yb, s)), 4),
            "roc": _thin(fpr, tpr),
            "pr": _thin(rec[::-1], prec[::-1]),
            "scores": [round(float(v), 3) for v in s],
        }
    return out
