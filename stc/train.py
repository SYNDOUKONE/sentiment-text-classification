"""End-to-end pipeline: EDA, baselines, transformer models, topic model, exports.

Usage: python -m stc.train
Writes reports/metrics.json, reports/figures/*.png, docs/assets/{results,model}.json.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from .data import DATA_DIR, SEED, SENTIMENTS, load_clean
from .eda import run_eda
from .metrics import evaluate
from .text import tokenize

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
ASSETS = ROOT / "docs" / "assets"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ZERO_SHOT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
C_GRID = [0.25, 0.5, 1, 2, 4, 8, 16]


def make_vectorizer(max_features=20000):
    return TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                           ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)


def tune_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        f = f1_score(yva, m.predict(Xva), average="macro")
        if best is None or f > best[0]:
            best = (f, C, m)
    return best


def embed(texts, cache_name):
    cache = DATA_DIR / cache_name
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(EMBED_MODEL, device="cpu").encode(
        list(texts), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(cache, emb)
    return emb


def zero_shot(texts):
    from transformers import pipeline
    clf = pipeline("text-classification", model=ZERO_SHOT_MODEL, top_k=None, device=-1, truncation=True)
    out = clf(list(texts), batch_size=32)
    return np.array([[next(d["score"] for d in row if d["label"].lower() == c) for c in SENTIMENTS] for row in out])


def plot_confusion(cm, classes, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(0.7 * len(classes) + 3, 0.6 * len(classes) + 2.5))
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def lr_contributions(vec, model, text_clean, cls_idx, k=8):
    x = vec.transform([text_clean]).tocoo()
    terms = vec.get_feature_names_out()
    contrib = [(terms[j], float(v * model.coef_[cls_idx, j])) for j, v in zip(x.col, x.data)]
    contrib.sort(key=lambda t: -abs(t[1]))
    return [[t, round(c, 4)] for t, c in contrib[:k]]


def export_linear(vec, model, decimals=4):
    vocab = vec.vocabulary_
    terms = sorted(vocab, key=vocab.get)
    return {
        "classes": list(model.classes_),
        "terms": terms,
        "idf": [round(float(v), decimals) for v in vec.idf_],
        "coef": [[round(float(v), decimals) for v in row] for row in model.coef_],
        "intercept": [round(float(v), decimals) for v in model.intercept_],
    }


def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(parents=True, exist_ok=True)
    df, clean_stats = load_clean()
    print("clean stats", clean_stats)
    eda = run_eda(df, FIG)

    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    ytr, yva, yte = tr.sentiment.values, va.sentiment.values, te.sentiment.values

    # ---------- Sentiment: TF-IDF baselines ----------
    vec = make_vectorizer()
    Xtr, Xva, Xte = vec.fit_transform(tr.clean), vec.transform(va.clean), vec.transform(te.clean)
    print("tfidf vocab", Xtr.shape)
    results, val_f1, info = {}, {}, {}

    f, C, lr = tune_lr(Xtr, ytr, Xva, yva)
    val_f1["Logistic Regression (TF-IDF)"] = f
    info["Logistic Regression (TF-IDF)"] = f"C={C}, class_weight=balanced, uni+bi-grams, {Xtr.shape[1]} features"
    results["Logistic Regression (TF-IDF)"] = evaluate(yte, lr.predict_proba(Xte), SENTIMENTS, "negative")
    print("LR", C, results["Logistic Regression (TF-IDF)"]["macro_f1"])

    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1,
                                random_state=SEED).fit(Xtr, ytr)
    val_f1["Random Forest (TF-IDF)"] = f1_score(yva, rf.predict(Xva), average="macro")
    info["Random Forest (TF-IDF)"] = "400 trees, class_weight=balanced_subsample"
    results["Random Forest (TF-IDF)"] = evaluate(yte, rf.predict_proba(Xte), SENTIMENTS, "negative")
    print("RF", results["Random Forest (TF-IDF)"]["macro_f1"])

    svd = TruncatedSVD(300, random_state=SEED).fit(Xtr)
    gb = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, early_stopping=True, scoring="loss",
                                        validation_fraction=0.15, n_iter_no_change=25, class_weight="balanced",
                                        random_state=SEED).fit(svd.transform(Xtr), ytr)
    val_f1["Gradient Boosting (TF-IDF→SVD)"] = f1_score(yva, gb.predict(svd.transform(Xva)), average="macro")
    info["Gradient Boosting (TF-IDF→SVD)"] = f"HistGradientBoosting on 300 SVD components, {gb.n_iter_} rounds (early stopping)"
    results["Gradient Boosting (TF-IDF→SVD)"] = evaluate(yte, gb.predict_proba(svd.transform(Xte)), SENTIMENTS, "negative")
    gb_curve = {"train": [round(-float(v), 4) for v in gb.train_score_],
                "val": [round(-float(v), 4) for v in gb.validation_score_]}
    print("GB", gb.n_iter_, results["Gradient Boosting (TF-IDF→SVD)"]["macro_f1"])

    # ---------- Sentiment: transformer models ----------
    emb = embed(df.clean, "emb_minilm.npy")
    Etr, Eva, Ete = emb[tr.index], emb[va.index], emb[te.index]
    f, C, elr = tune_lr(Etr, ytr, Eva, yva)
    val_f1["MiniLM embeddings + LR"] = f
    info["MiniLM embeddings + LR"] = f"{EMBED_MODEL} (384-d contextual sentence embeddings, frozen), C={C}"
    results["MiniLM embeddings + LR"] = evaluate(yte, elr.predict_proba(Ete), SENTIMENTS, "negative")
    print("EMB", results["MiniLM embeddings + LR"]["macro_f1"])

    zs_cache = DATA_DIR / "zero_shot_test.npy"
    if not zs_cache.exists():
        np.save(zs_cache, zero_shot(te.clean))
    results["Twitter-RoBERTa (zero-shot)"] = evaluate(yte, np.load(zs_cache), SENTIMENTS, "negative")
    info["Twitter-RoBERTa (zero-shot)"] = f"{ZERO_SHOT_MODEL}, pretrained on ~124M tweets, no training on this data"
    print("ZS", results["Twitter-RoBERTa (zero-shot)"]["macro_f1"])

    ft_path = ROOT / "reports" / "finetune.json"
    ft = None
    if ft_path.exists():
        ft = json.loads(ft_path.read_text())
        name = ft["name"]
        results[name] = evaluate(yte, np.array(ft["test_proba"]), SENTIMENTS, "negative")
        val_f1[name] = ft["best_val_macro_f1"]
        info[name] = ft["info"]
        print("FT", results[name]["macro_f1"])

    for name, r in results.items():
        r["val_macro_f1"] = round(float(val_f1[name]), 4) if name in val_f1 else None
        r["info"] = info[name]
        plot_confusion(r["confusion"], SENTIMENTS, name, FIG / f"cm_{name.split(' ')[0].lower()}.png")
    best_name = max(results, key=lambda n: results[n]["macro_f1"])

    # ROC / PR (negative vs rest)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in results.items():
        b = r["binary"]
        axes[0].plot(*zip(*b["roc"]), label=f"{name} (AUC {b['roc_auc']:.3f})")
        axes[1].plot(*zip(*b["pr"]), label=f"{name} (AP {b['ap']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — negative vs rest")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall — negative vs rest")
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "roc_pr.png", dpi=120); plt.close(fig)

    # Model comparison bar chart
    names = list(results)
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(len(metric_keys)) + i * w, [results[n][k] for k in metric_keys], w, label=n)
    ax.set_xticks(np.arange(len(metric_keys)) + 0.4 - w / 2, metric_keys); ax.set_ylim(0.4, 1)
    ax.legend(fontsize=7, ncol=2); ax.set_title("Sentiment — held-out test set")
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=120); plt.close(fig)

    # Feature importance + LR coefficients
    terms = vec.get_feature_names_out()
    top = np.argsort(rf.feature_importances_)[::-1][:15]
    rf_importance = [[terms[i], round(float(rf.feature_importances_[i]), 5)] for i in top]
    lr_top_terms = {c: [[terms[j], round(float(lr.coef_[k, j]), 3)] for j in np.argsort(lr.coef_[k])[::-1][:12]]
                    for k, c in enumerate(lr.classes_)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t, v = zip(*rf_importance)
    sns.barplot(x=list(v), y=list(t), color="#4c78a8", ax=axes[0]); axes[0].set_title("Random forest — impurity importance")
    t, v = zip(*lr_top_terms["negative"])
    sns.barplot(x=list(v), y=list(t), color="#e45756", ax=axes[1]); axes[1].set_title("Logistic regression — top 'negative' terms")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=120); plt.close(fig)

    # Loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gb_curve["train"], label="training loss"); ax.plot(gb_curve["val"], label="validation loss")
    ax.set(xlabel="Boosting round", ylabel="log-loss", title="Gradient boosting — training vs validation loss"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "loss_curve.png", dpi=120); plt.close(fig)

    # Explanations + error analysis on the LR baseline
    lr_proba = lr.predict_proba(Xte)
    lr_pred = lr.classes_[lr_proba.argmax(1)]
    explanations = []
    for idx in [np.argmax(lr_proba[:, 0]), np.argmax(lr_proba[:, 2]), int(np.argmin(np.abs(lr_proba[:, 0] - 0.5)))]:
        k = int(lr_proba[idx].argmax())
        explanations.append({"text": te.text.iloc[idx], "true": yte[idx], "pred": lr.classes_[k],
                             "proba": [round(float(p), 3) for p in lr_proba[idx]],
                             "contributions": lr_contributions(vec, lr, te.clean.iloc[idx], k)})
    best_proba = np.load(zs_cache) if best_name.startswith("Twitter") else None
    wrong = np.where(lr_pred != yte)[0]
    conf = lr_proba.max(1)[wrong]
    errors = [{"text": te.text.iloc[i], "true": yte[i], "pred": lr_pred[i], "confidence": round(float(lr_proba[i].max()), 3)}
              for i in wrong[np.argsort(-conf)][:12]]

    # ---------- Topic classification (negative tweets) ----------
    neg = df[(df.sentiment == "negative") & df.topic.notna() & (df.topic != "Can't Tell")]
    ntr, nva, nte = (neg[neg.split == s] for s in ["train", "val", "test"])
    topics = sorted(neg.topic.unique())
    tvec = make_vectorizer(8000)
    Ttr, Tva, Tte = tvec.fit_transform(ntr.clean), tvec.transform(nva.clean), tvec.transform(nte.clean)
    topic_results = {}
    f, C, tlr = tune_lr(Ttr, ntr.topic, Tva, nva.topic)
    topic_results["Logistic Regression (TF-IDF)"] = evaluate(nte.topic, tlr.predict_proba(Tte), topics)
    topic_results["Logistic Regression (TF-IDF)"]["info"] = f"C={C}"
    trf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1,
                                 random_state=SEED).fit(Ttr, ntr.topic)
    topic_results["Random Forest (TF-IDF)"] = evaluate(nte.topic, trf.predict_proba(Tte), topics)
    topic_results["Random Forest (TF-IDF)"]["info"] = "400 trees"
    f, C, telr = tune_lr(emb[ntr.index], ntr.topic, emb[nva.index], nva.topic)
    topic_results["MiniLM embeddings + LR"] = evaluate(nte.topic, telr.predict_proba(emb[nte.index]), topics)
    topic_results["MiniLM embeddings + LR"]["info"] = f"C={C}"
    best_topic = max(topic_results, key=lambda n: topic_results[n]["macro_f1"])
    plot_confusion(topic_results[best_topic]["confusion"], topics, f"Topic — {best_topic}", FIG / "cm_topic.png")
    for n, r in topic_results.items():
        print("TOPIC", n, r["macro_f1"])

    # ---------- Exports ----------
    summary = {k: {m: r[m] for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_f1_ci95",
                                     "weighted_f1", "roc_auc_ovr", "per_class", "val_macro_f1", "info"]}
               for k, r in results.items()}
    topic_summary = {k: {m: r[m] for m in ["accuracy", "macro_f1", "macro_f1_ci95", "weighted_f1", "per_class", "info"]}
                     for k, r in topic_results.items()}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(
        {"data": clean_stats, "sentiment": summary, "best_sentiment_model": best_name,
         "topic": topic_summary, "best_topic_model": best_topic}, indent=2))

    out = {
        "data": clean_stats, "eda": eda, "sentiment": results, "best_sentiment_model": best_name,
        "y_test_negative": (yte == "negative").astype(int).tolist(),
        "topic": topic_results, "topic_classes": topics, "best_topic_model": best_topic,
        "topic_sizes": {"train": len(ntr), "val": len(nva), "test": len(nte)},
        "rf_importance": rf_importance, "lr_top_terms": lr_top_terms, "gb_loss": gb_curve,
        "finetune_history": ft["history"] if ft else None,
        "explanations": explanations, "errors": errors,
    }
    (ASSETS / "results.json").write_text(json.dumps(out, separators=(",", ":")))
    (ASSETS / "model.json").write_text(json.dumps(
        {"sentiment": export_linear(vec, lr), "topic": export_linear(tvec, tlr)}, separators=(",", ":")))
    print(f"done in {time.time() - t0:.0f}s; best sentiment={best_name}; best topic={best_topic}")


if __name__ == "__main__":
    main()
