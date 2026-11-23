"""Exploratory data analysis: class balance, length distribution, common n-grams."""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import SENTIMENTS

PALETTE = {"negative": "#e45756", "neutral": "#9d9da1", "positive": "#54a24b"}
# Display-only stop list (models keep every token, negations included).
STOP = set("""a an the to and or of in on for at is it its this that be was were are am i me my we our you your
he she they them so do did does have has had with from as by just but if about up out get got im would will can
could there their what when how all one u us flight flights 2 amp""".split())


def ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def top_ngrams(df: pd.DataFrame, n: int, k: int = 12) -> dict:
    out = {}
    for s in SENTIMENTS:
        c = Counter()
        for toks in df.loc[df.sentiment == s, "tokens"]:
            toks = [t for t in toks if t not in STOP and t not in "!?"]
            c.update(ngrams(toks, n))
        out[s] = c.most_common(k)
    return out


def run_eda(df: pd.DataFrame, fig_dir) -> dict:
    sns.set_theme(style="whitegrid")
    eda = {}
    eda["class_counts"] = df.sentiment.value_counts().reindex(SENTIMENTS).to_dict()
    eda["split_counts"] = {sp: df.loc[df.split == sp, "sentiment"].value_counts().reindex(SENTIMENTS).to_dict()
                           for sp in ["train", "val", "test"]}
    eda["topic_counts"] = df.topic.dropna().value_counts().to_dict()
    eda["airline_sentiment"] = (pd.crosstab(df.airline, df.sentiment, normalize="index").round(3)
                                .reindex(columns=SENTIMENTS).to_dict(orient="index"))
    bins = list(range(0, 41, 2))
    eda["length_bins"] = bins
    eda["length_hist"] = {s: np.histogram(df.loc[df.sentiment == s, "n_tokens"].clip(upper=39), bins=bins)[0].tolist()
                          for s in SENTIMENTS}
    eda["length_median"] = df.groupby("sentiment").n_tokens.median().to_dict()
    eda["top_unigrams"] = top_ngrams(df, 1)
    eda["top_bigrams"] = top_ngrams(df, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="sentiment", order=SENTIMENTS, palette=PALETTE, hue="sentiment", ax=axes[0], legend=False)
    axes[0].set_title("Sentiment class balance")
    topics = df.topic.dropna().value_counts()
    sns.barplot(x=topics.values, y=topics.index, color="#e45756", ax=axes[1])
    axes[1].set_title("Complaint topics (negative tweets)")
    fig.tight_layout(); fig.savefig(fig_dir / "class_balance.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SENTIMENTS:
        sns.kdeplot(df.loc[df.sentiment == s, "n_tokens"], ax=ax, label=s, color=PALETTE[s], fill=True, alpha=.2)
    ax.set_xlabel("tokens per tweet"); ax.set_title("Length distribution by sentiment"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "length_distribution.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, s in zip(axes, SENTIMENTS):
        terms, counts = zip(*eda["top_bigrams"][s][:10])
        sns.barplot(x=list(counts), y=list(terms), color=PALETTE[s], ax=ax)
        ax.set_title(f"Top bigrams: {s}")
    fig.tight_layout(); fig.savefig(fig_dir / "top_bigrams.png", dpi=120); plt.close(fig)
    return eda
