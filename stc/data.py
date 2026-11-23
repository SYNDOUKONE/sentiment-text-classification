"""Load the Twitter US Airline Sentiment dataset, clean it and make stratified splits."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from langdetect import DetectorFactory, detect_langs
from sklearn.model_selection import train_test_split

from .text import clean, tokenize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HF_DATASET = "osanseviero/twitter-airline-sentiment"
SEED = 42
SENTIMENTS = ["negative", "neutral", "positive"]
DetectorFactory.seed = 0


def detect_lang(text: str) -> str:
    if len(text.split()) < 4:
        return "short"  # too short for reliable detection
    try:
        best = detect_langs(text)[0]
    except Exception:
        return "unknown"
    return best.lang if best.prob > 0.9 else "uncertain"


def load_clean(force: bool = False) -> tuple[pd.DataFrame, dict]:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "clean.parquet"
    stats_cache = DATA_DIR / "clean_stats.json"
    if cache.exists() and stats_cache.exists() and not force:
        return pd.read_parquet(cache), pd.read_json(stats_cache, typ="series").to_dict()

    raw = load_dataset(HF_DATASET)["train"].to_pandas()
    stats = {"raw_rows": len(raw)}
    df = raw[["tweet_id", "text", "airline", "airline_sentiment", "negativereason", "tweet_created"]].rename(
        columns={"airline_sentiment": "sentiment", "negativereason": "topic"}
    )
    df["clean"] = df["text"].map(clean)
    df["tokens"] = df["clean"].map(tokenize)
    df["n_tokens"] = df["tokens"].map(len)

    before = len(df)
    df = df[df["n_tokens"] > 0]
    stats["dropped_empty"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset="clean")
    stats["dropped_duplicates"] = before - len(df)

    # Language is detected and reported but not used as a filter: manual review showed that the tweets
    # langdetect flags as non-English are English slang/emoji ("sounds like fun !", "i ❤️ flying").
    df["lang"] = df["clean"].map(detect_lang)
    stats["flagged_non_english"] = int((~df["lang"].isin(["en", "short", "unknown", "uncertain"])).sum())

    df = df.reset_index(drop=True)
    stats["clean_rows"] = len(df)

    train, rest = train_test_split(df.index, test_size=0.30, stratify=df["sentiment"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=df.loc[rest, "sentiment"], random_state=SEED)
    df["split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"
    df["tokens"] = df["tokens"].map(list)
    df.to_parquet(cache)
    pd.Series(stats).to_json(stats_cache)
    return df, stats
