"""Text cleaning and tokenisation.

Kept deliberately simple and dependency-free so the exact same logic can be
re-implemented in JavaScript (docs/app.js) for in-browser inference.
"""
import re

ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?]|[\U0001F300-\U0001FAFF\u2600-\u27BF]")
CONTRACTIONS = [("can't", "can not"), ("won't", "will not"), ("n't", " not")]


def clean(text: str) -> str:
    for k, v in ENTITIES.items():
        text = text.replace(k, v)
    text = text.lower().replace("\u2019", "'")
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    for k, v in CONTRACTIONS:
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in (tok.strip("'") for tok in TOKEN_RE.findall(text)) if t]
