"""Fine-tune a small transformer (DistilRoBERTa) for 3-class sentiment on CPU.

Usage: python -m stc.finetune [--epochs 3] [--model distilroberta-base]
Writes reports/finetune.json (val history + test probabilities), consumed by stc.train.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import SEED, SENTIMENTS, load_clean

ROOT = Path(__file__).resolve().parent.parent


def encode(tok, texts, labels, max_len):
    enc = tok(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    y = torch.tensor([SENTIMENTS.index(l) for l in labels])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, losses = [], []
    for ids, mask, y in loader:
        out = model(input_ids=ids, attention_mask=mask, labels=y)
        losses.append(out.loss.item() * len(y))
        probs.append(out.logits.softmax(-1).numpy())
    return np.concatenate(probs), sum(losses) / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    df, _ = load_clean()
    # Transformers get the cleaned text (URLs/handles removed) but keep their own sub-word tokeniser.
    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    counts = tr.sentiment.value_counts().reindex(SENTIMENTS).values
    weights = torch.tensor(len(tr) / (3 * counts), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    dl_tr = DataLoader(encode(tok, tr.clean, tr.sentiment, args.max_len), batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(encode(tok, va.clean, va.sentiment, args.max_len), batch_size=128)
    dl_te = DataLoader(encode(tok, te.clean, te.sentiment, args.max_len), batch_size=128)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(dl_tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    history, best, best_state = [], -1, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for step, (ids, mask, y) in enumerate(dl_tr, 1):
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item() * len(y); n += len(y)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(dl_tr)} loss {running / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        va_proba, va_loss = predict(model, dl_va)
        va_f1 = f1_score(va.sentiment, np.array(SENTIMENTS)[va_proba.argmax(1)], average="macro")
        history.append({"epoch": epoch, "train_loss": round(running / n, 4), "val_loss": round(va_loss, 4),
                        "val_macro_f1": round(float(va_f1), 4)})
        print(history[-1], flush=True)
        if va_f1 > best:
            best, best_state = va_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te_proba, _ = predict(model, dl_te)
    out = {
        "name": "DistilRoBERTa (fine-tuned)",
        "info": f"{args.model}, {args.epochs} epochs, lr={args.lr}, max_len={args.max_len}, class-weighted loss, "
                f"best epoch by val macro-F1, trained on CPU in {(time.time() - t0) / 60:.0f} min",
        "history": history,
        "best_val_macro_f1": round(float(best), 4),
        "test_proba": [[round(float(p), 5) for p in row] for row in te_proba],
    }
    (ROOT / "reports" / "finetune.json").write_text(json.dumps(out))
    print("test macro-F1", f1_score(te.sentiment, np.array(SENTIMENTS)[te_proba.argmax(1)], average="macro"))


if __name__ == "__main__":
    main()
