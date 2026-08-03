#!/usr/bin/env python3
"""Inspect confidently-wrong FP-hard eval docs (ANALYSIS-r2 follow-up #1).

The r4 threshold sweep showed FP-hard is threshold-insensitive: ~12 benign
eval docs score >0.60 — not borderline, confidently wrong. No threshold fixes
those; only targeted training data can. Step one is reading them: this tool
loads a trained student checkpoint, scores the held-out eval split, and dumps
every false-positive benign doc above --min-score with its metadata and the
text of its highest-scoring window (the window that triggered max-pool — the
pattern the model is actually reacting to).

Read-only analysis — no training, nothing written unless --out is given.

Run (shared venv):
    source ../.venv/bin/activate
    python3 inspect_fphard.py                          # FP-hard only, >0.60
    python3 inspect_fphard.py --min-score 0.5 --all-fp # every benign FP ≥0.5
    python3 inspect_fphard.py --out work/fphard_dump.jsonl
"""
from __future__ import annotations

import argparse
import json

from train import (CLASSES, CLASS_IX, EVAL_JSONL, SEQ_LEN, VOCAB_PATH, WORK,
                   encode, load_docs)
from windowing import char_windows


def build_student(torch, vocab_list, vocab, hidden, heads=4, layers=2):
    """Mirror of train.run_train's Classifier — keep in lockstep."""
    import torch.nn as nn

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab_list), hidden, padding_idx=vocab["[PAD]"])
            self.pos = nn.Embedding(SEQ_LEN, hidden)
            enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4,
                                             batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, len(CLASSES))

        def forward(self, ids, mask):
            pos = torch.arange(SEQ_LEN, device=ids.device).unsqueeze(0)
            x = self.emb(ids) + self.pos(pos)
            x = self.enc(x, src_key_padding_mask=(mask == 0))
            pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return self.head(self.norm(pooled))

    return Classifier()


def snippet(text: str, n: int = 200) -> str:
    return " ".join(text.split())[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(WORK / "student_r4a.pt"))
    ap.add_argument("--hidden", type=int, default=192,
                    help="must match the checkpoint's architecture (r4+: 192)")
    ap.add_argument("--min-score", type=float, default=0.60,
                    help="only show benign docs scoring >= this (confidently wrong)")
    ap.add_argument("--all-fp", action="store_true",
                    help="include ALL benign false-positives, not just fp_hard-tagged")
    ap.add_argument("--out", default="", help="optional JSONL dump path")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("torch missing — source ../.venv/bin/activate")

    vocab_list = VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    vocab = {t: i for i, t in enumerate(vocab_list)}
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = build_student(torch, vocab_list, vocab, args.hidden).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"checkpoint {args.ckpt} | hidden {args.hidden} | device {device}")

    hits = []
    n_benign = n_fphard = 0
    with torch.no_grad():
        for d in load_docs(EVAL_JSONL):
            if d["label"] != "benign":
                continue
            n_benign += 1
            fph = bool(d.get("fp_hard"))
            if fph:
                n_fphard += 1
            if not fph and not args.all_fp:
                continue
            wins = char_windows(d["text"])
            best_prob, best_text = -1.0, ""
            for w in wins:
                ids, mask = encode(w.text, vocab)
                logit = model(torch.tensor([ids], device=device),
                              torch.tensor([mask], device=device))
                p = torch.softmax(logit, -1)[0, CLASS_IX["injection"]].item()
                if p > best_prob:
                    best_prob, best_text = p, w.text
            if best_prob >= args.min_score:
                hits.append({"score": round(best_prob, 4), "fp_hard": fph,
                             "lang": d.get("lang"), "source": d.get("source"),
                             "channel": d.get("channel"), "id": d.get("id"),
                             "doc_head": snippet(d["text"], 160),
                             "top_window": snippet(best_text, 300)})

    hits.sort(key=lambda h: -h["score"])
    scope = "ALL benign FPs" if args.all_fp else "FP-hard-tagged only"
    print(f"eval benign {n_benign} (fp_hard-tagged {n_fphard}) | scope: {scope} | "
          f"confidently-wrong (>= {args.min_score}): {len(hits)}\n")
    by_src = {}
    for h in hits:
        by_src[h["source"]] = by_src.get(h["source"], 0) + 1
    print(f"by source: {by_src}\n" + "=" * 72)
    for i, h in enumerate(hits, 1):
        tag = "FP-HARD" if h["fp_hard"] else "fp"
        print(f"\n#{i} [{tag}] score {h['score']} | {h['lang']} | {h['source']} | "
              f"{h['channel']} | id {h['id'][:10]}")
        print(f"  doc: {h['doc_head']}")
        print(f"  TOP WINDOW → {h['top_window']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for h in hits:
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(f"\ndump → {args.out} ({len(hits)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
