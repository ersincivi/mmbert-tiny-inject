#!/usr/bin/env python3
"""Reconstruct the round-1 vocab (work/vocab_r1.txt) — torch-free, one-shot.

Why: `train.py --prepare` after the 2026-07-24 data-hygiene round overwrote
work/vocab.txt with a vocab built from the NEW corpus/split. The round-1
checkpoints (work/student_r4a.pt etc.) were trained with the OLD vocab — a
model and its vocab are a unit, so re-scoring r4a against the new vocab would
silently produce garbage. This script rebuilds the old vocab exactly:

  * old corpus = the first 15,513 rows of inject.jsonl (today's session only
    APPENDED 559 sms-ham rows; repair_lang rewrote lang/notes in place but
    never text or ORDER, and vocab/split depend only on text),
  * old split  = eval_oracle's round-1 rules (base fraction 0.18, no FP-hard
    boost, no label-suspect exclusion, weak/synthetic forced to train),
  * old vocab  = build_vocab over the old train windows (deterministic).

Self-check: the reconstructed eval must have exactly 2,522 rows (the round-1
eval size recorded in TRAINING-LOG) — the script fails loudly otherwise.

Usage:  python3 rebuild_r1_artifacts.py
Then:   python3 train.py --eval-checkpoint work/student_r4a.pt --vocab work/vocab_r1.txt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE))

from train import expand_training, build_vocab, WORK  # noqa: E402
from corpus_common import dedup_key  # noqa: E402

CORPUS = HERE / "corpus_work" / "inject.jsonl"
R1_CORPUS_ROWS = 15_513   # corpus size when round 1 was trained
R1_EVAL_ROWS = 2_522      # round-1 eval size (TRAINING-LOG)
R1_EVAL_FRACTION = 0.18
VOCAB_R1 = WORK / "vocab_r1.txt"


def main() -> int:
    rows = []
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) < R1_CORPUS_ROWS:
        sys.exit(f"corpus has {len(rows)} rows < expected {R1_CORPUS_ROWS} — "
                 "was it rebuilt with --fresh? Cannot reconstruct r1 state.")
    old = rows[:R1_CORPUS_ROWS]

    def group_key(r: dict) -> str:
        g = r.get("group")
        return g if g else "row-" + dedup_key(r["text"])

    def in_eval(gk: str) -> bool:
        h = int(hashlib.sha1(gk.encode()).hexdigest()[:8], 16)
        return (h % 100) < int(R1_EVAL_FRACTION * 100)

    train_docs, n_eval = [], 0
    for r in old:
        forced_train = r.get("weak_seed") or r.get("is_synthetic")
        if not forced_train and in_eval(group_key(r)):
            n_eval += 1
        else:
            train_docs.append(r)

    if n_eval != R1_EVAL_ROWS:
        sys.exit(f"reconstructed eval = {n_eval} rows, expected {R1_EVAL_ROWS} "
                 "— split rules or corpus prefix drifted; refusing to write a "
                 "vocab that may not match the r1 checkpoints.")

    pairs = expand_training(train_docs)
    vocab_list = build_vocab(pairs)
    WORK.mkdir(exist_ok=True)
    VOCAB_R1.write_text("\n".join(vocab_list), encoding="utf-8")
    print(f"r1 split reconstructed: train {len(train_docs)} / eval {n_eval} ✓")
    print(f"r1 vocab ({len(vocab_list)} tokens) -> {VOCAB_R1}")
    print("\nnext: python3 train.py --eval-checkpoint work/student_r4a.pt "
          "--vocab work/vocab_r1.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
