#!/usr/bin/env python3
"""Stream C — benign negatives from the SMS ham corpus + FP-hard mining.

Two kinds of negative:
  1. Plain benign — sampled ham text (label!=spam) for base-rate benign.
  2. FP-HARD — benign text that *contains* injection-shaped words ("ignore",
     "verify", "password", "instructions", "api key" …). These are the
     hardest negatives: they teach the classifier that "scary words in
     legitimate text" != injection (the transactional-FP-guard lesson from
     the SMS model).

FP-hard rows are oversampled relative to their natural frequency because they
carry the most training signal per row.

Local, stdlib-only. Needs an SMS corpus from the companion SMS classifier
project; point at it with SMS_CORPUS=/path/to/corpus.jsonl (rows need
`text` + `label_3class`).
Deterministic sampling (hash-based) — no RNG, reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os

from corpus_common import CorpusWriter, Row, is_fp_hard

HERE = os.path.dirname(os.path.abspath(__file__))
SMS_CORPUS = os.environ.get("SMS_CORPUS") or os.path.normpath(
    os.path.join(HERE, "corpus_work", "sms_corpus.jsonl")
)

# Budget: keep the negative side from swamping the (small) positive anchor.
PLAIN_TARGET = 1500
FP_HARD_TARGET = 1500


def sample_bucket(text: str, modulo: int) -> int:
    """Deterministic 0..modulo-1 bucket from text hash (reproducible sampling)."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


def main() -> int:
    if not os.path.exists(SMS_CORPUS):
        print(f"[sms-ham] corpus not found: {SMS_CORPUS}")
        return 1

    writer = CorpusWriter()
    plain = fp_hard = 0
    seen_plain = seen_fp = 0

    with open(SMS_CORPUS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only benign SMS (ham) becomes a negative for the injection model. The SMS corpus
            # labels malicious rows as "phishing"/"spam" — WHITELIST ham so no
            # smishing leaks in as a benign negative (would teach the model
            # that phishing text is safe).
            if obj.get("label") != "ham" or obj.get("label_3class") != "ham":
                continue
            text = obj.get("text", "")
            if not text:
                continue

            hard = is_fp_hard(text)
            row = Row(
                text=text,
                label="benign",
                source="sms-ham",
                license=obj.get("license", "unknown"),
                lang=obj.get("language", "") or "",
                channel="sms",
                notes="fp-hard" if hard else "plain-benign",
            )

            if hard:
                seen_fp += 1
                if fp_hard < FP_HARD_TARGET:  # take all FP-hard up to budget
                    if writer.add(row):
                        fp_hard += 1
            else:
                seen_plain += 1
                if plain < PLAIN_TARGET and sample_bucket(text, 40) == 0:
                    if writer.add(row):
                        plain += 1

    writer.report("sms-ham")
    print(f"[sms-ham] plain {plain}/{seen_plain} available | "
          f"fp-hard {fp_hard}/{seen_fp} available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
