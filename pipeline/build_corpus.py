#!/usr/bin/env python3
"""mmbert-tiny-inject corpus orchestrator — assemble + composition report.

Runs every ingest adapter into corpus_work/inject.jsonl and prints a composition
report (label balance, per-source, per-lang, weak-seed share, honest warnings).

The honest train/eval split is not done here — it belongs to `eval_oracle.py`
(group-aware, so an assembled source's poisoned+clean twins never straddle the
split). Run `python3 eval_oracle.py build` after.

Usage:
  python3 build_corpus.py            # ingest all + report
  python3 build_corpus.py --report   # report only (no re-ingest)
  python3 build_corpus.py --fresh    # delete inject.jsonl first, then rebuild
"""
from __future__ import annotations

import collections
import os
import subprocess
import sys

from corpus_common import CORPUS_PATH, iter_rows

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = (
    "ingest_deepset.py",    # Stream B anchor — real labeled injection (direct chat)
    "ingest_jackhhao.py",   # Stream B — jailbreak-class positives + benign (Apache-2.0)
    "ingest_bipia.py",      # Stream B — indirect injection (isolated instruction) MIT
    "ingest_bipia_context.py",  # Stream B — indirect injection spliced into email context
    "ingest_nemotron.py",   # Stream B — indirect agentic injection (CC-BY-4.0, exfiltration)
    "ingest_llmail.py",     # Stream B — real adaptive indirect email injection (LLMail, MIT, bounded 4k)
    "ingest_polyguard.py",  # Stream D — real multilingual (PolyGuard: benign negs + jailbreak-class, capped)
    "ingest_sms_ham.py",    # Stream C — benign negatives + FP-hard mining
    "synthesize.py",        # Stream E — templated FP-hard negatives + injection paraphrase
    "synthesize_email_benign.py",  # Stream E — benign email negatives (LLMail channel-confound guard)
    "synthesize_multilingual.py",  # Stream D+E — multilingual injection supplement (reduced 45→20/lang)
)


def run_adapters() -> None:
    for a in ADAPTERS:
        print(f"\n=== {a} ===")
        rc = subprocess.call([sys.executable, os.path.join(HERE, a)], cwd=HERE)
        if rc != 0:
            print(f"  ! {a} exited {rc} (continuing)")


def report() -> None:
    rows = list(iter_rows(CORPUS_PATH))
    if not rows:
        print("no rows — run adapters first")
        return

    by_label = collections.Counter()
    by_source = collections.Counter()
    by_lang = collections.Counter()
    weak = 0
    for r in rows:
        by_label[r["label"]] += 1
        by_source[r["source"]] += 1
        by_lang[r["lang"]] += 1
        if r.get("weak_seed"):
            weak += 1

    total = len(rows)
    inj = by_label["injection"]
    ben = by_label["benign"]
    print("\n" + "=" * 56)
    print(f"mmbert-tiny-inject corpus report  ({total} rows)")
    print("=" * 56)
    print(f"  injection : {inj:6d}  ({100*inj/total:.1f}%)")
    print(f"  benign    : {ben:6d}  ({100*ben/total:.1f}%)")
    print(f"  weak_seed : {weak:6d}  ({100*weak/total:.1f}% of all; positives only)")
    print("\n  by source:")
    for s, n in by_source.most_common():
        print(f"    {s:16s} {n:6d}")
    print("\n  by lang:")
    for lg, n in by_lang.most_common():
        print(f"    {lg:16s} {n:6d}")

    # Honest-limit warnings.
    print("\n  notes:")
    real_pos = inj - weak
    print(f"    real (non-weak) positives: {real_pos}")
    if real_pos < 500:
        print("    ⚠ real positive count is LOW — expand Stream B before training.")
    non_en_pos = sum(1 for r in rows if r["label"] == "injection" and r["lang"] != "en")
    print(f"    non-EN positives: {non_en_pos}  "
          f"({'EN-dominant — Stream D transfer strategy (see SOURCES.md)' if non_en_pos < real_pos*0.2 else 'ok'})")
    print("\n  next: python3 eval_oracle.py build   (group-aware train/eval split)")


def main() -> int:
    args = set(sys.argv[1:])
    if "--fresh" in args and os.path.exists(CORPUS_PATH):
        os.remove(CORPUS_PATH)
        print(f"removed {CORPUS_PATH}")
    if "--report" not in args:
        run_adapters()
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
