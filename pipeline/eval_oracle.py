#!/usr/bin/env python3
"""mmbert-tiny-inject FP-eval oracle — honest held-out split + false-positive-cost scorer.

Why this exists: a classifier
that flags legitimate content as injection is worse than useless — it trains
users to ignore the shield. So the eval must measure **FP cost** honestly, on
the *hardest* negatives (benign text that looks injection-shaped), and must
measure whether injection detection **transfers across languages** — not just
report a single aggregate accuracy.

Two modes:

  python3 eval_oracle.py build            # group-aware train/eval split
  python3 eval_oracle.py score PREDS.jsonl  # score predictions vs eval set
  python3 eval_oracle.py score --baseline   # score the built-in keyword baseline
                                            # (sanity check that the harness
                                            #  works BEFORE mmbert-tiny-inject exists)

Honest-eval guarantees in `build`:
  - GROUP-AWARE split: rows sharing a `group` key (e.g. bipia-ctx poisoned +
    clean twins) always land on the same side — the model can't learn the email
    body in train and be tested on its poisoned twin in eval.
  - weak_seed + synthetic rows NEVER enter eval (auxiliary, not honest truth).
  - FP-HARD tagging: benign eval rows whose text carries injection-shaped words
    are marked so the scorer can report FP-rate on them specifically.

`id` for a row = sha1 of its normalized text (stable). A predictions file is
JSONL of {"id": <row id>, "pred": "injection"|"benign"} OR {"id","score":0..1}.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

from corpus_common import CORPUS_PATH, dedup_key, is_fp_hard as _text_fp_hard, iter_rows

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "corpus_work", "inject_train.jsonl")
EVAL_PATH = os.path.join(HERE, "corpus_work", "inject_eval.jsonl")
BASELINE_PRED_PATH = os.path.join(HERE, "corpus_work", "baseline_preds.jsonl")
SUSPECTS_PATH = os.path.join(HERE, "label_suspects.jsonl")

EVAL_FRACTION = 0.18
# FP-hard benign rows are the money metric but are scarce (~550 real rows) —
# at the base fraction the eval lands ~36 of them and single-doc differences
# swamp the FP-hard trend (TRAINING-LOG round-close finding). Groupless real
# FP-hard rows therefore get a boosted eval fraction to reach the >=150
# statistical floor. Grouped rows (bipia-ctx twins) keep the base fraction so
# group-side decisions stay uniform within a group.
FP_HARD_EVAL_FRACTION = 0.30
SCORE_THRESHOLD = 0.5
MIN_LANG_POS = 30  # below this, per-language recall is "insufficient to measure"


def row_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def group_key(r: dict) -> str:
    g = r.get("group")
    return g if g else "row-" + dedup_key(r["text"])


def is_fp_hard(r: dict) -> bool:
    return r["label"] == "benign" and _text_fp_hard(r["text"])


def load_suspects() -> dict[str, str]:
    """Provenance'd label-suspect exclusion list (id -> reason).

    Rows whose source label is credibly WRONG (e.g. jailbreak-shaped text a
    dataset marks "benign") are excluded from BOTH sides of the split: in eval
    they corrupt the FP metric, in train they teach the wrong label. The list
    lives in label_suspects.jsonl — never silent row surgery on the corpus.
    """
    if not os.path.exists(SUSPECTS_PATH):
        return {}
    out = {}
    for line in open(SUSPECTS_PATH, "r", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        o = json.loads(line)
        out[o["id"]] = o.get("reason", "")
    return out


# ── build ────────────────────────────────────────────────────────────────────

def build() -> int:
    rows = list(iter_rows(CORPUS_PATH))
    if not rows:
        print("no corpus — run build_corpus.py first")
        return 1

    # Decide each group's side once (deterministic hash), then place every row
    # of the group there. weak/synthetic are forced to train regardless.
    def group_in_eval(gk: str, fraction: float) -> bool:
        h = int(hashlib.sha1(gk.encode()).hexdigest()[:8], 16)
        return (h % 100) < int(fraction * 100)

    suspects = load_suspects()
    excluded = []
    train, ev = [], []
    for r in rows:
        r = dict(r)
        r["id"] = row_id(r["text"])
        r["fp_hard"] = is_fp_hard(r)
        if r["id"] in suspects:
            excluded.append(r)
            continue
        forced_train = r.get("weak_seed") or r.get("is_synthetic")
        boosted = r["fp_hard"] and not r.get("group") and not forced_train
        fraction = FP_HARD_EVAL_FRACTION if boosted else EVAL_FRACTION
        if not forced_train and group_in_eval(group_key(r), fraction):
            ev.append(r)
        else:
            train.append(r)

    for path, subset in ((TRAIN_PATH, train), (EVAL_PATH, ev)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in subset:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report eval composition.
    ev_pos = [r for r in ev if r["label"] == "injection"]
    ev_neg = [r for r in ev if r["label"] == "benign"]
    fp_hard = [r for r in ev_neg if r["fp_hard"]]
    print("=" * 56)
    print(f"held-out eval set  ({len(ev)} rows | train {len(train)})")
    print("=" * 56)
    print(f"  injection : {len(ev_pos)}")
    print(f"  benign    : {len(ev_neg)}  (of which FP-hard: {len(fp_hard)})")
    if len(fp_hard) < 150:
        print("  ⚠ FP-hard eval below the 150 statistical floor — trend unreadable")
    if excluded:
        by_src = collections.Counter(r["source"] for r in excluded)
        print(f"  label-suspects excluded from BOTH sides: {len(excluded)} {dict(by_src)}")
    print("\n  eval positives by language (transfer probe):")
    lang_pos = collections.Counter(r["lang"] for r in ev_pos)
    for lg, n in lang_pos.most_common():
        flag = "" if n >= MIN_LANG_POS else "  ⚠ too few to measure recall honestly"
        print(f"    {lg:6s} {n:4d}{flag}")
    # Leakage guard self-check: no group spans both sides.
    tg = {group_key(r) for r in train}
    eg = {group_key(r) for r in ev}
    overlap = tg & eg
    print(f"\n  group-leakage check: {len(overlap)} groups span both sides "
          f"({'OK' if not overlap else 'BUG'})")
    print(f"\n  wrote {TRAIN_PATH}\n        {EVAL_PATH}")
    return 0


# ── score ────────────────────────────────────────────────────────────────────

def load_eval() -> list[dict]:
    if not os.path.exists(EVAL_PATH):
        print("no eval set — run: python3 eval_oracle.py build")
        sys.exit(1)
    return list(iter_rows(EVAL_PATH))


def baseline_predict(rows: list[dict]) -> dict:
    """Keyword baseline — a SANITY predictor, NOT the real mmbert-tiny-inject classifier.

    Flags a row as injection if it contains any obvious override/exfil marker.
    Deliberately crude: its whole purpose is to prove the scorer computes
    sensible metrics end-to-end before a trained model exists.
    """
    markers = (
        "ignore previous", "ignore all previous", "disregard previous",
        "forget all previous", "you are now", "system prompt", "reveal your",
        "print your", "as an ai", "new task:", "before completing",
        "before finishing", "you must", "send patient", "exfiltrate",
    )
    preds = {}
    for r in rows:
        low = r["text"].lower()
        preds[r["id"]] = "injection" if any(m in low for m in markers) else "benign"
    return preds


def load_preds(path: str) -> dict:
    preds = {}
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "pred" in o:
            preds[o["id"]] = o["pred"]
        elif "score" in o:
            preds[o["id"]] = "injection" if float(o["score"]) >= SCORE_THRESHOLD else "benign"
    return preds


def score(preds: dict, rows: list[dict], label: str) -> int:
    tp = fp = tn = fn = 0
    missing = 0
    fp_hard_total = fp_hard_fp = 0
    lang_pos = collections.Counter()
    lang_tp = collections.Counter()

    for r in rows:
        gold = r["label"]
        pred = preds.get(r["id"])
        if pred is None:
            missing += 1
            continue
        if gold == "injection":
            lang_pos[r["lang"]] += 1
            if pred == "injection":
                tp += 1
                lang_tp[r["lang"]] += 1
            else:
                fn += 1
        else:  # benign
            if r.get("fp_hard"):
                fp_hard_total += 1
            if pred == "injection":
                fp += 1
                if r.get("fp_hard"):
                    fp_hard_fp += 1
            else:
                tn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    fp_hard_rate = fp_hard_fp / fp_hard_total if fp_hard_total else 0.0

    print("=" * 56)
    print(f"FP-eval oracle — scoring [{label}]")
    print("=" * 56)
    print(f"  precision {prec:.3f} | recall {rec:.3f} | F1 {f1:.3f}")
    print(f"  confusion: TP {tp}  FP {fp}  TN {tn}  FN {fn}"
          + (f"  (missing preds: {missing})" if missing else ""))
    print("\n  ── false-positive cost (the money metric) ──")
    print(f"  FP-rate overall         : {fp_rate:.3f}  ({fp}/{fp+tn} benign flagged)")
    print(f"  FP-rate on FP-HARD negs : {fp_hard_rate:.3f}  ({fp_hard_fp}/{fp_hard_total})")
    print("\n  ── per-language recall (transfer probe) ──")
    for lg, npos in lang_pos.most_common():
        r = lang_tp[lg] / npos if npos else 0.0
        flag = "" if npos >= MIN_LANG_POS else "  ⚠ n<%d, not honest" % MIN_LANG_POS
        print(f"    {lg:6s} recall {r:.3f}  ({lang_tp[lg]}/{npos}){flag}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "score"):
        print(__doc__)
        return 1
    if sys.argv[1] == "build":
        return build()
    # score
    rows = load_eval()
    if "--baseline" in sys.argv[2:]:
        preds = baseline_predict(rows)
        with open(BASELINE_PRED_PATH, "w", encoding="utf-8") as fh:
            for rid, p in preds.items():
                fh.write(json.dumps({"id": rid, "pred": p}) + "\n")
        return score(preds, rows, "keyword-baseline (sanity, NOT mmbert-tiny-inject)")
    pred_paths = [a for a in sys.argv[2:] if not a.startswith("-")]
    if not pred_paths:
        print("usage: eval_oracle.py score PREDS.jsonl | --baseline")
        return 1
    return score(load_preds(pred_paths[0]), rows, os.path.basename(pred_paths[0]))


if __name__ == "__main__":
    raise SystemExit(main())
