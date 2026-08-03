#!/usr/bin/env python3
"""Train the on-device prompt-injection classifier (mmbert-tiny-inject, 2-class).

Successor to `train.py` in the SMS classifier project, NOT a copy — prompts are large, so the core
difference is length handling: instead of head-truncating (which drops tail
injections), documents are split into overlapping character windows
(`windowing.py`) and the document score is a MAX-POOL over its windows. See
that module + MODEL_CONTRACT.md for the contract the on-device Rust engine
mirrors.

Data: `corpus_work/inject_{train,eval}.jsonl` (produced by
`eval_oracle.py build` — group-aware, honest split). Labels are
2-class: benign | injection.

Two modes:
  python3 train.py --prepare   # torch-FREE: expand windows, build vocab, encode,
                               #   report length/window stats. Validates the
                               #   char-counted pipeline without a GPU.
  python3 train.py --train     # needs torch (M3/MPS). Trains + max-pool eval.
  python3 train.py --train --smoke   # 2k windows, 1 epoch, CPU — pipeline check
  python3 train.py --train --teacher-logits work/teacher_logits.jsonl
                               # distillation (optional lever, AFTER a clean
                               #   baseline run): adds KL(student||teacher) per
                               #   WINDOW, keyed by window_hash — see
                               #   distill_teacher.py.

LOCKSTEP: tokenization = `build_vocab.pre_tokenize` in the SMS classifier project (mirror of
the Rust inference wrapper), imported not re-implemented. The
on-device prompt engine will reuse the same pre_tokenize + greedy encode.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus_work"
TRAIN_JSONL = CORPUS / "inject_train.jsonl"
EVAL_JSONL = CORPUS / "inject_eval.jsonl"
WORK = HERE / "work"
VOCAB_PATH = WORK / "vocab.txt"

# Lockstep tokenizer (byte-level mirror of the Rust inference wrapper).
# `build_vocab.py` is shared with the SMS classifier project and vendored here
# so this repository stays self-contained; keep the two copies in sync.
from build_vocab import pre_tokenize, SPECIALS  # noqa: E402

from windowing import (  # noqa: E402
    SEQ_LEN, WINDOW_CHARS, STRIDE_CHARS, OVERLAP_CHARS,
    char_windows, training_windows, max_pool,
)

CLASSES = ["benign", "injection"]        # logits order (MODEL_CONTRACT)
CLASS_IX = {c: i for i, c in enumerate(CLASSES)}
VOCAB_SIZE = 20000
MIN_LANG_POS = 30


# ── data ──────────────────────────────────────────────────────────────────────

def load_docs(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path} — run: python3 eval_oracle.py build")
    # NOTE: split on "\n" only — str.splitlines() also breaks on Unicode line
    # separators ( , \x85, …) that occur INSIDE text fields, which would
    # shatter a JSON record. json.dumps escapes real newlines, so "\n" is safe.
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_hint(d: dict) -> str:
    """Sparse-source window position hint ("start"/"middle"/"end") from notes."""
    if d.get("source") == "bipia-ctx" and "/" in d.get("notes", ""):
        return d["notes"].split("/")[1]
    return ""


def window_hash(text: str) -> str:
    """Distillation lookup key: sha1 of the EXACT window text. Teacher
    (distill_teacher.py) and student compute this identically, so KL aligns
    window-for-window without sharing a tokenizer."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def expand_training(docs: list[dict]) -> list[tuple[str, str]]:
    """Documents → (window_text, label) pairs via the windowing label rules."""
    out: list[tuple[str, str]] = []
    for d in docs:
        out.extend(training_windows(d["text"], d["label"], d.get("source", ""), doc_hint(d)))
    return out


# ── vocab + encode (greedy wordpiece-lite; shares pre_tokenize with SMS) ───────

def build_vocab(pairs: list[tuple[str, str]]) -> list[str]:
    word_freq: collections.Counter = collections.Counter()
    chars: set[str] = set()
    for text, _ in pairs:
        for w in pre_tokenize(text):
            word_freq[w] += 1
            chars.update(w)
    vocab = list(SPECIALS)
    vocab += ["##" + c for c in sorted(chars)]      # char continuations (fallback)
    vocab += [c for c in sorted(chars)]             # single chars (word start)
    seen = set(vocab)
    for w, _ in word_freq.most_common():
        if len(vocab) >= VOCAB_SIZE:
            break
        if w not in seen:
            vocab.append(w)
            seen.add(w)
    return vocab


def encode(text: str, vocab: dict[str, int]) -> tuple[list[int], list[int]]:
    """Greedy longest-match to SEQ_LEN ids (mirror of the engine encode)."""
    unk, cls, sep, pad = vocab["[UNK]"], vocab["[CLS]"], vocab["[SEP]"], vocab["[PAD]"]
    ids = [cls]
    for word in pre_tokenize(text):
        if len(ids) >= SEQ_LEN - 1:
            break
        if len(word) > 100:
            ids.append(unk)
            continue
        n, start, pieces, ok = len(word), 0, [], True
        while start < n:
            end = n
            while end > start:
                cand = word[start:end] if start == 0 else "##" + word[start:end]
                if cand in vocab:
                    pieces.append(vocab[cand])
                    break
                end -= 1
            else:
                ok = False
                break
            start = end
        ids.extend(pieces if ok else [unk])
    ids = ids[: SEQ_LEN - 1] + [sep]
    mask = [1] * len(ids) + [0] * (SEQ_LEN - len(ids))
    ids = ids + [pad] * (SEQ_LEN - len(ids))
    return ids[:SEQ_LEN], mask[:SEQ_LEN]


# ── prepare (torch-free validation of the length-aware pipeline) ───────────────

def prepare() -> int:
    WORK.mkdir(exist_ok=True)
    train_docs = load_docs(TRAIN_JSONL)
    eval_docs = load_docs(EVAL_JSONL)
    pairs = expand_training(train_docs)

    vocab_list = build_vocab(pairs)
    VOCAB_PATH.write_text("\n".join(vocab_list), encoding="utf-8")
    vocab = {t: i for i, t in enumerate(vocab_list)}

    # window inflation
    docs_by_lbl = collections.Counter(d["label"] for d in train_docs)
    win_by_lbl = collections.Counter(lbl for _, lbl in pairs)

    # within-window token fit: how many windows still hit the SEQ_LEN cap?
    over = 0
    fill = []
    for text, _ in pairs:
        ids, mask = encode(text, vocab)
        used = sum(mask)
        fill.append(used)
        if used >= SEQ_LEN:      # window itself filled the token budget
            over += 1

    # multi-window documents (the length-aware path actually exercised)
    multi = sum(1 for d in train_docs if len(char_windows(d["text"])) > 1)

    print("=" * 60)
    print("AI-prompt classifier — prepare (length-aware pipeline)")
    print("=" * 60)
    print(f"  train docs   : {len(train_docs)}  ({dict(docs_by_lbl)})")
    print(f"  eval docs    : {len(eval_docs)}")
    print(f"  multi-window docs: {multi}  ({100*multi/len(train_docs):.1f}% needed windowing)")
    print(f"  training windows : {len(pairs)}  ({dict(win_by_lbl)})")
    print(f"  vocab size   : {len(vocab_list)}  -> {VOCAB_PATH}")
    print(f"\n  window token fill (SEQ_LEN={SEQ_LEN}):")
    fill.sort()
    if fill:
        q = lambda p: fill[min(len(fill) - 1, int(len(fill) * p))]
        print(f"    p50 {q(.5)}  p90 {q(.9)}  p99 {q(.99)}  max {fill[-1]}")
        print(f"    windows hitting the cap: {over} ({100*over/len(fill):.1f}%) "
              f"— overlap {OVERLAP_CHARS}c ensures a capped window's tail is "
              f"re-covered by the next window, so no injection is lost")
    print("\n  next: python3 train.py --train   (needs torch)")
    return 0


# ── train (torch) ──────────────────────────────────────────────────────────────

def run_train(args) -> int:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("torch not installed — `pip install torch onnx onnxruntime` "
                 "in a venv (M3/MPS). Use --prepare to validate the pipeline "
                 "without torch.")

    if not VOCAB_PATH.exists():
        prepare()
    # --vocab: score a checkpoint with the vocab it was TRAINED with. A model
    # and its vocab are a unit — token ids shift whenever the corpus changes,
    # so re-scoring an old checkpoint against a freshly built vocab.txt would
    # silently produce garbage (see rebuild_r1_artifacts.py).
    vocab_src = Path(args.vocab) if args.vocab else VOCAB_PATH
    vocab_list = vocab_src.read_text(encoding="utf-8").splitlines()
    vocab = {t: i for i, t in enumerate(vocab_list)}

    train_pairs = [] if args.eval_checkpoint else expand_training(load_docs(TRAIN_JSONL))
    eval_docs = load_docs(EVAL_JSONL)
    if args.smoke:
        train_pairs = train_pairs[:2000]

    device = ("cpu" if args.smoke                      # smoke = CPU (per docstring)
              else "mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"vocab {len(vocab_list)} | device {device} | seq_len {SEQ_LEN} | "
          f"train windows {len(train_pairs)} | eval docs {len(eval_docs)}")

    # Distillation (optional): precomputed per-WINDOW teacher soft-labels, keyed
    # by window_hash. Uncovered windows fall back to CE-only (KL-masked) so a
    # --limit'ed (partial) teacher export degrades gracefully instead of crashing.
    teacher = None
    if args.teacher_logits:
        teacher = {}
        for line in Path(args.teacher_logits).open(encoding="utf-8"):
            d = json.loads(line)
            teacher[d["hash"]] = d["logits"]
        covered = sum(1 for t, _ in train_pairs if window_hash(t) in teacher)
        print(f"distillation ON: {len(teacher):,} teacher soft-labels | "
              f"alpha {args.distill_alpha} temp {args.distill_temp} | "
              f"window coverage {covered:,}/{len(train_pairs):,} "
              f"({covered/max(1, len(train_pairs)):.0%})")
        if covered < len(train_pairs):
            print("  ⚠ uncovered windows train CE-only — expected ONLY for a "
                  "--limit'ed teacher run; for a real train, re-export in full")

    # tiny 2-layer transformer (same family as make_smoke_model; 2-class head)
    # r4: hidden 128→192, feedforward hidden*2→hidden*4, +LayerNorm — matches
    # SMS v8 capacity band more closely (SMS: hidden=256, ff=1024; mmbert-tiny-inject now:
    # hidden=192, ff=768). The mmbert-tiny-inject r2/r3 distillation failed at hidden=128
    # because the student lacked capacity to follow the teacher across languages.
    hidden, heads, layers = args.hidden, 4, 2

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab_list), hidden, padding_idx=vocab["[PAD]"])
            self.pos = nn.Embedding(SEQ_LEN, hidden)
            enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4,
                                             batch_first=True, dropout=0.1)
            # enable_nested_tensor=False: the nested-tensor fast path calls
            # aten::_nested_tensor_from_mask_left_aligned, unimplemented on MPS
            # (M3 is the training device) — hit in the 2026-07-24 smoke.
            self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden)   # r4: stabilize pooled repr (SMS has this)
            self.head = nn.Linear(hidden, len(CLASSES))

        def forward(self, ids, mask):
            pos = torch.arange(SEQ_LEN, device=ids.device).unsqueeze(0)
            x = self.emb(ids) + self.pos(pos)
            x = self.enc(x, src_key_padding_mask=(mask == 0))
            pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return self.head(self.norm(pooled))

    torch.manual_seed(0)  # reproducible weights/dropout — r-rounds must differ
                          # by their KNOB, not by init noise (TRAINING-LOG r3)
    model = Classifier().to(device)

    # Eval-only: score an existing checkpoint against the CURRENT eval set.
    # Purpose: after a data round changes the eval set, the old headline number
    # is not comparable — the honest baseline for the next round is the old
    # checkpoint RE-SCORED on the new eval (TRAINING-LOG "COMPARABILITY BREAK").
    if args.eval_checkpoint:
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device))
        model.eval()
        print(f"eval-only: {args.eval_checkpoint} (vocab {vocab_src}) "
              f"vs {EVAL_JSONL.name} — no training")
        evaluate_maxpool(model, eval_docs, vocab, device, torch)
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    import torch.nn.functional as F
    import math
    from torch.optim.lr_scheduler import LambdaLR

    # r4: LR warmup (first 10% of steps) + cosine decay. Warmup lets the model
    # learn basic structure (embeddings, common patterns) before distillation
    # gradients push it toward the teacher's distribution.
    total_steps = (len(train_pairs) // args.batch + 1) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    scheduler = LambdaLR(opt, lr_lambda)

    # r4: focal loss (ported from the SMS classifier project). Down-weights easy examples
    # (where the model is already confident), helping it focus on hard non-EN
    # cases instead of coasting on the EN-dominant mass. gamma=0 → plain CE.
    def focal_loss(logits, y, gamma):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma) * ce

    def to_batch(pairs):
        ids, masks, ys, soft = [], [], [], []
        for text, lbl in pairs:
            i, m = encode(text, vocab)
            ids.append(i); masks.append(m); ys.append(CLASS_IX[lbl])
            if teacher is not None:
                soft.append(teacher.get(window_hash(text), [0.0] * len(CLASSES)))
        return (torch.tensor(ids, device=device),
                torch.tensor(masks, device=device),
                torch.tensor(ys, device=device),
                torch.tensor(soft, dtype=torch.float, device=device)
                if teacher is not None else None)

    import random
    rng = random.Random(0)
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train_pairs)
        # r4: two-phase training — CE-only for the first N epochs (learn hard-label
        # structure, esp. non-EN), then add distillation. This is the pretrain-then-
        # distill pattern; it prevents the KL gradient from dominating before the
        # model has established its representation.
        distill_active = (teacher is not None and args.distill_alpha > 0
                          and ep >= args.distill_start_epoch)
        phase = f"distill α{args.distill_alpha}" if distill_active else "CE-only"
        tot = 0.0
        for k in range(0, len(train_pairs), args.batch):
            ids, masks, ys, soft = to_batch(train_pairs[k:k + args.batch])
            opt.zero_grad()
            logits = model(ids, masks)
            ce = focal_loss(logits, ys, args.focal_gamma)   # r4: focal instead of plain CE
            if distill_active:
                # KL(student||teacher) at temperature T; all-zero soft row =
                # no teacher logit for this window → KL-masked (CE-only).
                T = args.distill_temp
                kl_mask = (soft.abs().sum(1) > 0).float()
                kl = F.kl_div(F.log_softmax(logits / T, dim=1),
                              F.softmax(soft / T, dim=1),
                              reduction="none").sum(1) * kl_mask
                loss = ((1.0 - args.distill_alpha) * ce
                        + args.distill_alpha * (T * T) * kl).mean()
            else:
                loss = ce.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # r4: grad clip (SMS has it)
            opt.step()
            scheduler.step()   # r4: LR warmup + cosine decay
            tot += loss.item()
        print(f"  epoch {ep+1}/{args.epochs}  loss {tot/max(1,len(train_pairs)//args.batch):.4f}  [{phase}]")

    evaluate_maxpool(model, eval_docs, vocab, device, torch)
    torch.save(model.state_dict(), WORK / "student.pt")
    print(f"\n  weights → {WORK / 'student.pt'} (pre-export checkpoint)")
    return 0


def evaluate_maxpool(model, eval_docs, vocab, device, torch) -> None:
    """Document-level eval that MIRRORS deployment: score every window, MAX-pool.

    This is the honest metric — it matches how the on-device engine will run, so
    a tail injection that head-truncation would miss is scored correctly here.
    """
    model.eval()
    scored = []  # (doc_score, gold_label, fp_hard, lang) — score once, judge many
    with torch.no_grad():
        for d in eval_docs:
            wins = char_windows(d["text"])
            scores = []
            for w in wins:
                ids, mask = encode(w.text, vocab)
                logit = model(torch.tensor([ids], device=device),
                              torch.tensor([mask], device=device))
                prob = torch.softmax(logit, -1)[0, CLASS_IX["injection"]].item()
                scores.append(prob)
            scored.append((max_pool(scores), d["label"],
                           bool(d.get("fp_hard")), d["lang"]))

    def metrics_at(thr):
        tp = fp = tn = fn = 0
        fph_tot = fph_fp = 0
        lang_pos = collections.Counter(); lang_tp = collections.Counter()
        for s, gold, fph, lang in scored:
            pred_inj = s >= thr
            if gold == "injection":
                lang_pos[lang] += 1
                if pred_inj: tp += 1; lang_tp[lang] += 1
                else: fn += 1
            else:
                if fph: fph_tot += 1
                if pred_inj:
                    fp += 1
                    if fph: fph_fp += 1
                else: tn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fphr = fph_fp / fph_tot if fph_tot else 0.0
        return prec, rec, f1, fpr, fphr, fph_fp, fph_tot, lang_pos, lang_tp

    prec, rec, f1, fpr, fphr, fph_fp, fph_tot, lang_pos, lang_tp = metrics_at(0.5)
    print("\n  ── max-pool document eval ──")
    print(f"  precision {prec:.3f} | recall {rec:.3f} | F1 {f1:.3f}")
    print(f"  FP-rate {fpr:.3f} | FP-hard {fphr:.3f} ({fph_fp}/{fph_tot})")
    for lg, n in lang_pos.most_common():
        flag = "" if n >= MIN_LANG_POS else "  ⚠ n<%d" % MIN_LANG_POS
        print(f"    {lg:5s} recall {lang_tp[lg]/n if n else 0:.3f} ({lang_tp[lg]}/{n}){flag}")

    # Threshold sweep — calibration diagnostic (TRAINING-LOG r2/r3: distill
    # shifts doc scores conservative; if the model still RANKS well, the fix is
    # the threshold, not the training). Scores are already collected — free.
    print("\n  ── threshold sweep (doc-level) ──")
    print("  thr    prec   rec    F1     FP-rate  FP-hard")
    best_thr, best_f1 = 0.5, f1
    for thr in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        p_, r_, f_, fpr_, fphr_, ffp_, ftot_, _, _ = metrics_at(thr)
        if f_ > best_f1:
            best_f1, best_thr = f_, thr
        print(f"  {thr:.2f}   {p_:.3f}  {r_:.3f}  {f_:.3f}  {fpr_:.3f}    "
              f"{fphr_:.3f} ({ffp_}/{ftot_})")
    if best_thr != 0.5:
        print(f"\n  best-F1 threshold {best_thr:.2f} — per-language recall there:")
        *_, lp, lt = metrics_at(best_thr)
        for lg, n in lp.most_common():
            flag = "" if n >= MIN_LANG_POS else "  ⚠ n<%d" % MIN_LANG_POS
            print(f"    {lg:5s} recall {lt[lg]/n if n else 0:.3f} ({lt[lg]}/{n}){flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-checkpoint", default="",
                    help="score this .pt checkpoint on the current eval set and "
                         "exit (no training). Pair with --vocab so the model is "
                         "decoded with the vocab it was trained with.")
    ap.add_argument("--vocab", default="",
                    help="vocab file to use instead of work/vocab.txt (e.g. "
                         "work/vocab_r1.txt from rebuild_r1_artifacts.py).")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    # r4: architecture + training aids (ported from SMS v8 that succeeded)
    ap.add_argument("--hidden", type=int, default=192,
                    help="transformer hidden dim (r1-r3 used 128; SMS v8 used 256). "
                         "192 is a compromise: ~5.5 MB INT8 vs r1's ~3 MB.")
    ap.add_argument("--focal-gamma", type=float, default=1.5,
                    help="focal-loss focusing param (0=plain CE; SMS v8 used 1.5). "
                         "Down-weights easy examples so the model focuses on hard non-EN.")
    ap.add_argument("--warmup-frac", type=float, default=0.1,
                    help="fraction of total steps for LR warmup (then cosine decay to 0).")
    ap.add_argument("--distill-start-epoch", type=int, default=2,
                    help="two-phase: CE-only for epochs < this, then add distillation. "
                         "Default 2 = first 2 epochs learn hard-label structure (esp. "
                         "non-EN), then distillation refines. Set 0 to disable (old behavior).")
    # Distillation (teacher = training-time only, never ships — see distill_teacher.py)
    ap.add_argument("--teacher-logits", default="",
                    help="JSONL of {hash, logits:[benign,injection]} from "
                         "distill_teacher.py. Adds KL(student||teacher) per window. "
                         "Empty = plain CE baseline (run this FIRST).")
    ap.add_argument("--distill-alpha", type=float, default=0.5,
                    help="loss = (1-a)*CE + a*T^2*KL. Only with --teacher-logits.")
    ap.add_argument("--distill-temp", type=float, default=2.0,
                    help="distill softmax temperature T.")
    args = ap.parse_args()
    if args.prepare and not args.train:
        return prepare()
    if args.train:
        return run_train(args)
    if args.eval_checkpoint:
        return run_train(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
