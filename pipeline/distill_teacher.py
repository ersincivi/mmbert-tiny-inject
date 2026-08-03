#!/usr/bin/env python3
"""mmbert-tiny-inject teacher (window-level) — fine-tune a multilingual encoder on the same
training windows the tiny student sees, then export per-window soft logits
keyed by window-hash for `train.py --train --teacher-logits`.

Adapted from `distill_teacher.py` in the SMS classifier project but not a mechanical copy. The one
real difference: SMS labels whole (short) messages, while mmbert-tiny-inject documents are
large, so the student trains on overlapping character windows (`windowing.py`).
The teacher must therefore be window-level too:

  - it applies the identical `training_windows` expansion (same dense/sparse/
    benign label discipline, same bipia-ctx hints via `train.doc_hint`);
  - every exported logit row is keyed by `train.window_hash(window_text)` — the
    student looks soft-labels up by the same hash, so the KL aligns
    window-for-window. Teacher & student keep their own tokenizers;
    distillation aligns on the 2-class [benign, injection] output, not tokens.

Length safety (the SMS lesson — measured, do not regress):
  900-char windows are not ≤256 mmBERT tokens. Measured over all 21,284 train
  windows: p50=158 p90=223 p99=327 max=749 tokens. A 256 cap would silently
  truncate 4.5% of windows — 674 of them injection, 574 from llmail (the most
  valuable adaptive-attack slice). Default --max-len is therefore 768 (= zero
  truncation on the current corpus) and the run measures the real distribution
  up front, warning loudly if the chosen cap cuts anything. Length-bucketing
  keeps the cost honest: most batches stay ≤256 wide; only the rare long
  windows pad wide. The teacher never ships, so a big cap costs training time
  only — zero effect on the on-device artifact.

Modes:
    python3 distill_teacher.py --prepare     # torch-free: expand + hash + stats
    python3 distill_teacher.py --limit 500   # bounded ~1-min timing run (keeps MPS)
    python3 distill_teacher.py               # full fine-tune + doc-eval + export
    python3 distill_teacher.py --smoke       # 500 windows, 1 epoch, CPU

Install (training venv, M3):  pip install torch transformers sentencepiece
Output:  work/teacher_logits.jsonl   {hash, logits:[benign,injection], label, lang}
Then (baseline first — proves the M3/MPS path and sets a shippable floor;
distillation was +0.02 on SMS: a lever, not a foundation):
    python3 train.py --train
    python3 train.py --train --teacher-logits work/teacher_logits.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

# Single source of truth: data path, label rules, and the hash key all come
# from the student's module — never re-implemented here.
from train import (CLASSES, CLASS_IX, MIN_LANG_POS, TRAIN_JSONL, EVAL_JSONL,
                   WORK, doc_hint, load_docs, window_hash)
from windowing import char_windows, training_windows, max_pool

TRUNC_WARN_FRAC = 0.005   # loud warning if the cap cuts >0.5% of windows


# ── data ──────────────────────────────────────────────────────────────────────

def window_rows(docs: list[dict]) -> list[tuple[str, str, str]]:
    """Documents → (window_text, label, lang) via the same expansion as the
    student's `expand_training` (same calls, plus lang carried for reporting)."""
    rows: list[tuple[str, str, str]] = []
    for d in docs:
        for text, label in training_windows(d["text"], d["label"],
                                            d.get("source", ""), doc_hint(d)):
            rows.append((text, label, d.get("lang", "")))
    return rows


def dedup_for_export(rows: list[tuple[str, str, str]]):
    """Unique windows by hash (identical text → identical logits). Returns
    (items, n_label_conflicts); a hash seen with different gold labels is a
    corpus smell worth surfacing, not an error (the teacher's soft label is
    text-derived either way)."""
    uniq: dict[str, tuple[str, str, str]] = {}
    conflicts = 0
    for text, label, lang in rows:
        h = window_hash(text)
        prev = uniq.get(h)
        if prev is None:
            uniq[h] = (text, label, lang)
        elif prev[1] != label:
            conflicts += 1
    return list(uniq.items()), conflicts


# ── prepare (torch-free validation) ───────────────────────────────────────────

def prepare() -> int:
    docs = load_docs(TRAIN_JSONL)
    rows = window_rows(docs)
    items, conflicts = dedup_for_export(rows)
    by_label = collections.Counter(lbl for _, lbl, _ in rows)
    by_lang = collections.Counter(lang for _, _, lang in rows)
    multi = sum(1 for d in docs if len(char_windows(d["text"])) > 1)
    chars = sorted(len(t) for t, _, _ in rows)
    q = lambda p: chars[min(len(chars) - 1, int(len(chars) * p))]
    print("=" * 60)
    print("mmbert-tiny-inject teacher — prepare (window expansion + hash keying)")
    print("=" * 60)
    print(f"  train docs       : {len(docs)}  ({multi} multi-window, "
          f"{100*multi/len(docs):.1f}%)")
    print(f"  training windows : {len(rows)}  ({dict(by_label)})")
    print(f"  unique hashes    : {len(items)}  "
          f"(dup windows {len(rows)-len(items)}, label-conflicts {conflicts})")
    print(f"  window chars     : p50 {q(.5)}  p90 {q(.9)}  max {chars[-1]}")
    print(f"  languages        : {dict(by_lang.most_common(12))}")
    print(f"  eval docs        : {len(load_docs(EVAL_JSONL))} (teacher doc-eval set; "
          f"NEVER trained on)")
    print("\n  hash-key contract: train.window_hash == sha1(window text) — the "
          "student resolves\n  teacher logits by the same function, so KL aligns "
          "window-for-window.")
    print("\n  next: python3 distill_teacher.py --limit 500   (bounded timing on MPS)")
    print("        python3 distill_teacher.py                 (full run)")
    return 0


# ── train + eval + export (torch) ─────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepare", action="store_true",
                    help="torch-free: validate window expansion + hash keying")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base",
                    help="HF teacher encoder (MIT). Default mmBERT (1800+ langs); "
                         "alt: microsoft/mdeberta-v3-base.")
    ap.add_argument("--out", default=str(WORK / "teacher_logits.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32,
                    help="frozen embedding + length-bucketing leaves headroom for "
                         "32-64 on 16GB MPS.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=768,
                    help="teacher token cap per window. 768 = ZERO truncation on "
                         "the current corpus (measured max 749; a 256 cap would cut "
                         "4.5%% of windows, mostly llmail injections — the SMS "
                         "truncation mistake, do not repeat). The run re-measures "
                         "and warns if the cap cuts >0.5%%.")
    # 16GB OOM fix (proven on the SMS teacher): freeze the 256k-vocab embedding
    # so Adam doesn't allocate momentum for it; pretrained embeddings already
    # know the languages. Teacher = training-time only → soft labels stay good.
    ap.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction,
                    default=True, help="freeze embedding params (default ON).")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="also freeze the first N encoder layers.")
    ap.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True,
                    help="after training, max-pool doc-eval on inject_eval.jsonl — "
                         "the teacher's F1 is the student's quality ceiling. "
                         "Auto-skipped for --smoke/--limit runs.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="500 windows, 1 epoch, CPU — plumbing check")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap train windows to N for a BOUNDED timing run on the "
                         "real device (keeps MPS). e.g. --limit 500 → read "
                         "rows/sec in ~1 min before the full multi-hour run.")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        return prepare()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise SystemExit("transformers/torch not installed — source ../.venv/bin/activate "
                         "(shared training venv; "
                         "or run --prepare, which is torch-free)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cpu" if (args.cpu or args.smoke)
              else "mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    rows = window_rows(load_docs(TRAIN_JSONL))
    full_n = len(rows)
    partial = False
    if args.smoke:
        rows = rows[:500]
    if args.limit and len(rows) > args.limit:
        # Random sample, not the corpus head — the file starts with short
        # deepset rows, so a head-slice would overestimate rows/sec badly and
        # give a false full-run ETA.
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
        partial = True
        print(f"⚠ --limit {args.limit}: BOUNDED timing run on a seeded random "
              f"sample (exported logits are partial — NOT for a real student train)")
    print(f"teacher {args.model} | device {device} | train windows {len(rows):,}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(CLASSES)).to(device)

    # Freeze big embedding (+ optional early layers); arch-agnostic name match.
    import re
    if args.freeze_embeddings:
        for name, p in model.named_parameters():
            if "embed" in name.lower():
                p.requires_grad = False
    if args.freeze_layers > 0:
        for name, p in model.named_parameters():
            m = re.search(r"\.layers?\.(\d+)\.", name)
            if m and int(m.group(1)) < args.freeze_layers:
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    nt = sum(p.numel() for p in trainable)
    ntot = sum(p.numel() for p in model.parameters())
    print(f"trainable {nt/1e6:.0f}M / {ntot/1e6:.0f}M params "
          f"(embed-frozen={args.freeze_embeddings}, "
          f"first-{args.freeze_layers}-layers-frozen)")

    texts = [r[0] for r in rows]
    labels = [CLASS_IX[r[1]] for r in rows]

    # Pre-tokenize once (no padding): per-window ids reused every epoch + the
    # lengths that drive bucketing and the truncation guard below.
    enc_all = tok(texts, truncation=True, max_length=args.max_len, padding=False)
    all_ids = enc_all["input_ids"]
    lengths = [len(x) for x in all_ids]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # Length-safety guard (the SMS lesson): measure what the cap actually cuts.
    # A truncated row tokenizes to exactly max_len, so re-check those candidates
    # without the cap — cheap (few rows) and exact.
    at_cap = [i for i, ln in enumerate(lengths) if ln >= args.max_len]
    truncated = 0
    if at_cap:
        full = tok([texts[i] for i in at_cap], truncation=False, padding=False)
        truncated = sum(1 for x in full["input_ids"] if len(x) > args.max_len)
    sl = sorted(lengths)
    q = lambda p: sl[min(len(sl) - 1, int(len(sl) * p))]
    print(f"teacher token fill (cap {args.max_len}): "
          f"p50 {q(.5)}  p90 {q(.9)}  p99 {q(.99)}  max {sl[-1]} | "
          f"TRUNCATED {truncated} ({100*truncated/max(1,len(rows)):.2f}%)")
    if truncated / max(1, len(rows)) > TRUNC_WARN_FRAC:
        print("  ⚠⚠ CAP IS CUTTING WINDOWS — injection tails may get wrong soft-"
              "labels.\n  Raise --max-len (bucketing keeps short batches cheap; "
              "teacher never ships).")

    # Length-bucketing (proven on the SMS teacher): sort by token length so each
    # batch pads only to its own max, quantized to PAD_MULTIPLE so MPS sees few
    # distinct shapes (a shape per batch would re-trigger graph recompiles).
    #
    # Token-budget cap on top (the 768-cap companion): --batch rows is the cap
    # for shapes ≤256; longer shapes automatically take fewer rows so one step
    # never exceeds batch×256 padded tokens. Without this, the rare 768-wide
    # batch at 32 rows would spike peak attention memory ~9× vs a 256-wide one
    # (O(L²)) — an OOM/latency hazard on 16GB MPS. Rows only ever shrink, never
    # grow, so SGD dynamics for the short/typical batches are unchanged.
    PAD_MULTIPLE = 32
    TOKEN_BUDGET = args.batch * 256

    def quantize(ln):
        return min(args.max_len, ((ln + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE)

    def make_batch(idx):
        blen = quantize(max(lengths[j] for j in idx))
        ids = torch.full((len(idx), blen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), blen), dtype=torch.long)
        for k, j in enumerate(idx):
            row = all_ids[j][:blen]
            ids[k, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[k, : len(row)] = 1
        return ids.to(device), mask.to(device)

    def length_batches(indices):
        s = sorted(indices, key=lambda j: lengths[j])
        out, cur = [], []
        for j in s:
            cand = cur + [j]
            # sorted ascending → lengths[j] is the running max of the batch
            if cur and (len(cand) > args.batch
                        or quantize(lengths[j]) * len(cand) > TOKEN_BUDGET):
                out.append(cur)
                cur = [j]
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    epochs = 1 if (args.smoke or args.limit) else args.epochs
    train_batches = length_batches(range(len(rows)))
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_batches)  # shuffle batch order; length stays homogeneous
        total = 0.0
        t0 = time.perf_counter()
        for idx in train_batches:
            ids, mask = make_batch(idx)
            y = torch.tensor([labels[j] for j in idx], dtype=torch.long, device=device)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += out.loss.item() * len(idx)
        dt = time.perf_counter() - t0
        # rows/sec = the honest cross-run metric (s/step isn't comparable across
        # batch/shape changes). Full-run ETA ≈ windows × epochs ÷ rows_per_sec.
        rps = len(rows) / max(dt, 1e-9)
        print(f"  epoch {epoch}/{epochs}: loss {total/len(rows):.4f} | "
              f"{dt:.1f}s | {rps:.1f} rows/sec "
              f"({len(train_batches)} batches)")
        if partial:
            # export ≈ forward-only over unique windows (~3× train speed);
            # doc-eval ≈ forward over ~4.3k eval windows.
            train_min = full_n * args.epochs / rps / 60
            extra_min = (full_n * 0.83 + 4335) / (3 * rps) / 60
            print(f"  → FULL-RUN ETA @ {rps:.0f} rows/sec: "
                  f"~{train_min:.0f} min train ({args.epochs} epochs × {full_n:,}) "
                  f"+ ~{extra_min:.0f} min export+eval")

    # Teacher quality ceiling: same max-pool doc-eval the student uses (mirrors
    # deployment), with the teacher's own tokenizer. Skipped on smoke/limit.
    if args.eval and not (args.smoke or partial):
        eval_maxpool_teacher(model, tok, device, torch, args.max_len)

    # Export unique-hash soft labels (identical window text → identical logits).
    items, conflicts = dedup_for_export(rows)
    if conflicts:
        print(f"⚠ {conflicts} window hash(es) carry conflicting gold labels "
              f"(corpus smell — soft label is text-derived, export keeps first)")
    items.sort(key=lambda kv: len(kv[1][0]))  # char-length sort ≈ token bucketing
    model.eval()
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(items), args.batch):
            chunk = items[i: i + args.batch]
            enc = tok([t for _, (t, _, _) in chunk], truncation=True,
                      max_length=args.max_len, padding=True,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits.float().cpu().tolist()
            for (h, (_, label, lang)), lg in zip(chunk, logits):
                f.write(json.dumps({"hash": h,
                                    "logits": [round(v, 5) for v in lg],
                                    "label": label, "lang": lang},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"exported {n:,} teacher soft-labels → {out_path}")
    print(f"Next: python3 train.py --train                    (baseline FIRST)")
    print(f"      python3 train.py --train --teacher-logits {out_path}")
    return 0


def eval_maxpool_teacher(model, tok, device, torch, max_len) -> None:
    """Document-level max-pool eval of the teacher on inject_eval.jsonl — the
    student's honest metric applied to the teacher, so the distillation ceiling
    (and the teacher→student gap) is directly readable."""
    eval_docs = load_docs(EVAL_JSONL)
    model.eval()
    tp = fp = tn = fn = 0
    fph_tot = fph_fp = 0
    lang_pos = collections.Counter()
    lang_tp = collections.Counter()
    with torch.no_grad():
        for d in eval_docs:
            wins = char_windows(d["text"])
            enc = tok([w.text for w in wins], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, -1)[:, CLASS_IX["injection"]].tolist()
            pred = "injection" if max_pool(probs) >= 0.5 else "benign"
            gold = d["label"]
            if gold == "injection":
                lang_pos[d["lang"]] += 1
                if pred == "injection":
                    tp += 1
                    lang_tp[d["lang"]] += 1
                else:
                    fn += 1
            else:
                fph = bool(d.get("fp_hard"))
                if fph:
                    fph_tot += 1
                if pred == "injection":
                    fp += 1
                    if fph:
                        fph_fp += 1
                else:
                    tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print("\n  ── TEACHER max-pool doc eval (student's quality ceiling) ──")
    print(f"  precision {prec:.3f} | recall {rec:.3f} | F1 {f1:.3f}")
    print(f"  FP-rate {fp/(fp+tn) if fp+tn else 0:.3f} | "
          f"FP-hard {fph_fp/fph_tot if fph_tot else 0:.3f} ({fph_fp}/{fph_tot})")
    for lg, cnt in lang_pos.most_common():
        flag = "" if cnt >= MIN_LANG_POS else "  ⚠ n<%d" % MIN_LANG_POS
        print(f"    {lg:5s} recall {lang_tp[lg]/cnt if cnt else 0:.3f} "
              f"({lang_tp[lg]}/{cnt}){flag}")


if __name__ == "__main__":
    raise SystemExit(main())
