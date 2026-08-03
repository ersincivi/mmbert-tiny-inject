#!/usr/bin/env python3
"""mmbert-tiny-inject student → ONNX export (fp32 + dynamic-quant INT8) + verification.

Adapts the PROVEN SMS-model export path (legacy TorchScript exporter,
opset-14, `quantize_dynamic`): the SMS v8 artifact produced this way is live
on the OTA channel and round-trips through `tract` on device. NOT the QDQ
static-quant format — `quantize_dynamic` emits dynamically-quantized ops,
which is the path validated in tract (the old "QDQ" label in
MODEL_CONTRACT/sms prints was a misnomer for this same call).

Contract (mirrors the Rust inference wrapper's model loader):
  inputs  input_ids:int64[1,256] + attention_mask:int64[1,256]
  output  logits:float32[1,2]  in [benign, injection] order

Verification (needs onnxruntime, runs by default):
  1. PARITY — torch vs fp32-onnx vs int8-onnx logits on a sample of real eval
     windows; reports max |Δ| and verdict flips at the doc threshold.
  2. INT8 DOC-EVAL — the full max-pool document eval (same metric as
     train.py) run through the INT8 artifact: the number that actually ships.
     Compare against the checkpoint's ledger entry before signing/OTA.

A model and its vocab are a unit: the vocab used at train time is copied next
to the artifacts and hashed into the manifest.

Usage:
  python3 export_onnx.py                          # exports work/student_r7.pt
  python3 export_onnx.py --checkpoint work/student.pt --vocab work/vocab.txt
  python3 export_onnx.py --skip-doc-eval          # parity only (fast)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from train import (  # noqa: E402
    CLASSES, EVAL_JSONL, SEQ_LEN, VOCAB_SIZE, WORK,
    encode, load_docs,
)
from windowing import char_windows  # noqa: E402

HEADS, LAYERS = 4, 2  # train.py run_train fixed values (only hidden is a knob)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(torch, nn, vocab_list: list[str], vocab: dict[str, int], hidden: int):
    """EXACT mirror of train.py's Classifier — any drift breaks load_state_dict
    (strict=True, fail-loud by design)."""

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab_list), hidden, padding_idx=vocab["[PAD]"])
            self.pos = nn.Embedding(SEQ_LEN, hidden)
            enc = nn.TransformerEncoderLayer(hidden, HEADS, hidden * 4,
                                             batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(enc, LAYERS, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, len(CLASSES))

        def forward(self, ids, mask):
            pos = torch.arange(SEQ_LEN, device=ids.device).unsqueeze(0)
            x = self.emb(ids) + self.pos(pos)
            x = self.enc(x, src_key_padding_mask=(mask == 0))
            pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return self.head(self.norm(pooled))

    return Classifier()


def ort_session(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_logits(sess, ids: list[int], mask: list[int]):
    import numpy as np
    out = sess.run(["logits"], {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.array([mask], dtype=np.int64),
    })
    return out[0][0]  # [2]


def softmax_inj(logits) -> float:
    import math
    a, b = float(logits[0]), float(logits[1])
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    return eb / (ea + eb)


def parity_check(torch, model, sess_fp32, sess_int8, eval_docs, vocab,
                 sample: int, threshold: float) -> dict:
    """Window-level logit agreement torch ↔ fp32-onnx ↔ int8-onnx."""
    windows = []
    for d in eval_docs:
        windows.extend(w.text for w in char_windows(d["text"]))
        if len(windows) >= sample:
            break
    windows = windows[:sample]

    max_d_fp32 = max_d_int8 = 0.0
    flips_fp32 = flips_int8 = 0
    with torch.no_grad():
        for w in windows:
            ids, mask = encode(w, vocab)
            t = model(torch.tensor([ids]), torch.tensor([mask]))[0].tolist()
            f = ort_logits(sess_fp32, ids, mask)
            q = ort_logits(sess_int8, ids, mask)
            max_d_fp32 = max(max_d_fp32, max(abs(t[0] - float(f[0])), abs(t[1] - float(f[1]))))
            max_d_int8 = max(max_d_int8, max(abs(t[0] - float(q[0])), abs(t[1] - float(q[1]))))
            pt = softmax_inj(t) >= threshold
            if pt != (softmax_inj(f) >= threshold):
                flips_fp32 += 1
            if pt != (softmax_inj(q) >= threshold):
                flips_int8 += 1
    return {
        "windows": len(windows),
        "max_abs_logit_delta_fp32": round(max_d_fp32, 6),
        "max_abs_logit_delta_int8": round(max_d_int8, 6),
        "verdict_flips_fp32": flips_fp32,
        "verdict_flips_int8": flips_int8,
    }


def doc_eval(sess, eval_docs, vocab, threshold: float) -> dict:
    """Max-pool document eval through an ONNX session — mirrors
    train.py::evaluate_maxpool so the INT8 number is directly comparable to
    the checkpoint's ledger entry."""
    import collections
    tp = fp = tn = fn = 0
    fph_tot = fph_fp = 0
    lang_pos = collections.Counter()
    lang_tp = collections.Counter()
    for d in eval_docs:
        score = 0.0
        for w in char_windows(d["text"]):
            ids, mask = encode(w.text, vocab)
            score = max(score, softmax_inj(ort_logits(sess, ids, mask)))
        pred_inj = score >= threshold
        if d["label"] == "injection":
            lang_pos[d.get("lang", "?")] += 1
            if pred_inj:
                tp += 1
                lang_tp[d.get("lang", "?")] += 1
            else:
                fn += 1
        else:
            if d.get("fp_hard"):
                fph_tot += 1
                if pred_inj:
                    fph_fp += 1
            if pred_inj:
                fp += 1
            else:
                tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "fp_rate": round(fp / (fp + tn) if fp + tn else 0.0, 3),
        "fp_hard": round(fph_fp / fph_tot if fph_tot else 0.0, 3),
        "fp_hard_n": f"{fph_fp}/{fph_tot}",
        "per_lang_recall": {
            lg: f"{lang_tp[lg]}/{n}" for lg, n in lang_pos.most_common()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=str(WORK / "student_r7.pt"))
    ap.add_argument("--vocab", default=str(WORK / "vocab.txt"),
                    help="MUST be the vocab the checkpoint was trained with")
    ap.add_argument("--out", default=str(WORK / "model"))
    ap.add_argument("--hidden", type=int, default=192,
                    help="must match the checkpoint (train.py --hidden)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--parity-sample", type=int, default=512)
    ap.add_argument("--skip-doc-eval", action="store_true")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("torch not installed — run inside the shared training venv "
                 "(source ../.venv/bin/activate)")

    ckpt = Path(args.checkpoint)
    vocab_path = Path(args.vocab)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_list = vocab_path.read_text(encoding="utf-8").splitlines()
    if len(vocab_list) != VOCAB_SIZE:
        print(f"⚠ vocab {vocab_path} has {len(vocab_list)} tokens (expected {VOCAB_SIZE})")
    vocab = {t: i for i, t in enumerate(vocab_list)}

    model = build_model(torch, nn, vocab_list, vocab, args.hidden)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))  # strict — fail loud
    model.eval().cpu()
    print(f"loaded {ckpt.name} (hidden {args.hidden}, vocab {len(vocab_list)})")

    # ── export (the proven SMS-model path) ────────────────────────────────────
    ids = torch.zeros(1, SEQ_LEN, dtype=torch.int64)
    mask = torch.ones(1, SEQ_LEN, dtype=torch.int64)
    fp32_path = out_dir / "model.onnx"
    # dynamo=False → legacy TorchScript exporter: no onnxscript dependency and
    # a predictable opset-14 graph that round-trips through tract.
    torch.onnx.export(
        model, (ids, mask), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        opset_version=14, do_constant_folding=True, dynamo=False,
    )
    print(f"exported {fp32_path} (fp32, {fp32_path.stat().st_size / 1e6:.2f} MB)")

    int8_path = out_dir / "model_int8.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"exported {int8_path} (dynamic-quant INT8, "
              f"{int8_path.stat().st_size / 1e6:.2f} MB) — tract-smoke THIS file")
    except ImportError:
        sys.exit("onnxruntime missing → cannot produce the shipping INT8 "
                 "artifact (pip install onnxruntime)")

    vocab_copy = out_dir / "vocab.txt"
    shutil.copyfile(vocab_path, vocab_copy)

    # ── verification ──────────────────────────────────────────────────────────
    eval_docs = load_docs(EVAL_JSONL)
    sess_fp32 = ort_session(fp32_path)
    sess_int8 = ort_session(int8_path)

    parity = parity_check(torch, model, sess_fp32, sess_int8, eval_docs, vocab,
                          args.parity_sample, args.threshold)
    print(f"\nparity ({parity['windows']} eval windows):")
    print(f"  torch↔fp32  max|Δlogit| {parity['max_abs_logit_delta_fp32']}"
          f"  verdict flips {parity['verdict_flips_fp32']}")
    print(f"  torch↔int8  max|Δlogit| {parity['max_abs_logit_delta_int8']}"
          f"  verdict flips {parity['verdict_flips_int8']}")

    int8_metrics = None
    if not args.skip_doc_eval:
        print("\nINT8 doc-eval (max-pool, full eval set — the shipping number):")
        int8_metrics = doc_eval(sess_int8, eval_docs, vocab, args.threshold)
        print(f"  precision {int8_metrics['precision']} | recall {int8_metrics['recall']} "
              f"| F1 {int8_metrics['f1']}")
        print(f"  FP-rate {int8_metrics['fp_rate']} | FP-hard {int8_metrics['fp_hard']} "
              f"({int8_metrics['fp_hard_n']})")
        for lg, frac in int8_metrics["per_lang_recall"].items():
            print(f"    {lg:5s} {frac}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": ckpt.name,
        "checkpoint_sha256": sha256(ckpt),
        "config": {"hidden": args.hidden, "heads": HEADS, "layers": LAYERS,
                   "seq_len": SEQ_LEN, "vocab_size": len(vocab_list),
                   "classes": CLASSES, "threshold": args.threshold,
                   "opset": 14, "quant": "dynamic-QInt8"},
        "artifacts": {
            "model.onnx": {"sha256": sha256(fp32_path),
                           "bytes": fp32_path.stat().st_size},
            "model_int8.onnx": {"sha256": sha256(int8_path),
                                "bytes": int8_path.stat().st_size},
            "vocab.txt": {"sha256": sha256(vocab_copy),
                          "bytes": vocab_copy.stat().st_size},
        },
        "parity": parity,
        "int8_doc_eval": int8_metrics,
    }
    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"\nmanifest → {manifest_path}")
    print("next: tract smoke (core, `sms_ml` feature loader family) → Ed25519 "
          "OTA sign → escalation-routing wire-in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
