# `mmbert-tiny-inject` — model contract

The on-device Rust engine and this training script must agree **exactly** on
tokenization, windowing, and aggregation, or the model that trains well offline
will misbehave on device. This file is the single source of that agreement.

## Classes

2-class, logits in this order (index = class id):

| id | class |
|----|-------|
| 0  | benign |
| 1  | injection |

## Tokenization (lockstep with SMS engine)

- `pre_tokenize` = `build_vocab.pre_tokenize` = byte-level mirror of the Rust
  inference wrapper's own `pre_tokenize` (NFKC + lowercase +
  whitespace/ASCII-punct splitting). Reused verbatim, never re-implemented.
- Greedy longest-match wordpiece to ids, `[CLS] … [SEP]` + `[PAD]`, capped at
  **SEQ_LEN = 256 tokens** per window (SMS was 128; prompts are larger).
- Vocab: `work/vocab.txt`, one token per line, index = line number.

## Windowing (THE difference from SMS — see windowing.py)

Prompts are large and an injection can sit anywhere, so we do **not**
head-truncate. A document is split into overlapping character windows:

| constant | value | meaning |
|----------|-------|---------|
| `WINDOW_CHARS` | 900 | characters per window (fits ~256 tokens; measured p99 fill 224) |
| `STRIDE_CHARS` | 600 | advance per window |
| `OVERLAP_CHARS` | 300 | `WINDOW_CHARS − STRIDE_CHARS` |

Guarantees (asserted in `windowing.py` self-test): full character coverage, the
final window is anchored to the text end (no dropped tail), and any injected
instruction shorter than `OVERLAP_CHARS` (300) is wholly inside at least one
window even if it straddles a boundary.

## Aggregation — MAX-POOL

```
document_injection_score = max( softmax(window_logits)[injection]  for each window )
verdict = injection  if document_injection_score >= THRESHOLD  else benign
```

`THRESHOLD` default 0.5 (calibrate against the FP-eval oracle, esp. the FP-hard
negative rate — the money metric). The engine MUST score every window and take
the max; scoring only the first window would reproduce the head-truncation bug
this design exists to avoid.

## Training-label discipline (windowing.training_windows)

When a long document is expanded into windows for training:
- **benign** → every window labeled benign.
- **dense injection** (jailbreak/chat/agentic/synthetic — the whole text is the
  attack) → every window labeled injection.
- **sparse injection** (`bipia-ctx`: benign body + spliced instruction) → only
  the injection-bearing window (chosen by the start/end/middle position hint) is
  labeled injection; benign windows of that document are NOT emitted as
  positives (that would teach the model that benign prose is injection → the
  FP-hard failure mode).

## Export

`model.onnx` (fp32) + `model_int8.onnx` (**dynamic-quant INT8** via
`onnxruntime quantize_dynamic`/QInt8 — NOT static QDQ; dynamic-quant is the
path proven to round-trip through `tract`), opset-14, inputs
`input_ids:int64[1,256]` + `attention_mask:int64[1,256]`, output
`logits:float32[1,2]` — same shape family as the SMS model, so the `tract`
loader + OTA pipeline (Ed25519-signed, atomic swap) are reused unchanged. The
model is a SEPARATE artifact from the SMS model (own contract, own OTA channel).
Export + verification: `export_onnx.py` (parity check + INT8 doc-eval +
sha256 manifest; ships `vocab.txt` next to the model — they are a unit).
