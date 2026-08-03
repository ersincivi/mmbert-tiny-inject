# `mmbert-tiny-inject` — prompt-injection semantic classifier

> **What this is.** `mmbert-tiny-inject` is the ML successor to the
> deterministic regex layer. It is a **binary text
> classifier** — input text → `{benign | injection}` + score — that sits
> as a second tier *above* the deterministic engine. It only
> owns the **semantic layer (intent)**; concealment (the lower, pattern-level
> layers) and PII stay fully deterministic.
>
> **Hard constraint (never violate).** On-device inference must be
> **zero-network, non-generative** (input → label + score). No cloud, no
> generative LLM. This is a classifier, not a chatbot.

## Training stack

| Stage | Choice |
|---|---|
| Teacher | `jhu-clsp/mmBERT-small`, embed-frozen — multilinguality comes from the teacher |
| Student | tiny 2-layer transformer, hidden 192, INT8 QDQ ~4.9 MB |
| Objective | class-weighted CE, then KL distillation (α 0.5, T 2.0), two-phase |
| Windows | 900 chars / 600 stride / 300 overlap, max-pooled to a document score |
| Runtime | `tract`, pure Rust, mmap, zero network |
| Delivery | signed download, atomic swap, bundled fallback |
| Contract | [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md) — `input_ids`/`attention_mask` → 2-class `logits` |

**The hard part is the data, not the training loop.** There is no ready
injection corpus, so most of the work in this repository is corpus engineering.

## Assembling a corpus

Five streams:

- **A — in-house attack taxonomy** *(not part of this public repository)*: a
  private catalogue of ~100 vectors. ⚠ Its entries are **descriptions +
  detection signatures, not weaponised payloads**, so they were usable only as
  **weak auxiliary positives** — quoted trigger-phrase fragments — never as
  bulk payload text. Training on descriptions of attacks would teach the model
  to flag *write-ups about* injection, which is the wrong target. The adapter
  and the catalogue are withheld here; the 152 rows it produced are excluded
  from the demo dataset.
- **B — license-safe public** (`SOURCES.md`): deepset (Apache-2.0, real
  labeled injection text — **primary anchor**), ProtectAI-22 blueprint,
  BIPIA indirect. Clean-room provenance discipline.
- **C — negatives** (FP-hard): benign SMS text from a companion corpus +
  legitimate text that *contains* injection-shaped words ("ignore",
  "verify", API tokens in dev docs). `ingest_sms_ham.py`.
- **D — multilingual coverage** via mmBERT cross-lingual transfer rather than
  translation.
- **E — synthetic balance** (Claude/Mistral) — later, to stop
  injection-only skew.

Plus a held-out **FP-eval oracle** (our own; PINT stays scoring-only).

## Normalized corpus schema (one JSON object per line)

Every ingest adapter emits rows in `corpus_work/inject.jsonl` with this shape
(mirrors SMS schema; `label` is 2-class here):

```json
{
  "text": "…",                      // the sample text
  "label": "injection",             // "benign" | "injection"
  "lang": "en",                     // ISO-639-1 best-effort
  "channel": "url_text",            // url_text|pdf|image|email|tool_result|sms|chat
  "source": "deepset",              // provenance key (matches SOURCES.md)
  "license": "apache-2.0",          // SPDX-ish; matches SOURCES.md
  "is_translated": false,           // true if machine-translated (Stream D)
  "is_synthetic": false,            // true if model-generated (Stream E)
  "weak_seed": false,               // true = weak/auxiliary positive (Stream A)
  "notes": ""                       // optional free text
}
```

`corpus_common.py` is the single writer + validator for this schema — every
ingest adapter imports it. Do **not** hand-emit rows.

## Layout

```
pipeline/
docs/PIPELINE.md           ← this file (pipeline authority)
docs/SOURCES.md            ← license inventory (READ before adding a source)
├── corpus_common.py         ← schema writer/validator/dedup + lang guess (stdlib-only)
├── ingest_deepset.py      ← Stream B anchor: deepset/prompt-injections (HF rows API)
├── ingest_sms_ham.py      ← Stream C negatives: SMS ham + FP-hard mining
├── build_corpus.py        ← orchestrator: run adapters → dedup → composition report
├── eval_oracle.py         ← honest held-out split (group-aware) + FP-cost scorer
└── corpus_work/           ← generated JSONL (git-ignored)
```

## FP-eval oracle (`eval_oracle.py`)

The eval harness that keeps us honest. Two modes:

```bash
python3 eval_oracle.py build             # group-aware train/eval split
python3 eval_oracle.py score --baseline  # validate scorer w/ keyword baseline
python3 eval_oracle.py score PREDS.jsonl # score a model's predictions
```

- **Group-aware split** — rows sharing a `group` key (bipia-ctx poisoned+clean
  twins) never straddle train/eval; the split self-checks 0 group leakage.
- **FP-cost is the money metric** — reports FP-rate overall AND on **FP-hard**
  negatives (benign text carrying injection-shaped words). A shield that cries
  wolf on legit content is worse than none.
- **Per-language recall probe** — measures whether detection transfers; flags
  languages with < 30 eval positives as "not honest to measure" (today only EN
  is measurable — the multilingual gap, quantified).
- **Baseline** — a crude keyword predictor proves the harness works before
  `mmbert-tiny-inject` exists. Current baseline (≈ the deterministic regex
  layer's reach): **precision 0.99, recall 0.35** → the regex floor misses ~65%
  of eval positives. That gap is exactly the classifier's job: lift recall
  (paraphrase + non-EN) without raising the FP-hard rate.

## Run

```bash
# individual adapters (each writes/append to corpus_work/inject.jsonl)
python3 ingest_deepset.py          # network: HF datasets-server rows API
SMS_CORPUS=/path/to/corpus.jsonl python3 ingest_sms_ham.py

# or the orchestrator (runs all + dedup + stratified split + prints report)
python3 build_corpus.py
```

All ingest scripts are **stdlib-only** (urllib, json, re) — matching the
`import_corpus.py` convention; no pyarrow/pandas/datasets needed. Only the
downstream `train.py` and distillation step needs torch.

## Status

- ✅ Scaffold + schema + `corpus_common.py` (2026-07-16).
- ✅ Stream B: `ingest_deepset.py` (direct-chat, 661), `ingest_jackhhao.py`
  (jailbreak Apache-2.0, 1203), `ingest_bipia.py` (**indirect** isolated, MIT,
  250), `ingest_bipia_context.py` (**indirect spliced-into-email**, MIT, 478),
  `ingest_nemotron.py` (**indirect agentic** CC-BY-4.0, exfiltration, 676).
- ✅ Stream C `ingest_sms_ham.py`. (Stream A used a private in-house taxonomy;
  its adapter and rows are withheld from this repository.)
- ✅ Stream D **decided = mmBERT transfer** (translation ruled out: the only
  multilingual set available is GPL). See SOURCES.md.
- ✅ **FP-eval oracle** `eval_oracle.py` — group-aware split (0 leakage),
  FP-cost + FP-hard + per-language transfer probe; baseline validated
  (regex-floor reach ≈ P0.99/R0.35).
- ✅ **Stream E** `synthesize.py` — template FP-hard dev-doc negatives (181) +
  paraphrase-varied injection (220), all `is_synthetic` → train-only. Adding
  them left the eval set **byte-identical** (proves synthetic can't inflate).
- ✅ **Stream D+E multilingual** `synthesize_multilingual.py` — 10 languages
  (de/fr/tr/es/it/pt/nl/pl/ru/zh), native-authored, ~510 rows, train-only.
  non-EN positives 110 → 560; vocab now covers Cyrillic + CJK subwords.
- Current build: **~5945 rows, ~2775 real positives, ~49% injection**; eval set
  892 rows (385 pos / 507 neg / 28 FP-hard), still only EN honestly measurable
  (real multilingual eval is a separate next step).
- ⏳ Stream B remainder (InjecAgent; qa/table/code context-assembly), keyed
  translation-augmentation (when an API key is available), more REAL dev-doc
  FP-hard negatives (to thicken eval FP-hard beyond 28).
- ⏳ 2-class train head + distill + tract wire-in + signed delivery (
  needs torch + M3 — the corpus + honest eval are ready and waiting).

> ⚠ **Honest-eval note (for the FP-eval oracle slice).** The current
> hash-on-text split can place an email's *clean* twin in train and its
> *poisoned* twin in val (same underlying `bipia-ctx` context). That risks the
> classifier learning the email body rather than the injection. The eval
> oracle must use **context-grouped or held-out-source** splitting, not
> per-row hashing.
