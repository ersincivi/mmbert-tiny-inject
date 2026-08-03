# `mmbert-tiny-inject` corpus — source & license inventory

> **Read this before adding any source.** Clean-room discipline: we record
> provenance + license for every row (`source` + `license` fields). A source
> whose license we cannot confirm as training-safe does **not** enter the
> corpus. Training-data licensing is a separate question from blocklist
> redistribution.

## Status legend

- ✅ **ingested** — adapter written + rows in corpus
- 🔨 **planned** — license OK, adapter not yet written
- 🔬 **review** — license or provenance needs confirmation before ingest
- ⛔ **scoring-only** — usable for eval, NOT for training

## Positive (injection) sources

| Source | Rows | License | Lang | Indirect? | Status | Notes |
|---|---|---|---|---|---|---|
| **deepset/prompt-injections** | 662 (546 train / 116 test) | ⚠ see note | EN/DE/FR | ❌ direct-chat | ✅ ingested | **Primary anchor** — real labeled injection text. HF card is contradictory: repo top-level `apache-2.0`, `dataset_info` block `cc-by-4.0`. Both are training-safe **with attribution**; we tag rows `license: apache-2.0` and keep this note as the attribution/provenance record. |
| **jackhhao/jailbreak-classification** | 1306 (1044/262) | **Apache-2.0** ✅ (own card + ProtectAI list) | EN | ❌ direct-chat jailbreak | ✅ ingested | `ingest_jackhhao.py`. jailbreak→injection, benign→benign. ⚠ jailbreak ≠ indirect injection (shares override surface) → tagged `channel:chat` + `notes:jailbreak-class` for distinct weighting. |
| **Microsoft BIPIA** (isolated) | text 15-cat + code 10-cat (train+test) | **MIT** ✅ (LICENSE file verified) | EN | ✅ **indirect** — matches our threat model exactly | ✅ ingested (250 rows) | `ingest_bipia.py`. Manipulative text-cats→injection, benign-task-cats→benign, all code-cats→injection (tool_result). ⚠ train/test files have DIFFERENT category sets — both classified by reading content (fail-loud on unknown). arXiv 2312.14197. |
| **Microsoft BIPIA** (context-assembly) | 100 email contexts × 4 attacks | **MIT** ✅ | EN | ✅ **true indirect** | ✅ ingested (478 rows) | `ingest_bipia_context.py`. Splices malicious instruction into real email bodies (mirrors BIPIA `insert_start/end/middle`), emits paired poisoned (injection) + clean (benign). **This is the real indirect signal.** ⚠ eval must use context-grouped split (poisoned/clean twins) — see README honest-eval note. Follow-up: qa/table/code contexts too. |
| **nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1** | 1272 (676 ingested) | **CC-BY-4.0** ✅ (per-row) | EN | ✅ **indirect + agentic** — matches threat model | ✅ ingested | `ingest_nemotron.py`. Realistic embedded payloads (healthcare/finance, exfiltration & unauthorized-action). Extracts `injection.injection_text` (fallback `goal`). channel=tool_result. EN-only → boosts indirect/agentic category, not multilingual. |
| **in-house attack taxonomy** *(withheld)* | ~100 vectors | ours | EN | ✅ (by design) | ⚠ not published here | ⚠ DESCRIPTIONS + detection signatures, **not payloads**. Ingested as `weak_seed:true` auxiliary positives (quoted trigger phrases only), never bulk payload. The catalogue, its adapter and the 152 rows it produced are excluded from this repository and from the demo dataset. |
| **ProtectAI-22 blueprint (remainder)** | ~20 more | mixed (MIT/CC0/no-license/CC-BY) | EN-heavy | mixed | 🔬 review | Card names only some: Apache-2.0 injection ones = jackhhao ✅ + `Harelix/Prompt-Injection-Mixed-Techniques-2024` (⛔ **now gated/removed** — real sourcing hazard). Remainder mostly benign/instruction (`chatbot_instruction_prompts`, `grok-conversation-harmless`, `open-instruct`, `Salad-Data`, `xstest`) = benign-side expansion. ⚠ ProtectAI v2 is **EN-only by design** → reinforces Stream-D need. |
| **InjecAgent** | tool-chain injection | benchmark | EN | ✅ agentic | 🔬 review | arXiv 2403.02691. Agentic/tool-result channel. |

## Negative (benign) sources

| Source | Rows | License | Lang | Status | Notes |
|---|---|---|---|---|---|
| **deepset benign half** | ~343 | (as above) | EN/DE/FR | ✅ ingested | label 0 rows from deepset. |
| **SMS ham** (external corpus, see `ingest_sms_ham.py`) | 158k avail | CC-BY-4.0 etc. (per-row) | multi | ✅ ingested (sampled) | Stream C negatives + **FP-hard mining**: ham that *contains* injection-shaped words ("ignore", "stop", "verify", "password") = the hardest negatives (teaches "legit text with scary words ≠ injection"). |
| dev-doc / legit "ignore previous" text | 181 | synthetic | EN | ✅ ingested | `synthesize.py` (Stream E). Template dev-doc/support/security-ed text carrying injection-shaped words ("set your API key", "ignore the previous step", "system prompt"). `is_synthetic` → **train-only** (never eval). Real harvested dev-docs still 🔨 planned to thicken eval FP-hard. |
| synthetic injection (paraphrase) | 220 | synthetic | EN | ✅ ingested | `synthesize.py` (Stream E). Override/exfil/persona grammar with varied fillers, uniform-sampled across the template product for diversity. `is_synthetic` → train-only. Modest cap so templated positives don't dominate real ones. |

## Eval-only (never train)

| Source | License | Status | Notes |
|---|---|---|---|
| **Lakera PINT** | private (anti-overfit) | ⛔ scoring-only | ~4,314 entries (1,298 non-EN). Use for benchmarking `mmbert-tiny-inject` FP-cost only. The "36.5% document / FP-cost-measuring" claims were **refuted** — do NOT treat PINT as a file-preview indirect benchmark. |
| **own held-out FP-eval** | ours | 🔨 planned | Per-language benign + trap-like-innocent. The honest-eval oracle (≥100 real rows per language). |

## Rejected / excluded (do NOT re-add)

| Source | Why excluded |
|---|---|
| **Octavio-Santana/prompt-injection-attack-detection-multilingual** | ⛔ **`license: gpl`.** 11 languages (pt/en/zh/fr/vi/de/hi/it/ja/es/th, 6339 rows) — would have been ideal multilingual coverage, but GPL copyleft on training data is incompatible with how the resulting model is redistributed. Clean-room discipline: excluded. (Could be revisited scoring-only, but not for training.) |
| **rikka-snow/prompt-injection-multilingual** | Rows are **identical to deepset** ("Refugee crisis in Europe solutions", …) — a re-host, not new data. Redundant; skip. |

## Multilingual synthetic (Stream D+E — `synthesize_multilingual.py`, 2026-07-17)

Authored directly (operator: no translation API) in **10 languages** — de, fr,
tr, es, it, pt, nl, pl, ru, zh — native-quality override/exfil grammar + FP-hard
benign carriers. Universal keys (API/token/URL/Base64) stay language-agnostic;
only per-language verbs/nouns are authored. ~510 rows (45 injection + 6 benign
per language). All `is_synthetic + is_translated` → **train-only** (eval_oracle
forces them out of eval).

Impact: non-EN positives **110 → 560**; vocab now carries **119 Cyrillic + 260
CJK** subword tokens (previously non-Latin → `[UNK]`; the v8 tokenizer blocker
for the student is now addressed). This is the multilingual TEXT the mmBERT
teacher will soft-label and the student will learn from.

⚠ Still needed (next): a REAL per-language **eval** set (≥100 real rows/lang) —
synthetic trains but must never be the yardstick. Eval remains EN-only
measurable today.

## Multilingual (Stream D — DECISION 2026-07-17: **mmBERT transfer**)

The open question was: translate vs trust-transfer. **Decided by
evidence, not preference** — both alternatives to transfer are blocked:
- **Machine-translation is not executable here** — no `ANTHROPIC_API_KEY` /
  `MISTRAL_API_KEY` in this environment. Deferred until the operator provides a
  key; when unblocked, target **German first** + non-Latin scripts (where
  transfer is weakest).
- **The one big multilingual injection dataset (Octavio, 11 langs) is GPL** →
  rejected above.

**→ Multilingual strategy = rely on the mmBERT teacher's inherent
cross-lingual transfer.** The SMS model's **v8 round already validated** that
Latin-script transfer is real (blocker was tokenizer, not transfer). Universal
trigger tokens (API/token/URL/Base64) are language-agnostic and the
deterministic rule layer already exploits them.

**Honest state of coverage:** the benign side already has real multilingual
data (French/Bosnian/Turkish SMS ham). The **positive** side is EN-dominant
(~110 non-EN of ~2100 real positives) — this is what the build report flags.
The FP-eval oracle (later slice) **must** include a per-language positive probe
(≥100 real rows/lang) to *measure* whether
injection transfers as well as SMS spam did — that measurement, not more
training data, is the real open question.

---

## Enrichment round — 2026-07-24 (operator: skip the license GATE, keep the LEDGER)

Operator decision: security/public-interest attack data — do not let the license
card BLOCK ingestion, but keep provenance here (attribution for CC-BY/ODC-BY is
cheap; model-provenance is shipped practice). Corpus **5,945 → 15,513 rows**;
real positives **2,325 → 8,523 (3.7×)**; non-EN positives **110 → 3,623 (33×)**.
Both build-report honest-warnings cleared.

| Source | HF id | License | Role | Bound |
|---|---|---|---|---|
| **LLMail-Inject** | `microsoft/llmail-inject-challenge` (Phase1+2, 461K) | MIT (repo) | ⭐ REAL adaptive INDIRECT email injection (SaTML 2025) — the exact gap | spread-sampled, dedup, **cap 4,000** (got 4,018) |
| **PolyGuardMix** | `ToxicityPrompts/PolyGuardMix` (1.91M, 17 lang) | card (skipped-gate) | REAL multilingual: benign negs + jailbreak-class positives (`channel:chat`, tagged) | **per-(lang,label) cap 200** (got 5,400) |
| **email-benign** | *(own synthetic)* `synthesize_email_benign.py` | own-synthetic-CC0 | benign email negatives — LLMail channel-confound guard (Subject/greeting/sign-off, EN+DE) | cap 420, train-only |
| synthetic-ml | *(own)* `synthesize_multilingual.py` | own | reduced 45→20/lang (real PolyGuard now covers multilingual) | 240 |

⭐ **The per-language measurement question is now ANSWERABLE.** `eval_oracle.py build`
holds out ≥30 eval positives for **10 Latin langs + en** (fr 50 · de 44 · es/it/cs/pl 34 ·
nl 33 · pt 32 · sv 30) → per-lang recall MEASURABLE for the first time (was 0). tr/ru/ar
still too few (<30). Fetch is resilient (skips 429'd pages, never crashes).

**Deferred / not collected:** WildJailbreak (`allenai/wildjailbreak`, gated — Ai2 form
accepted + HF_TOKEN ready, but datasets-server `/rows` doesn't serve it → needs raw-TSV
download; jailbreak-class = lower value) · NotInject/InjecGuard + CAPTURE (not accessible
via datasets-server). **Tuning knob (next --fresh):** PolyGuard non-Latin benign ~3,000 =
[UNK] dead-weight (the Rust inference wrapper abstains on non-Latin at inference) → non-Latin cap ~50.

---

## Data-hygiene round — 2026-07-24 (pre-rerun audit, second session)

Audit before the r4a-config re-run found the corpus SOURCES sound but the
LANG TAGS broken, and fixed three things (all scripted, all provenance'd):

1. **Language repair** (`repair_lang.py`, idempotent). `guess_lang` was
   presence-based: one CJK char claimed "zh", one diacritic claimed "fr"/"de"/
   "tr". LLMail adaptive attacks smuggle short foreign snippets inside English
   emails → **1,322 English emails were tagged "zh"** (median CJK ratio 3%) and
   the previous round's "zh recall 0.992" measured English, not Chinese.
   `guess_lang` is now dominance-based (script ratio ≥0.30; Latin langs scored
   by unique chars + function words with a length-scaled floor, argmax) — TR no
   longer steals German rows via shared ö/ü (deepset de restored 44→134).
   Rows repaired in place with a `lang-fix:orig->final` note. Honest fallout:
   zh eval positives 248 (fake) → 5 (real) = zh recall is now UNMEASURABLE
   until real Chinese injection eval data is sourced (PolyGuard non-Latin
   positives are skipped by design — the Rust inference wrapper abstains on
   non-Latin).

2. **Label-suspect exclusion** (`label_suspects.jsonl`, consumed by
   `eval_oracle.py build`). 5 PolyGuard "benign" rows with explicit
   restriction-removal / disguised-elicitation framing (incl. the 2 known from
   TRAINING-LOG r5-inspect) are excluded from BOTH sides. Review bar kept
   tight on purpose — plain roleplay benigns stay (they ARE the Pattern-B
   FP-hard class).

3. **FP-hard eval 36 → 183** (≥150 statistical floor met). Two levers:
   (a) FP_HARD_WORDS moved to `corpus_common.py` as the SINGLE shared
   definition (miner and oracle had drifted apart) and made multilingual
   (de/fr/tr/bs/es/it/pt/nl/pl credential + override vocabulary) — the
   EN-only list found just 114 rows in a fr/bs/tr-heavy ham pool, the
   multilingual list mines 518 (+559 new sms-ham rows ingested);
   (b) groupless real FP-hard benigns get a boosted eval fraction
   (`FP_HARD_EVAL_FRACTION = 0.30` vs base 0.18) in `eval_oracle.py`.

Corpus now **16,072 rows** (train 13,377 / eval 2,690 after suspects).
Eval: 1,448 pos / 1,242 neg (183 FP-hard); 10 Latin langs + en all ≥30 pos.
⚠ Teacher logits from any earlier run are STALE (window set changed) — re-run
`distill_teacher.py` from scratch; never reuse smoke-run logits.

**Still open (deferred by operator choice this round):** it/nl/sv real
positives beyond single-source PolyGuard (≥500 target) · REAL zh/tr/ru eval
positives · Pattern-A/B curated negatives (train-side synthesis + real dev-doc
eval harvest).
