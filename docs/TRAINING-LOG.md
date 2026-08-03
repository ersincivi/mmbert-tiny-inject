# `mmbert-tiny-inject` — training log

> Ledger rule: every training run gets an entry (mirror of the SMS model's
> TRAINING-LOG discipline). Command · corpus state · metrics ·
> verdict — so the next round argues against RECORDED numbers, not memory.

## r1 — BASELINE (no teacher) — 2026-07-24, operator-run

**Command:** `python3 train.py --train` (shared venv, torch 2.12 MPS)
**Corpus:** post-enrichment 15,513 rows (enrichment round 2026-07-24: LLMail + PolyGuardMix
+ email-benign); train 12,991 docs → 21,284 windows (900-char / 300-overlap);
held-out eval 2,522 docs. Vocab 20,000 · SEQ_LEN 256 · 4 epochs (loss 0.293→0.138).

**Max-pool document eval (held-out):**

| metric | value |
|---|---|
| precision | 0.891 |
| recall | 0.887 |
| F1 | 0.889 |
| FP-rate | **0.147** ⚠ |
| FP-hard | **0.389** (14/36) ⚠⚠ |

Per-language recall (≥30 eval pos measurable): en 0.909 (769/846) · zh 0.992
(246/248) · fr 0.900 · it 0.882 · nl 0.848 · pl 0.765 · es 0.765 · pt 0.750 ·
de **0.705** · sv **0.633** · cs **0.588**. Unmeasurable (n<30): tr 0.650(20) ·
ru(8) · ar(1).

**Reading (r1 verdict):**
1. **Recall 0.35 → 0.887 vs the keyword/regex-floor baseline** — the classifier's
   core job (paraphrase + non-EN lift) demonstrably works even without a teacher.
2. **The money metric is the problem: FP-hard 0.389.** ~4/10 injection-shaped
   benign texts (dev docs, "ignore my previous email") get flagged. Doctrine:
   a shield that cries wolf is worse than none. Caveat: only 36 FP-hard eval
   rows — thin sample, thicken before over-fitting to it.
3. **Low-resource Latin langs lag** (de/sv/cs 0.59-0.71) — exactly the gap
   mmBERT cross-lingual distillation closed on SMS v8.
4. zh 0.992 likely reflects PolyGuard homogeneity, not true zh mastery — treat
   with suspicion until source-diverse zh eval exists.

**Decision:** r1 = frozen control number. → r2 = distill (mmBERT-small
`--batch 16`, measured ~1 h) targeting FP-hard ↓ + low-resource recall ↑.
Post-r2 levers if FP-hard stays high: per-class threshold calibration
+ more REAL dev-doc FP-hard negatives.

## r2 — DISTILL mmBERT-small α0.5 T2.0 — 2026-07-24, operator-run

**Command:** `distill_teacher.py --model jhu-clsp/mmBERT-small --batch 16 &&
train.py --train --teacher-logits work/teacher_logits.jsonl`
**Teacher:** embed-frozen (42M/141M trainable), token-fill p99 327/cap 768,
TRUNCATED 0; 3 epochs (loss 0.240→0.059, 20.1 rows/s, ~53 min); exported
17,633 soft-labels → window coverage 21,284/21,284 (100%, hash-dedup).

**TEACHER's own max-pool doc eval (= student's quality ceiling):**
precision 0.973 | recall 0.943 | F1 0.958 | FP-rate 0.035 | **FP-hard 0.083**
(3/36). Per-lang recall: en 0.983 · zh 1.000 · fr 0.940 · pt 0.844 · de 0.818 ·
it 0.794 · pl 0.794 · nl 0.788 · es 0.765 · cs 0.735 · **sv 0.633** (n<30:
tr 0.850 · ru 1.000 · ar 1.000). → Teacher is strong across languages —
**hypothesis (a) "EN-skewed teacher" REJECTED.** sv 0.633 = same as both
students → sv is a CORPUS gap, not a distill artifact.

**Max-pool document eval (held-out) — r2 vs r1:**

| metric | r1 (no teacher) | r2 (distill) | Δ |
|---|---|---|---|
| precision | 0.891 | 0.913 | +0.022 |
| recall | 0.887 | 0.859 | −0.028 |
| F1 | 0.889 | 0.885 | −0.004 |
| FP-rate | 0.147 | **0.111** | ✅ −0.036 |
| FP-hard | 0.389 (14/36) | **0.250** (9/36) | ✅ −0.139 |

Per-language recall r1→r2: en 0.909→**0.934** ✅ · zh 0.992→0.996 ·
fr 0.900→**0.720** 🔻 · de 0.705→**0.545** 🔻 · it 0.882→**0.500** 🔻🔻 ·
es 0.765→0.559 🔻 · pl 0.765→0.559 🔻 · nl 0.848→0.545 🔻 · pt 0.750→0.531 🔻 ·
cs 0.588→0.500 🔻 · sv 0.633→0.600 🔻. (n<30: tr 0.650= · ru 0.875→1.000 · ar 1.000=)

**Reading (r2 verdict):** NOT the SMS-v8 story. Money metric improved
(FP-hard 0.389→0.250, FP-rate ↓, en +0.025) but **non-EN Latin recall
collapsed uniformly across 9 languages** (−0.03..−0.38) — systematic, not
noise. With the teacher eval in hand: the student dropped BELOW both its own
r1 baseline AND the teacher in non-EN → the teacher is not the problem; the
**student failed to follow it**. Working hypotheses now:
(b) **calibration**: distill smooths scores; the fixed 0.5 doc threshold sits
above many mid-confidence non-EN positives;
(c) **capacity/loss-budget**: α0.5 with T=2 scales KL by T²=4 → hard-label CE
(the only place thin non-EN positives speak with full weight) is drowned; the
tiny 20k-vocab student spends its capacity matching teacher distributions on
the EN/zh-dominant mass. Distill loss was still falling at epoch 4 (0.40→0.33)
— possibly undertrained for the harder objective.

**Decision: r2 NOT accepted as final.** One knob per round (honest
attribution), teacher logits reusable: **r3 = `--distill-alpha 0.3`**
(student-only, minutes). If non-EN recovers only partially: r4 candidates =
`--distill-temp 1.0` (kills the ×4 KL scale) · `--epochs 6` · doc-threshold
sweep (knob doesn't exist in train.py yet — small addition if (b) confirmed).
sv needs corpus work regardless (teacher ceiling 0.633).

## r4a/r4b — NEW ARCHITECTURE (hidden 192 + recipe port) — 2026-07-24, operator-run

**Config (both):** hidden 192 · ff 768 (×4) · LayerNorm-before-head · focal γ1.5 ·
grad-clip 1.0 · LR warmup 10%+cosine · seed 0 · 6 epochs. GLM-5.2 ANALYSIS-r2.md
P0-P2 bundle. r4a = no teacher; r4b = + `--distill-alpha 0.15
--distill-start-epoch 2` (two-phase: 2 CE + 4 distill; teacher logits reused).
Checkpoints: `work/student_r4a.pt` / `work/student_r4b.pt`.

| metric | r1 | r2(α.5) | r3(α.3) | **r4a (CE)** | **r4b (α.15 2ph)** |
|---|---|---|---|---|---|
| precision | 0.891 | 0.913 | 0.925 | 0.883 | 0.883 |
| recall | 0.887 | 0.859 | 0.851 | **0.910** | **0.911** |
| F1 | 0.889 | 0.885 | 0.886 | **0.897** | **0.897** |
| FP-rate | 0.147 | 0.111 | 0.093 | 0.162 | 0.162 |
| FP-hard | 0.389 | 0.250 | 0.250 | 0.333 | 0.333 |

Per-lang recall (r1 → r4a → r4b): en 0.909→0.937→0.942 ✅ · fr 0.900→0.920→0.940 ✅ ·
de 0.705→**0.773**→0.750 ✅ · sv 0.633→**0.800**→0.733 ✅ · cs 0.588→**0.706**→0.676 ✅ ·
pl 0.765→**0.824**→0.794 ✅ · es 0.765→0.794→0.794 ✅ · pt 0.750→0.719→0.750 = ·
it 0.882→0.735→0.735 🔻 · nl 0.848→0.697→0.697 🔻 · (tr n<30: 0.650→0.800→0.800)

**Sweep (both):** F1 flat 0.45-0.60; FP-hard threshold-INSENSITIVE (12 docs score
>0.6 — confidently-wrong errors). Threshold is not the lever for this failure
mode; deployment threshold stays 0.5 (contract default).

**Reading (r4 verdict):**
1. **Architecture was the real fix for multilinguality** — r4a lifts 7/11
   measurable langs over r1 (sv +0.17, de +0.07, cs +0.12, pl +0.06), F1 0.897
   best-of-all-rounds. it/nl regressed (r1 was unseeded — attribution fuzzy).
2. **Distill at α0.15 is a NO-OP** — r4b ≈ r4a on every aggregate (identical
   F1/FP/FP-hard; per-lang shuffles within noise). The FP-hard cut seen in
   r2/r3 (0.250) was bought by strong KL — which also collapsed non-EN.
   On this corpus, distill strength trades FP-hard against non-EN recall;
   no tested α wins both.
3. **⚠ FP-hard n=36 — every FP-hard delta in this table is 1-3 documents.**
   Further knob-tuning against it = fitting noise. STOP.

**Decision (r4):** **r4a config = v1 CANDIDATE** (simpler: no teacher
dependency; distill shelved until data changes). Deployment threshold 0.5.
Next lever is DATA, not knobs (GLM Tier-4 corpus work):
(1) thicken FP-hard eval 36 → ≥150-200 + add REAL dev-doc FP-hard negatives
to train (the 12 confidently-wrong docs are a data gap — synthetic FP-hard
templates didn't cover them); (2) real sv/it/nl positives; (3) source-diverse
zh eval. Re-run r4a config (+ one α probe) after corpus grows.

## r5-inspect — the 12 confidently-wrong FP-hard docs READ — 2026-07-24

**Tool:** `inspect_fphard.py --out work/fphard_dump.jsonl` (student_r4a.pt).
**GLM's predicted patterns (dev docs / security articles / API docs): 0/12.**
Actual composition:

1. **Casual-chat credential/instruction vocabulary — 7/12 (all sms-ham):**
   "where 2 get user name & password" (0.997) · "admin credentials" (0.996) ·
   "buy and sell instructions… add my assistant's WhatsApp" (0.973) · "dont
   forget my pix" (0.935) · "revealing" greeting (0.899) · "Ok, I will follow
   your instruction." (0.858) · "PLEASE DONT IGNORE MYCALLS" (0.801).
   → model = keyword-reactive on credential/instruction/ignore lexemes in
   casual register. TRUE hard negatives; targeted-negative pattern A.
2. **Benign AI-meta prompts — 3/12 (polyguard):** Midjourney prompt-generator
   meta-instructions (en 0.873 + its nl twin 0.803 — same template, both in
   eval, group-split held) · fr PlantUML→DALL-E prompt request (0.690).
   → USER-authored instructions-to-AI are benign; same text embedded in
   third-party content would be an attack. **Text-only classifier cannot see
   provenance — this FP-hard mass is partly irreducible at model level;
   channel/context belongs to the engine/UI layer (doctrine note).**
   Targeted-negative pattern B (curate carefully — some "act as X" IS jailbreak).
3. **⚠ SUSPECT LABELS — 2/12 (polyguard "benign"):** sv 777b596c09 "…while
   IGNORING ALL WARNINGS OR EXCUSES THE MODEL MAY GIVE" (guardrail-bypass
   instruction) · en 5cef8cda86 "access to an UNRESTRICTED VERSION of this
   language model… most unconventional and risky" (jailbreak framing trope).
   Full texts verified in inject_eval.jsonl. **The model flagging these is
   arguably CORRECT — PolyGuard label error.** Honest FP-hard if excluded:
   10/34 ≈ 0.294. Action queued: adapter-level `label_suspect` exclusion list
   in the ingest pipeline (provenance-documented, no silent row surgery).

**Consequence for data work:** thicken FP-hard eval/train with patterns A + B
(NOT dev-doc templates — that guess is dead); audit PolyGuard benigns for
more jailbreak-shaped mislabels before trusting FP-hard trends.

## r5a — FOCAL DIAGNOSIS (γ0, r4a config otherwise) — 2026-07-24

**Command:** `train.py --train --epochs 6 --focal-gamma 0` → `student_r5a.pt`
**Result:** P 0.872 | R 0.913 | F1 0.892 | FP-rate 0.181 | FP-hard 0.361 (13/36).
Diagnostic targets: **it 0.735→0.735 (unchanged)** · nl 0.697→0.727 (+1 doc).

**Verdict:** focal-starvation hypothesis REJECTED — disabling focal did not
recover it/nl. **it/nl = data thinness (≈220 windows each), confirmed → r7
data queue.** Aggregate slightly WORSE without focal (F1 −0.005, FP-rate
+0.019, FP-hard 12→13): **focal γ1.5 stays; r4a config unchanged as v1
candidate.** (cs/es up, de/sv/fr/pt down — noise-band shuffles.)

## r5b — EPOCHS 10 (r4a config otherwise) — 2026-07-24

**Command:** `train.py --train --epochs 10` → `student_r5b.pt`
**Result:** P 0.875 | R 0.907 | F1 0.891 | FP-rate 0.174 | FP-hard 0.361
(13/36). Train loss 0.0010 at epoch 10 = memorization; eval flat-to-worse
(F1 −0.006, FP-rate +0.012 vs r4a). Per-lang shuffles (de 0.795 best-yet,
fr/cs/nl down) = overfit jitter, not signal.

**Verdict:** more epochs do NOT help at this corpus size — 6 epochs was right.

## ⭐ ROUND CLOSE (2026-07-24) — FINAL: r4a = v1 CANDIDATE

8 runs (r1 · r2 · r3 · r4a · r4b · r5a · r5b + smoke). **Winner: r4a**
(hidden 192 · ff 768 · LayerNorm · focal γ1.5 · grad-clip · warmup+cosine ·
seed 0 · 6 epochs · NO teacher · threshold 0.5) — F1 0.897 / recall 0.910 /
FP-hard 0.333. Checkpoint `work/student_r4a.pt`.

Locked findings: architecture (not distill) fixed multilinguality · distill =
FP-hard↔non-EN trade at every α → SHELVED until corpus grows · focal γ1.5
confirmed (r5a) · 6 epochs confirmed (r5b) · it/nl = data thinness (r5a) ·
FP-hard n=36 too thin + 2/12 suspect PolyGuard labels + patterns = casual-SMS
credential talk (7) / benign AI-meta prompts (3), NOT dev-docs.

**Still pending (not this round):** ONNX/INT8 export + tract smoke +
escalation wire-in in the Rust inference wrapper · data work (patterns A+B
negatives, label audit, it/nl/sv positives, FP-hard eval ≥150, zh source
diversity) · per-channel threshold calibration.

**r5 queue (GLM post-r4 review, agreed 2026-07-24):**
- **r5-inspect** (FIRST): `inspect_fphard.py` — read the ~12 confidently-wrong
  FP-hard docs (top-window text) → pattern list → TARGETED negatives, not more
  generic templates. Tool added post-r4 (mirrors run_train arch; lockstep).
- **r5a:** r4a config + `--focal-gamma 0 --epochs 6` — it/nl diagnosis (focal
  starving easy templated langs vs data thinness). Read only BIG consistent
  deltas (per-lang n≈33, ±0.03/doc); ignore FP-hard column (n=36 rule).
- **r5b:** r4a config + `--epochs 10` — cheapest probe (loss 0.023 still falling).
- **r6:** hard-negative mining in train.py (implement AFTER r5-inspect shows
  patterns) · **r7:** corpus-grown re-run. ⚠ sv 0.800 > teacher ceiling 0.633
  → sv eval likely homogeneous (PolyGuard) — validate before trusting.

## r3 — DISTILL α0.3 T2.0 — 2026-07-24, operator-run

**Command:** `python3 train.py --train --teacher-logits work/teacher_logits.jsonl
--distill-alpha 0.3`
**Teacher logits:** reused (r2 export, 17,633 soft-labels, 100% coverage).
4 epochs (loss 0.549→0.258, still falling — undertrained).

**Max-pool document eval (held-out) — r3 vs r1 vs r2:**

| metric | r1 (no teacher) | r2 (α0.5) | r3 (α0.3) | r3 vs r1 | r3 vs r2 |
|---|---|---|---|---|---|
| precision | 0.891 | 0.913 | **0.925** | +0.034 ✅ | +0.012 ✅ |
| recall | 0.887 | 0.859 | 0.851 | −0.036 | −0.008 |
| F1 | 0.889 | 0.885 | 0.886 | −0.003 | +0.001 |
| FP-rate | 0.147 | 0.111 | **0.093** | −0.054 ✅ | −0.018 ✅ |
| FP-hard | 0.389 (14/36) | 0.250 (9/36) | 0.250 (9/36) | −0.139 ✅ | = flat |

Per-language recall r1→r2→r3:

| lang | r1 | r2(α.5) | r3(α.3) | r3 vs r1 | r3 vs r2 |
|---|---|---|---|---|---|
| en | 0.909 | 0.934 | 0.905 | −0.004 | −0.029 🔻 |
| zh | 0.992 | 0.996 | 1.000 | +0.008 | +0.004 |
| fr | 0.900 | 0.720 | 0.780 | −0.120 🔻 | +0.060 ✅ |
| de | 0.705 | 0.545 | 0.523 | −0.182 🔻 | −0.022 🔻 |
| es | 0.765 | 0.559 | 0.529 | −0.236 🔻 | −0.030 🔻 |
| it | 0.882 | 0.500 | 0.618 | −0.264 🔻 | +0.118 ✅ |
| cs | 0.588 | 0.500 | 0.676 | +0.088 ✅ | +0.176 ✅ |
| pl | 0.765 | 0.559 | 0.647 | −0.118 🔻 | +0.088 ✅ |
| nl | 0.848 | 0.545 | 0.576 | −0.272 🔻 | +0.031 ✅ |
| pt | 0.750 | 0.531 | 0.562 | −0.188 🔻 | +0.031 ✅ |
| sv | 0.633 | 0.600 | 0.500 | −0.133 🔻 | −0.100 🔻 |
| tr | 0.650 | 0.650 | 0.550 | −0.100 🔻 | −0.100 🔻 |

**Reading (r3 verdict):** α tuning alone is INSUFFICIENT.
1. **Partial recovery vs r2:** 6/9 non-EN langs improved (fr +0.06, it +0.12,
   cs +0.18, pl +0.09, nl +0.03, pt +0.03) — lowering α helps, as predicted.
2. **Zero recovery vs r1:** NOT ONE non-EN lang returned to its no-teacher
   baseline. Distillation still HURTS non-EN at every α tested.
3. **New regressions:** de −0.022, es −0.030, sv −0.100, en −0.029 — lowering α
   traded EN for some non-EN, didn't lift the overall.
4. **FP-hard stuck at 0.250** (same 9/36) — the FP-hard gain came from distillation
   itself, not from α strength.
5. **Loss still falling** at epoch 4 (0.317→0.258) — undertrained for the harder
   distillation objective.

**Root-cause confirmed (see ANALYSIS-r2.md):** the student at hidden=128 without
focal loss, grad clipping, or LayerNorm simply lacks the capacity to follow the
teacher across languages. The SMS v8 distillation succeeded at α=0.5/T=2.0 because
it had hidden=256, feedforward=1024, focal loss, class weights, grad clip, and
LayerNorm. This model has half the capacity and none of those training aids.

**Decision: stop tuning α. Fix the architecture first.**
- r4 = baseline (no teacher) WITH architecture fixes: hidden=192, ff=hidden×4,
  LayerNorm, focal loss, grad clip, LR warmup — prove the architecture is stronger
  on its own before re-introducing distillation.
- r5 = distill α=0.15 on the r4 architecture — if the capacity is sufficient,
  distillation should now help (as it did on SMS v8).
- Teacher logits are reusable (17,633, 100% coverage) — no teacher re-run needed.
- sv needs corpus work regardless (teacher ceiling 0.633 = corpus gap).

---

## DATA HYGIENE ROUND — 2026-07-24 (between round 1 and round 2; no training)

Corpus/eval repaired before the r4a-config re-run (detail:
`SOURCES.md` §"Data-hygiene round"): lang tags fixed
(1,322 English LLMail emails were "zh" — round-1 per-lang numbers for
zh/ru/ar/tr are UNRELIABLE), 5 PolyGuard label-suspects excluded, FP-hard
eval 36 → 183, split rebuilt (train 13,377 / eval 2,690 / 21,663 windows).

⚠ **COMPARABILITY BREAK:** the eval set changed. Round-1 numbers (r1–r5b,
incl. v1-candidate r4a F1 0.897) were measured on the OLD eval and must NOT
be compared against runs on the new eval. First step of round 2 is to
RE-SCORE `work/student_r4a.pt` on the new eval — that number, not 0.897, is
the baseline the re-run must beat. Teacher logits from any earlier run are
stale (window set changed) — regenerate from scratch.

## r4a-RESCORE — 2026-07-24 (round-2 baseline anchor; no training)

`train.py --eval-checkpoint work/student_r4a.pt --vocab work/vocab_r1.txt` on
the NEW (post-hygiene) eval — **this is the number round-2 runs must beat**:

- **F1 0.874 | precision 0.845 | recall 0.904 | FP-rate 0.193 | FP-hard 0.235 (43/183)** @ thr 0.5
- Per-lang recall finally readable (old zh/ru/ar/tr numbers were mislabeled-data
  artifacts): en 0.954 · fr 0.850 · it 0.824 · es 0.794 · pt 0.750 · cs 0.706 ·
  pl 0.676 · sv 0.667 · nl 0.636 · **de 0.625** (weakest measurable — despite
  de being the 2nd-largest eval slice; watch in round 2).
- Threshold sweep: F1 rises monotonically to **0.884 @ 0.60** but non-EN recall
  collapses there (de 0.562, sv 0.533, nl 0.545) — the sweep's F1 gain is an
  EN-mass effect, NOT a free win. Per-channel calibration
  remains the right lever; do not chase 0.60 globally.
- FP-hard 0.235 on n=183 is statistically meaningful for the first time
  (round-1's 0.333 was on n=36).

## r6 — ROUND-2 BASELINE (post-hygiene data, 4 epochs) — 2026-07-24

**Command:** `train.py --train` (defaults: hidden 192, focal γ1.5, 4 epochs,
seed 0, no teacher) on the hygiene-round data (train 13,377 / 21,663 windows).
→ `work/student.pt`

**vs r4a-rescore anchor (same eval):**

| metric | r4a-rescore | r6 | Δ |
|---|---|---|---|
| F1 @0.5 | 0.874 | **0.892** | +0.018 |
| recall | 0.904 | 0.903 | ≈ |
| FP-rate | 0.193 | **0.143** | −0.050 |
| **FP-hard** | **0.235 (43/183)** | **0.104 (19/183)** | **−56% rel.** |
| de | 0.625 | **0.750** | +0.125 |
| pl | 0.676 | **0.824** | +0.148 |
| pt | 0.750 | 0.812 | +0.062 |
| fr | 0.850 | 0.725 | −0.125 |
| es | 0.794 | 0.706 | −0.088 |
| sv | 0.667 | 0.600 | −0.067 |

**Reading:** the data round paid off exactly where it aimed — FP-hard halved+
(the +559 multilingual FP-hard ham rows taught "scary words in legit text ≠
injection") and de jumped +0.125 (the 90 real German rows restored by the
lang repair). The fr/es dips are the visible cost of the same lever: the new
FP-hard mining was fr-heavy (105 fr rows), pushing the model conservative on
Romance-language credential vocabulary. Recall held at 0.903 overall.

**Caveat:** 4 epochs (default) ≠ r4a config (6 epochs, r5b-validated); train
loss still falling at epoch 4 (0.0558→0.0445) — r6 is likely undertrained.
Next: **r6b = `--epochs 6`** for the true r4a-config-on-new-data comparison,
then teacher + distill probe.

## r6b — r4a-CONFIG (6 epochs) ON NEW DATA — 2026-07-24

**Command:** `train.py --train --epochs 6` → `work/student.pt` (overwrote r6's
4-epoch weights; both runs are seed-0 deterministic and reproducible).

**vs r6 (4 epochs, same data/eval), both @0.5:**

| metric | r6 (4ep) | r6b (6ep) |
|---|---|---|
| F1 | 0.892 | 0.892 |
| recall | 0.903 | 0.907 |
| FP-rate | 0.143 | 0.147 |
| FP-hard | **0.104 (19/183)** | 0.131 (24/183) |
| de | **0.750** | 0.688 |
| loss @ last epoch | 0.0445 | 0.0245 |

**Verdict:** the r5b-era "6 epochs is right" finding does NOT carry to the
post-hygiene corpus — F1 identical, FP-hard 5 docs worse, per-lang shifts all
within the 1-3-doc noise band, and loss 0.0245 is drifting toward the
memorization territory r5b mapped (0.0010 @ 10ep). With more + cleaner data,
4 epochs reaches the same place. **Round-2 default = 4 epochs (r6).** Do NOT
chase the thr-0.45 sweep row (F1 0.894, prettier per-lang) — picking the
threshold that flatters the eval is fitting to noise; per-channel calibration
stays the principled lever.

**Next:** teacher (mmBERT-small) + distill probe at DEFAULT 4 epochs so the
only variable vs r6 is the teacher signal.

## TEACHER (round 2) — mmBERT-small, post-hygiene corpus — 2026-07-24

**Command:** `distill_teacher.py --model jhu-clsp/mmBERT-small --batch 16`
(3 epochs, embed-frozen, max-len 768, TRUNCATED 0). ~64 min (16-17 rows/sec,
slightly under the 22.6 solo-run measurement). Exported 18,010 soft-labels
(unique window hashes of 21,663 windows) → `work/teacher_logits.jsonl`.

**Teacher doc-eval (= student's quality ceiling):**
F1 **0.928** | precision 0.878 | recall 0.985 | FP-rate 0.159 | FP-hard 0.109 (20/183)
Per-lang recall: en 0.992 · fr 1.000 · pt 1.000 · es 0.971 · pl 0.971 · nl 0.970 ·
**de 0.958** · it 0.941 · cs 0.912 · sv 0.900.

**Reading:** ceiling is +0.036 F1 over r6 and the non-EN gap is where the
headroom lives (de 0.958 vs student 0.750; sv 0.900 vs 0.600). BUT teacher
FP-hard (0.109) ≈ student's (0.104) — the teacher has nothing to teach on
FP-hard, so the round-1 risk (distill trades FP-hard for non-EN) is still
live; judge the distill probe on "non-EN up, FP-hard NOT worse".

⚠ "2 window hashes carry conflicting gold labels" — expected artifact, not a
new bug: bipia-ctx poisoned/clean twins share body text, so a window that
misses the injected span is byte-identical across both docs yet carries each
doc's label. 2/18,010 = negligible; export keeps first.

## r7 — DISTILL PROBE (α0.5, round-2 corpus) — 2026-07-24

**Command:** `train.py --train --teacher-logits work/teacher_logits.jsonl`
(defaults: 4 epochs two-phase = 2 CE + 2 distill, α0.5, T2.0; coverage
21,663/21,663 = 100%). → `work/student.pt` (overwrote r6b).

**Three-number comparison (all @0.5, same eval):**

| metric | r6 (CE baseline) | r7 (distill) | teacher ceiling |
|---|---|---|---|
| F1 | 0.892 | 0.891 | 0.928 |
| recall | 0.903 | **0.918** | 0.985 |
| precision | 0.881 | 0.866 | 0.878 |
| FP-rate | 0.143 | 0.166 | 0.159 |
| FP-hard | **0.104 (19/183)** | 0.120 (22/183) | 0.109 |
| de | 0.750 | 0.750 | 0.958 |
| sv | 0.600 | **0.800** | 0.900 |
| nl | 0.667 | **0.788** | 0.970 |
| it | 0.735 | **0.824** | 0.941 |
| cs | 0.735 | **0.824** | 0.912 |
| pt | 0.812 | 0.781 | 1.000 |

**Reading:** the round-1 shelving verdict ("distill trades FP-hard for non-EN, no α
wins both") is BROKEN on the bigger/cleaner corpus — non-EN moved sharply
toward the teacher (sv +0.200, nl +0.121, it/cs +0.089) while FP-hard gave up
only 3 docs (19→22/183) and F1 held. This was exactly the "retry when the
corpus grows" condition. Residual trade is mild, visible in precision/FP-rate.
de is the outlier: unmoved at 0.750 vs teacher 0.958 — the student's remaining
de gap is NOT a soft-label problem (likely capacity/tokenizer; note for
round 3, ties into the German-source data work).
Distill loss still falling (0.359→0.262 over 2 distill epochs) — a 2+4-epoch
variant is plausible headroom but epoch-extension flirted with memorization in
r6b; park it, don't knob-chase now.

## ═══ ROUND 2 CLOSE — 2026-07-24 ═══

**v2-candidate = r7 (distill α0.5)** — OPERATOR DECISION 2026-07-24.
Checkpoint preserved: `work/student_r7.pt` (copy of the r7 `student.pt`).
F1 0.891 / recall 0.918 / FP-hard 0.120 (22/183) / sv 0.800 · nl 0.788 ·
it/cs 0.824 · de 0.750 @ thr 0.5, vocab = `work/vocab.txt` (round-2).

Round-2 arc in one line: data hygiene (lang-fix + suspects + FP-hard 183) →
r4a re-scored 0.874 (honest anchor) → r6 CE baseline 0.892 (FP-hard halved
by mined multilingual ham) → r6b showed 6-epochs no longer helps → teacher
ceiling 0.928 → r7 distill broke round-1's FP-hard↔non-EN trade (non-EN
sharply up for 3 FP-hard docs).

**Rescinded:** round-1 "distill SHELVED" verdict (conditional on corpus size —
condition met and tested). **Standing findings:** 4 epochs default · don't
chase sweep thresholds · de-gap is not a soft-label problem (round-3 data
work: real de positives).

**Remaining to ship r7:** ONNX export (dynamic-quant, NOT
QDQ) → tract smoke → Ed25519 OTA artifact → escalation-routing
wire-in → deterministic-engine threshold calibration (same eval infra). Export
script does not exist yet here.

## EXPORT (r7 → ONNX) — 2026-07-24

**Command:** `export_onnx.py` (defaults) → `work/model/`
{model.onnx 19.16 MB fp32 · **model_int8.onnx 4.86 MB dynamic-quant** ·
vocab.txt · export_manifest.json (sha256s)}.

- Parity (512 eval windows): torch↔fp32 max|Δlogit| 4e-06, 0 flips (exact);
  torch↔int8 max|Δlogit| 0.224, **2/512 flips** (normal quant noise).
- **INT8 doc-eval (the shipping number): F1 0.891 | recall 0.919 |
  FP-hard 0.120 (22/183)** — identical to the r7 checkpoint (Δ ≈ +1 doc);
  per-lang table unchanged. Quantization cost ≈ zero.
- 4.86 MB sits in the SMS-model band (4.5 MB) → device budget holds.

**Remaining chain:** tract smoke (shared loader family, [1,2] head) → Ed25519
OTA artifact → escalation-routing wire-in → per-channel calibration.

## TRACT SMOKE (Rust) — 2026-07-24 ✅ + TOPIC-CONFOUND FINDING ⚠

**New:** a Rust inference module for this model (own feature flag, same deps
as the SMS one): `char_windows` Rust mirror (CHAR-counted, tail-anchored; 5 unit
tests, feature-less) + the tract backend (SEQ_LEN 256, [1,2] head,
reuses the lockstep WordPieceTokenizer) + `score_document` max-pool +
unk-abstain signal + env-var self-skip smoke.

- **Smoke PASSED** against the real `model_int8.onnx` (~1.5 s load+infer):
  injection 1.000 · neutral benign 0.023 · benign+LLMail-exfil-line 0.999.
- **tract ↔ onnxruntime parity CONFIRMED** on identical texts (0.9999/1.0000
  matched to 4 decimals) — the Rust windowing+tokenizer port is faithful.

⚠ **TOPIC CONFOUND (round-3 data item, found BY the smoke):** benign
corporate budget-email prose ("This quarter's budget review covers vendor
spend and hiring…") scores **0.9999** — ALL 4,018 LLMail positives are
budget-themed and the corpus has NO budget-themed benign email (email-benign
is order/statement-themed). The eval can't see this either (no such benign
slice) — a distribution blind spot, exactly the honest-eval caveat. Real-user
impact: legit corporate email about budgets/planning → FP risk. Fix is DATA:
benign corporate-email negatives on LLMail themes (budget/planning/meeting)
+ an eval slice for them. Deterministic rule-layer gating + the inform-only
doctrine bound the blast radius until then.

## Packaging and naming — 2026-07-24

**Naming.** The two students trained in this repository are:

- **`mmbert-tiny-sms`** — the SMS classifier (lineage `v8-mmbert-distill`).
- **`mmbert-tiny-inject`** — this model (internal `r7` distill, round-2).

Both are students distilled from an mmBERT teacher, hence the shared prefix;
"tiny" is literal — 4.4M parameters, ~4.5 MB after INT8 quantisation.

Architecturally each model is the *semantic layer* that sits above a
deterministic rule layer. Layer and model are separate things: the layer is
where a verdict is formed, the model is one implementation of it.

**Packaging.** Both students ship as INT8 ONNX plus their vocabulary file
(~4.9 MB for this one), memory-mapped through `tract`. Updates are pulled, never
pushed: a manifest lists the files for a given pack and version, every file
carries a detached Ed25519 signature verified against a key pinned in the
client, and the swap is atomic with the bundled copy as fallback. Version
strings are pack-prefixed because the publish table is keyed on version alone,
so an unprefixed "v2" would collide across packs.

**Remaining chain:** per-channel calibration on the same eval infrastructure →
round-3 data (the budget-theme confound first).
