# `mmbert-tiny-inject` — training log

> Ledger rule: every training run gets an entry, mirroring the SMS model's
> discipline. Command, corpus state, metrics, verdict — so the next round argues
> against recorded numbers rather than memory.

## r1 — baseline, no teacher

**Command:** `python3 train.py --train` (shared venv, torch 2.12 MPS)
**Corpus:** post-enrichment 15,513 rows (LLMail + PolyGuardMix + email-benign);
train 12,991 docs → 21,284 windows (900-char / 300-overlap); held-out eval 2,522
docs. Vocab 20,000 · SEQ_LEN 256 · 4 epochs (loss 0.293→0.138).

**Max-pool document eval (held-out):**

| metric | value |
|---|---|
| precision | 0.891 |
| recall | 0.887 |
| F1 | 0.889 |
| FP-rate | **0.147** |
| FP-hard | **0.389** (14/36) |

Per-language recall (≥30 eval positives = measurable): en 0.909 (769/846) ·
zh 0.992 (246/248) · fr 0.900 · it 0.882 · nl 0.848 · pl 0.765 · es 0.765 ·
pt 0.750 · de **0.705** · sv **0.633** · cs **0.588**. Unmeasurable (n<30):
tr 0.650 (20) · ru (8) · ar (1).

**Reading:**

1. Recall 0.35 → 0.887 against the keyword/regex floor — the classifier's core
   job (paraphrase plus non-English lift) demonstrably works even without a
   teacher.
2. The money metric is the problem: FP-hard 0.389. Roughly four in ten
   injection-shaped benign texts (dev docs, "ignore my previous email") get
   flagged, and a shield that cries wolf is worse than none. Caveat: only 36
   FP-hard eval rows — a thin sample, thicken it before over-fitting to it.
3. Low-resource Latin languages lag (de/sv/cs 0.59-0.71) — exactly the gap
   mmBERT cross-lingual distillation closed on SMS v8.
4. zh 0.992 likely reflects PolyGuard homogeneity rather than true Chinese
   mastery; treat with suspicion until a source-diverse zh eval exists.

**Decision:** r1 is the frozen control number. Next, r2 = distillation
(mmBERT-small, `--batch 16`, measured ~1 h) targeting FP-hard down and
low-resource recall up. If FP-hard stays high afterwards, the levers are
per-class threshold calibration and more real dev-doc hard negatives.

## r2 — distill mmBERT-small, α0.5 T2.0

**Command:** `distill_teacher.py --model jhu-clsp/mmBERT-small --batch 16 &&
train.py --train --teacher-logits work/teacher_logits.jsonl`
**Teacher:** embed-frozen (42M/141M trainable), token fill p99 327 against a cap
of 768, 0 truncated; 3 epochs (loss 0.240→0.059, 20.1 rows/s, ~53 min); exported
17,633 soft labels → window coverage 21,284/21,284 (100%, hash-deduped).

**Teacher's own max-pool doc eval, i.e. the student's quality ceiling:**
precision 0.973 | recall 0.943 | F1 0.958 | FP-rate 0.035 | **FP-hard 0.083**
(3/36). Per-language recall: en 0.983 · zh 1.000 · fr 0.940 · pt 0.844 ·
de 0.818 · it 0.794 · pl 0.794 · nl 0.788 · es 0.765 · cs 0.735 · **sv 0.633**
(n<30: tr 0.850 · ru 1.000 · ar 1.000). The teacher is strong across languages,
so hypothesis (a), an English-skewed teacher, is rejected. sv 0.633 matches both
students, making sv a corpus gap rather than a distillation artifact.

**Max-pool document eval (held-out), r2 vs r1:**

| metric | r1 (no teacher) | r2 (distill) | Δ |
|---|---|---|---|
| precision | 0.891 | 0.913 | +0.022 |
| recall | 0.887 | 0.859 | −0.028 |
| F1 | 0.889 | 0.885 | −0.004 |
| FP-rate | 0.147 | **0.111** | −0.036 |
| FP-hard | 0.389 (14/36) | **0.250** (9/36) | −0.139 |

Per-language recall r1→r2: en 0.909→**0.934** · zh 0.992→0.996 ·
fr 0.900→**0.720** · de 0.705→**0.545** · it 0.882→**0.500** · es 0.765→0.559 ·
pl 0.765→0.559 · nl 0.848→0.545 · pt 0.750→0.531 · cs 0.588→0.500 ·
sv 0.633→0.600. (n<30: tr 0.650 flat · ru 0.875→1.000 · ar 1.000 flat)

**Reading:** this is not the SMS v8 story. The money metric improved (FP-hard
0.389→0.250, FP-rate down, en +0.025) but non-English Latin recall collapsed
uniformly across nine languages (−0.03 to −0.38) — systematic, not noise. With
the teacher eval in hand, the student dropped below both its own r1 baseline and
the teacher on non-English, so the teacher is not the problem: the student
failed to follow it. Working hypotheses:

- (b) **calibration** — distillation smooths scores, and the fixed 0.5 document
  threshold sits above many mid-confidence non-English positives;
- (c) **capacity / loss budget** — α0.5 with T=2 scales KL by T²=4, so the
  hard-label CE term (the only place thin non-English positives speak at full
  weight) is drowned, and the tiny 20k-vocab student spends its capacity
  matching teacher distributions over the English/Chinese-dominant mass.
  Distillation loss was still falling at epoch 4 (0.40→0.33), so it may also be
  undertrained for the harder objective.

**Decision:** r2 is not accepted as final. One knob per round for honest
attribution, and the teacher logits are reusable, so r3 = `--distill-alpha 0.3`
(student only, minutes). If non-English recovers only partially, r4 candidates
are `--distill-temp 1.0` (removes the ×4 KL scale), `--epochs 6`, and a document
threshold sweep — the last needs a small addition to `train.py` if (b) is
confirmed. sv needs corpus work regardless (teacher ceiling 0.633).

## r3 — distill α0.3 T2.0

**Command:** `python3 train.py --train --teacher-logits work/teacher_logits.jsonl
--distill-alpha 0.3`
**Teacher logits:** reused from the r2 export (17,633 soft labels, 100%
coverage). 4 epochs (loss 0.549→0.258, still falling — undertrained).

**Max-pool document eval (held-out), r3 vs r1 vs r2:**

| metric | r1 (no teacher) | r2 (α0.5) | r3 (α0.3) | r3 vs r1 | r3 vs r2 |
|---|---|---|---|---|---|
| precision | 0.891 | 0.913 | **0.925** | +0.034 | +0.012 |
| recall | 0.887 | 0.859 | 0.851 | −0.036 | −0.008 |
| F1 | 0.889 | 0.885 | 0.886 | −0.003 | +0.001 |
| FP-rate | 0.147 | 0.111 | **0.093** | −0.054 | −0.018 |
| FP-hard | 0.389 (14/36) | 0.250 (9/36) | 0.250 (9/36) | −0.139 | flat |

Per-language recall r1→r2→r3:

| lang | r1 | r2 (α.5) | r3 (α.3) | r3 vs r1 | r3 vs r2 |
|---|---|---|---|---|---|
| en | 0.909 | 0.934 | 0.905 | −0.004 | −0.029 |
| zh | 0.992 | 0.996 | 1.000 | +0.008 | +0.004 |
| fr | 0.900 | 0.720 | 0.780 | −0.120 | +0.060 |
| de | 0.705 | 0.545 | 0.523 | −0.182 | −0.022 |
| es | 0.765 | 0.559 | 0.529 | −0.236 | −0.030 |
| it | 0.882 | 0.500 | 0.618 | −0.264 | +0.118 |
| cs | 0.588 | 0.500 | 0.676 | +0.088 | +0.176 |
| pl | 0.765 | 0.559 | 0.647 | −0.118 | +0.088 |
| nl | 0.848 | 0.545 | 0.576 | −0.272 | +0.031 |
| pt | 0.750 | 0.531 | 0.562 | −0.188 | +0.031 |
| sv | 0.633 | 0.600 | 0.500 | −0.133 | −0.100 |
| tr | 0.650 | 0.650 | 0.550 | −0.100 | −0.100 |

**Reading:** α tuning alone is insufficient.

1. Partial recovery against r2: 6 of 9 non-English languages improved (fr +0.06,
   it +0.12, cs +0.18, pl +0.09, nl +0.03, pt +0.03) — lowering α helps, as
   predicted.
2. Zero recovery against r1: not one non-English language returned to its
   no-teacher baseline. Distillation still hurts non-English at every α tested.
3. New regressions: de −0.022, es −0.030, sv −0.100, en −0.029. Lowering α
   traded English for some non-English without lifting the overall.
4. FP-hard stuck at 0.250 (the same 9/36) — that gain came from distillation
   itself, not from α strength.
5. Loss still falling at epoch 4 (0.317→0.258): undertrained for the harder
   objective.

**Root cause (see ANALYSIS-r2.md):** the student at hidden=128, without focal
loss, gradient clipping or LayerNorm, lacks the capacity to follow the teacher
across languages. SMS v8 distillation succeeded at α=0.5 / T=2.0 because it had
hidden=256, feedforward=1024, focal loss, class weights, gradient clipping and
LayerNorm. This model has half the capacity and none of those training aids.

**Decision: stop tuning α, fix the architecture first.**

- r4 = baseline (no teacher) with the architecture fixes — hidden=192,
  ff=hidden×4, LayerNorm, focal loss, gradient clipping, LR warmup — to prove
  the architecture is stronger on its own before re-introducing distillation.
- r5 = distill α=0.15 on the r4 architecture; if capacity is now sufficient,
  distillation should help as it did on SMS v8.
- Teacher logits are reusable (17,633, 100% coverage), so no teacher re-run.
- sv needs corpus work regardless (teacher ceiling 0.633 = corpus gap).

## r4a / r4b — new architecture (hidden 192 + recipe port)

**Config (both):** hidden 192 · ff 768 (×4) · LayerNorm before head · focal γ1.5
· grad-clip 1.0 · LR warmup 10% + cosine · seed 0 · 6 epochs — the ANALYSIS-r2
P0-P2 bundle. r4a = no teacher; r4b = plus `--distill-alpha 0.15
--distill-start-epoch 2` (two-phase: 2 CE + 4 distill, teacher logits reused).
Checkpoints: `work/student_r4a.pt` / `work/student_r4b.pt`.

| metric | r1 | r2 (α.5) | r3 (α.3) | **r4a (CE)** | **r4b (α.15, 2-phase)** |
|---|---|---|---|---|---|
| precision | 0.891 | 0.913 | 0.925 | 0.883 | 0.883 |
| recall | 0.887 | 0.859 | 0.851 | **0.910** | **0.911** |
| F1 | 0.889 | 0.885 | 0.886 | **0.897** | **0.897** |
| FP-rate | 0.147 | 0.111 | 0.093 | 0.162 | 0.162 |
| FP-hard | 0.389 | 0.250 | 0.250 | 0.333 | 0.333 |

Per-language recall (r1 → r4a → r4b): en 0.909→0.937→0.942 · fr 0.900→0.920→0.940
· de 0.705→**0.773**→0.750 · sv 0.633→**0.800**→0.733 · cs 0.588→**0.706**→0.676
· pl 0.765→**0.824**→0.794 · es 0.765→0.794→0.794 · pt 0.750→0.719→0.750 ·
it 0.882→0.735→0.735 · nl 0.848→0.697→0.697 · (tr, n<30: 0.650→0.800→0.800)

**Sweep (both):** F1 flat between thresholds 0.45 and 0.60; FP-hard is
threshold-insensitive because 12 documents score above 0.6 — these are
confidently-wrong errors. Threshold is not the lever for this failure mode; the
deployment threshold stays at the contract default of 0.5.

**Reading:**

1. Architecture was the real fix for multilinguality. r4a lifts 7 of 11
   measurable languages over r1 (sv +0.17, de +0.07, cs +0.12, pl +0.06) and F1
   0.897 is the best of all rounds. it/nl regressed, though r1 was unseeded so
   attribution there is fuzzy.
2. Distillation at α0.15 is a no-op: r4b matches r4a on every aggregate
   (identical F1, FP-rate and FP-hard; per-language shuffles within noise). The
   FP-hard cut seen in r2/r3 was bought by strong KL, which also collapsed
   non-English. On this corpus, distillation strength trades FP-hard against
   non-English recall and no tested α wins both.
3. FP-hard n=36 means every FP-hard delta in this table is 1-3 documents.
   Further knob-tuning against it is fitting noise. Stop.

**Decision:** r4a config is the v1 candidate — simpler, with no teacher
dependency; distillation is shelved until the data changes. Deployment threshold
0.5. The next lever is data, not knobs (the analysis document's Tier-4 corpus
work): (1) thicken FP-hard eval from 36 to ≥150-200 and add real dev-doc hard
negatives to training, since the 12 confidently-wrong documents are a data gap
that synthetic FP-hard templates did not cover; (2) real sv/it/nl positives;
(3) a source-diverse zh eval. Re-run the r4a config, plus one α probe, after the
corpus grows.

## r5-inspect — reading the 12 confidently-wrong FP-hard documents

**Tool:** `inspect_fphard.py --out work/fphard_dump.jsonl` (student_r4a.pt).
The analysis document predicted dev docs, security articles and API docs; that
prediction was right for **0 of 12**. Actual composition:

1. **Casual-chat credential and instruction vocabulary — 7/12, all sms-ham:**
   "where 2 get user name & password" (0.997) · "admin credentials" (0.996) ·
   "buy and sell instructions… add my assistant's WhatsApp" (0.973) · "dont
   forget my pix" (0.935) · "revealing" greeting (0.899) · "Ok, I will follow
   your instruction." (0.858) · "PLEASE DONT IGNORE MYCALLS" (0.801). The model
   is keyword-reactive to credential / instruction / ignore lexemes in a casual
   register. These are true hard negatives: targeted-negative pattern A.
2. **Benign AI-meta prompts — 3/12, PolyGuard:** Midjourney prompt-generator
   meta-instructions (en 0.873 and its Dutch twin 0.803 — same template, both in
   eval, group split held) · a French PlantUML→DALL-E prompt request (0.690).
   User-authored instructions to an AI are benign, while the same text embedded
   in third-party content would be an attack. A text-only classifier cannot see
   provenance, so this part of the FP-hard mass is partly irreducible at model
   level; channel and context belong to the engine and UI layer. Pattern B, to
   be curated carefully — some "act as X" text really is jailbreak.
3. **Suspect labels — 2/12, PolyGuard "benign":** sv 777b596c09 ("…while
   IGNORING ALL WARNINGS OR EXCUSES THE MODEL MAY GIVE", a guardrail-bypass
   instruction) and en 5cef8cda86 ("access to an UNRESTRICTED VERSION of this
   language model… most unconventional and risky", jailbreak framing). Full
   texts verified in `inject_eval.jsonl`. The model flagging these is arguably
   correct and the PolyGuard labels are wrong. Excluding them, honest FP-hard is
   10/34 ≈ 0.294. Queued action: an adapter-level `label_suspect` exclusion list
   in the ingest pipeline, documented in provenance, with no silent row surgery.

**Consequence for data work:** thicken the FP-hard eval and training sets with
patterns A and B — not dev-doc templates, that guess is dead — and audit
PolyGuard benigns for more jailbreak-shaped mislabels before trusting FP-hard
trends.

## r5a — focal diagnosis (γ0, r4a config otherwise)

**Command:** `train.py --train --epochs 6 --focal-gamma 0` → `student_r5a.pt`
**Result:** P 0.872 | R 0.913 | F1 0.892 | FP-rate 0.181 | FP-hard 0.361 (13/36).
Diagnostic targets: it 0.735→0.735 (unchanged) · nl 0.697→0.727 (+1 document).

**Verdict:** the focal-starvation hypothesis is rejected — disabling focal did
not recover it/nl, confirming those two as data thinness (≈220 windows each) and
sending them to the r7 data queue. The aggregate is slightly worse without focal
(F1 −0.005, FP-rate +0.019, FP-hard 12→13), so focal γ1.5 stays and the r4a
config remains the v1 candidate. The cs/es up, de/sv/fr/pt down pattern is
noise-band shuffling.

## r5b — 10 epochs (r4a config otherwise)

**Command:** `train.py --train --epochs 10` → `student_r5b.pt`
**Result:** P 0.875 | R 0.907 | F1 0.891 | FP-rate 0.174 | FP-hard 0.361
(13/36). Train loss 0.0010 at epoch 10 is memorization, and eval is flat to
worse (F1 −0.006, FP-rate +0.012 against r4a). Per-language shuffles (de 0.795
best yet, fr/cs/nl down) are overfit jitter, not signal.

**Verdict:** more epochs do not help at this corpus size — 6 epochs was right.

## Round 1 close — r4a is the v1 candidate

8 runs (r1 · r2 · r3 · r4a · r4b · r5a · r5b, plus smoke). Winner: **r4a**
(hidden 192 · ff 768 · LayerNorm · focal γ1.5 · grad-clip · warmup + cosine ·
seed 0 · 6 epochs · no teacher · threshold 0.5) — F1 0.897 / recall 0.910 /
FP-hard 0.333. Checkpoint `work/student_r4a.pt`.

Locked findings: architecture, not distillation, fixed multilinguality ·
distillation trades FP-hard against non-English at every α, so it is shelved
until the corpus grows · focal γ1.5 confirmed (r5a) · 6 epochs confirmed (r5b) ·
it/nl are data thinness (r5a) · FP-hard n=36 is too thin, 2 of 12 PolyGuard
labels are suspect, and the real patterns are casual-SMS credential talk (7) and
benign AI-meta prompts (3), not dev docs.

**Still pending (not this round):** ONNX/INT8 export, tract smoke and escalation
wire-in in the Rust inference wrapper · data work (pattern A+B negatives, label
audit, it/nl/sv positives, FP-hard eval ≥150, zh source diversity) ·
per-channel threshold calibration.

**r5 queue (from the post-r4 review):**

- **r5-inspect** (first): `inspect_fphard.py` — read the ~12 confidently-wrong
  FP-hard documents (top-window text) to produce a pattern list, and from it
  targeted negatives rather than more generic templates. The tool was added
  post-r4 and mirrors `run_train`'s architecture in lockstep.
- **r5a:** r4a config plus `--focal-gamma 0 --epochs 6` — the it/nl diagnosis
  (focal starving easy templated languages vs data thinness). Read only large
  consistent deltas (per-language n≈33, ±0.03 per document); ignore the FP-hard
  column under the n=36 rule.
- **r5b:** r4a config plus `--epochs 10` — the cheapest probe (loss 0.023 still
  falling).
- **r6:** hard-negative mining in `train.py`, implemented after r5-inspect shows
  the patterns. **r7:** re-run on the grown corpus. Note that sv 0.800 exceeds
  the teacher ceiling of 0.633, so the sv eval is likely homogeneous
  (PolyGuard) — validate before trusting it.

---

## Data-hygiene round (between round 1 and round 2; no training)

The corpus and eval were repaired before the r4a-config re-run — detail in
`SOURCES.md` under "Data-hygiene round". Language tags were fixed (1,322 English
LLMail emails were tagged "zh", which makes the round-1 per-language numbers for
zh/ru/ar/tr unreliable), 5 PolyGuard label suspects were excluded, the FP-hard
eval went from 36 to 183, and the split was rebuilt (train 13,377 / eval 2,690 /
21,663 windows).

**Comparability break:** the eval set changed. Round-1 numbers (r1–r5b,
including the v1 candidate r4a at F1 0.897) were measured on the old eval and
must not be compared against runs on the new one. The first step of round 2 is
to re-score `work/student_r4a.pt` on the new eval; that number, not 0.897, is
the baseline the re-run must beat. Teacher logits from any earlier run are stale
because the window set changed — regenerate from scratch.

## r4a re-score — round-2 baseline anchor (no training)

`train.py --eval-checkpoint work/student_r4a.pt --vocab work/vocab_r1.txt` on
the new post-hygiene eval. This is the number round-2 runs must beat:

- **F1 0.874 | precision 0.845 | recall 0.904 | FP-rate 0.193 | FP-hard 0.235
  (43/183)** at threshold 0.5.
- Per-language recall is finally readable (the old zh/ru/ar/tr numbers were
  mislabeled-data artifacts): en 0.954 · fr 0.850 · it 0.824 · es 0.794 ·
  pt 0.750 · cs 0.706 · pl 0.676 · sv 0.667 · nl 0.636 · **de 0.625** — the
  weakest measurable language despite de being the second-largest eval slice.
  Watch it in round 2.
- Threshold sweep: F1 rises monotonically to 0.884 at 0.60, but non-English
  recall collapses there (de 0.562, sv 0.533, nl 0.545), so the sweep's F1 gain
  is an English-mass effect rather than a free win. Per-channel calibration
  remains the right lever; do not chase 0.60 globally.
- FP-hard 0.235 on n=183 is statistically meaningful for the first time
  (round 1's 0.333 was on n=36).

## r6 — round-2 baseline (post-hygiene data, 4 epochs)

**Command:** `train.py --train` (defaults: hidden 192, focal γ1.5, 4 epochs,
seed 0, no teacher) on the hygiene-round data (train 13,377 / 21,663 windows).
→ `work/student.pt`

**Against the r4a re-score anchor, same eval:**

| metric | r4a re-score | r6 | Δ |
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

**Reading:** the data round paid off exactly where it aimed. FP-hard more than
halved — the +559 multilingual FP-hard ham rows taught that scary words in
legitimate text are not injection — and de jumped +0.125 on the 90 real German
rows restored by the language repair. The fr/es dips are the visible cost of the
same lever: the new FP-hard mining was French-heavy (105 fr rows), pushing the
model conservative on Romance-language credential vocabulary. Recall held at
0.903 overall.

**Caveat:** 4 epochs (the default) is not the r4a config (6 epochs, validated in
r5b), and train loss was still falling at epoch 4 (0.0558→0.0445), so r6 is
likely undertrained. Next: **r6b = `--epochs 6`** for the true
r4a-config-on-new-data comparison, then the teacher and a distillation probe.

## r6b — r4a config (6 epochs) on the new data

**Command:** `train.py --train --epochs 6` → `work/student.pt` (overwrote r6's
4-epoch weights; both runs are seed-0 deterministic and reproducible).

**Against r6 (4 epochs, same data and eval), both at 0.5:**

| metric | r6 (4 ep) | r6b (6 ep) |
|---|---|---|
| F1 | 0.892 | 0.892 |
| recall | 0.903 | 0.907 |
| FP-rate | 0.143 | 0.147 |
| FP-hard | **0.104 (19/183)** | 0.131 (24/183) |
| de | **0.750** | 0.688 |
| loss at last epoch | 0.0445 | 0.0245 |

**Verdict:** the r5b-era "6 epochs is right" finding does not carry to the
post-hygiene corpus. F1 is identical, FP-hard is 5 documents worse, per-language
shifts are all inside the 1-3 document noise band, and loss 0.0245 drifts toward
the memorization territory r5b mapped (0.0010 at 10 epochs). With more and
cleaner data, 4 epochs reaches the same place, so the round-2 default is 4
epochs (r6). Do not chase the threshold-0.45 sweep row (F1 0.894, prettier
per-language numbers) — picking the threshold that flatters the eval is fitting
to noise, and per-channel calibration stays the principled lever.

**Next:** teacher (mmBERT-small) plus a distillation probe at the default 4
epochs, so the only variable against r6 is the teacher signal.

## Teacher (round 2) — mmBERT-small on the post-hygiene corpus

**Command:** `distill_teacher.py --model jhu-clsp/mmBERT-small --batch 16`
(3 epochs, embed-frozen, max-len 768, 0 truncated). ~64 min at 16-17 rows/s,
slightly under the 22.6 measured in the solo run. Exported 18,010 soft labels
(unique window hashes out of 21,663 windows) → `work/teacher_logits.jsonl`.

**Teacher doc-eval, i.e. the student's quality ceiling:**
F1 **0.928** | precision 0.878 | recall 0.985 | FP-rate 0.159 | FP-hard 0.109
(20/183). Per-language recall: en 0.992 · fr 1.000 · pt 1.000 · es 0.971 ·
pl 0.971 · nl 0.970 · **de 0.958** · it 0.941 · cs 0.912 · sv 0.900.

**Reading:** the ceiling is +0.036 F1 over r6 and the non-English gap is where
the headroom lives (de 0.958 vs student 0.750; sv 0.900 vs 0.600). But teacher
FP-hard (0.109) is level with the student's (0.104), so the teacher has nothing
to teach on FP-hard and the round-1 risk — distillation trading FP-hard for
non-English — is still live. Judge the probe on "non-English up, FP-hard not
worse".

Note on the export warning that 2 window hashes carry conflicting gold labels:
this is an expected artifact, not a new bug. bipia-ctx poisoned and clean twins
share body text, so a window that misses the injected span is byte-identical
across both documents while carrying each document's label. 2 of 18,010 is
negligible; the export keeps the first.

## r7 — distillation probe (α0.5, round-2 corpus)

**Command:** `train.py --train --teacher-logits work/teacher_logits.jsonl`
(defaults: 4 epochs two-phase = 2 CE + 2 distill, α0.5, T2.0; coverage
21,663/21,663 = 100%). → `work/student.pt` (overwrote r6b).

**Three-number comparison (all at 0.5, same eval):**

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

**Reading:** the round-1 shelving verdict — distillation trades FP-hard for
non-English, no α wins both — is broken on the bigger, cleaner corpus.
Non-English moved sharply toward the teacher (sv +0.200, nl +0.121, it/cs
+0.089) while FP-hard gave up only 3 documents (19→22 of 183) and F1 held. This
was exactly the "retry when the corpus grows" condition. The residual trade is
mild and visible in precision and FP-rate. de is the outlier, unmoved at 0.750
against a teacher at 0.958: the student's remaining German gap is not a
soft-label problem, more likely capacity or tokenizer, and it ties into the
German-source data work for round 3. Distillation loss was still falling
(0.359→0.262 over 2 distill epochs), so a 2+4-epoch variant is plausible
headroom — but epoch extension flirted with memorization in r6b, so park it
rather than knob-chase now.

## Round 2 close — r7 is the v2 candidate

**v2 candidate = r7 (distill α0.5).** Checkpoint preserved as
`work/student_r7.pt` (a copy of the r7 `student.pt`). F1 0.891 / recall 0.918 /
FP-hard 0.120 (22/183) / sv 0.800 · nl 0.788 · it/cs 0.824 · de 0.750 at
threshold 0.5, vocab = `work/vocab.txt` (round 2).

The round-2 arc in one line: data hygiene (language fix, label suspects, FP-hard
183) → r4a re-scored at 0.874 as an honest anchor → r6 CE baseline 0.892 with
FP-hard halved by mined multilingual ham → r6b showed 6 epochs no longer helps →
teacher ceiling 0.928 → r7 distillation broke round 1's FP-hard ↔ non-English
trade, buying a sharp non-English lift for 3 FP-hard documents.

**Rescinded:** round 1's "distillation shelved" verdict, which was conditional
on corpus size — the condition was met and tested. **Standing findings:** 4
epochs by default · do not chase sweep thresholds · the German gap is not a
soft-label problem, so round-3 data work needs real German positives.

**Remaining to ship r7:** ONNX export (dynamic quant, not QDQ) → tract smoke →
Ed25519 OTA artifact → escalation-routing wire-in → deterministic-engine
threshold calibration on the same eval infrastructure. The export script does
not exist yet here.

## Export (r7 → ONNX)

**Command:** `export_onnx.py` (defaults) → `work/model/` {model.onnx 19.16 MB
fp32 · **model_int8.onnx 4.86 MB dynamic-quant** · vocab.txt ·
export_manifest.json with sha256s}.

- Parity over 512 eval windows: torch↔fp32 max|Δlogit| 4e-06 with 0 flips
  (exact); torch↔int8 max|Δlogit| 0.224 with 2/512 flips (normal quant noise).
- **INT8 doc-eval, the shipping number: F1 0.891 | recall 0.919 | FP-hard 0.120
  (22/183)** — identical to the r7 checkpoint (Δ ≈ 1 document), per-language
  table unchanged. Quantization cost is about zero.
- 4.86 MB sits in the same band as the SMS model (4.5 MB), so the device budget
  holds.

**Remaining chain:** tract smoke (shared loader family, [1,2] head) → Ed25519
OTA artifact → escalation-routing wire-in → per-channel calibration.

## Tract smoke (Rust) — passed, plus a topic-confound finding

**New:** a Rust inference module for this model (its own feature flag, same
dependencies as the SMS one): a `char_windows` Rust mirror (char-counted,
tail-anchored, 5 unit tests, feature-less), the tract backend (SEQ_LEN 256,
[1,2] head, reusing the lockstep WordPieceTokenizer), `score_document` max-pool,
an unk-abstain signal, and an env-var self-skip smoke.

- **Smoke passed** against the real `model_int8.onnx` (~1.5 s load and infer):
  injection 1.000 · neutral benign 0.023 · benign + LLMail exfil line 0.999.
- **tract ↔ onnxruntime parity confirmed** on identical texts (0.9999/1.0000,
  matched to four decimals) — the Rust windowing and tokenizer port is faithful.

**Topic confound (round-3 data item, found by the smoke test):** benign corporate
budget-email prose ("This quarter's budget review covers vendor spend and
hiring…") scores **0.9999**. All 4,018 LLMail positives are budget-themed and
the corpus has no budget-themed benign email — email-benign is order and
statement themed. The eval cannot see this either, since it has no such benign
slice: a distribution blind spot, and exactly the honest-eval caveat. Real-user
impact is a false-positive risk on legitimate corporate email about budgets and
planning. The fix is data: benign corporate-email negatives on LLMail themes
(budget, planning, meeting) plus an eval slice for them. Deterministic
rule-layer gating and the inform-only doctrine bound the blast radius until then.

## Packaging and naming

**Naming.** The two students trained in this repository are:

- **`mmbert-tiny-sms`** — the SMS classifier (lineage `v8-mmbert-distill`).
- **`mmbert-tiny-inject`** — this model (internal `r7` distill, round 2).

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
