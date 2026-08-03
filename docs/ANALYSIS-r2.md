# `mmbert-tiny-inject` — r1→r2 distillation regression + improvement roadmap

**Scope:** train.py + distill_teacher.py + injection corpus + process
**Verdict:** r2 distillation regressed non-English recall by −0.03..−0.38 across 9 languages while
improving English (+0.025) and FP-hard (0.389→0.250). Root cause: **three compounding factors**, not one.

---

## 1. Results summary

### 1.1 r1 (baseline, no teacher)

| metric | value |
|---|---|
| precision | 0.891 |
| recall | 0.887 |
| F1 | 0.889 |
| FP-rate | 0.147 |
| FP-hard | **0.389** (14/36) |

Per-lang recall (measurable): en 0.909 · zh 0.992 · fr 0.900 · it 0.882 · nl 0.848 ·
pl 0.765 · es 0.765 · pt 0.750 · de 0.705 · sv 0.633 · cs 0.588

### 1.2 r2 (distill mmBERT-small α0.5 T2.0)

| metric | r1 | r2 | Δ |
|---|---|---|---|
| precision | 0.891 | 0.913 | +0.022 |
| recall | 0.887 | 0.859 | −0.028 |
| F1 | 0.889 | 0.885 | −0.004 |
| FP-rate | 0.147 | **0.111** | −0.036 |
| FP-hard | 0.389 | **0.250** | −0.139 |

Per-lang recall r1→r2:
- en 0.909 → **0.934** (+0.025)
- zh 0.992 → 0.996 (flat)
- fr 0.900 → **0.720** (−0.180)
- de 0.705 → **0.545** (−0.160)
- it 0.882 → **0.500** (−0.382)
- es 0.765 → 0.559 (−0.206)
- pl 0.765 → 0.559 (−0.206)
- nl 0.848 → 0.545 (−0.303)
- pt 0.750 → 0.531 (−0.219)
- cs 0.588 → 0.500 (−0.088)
- sv 0.633 → 0.600 (flat)

**Pattern:** English improved while all non-English Latin languages collapsed uniformly. Systematic, not noise.

### 1.3 Teacher quality ceiling

| metric | teacher | r1 student | r2 student |
|---|---|---|---|
| F1 | 0.958 | 0.889 | 0.885 |
| FP-hard | 0.083 | 0.389 | 0.250 |
| en recall | 0.983 | 0.909 | 0.934 |
| de recall | 0.818 | 0.705 | 0.545 |
| it recall | 0.794 | 0.882 | 0.500 |

The teacher is strong across languages — the student failed to follow it.

---

## 2. Root-cause analysis — three compounding factors

### Factor A: loss-weighting imbalance (primary)

**The math:** with α=0.5, T=2.0:

```
loss = (1-α)×CE + α×T²×KL
     = 0.5×CE + 0.5×4×KL
     = 0.5×CE + 2.0×KL
```

- **KL weight = 2.0, CE weight = 0.5 → ratio 4:1**
- **80% of the gradient comes from KL, only 20% from CE**
- CE is the only place where thin non-English hard labels speak at full weight
- KL matches the teacher's near-degenerate distribution (P(inj)≈1.0 or ≈0.0),
  which is dominated by EN/zh mass → student optimizes for EN/zh

**Evidence:** teacher soft-labels measured at P(injection):
- EN injection windows: mean 0.995, low-confidence (<0.9) = 1%
- DE injection windows: mean 0.937, low-confidence (<0.9) = 9%
- IT injection windows: mean 0.937, low-confidence (<0.9) = 9%
- NL injection windows: mean 0.937, low-confidence (<0.9) = 10%

The teacher is slightly less confident on non-English, but the KL loss pushes the student
to match all distributions equally — the non-English signal, already thinner, gets drowned
by the 4:1 KL:CE ratio.

### Factor B: architecture at half the SMS model (contributing)

| aspect | SMS v8 (successful) | mmbert-tiny-inject r2 (regressed) |
|---|---|---|
| hidden | **256** | **128** (half) |
| feedforward | **1024** (hidden×4) | **256** (hidden×2, quarter) |
| LayerNorm before head | **yes** | **no** |
| focal loss | **yes** (α-balanced, γ=1.5) | **no** (plain CE) |
| class-imbalance weights | **yes** (per-class α) | **no** |
| gradient clipping | **yes** (1.0) | **no** |
| learning rate schedule | implicit (8 epochs) | none (fixed 3e-4) |
| SEQ_LEN | 128 | 256 (double the sequence, half the capacity) |
| vocab | ~10-12k | 20k (double the embedding params to train) |

The mmbert-tiny-inject student processes **2× longer sequences** with **half the hidden size** and
**a quarter of the feedforward capacity**, while also training a **2× larger vocab
embedding**. The model simply lacks capacity to simultaneously match the teacher's
distribution on EN-dominant mass AND learn the non-EN signal.

### Factor C: training-data imbalance (contributing)

Window distribution (training set, 21,284 windows):

| lang | windows | % of total | injection | benign |
|---|---|---|---|---|
| en | 9,740 | **45.7%** | 7,699 | 2,033 |
| zh | 4,292 | 20.1% | 4,120 | 172 |
| fr | 1,629 | 7.6% | 703 | 902 |
| de | 723 | 3.4% | 317 | 406 |
| nl | 467 | 2.2% | 241 | 226 |
| it | 452 | 2.1% | 233 | 219 |
| es | 438 | 2.1% | 236 | 202 |
| pt | 418 | 2.0% | 227 | 191 |
| pl | 412 | 1.9% | 223 | 189 |
| cs | 408 | 1.9% | 204 | 204 |
| sv | 386 | 1.8% | 198 | 188 |

English plus Chinese make up **65.8%** of all training windows. Non-English Latin languages
have 400-700 windows each, 15-25× fewer than English, and no per-language oversampling or
class weighting exists.

Additional data issues:
- **PolyGuard (5,400 rows)** is all `channel:chat` jailbreak-class — not indirect injection.
  May teach the model jailbreak patterns rather than the indirect injection threat model.
- **Non-Latin benign-only languages** (hi, ja, ko, th, bs — 0 injection, 200+ benign each)
  may teach the model that these scripts are always safe.
- **synthetic-ml** (20 injection/lang) is highly templated — limited diversity.

---

## 3. Improvement roadmap

### Tier 1: immediate code changes (minutes, no teacher re-training)

#### 3.1 Fix loss weighting — `--distill-alpha 0.15`

**Change:** `python3 train.py --train --teacher-logits work/teacher_logits.jsonl --distill-alpha 0.15`

**Math:** loss = 0.85×CE + 0.15×4×KL = 0.85×CE + 0.6×KL → KL:CE = 0.71:1 (41% KL)

This restores CE as the dominant signal while keeping distillation as a regularizer.
The TRAINING-LOG proposed α=0.3 (KL:CE = 1.7:1, 63% KL) — still too KL-heavy.
The SMS v8 success used α=0.5 on a model with focal loss + class weights + 2× capacity.
Without those, the mmbert-tiny-inject student needs α much lower.

**Why not α=0.3:** 0.3×4=1.2 KL vs 0.7 CE → ratio 1.7:1 → 63% KL. This is still
dominated by KL. The goal is to let CE (hard labels) lead, with KL as a soft prior.
α=0.15 gives 41% KL, which is closer to a "regularization" role.

#### 3.2 Add gradient clipping

```python
# After loss.backward(), before opt.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

The SMS classifier has this. The mmbert-tiny-inject train.py is missing
it. Without clipping, distillation gradients (scaled by T²=4) can cause large updates
that destabilize non-EN representations.

#### 3.3 Add LayerNorm before classification head

```python
# In Classifier.__init__:
self.norm = nn.LayerNorm(hidden)
# In forward, before self.head:
pooled = self.norm(pooled)
return self.head(pooled)
```

The SMS model has this (make_smoke_model.py line 74). It stabilizes the pooled
representation before the linear head, especially important when the loss landscape
is dominated by KL (which produces different gradient dynamics than CE).

### Tier 2: short-term code changes (hours)

#### 3.4 Add learning rate warmup + decay

```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_scheduler(opt, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return LambdaLR(opt, lr_lambda)

# In run_train:
total_steps = (len(train_pairs) // args.batch + 1) * args.epochs
scheduler = get_scheduler(opt, warmup_steps=total_steps // 10, total_steps=total_steps)
# After opt.step():
scheduler.step()
```

Warmup lets the model first learn basic structure (token embeddings, common patterns)
before distillation gradients push it toward the teacher's distribution.

#### 3.5 Add per-language oversampling

```python
# In run_train, after expanding train_pairs:
from collections import Counter
lang_counts = Counter()
for text, lbl in train_pairs:
    # crude lang detection: check if text matches known non-EN patterns
    # or better: carry lang from the doc
    pass

# Simpler approach: weight windows by inverse language frequency
# Oversample non-EN windows to ~2× their natural frequency
TARGET_MIN = 2000  # target windows per language
oversampled = []
lang_win = collections.defaultdict(list)
for text, lbl in train_pairs:
    # need lang tracking — see implementation note below
    lang_win[lang].append((text, lbl))

for lang, wins in lang_win.items():
    if len(wins) < TARGET_MIN:
        # repeat with different shuffles to reach target
        import random
        rng = random.Random(42)
        pool = list(wins)
        while len(pool) < TARGET_MIN:
            rng.shuffle(wins)
            pool.extend(wins[:TARGET_MIN - len(pool)])
        oversampled.extend(pool[:TARGET_MIN])
    else:
        oversampled.extend(wins)
train_pairs = oversampled
```

**Implementation note:** This requires carrying `lang` through `expand_training`.
Currently `expand_training` returns `(window_text, label)` tuples. Change to
`(window_text, label, lang)` and thread it through.

#### 3.6 Add doc-threshold sweep

```python
# In evaluate_maxpool, after the main eval loop, add:
for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    tp = fp = tn = fn = 0
    for d in eval_docs:
        # ... same scoring loop ...
        pred = "injection" if max_pool(scores) >= thresh else "benign"
        # ... same counting ...
    # print threshold + metrics
```

This tests hypothesis (b) from the TRAINING-LOG: if distillation smooths scores,
a lower threshold may recover non-EN recall without sacrificing FP-hard.

### Tier 3: architecture changes (half a day, needs a retrain)

#### 3.7 Increase model capacity to match SMS v8

```python
# Change from:
hidden, heads, layers = 128, 4, 2
# To:
hidden, heads, layers = 192, 4, 2  # compromise: bigger than current, smaller than SMS

# And change feedforward from hidden*2 to hidden*4:
enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4,  # was hidden * 2
                                 batch_first=True, dropout=0.1)
```

**Trade-off:** hidden=192 → ~5.5 MB INT8 (vs current ~3 MB, SMS is ~4.5 MB).
This is still well within the on-device budget. The feedforward change (hidden×4
vs hidden×2) is the more impactful one — it doubles the model's capacity to
represent non-linear boundaries, which is critical for multilingual injection detection.

#### 3.8 Add focal loss (port from SMS model)

```python
def focal_loss(logits, y, alpha, gamma=1.5):
    """Alpha-balanced focal loss, per-sample (reduction='none')."""
    ce = F.cross_entropy(logits, y, reduction='none')
    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma
    return alpha[y] * focal

# Usage:
alpha = torch.tensor([1.0, 1.0], device=device)  # 2-class; tune per-class
ce = focal_loss(logits, ys, alpha, args.focal_gamma)
```

Focal loss down-weights easy examples (where the model is already confident), which
helps the model focus on hard non-EN cases instead of coasting on the EN-dominant mass.

### Tier 4: data improvements (days)

#### 3.9 Expand non-EN real injection data

Current state:
- Non-EN injection windows: ~200-400 per language (vs 7,699 for EN)
- Only zh has meaningful volume (4,120) thanks to PolyGuard

Priority targets:
1. **German** (highest-priority non-EN language): de has only 317 injection windows. Need ≥1,000.
   - Harvest real German injection examples from security blogs, CTF writeups
   - Translate the deepset corpus (662 rows) to German using a local model
   - Add German-specific FP-hard negatives (dev docs with "ignore", "Anweisungen")
2. **Swedish** (sv 0.633 = corpus gap, confirmed by teacher ceiling): need real sv
   injection text, not just synthetic templates.
3. **Turkish** (tr only 20 eval positives, n<30): need ≥100 real tr eval rows.

#### 3.10 Remove non-Latin benign-only languages from training

hi (211), ja (200), ko (200), th (200), bs (274) — all benign-only, 0 injection.
The Rust inference wrapper abstains on non-Latin at inference time, so these windows:
- Don't help the model (no positive signal)
- Waste training capacity (the model tries to learn these as "always benign")
- May bias the model toward "unfamiliar script = benign"

**Action:** either remove these from training, or add injection examples for them.
Removing is simpler and honest — the model will not process them at inference.

#### 3.11 Add InjecAgent dataset

Currently in "review" status (SOURCES.md). InjecAgent (arXiv 2403.02691) provides
agentic/tool-chain injection — a different attack surface than chat jailbreaks.
This would diversify the injection signal beyond PolyGuard's jailbreak-class.

#### 3.12 Thicken FP-hard eval set

Current: 36 FP-hard eval negatives. Too thin for statistical significance.
Target: ≥100 FP-hard negatives per language (or at least 200 total for EN).

### Tier 5: process improvements

#### 3.13 Two-phase training schedule

Instead of training with distillation from epoch 1:

```python
# Phase 1: CE-only (first 2 epochs) — learn hard-label structure
# Phase 2: Add distillation (remaining epochs) — refine with teacher
if ep < 2:
    loss = ce.mean()  # no distillation
else:
    loss = ((1.0 - args.distill_alpha) * ce
            + args.distill_alpha * (T * T) * kl).mean()
```

This lets the model first establish its representation (especially for non-EN)
using hard labels, then refine with the teacher's soft labels. This is the
"pretrain then distill" pattern, proven in NLP distillation literature.

#### 3.14 Separate baseline + distill comparison per language

Add per-language F1/recall tracking to the comparison table, not just aggregate.
The aggregate F1 can hide language-specific regressions (as it did in r2: F1
dropped only 0.004, but non-EN recall collapsed).

#### 3.15 Distillation-specific smoke test

Before running full distillation, run a 500-window smoke test with `--smoke --teacher-logits`
to verify the KL loss is computing correctly and the gradient magnitudes are reasonable.
The current smoke test doesn't exercise distillation.

---

## 4. Priority matrix

| # | Change | Effort | Impact | Risk | Priority |
|---|---|---|---|---|---|
| 3.1 | `--distill-alpha 0.15` | 1 min | high | low | **P0** (immediate) |
| 3.2 | Gradient clipping | 1 line | medium | none | **P0** |
| 3.3 | LayerNorm before head | 2 lines | medium | low | **P0** |
| 3.6 | Doc-threshold sweep | 20 min | high (tests hypothesis b) | none | **P0** |
| 3.4 | LR warmup + cosine decay | 15 min | medium | low | **P1** |
| 3.13 | Two-phase training | 10 min | high | low | **P1** |
| 3.5 | Per-lang oversampling | 1 h | high | medium (need lang threading) | **P1** |
| 3.8 | Focal loss port | 30 min | medium | low | **P1** |
| 3.7 | hidden=192, ff=hidden×4 | 30 min + retrain | high | medium (size ↑) | **P2** |
| 3.10 | Remove non-Latin benign-only | 5 min | low | none | **P2** |
| 3.9 | Expand non-EN real data | days | high (long-term) | low | **P2** |
| 3.12 | Thicken FP-hard eval | days | medium | none | **P2** |
| 3.11 | InjecAgent dataset | hours | medium | low | **P3** |
| 3.15 | Distill smoke test | 30 min | low | none | **P3** |

---

## 5. Recommended experiment sequence

**r3 (P0, minutes):** `--distill-alpha 0.15` + gradient clipping + LayerNorm + threshold sweep
- If non-EN recovers → accept as v1 candidate
- If non-EN partially recovers → r4

**r4 (P1, ~1 h):** r3 + LR warmup + two-phase training + focal loss
- If still insufficient → r5

**r5 (P2, ~2 h):** r4 + hidden=192, ff=hidden×4 + per-lang oversampling
- This is the "SMS v8 equivalent" configuration

**r6 (P2, days):** r5 + expanded non-EN data + InjecAgent + remove non-Latin benign-only

---

## 6. What not to change

- **Windowing algorithm** (`windowing.py`): the char-counted sliding window + max-pool
  is correct and well-tested. The tail-injection problem is solved.
- **Teacher model** (mmBERT-small): the teacher is strong (F1 0.958, FP-hard 0.083).
  The problem is the student, not the teacher.
- **Teacher logits**: 17,633 soft-labels, 100% window coverage, zero truncation.
  Reusable for all experiments without re-running the teacher.
- **Eval oracle** (`eval_oracle.py`): group-aware split, FP-hard tagging, per-language
  probe — all correct and honest.
- **Vocab size** (20k): adequate. The vocab includes multilingual subwords (Cyrillic,
  CJK, Arabic). The issue is model capacity, not vocab coverage.

---

## 7. Comparison with the SMS v8 result

| factor | SMS v8 | mmbert-tiny-inject r2 | lesson |
|---|---|---|---|
| hidden | 256 | 128 | mmbert-tiny-inject needs ≥192 |
| feedforward | 1024 | 256 | mmbert-tiny-inject needs ≥768 |
| focal loss | yes | no | port it |
| class weights | yes | no | add them |
| grad clip | yes | no | add it |
| LayerNorm | yes | no | add it |
| LR schedule | implicit (8 ep) | none | add warmup |
| data volume | 150k rows | 15.5k rows | mmbert-tiny-inject has 10× less |
| data balance | massive multilingual | EN-dominant (65.8%) | oversample non-EN |
| distill result | +0.021 F1 | −0.004 F1 | distillation needs the above to work |
| distill α/T | 0.5 / 2.0 | 0.5 / 2.0 | same params, different capacity → fails |

The SMS v8 distillation worked because the model had enough capacity (hidden=256, ff=1024)
and training aids (focal loss, class weights, grad clip, LayerNorm) to follow the teacher
across all languages. The mmbert-tiny-inject student at half the capacity and without those aids simply
couldn't follow — the KL gradient dominated and the model optimized for the EN-dominant mass.

---

## 8. Honest limits

1. **Only 36 FP-hard eval negatives** — every FP-hard metric has a wide confidence interval,
   and the r1→r2 improvement (14→9) could be noise. Thicken it before drawing conclusions.
2. **Per-lang eval is thin** — many languages have exactly 30-34 eval positives (the minimum
   threshold). A single misclassified example swings recall by ±0.03.
3. **sv 0.633 is a corpus gap** — confirmed by teacher ceiling (also 0.633). Not fixable
   by training changes; needs real Swedish injection data.
4. **zh 0.992 may be PolyGuard homogeneity** — PolyGuard zh is all jailbreak-class from
   the same source. Source-diverse zh eval doesn't exist yet.
5. **Teacher (mmBERT-small) is not the ceiling** — mmBERT-base would be stronger but
   OOMs on 16GB MPS (README measured 1.6 rows/sec, ~12 h). Consider overnight run
   only if small teacher proves insufficient after all code fixes.
