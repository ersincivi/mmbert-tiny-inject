#!/usr/bin/env python3
"""G4 — German-first WordPiece vocab from corpus.jsonl.

Produces `vocab.txt` in the exact format the Rust inference wrapper
loads: one token per line, id = line number, NO empty lines, specials
[PAD]/[UNK]/[CLS]/[SEP] present (engine errors without them).

Pre-tokenization MIRRORS the engine's `pre_tokenize` exactly: NFKC + lowercase,
whitespace split, ASCII punctuation as standalone tokens (non-ASCII punctuation
stays inside words — same as Rust `char::is_ascii_punctuation`).

Training: frequency-scored BPE merges emitted in WordPiece form (`##`
continuations) — compatible with the engine's greedy longest-match decoder.
German-first: de/fr/it rows are upweighted (MODEL_CONTRACT: de/en/fr/it);
en arrives at full natural volume, everything else rides along at weight 1
and falls back to char-level pieces or [UNK].

stdlib only. Run AFTER import_corpus.py:
  python3 build_vocab.py                       # corpus_work/vocab.txt, 11k
  python3 build_vocab.py --size 12000 --lang-weights de=20,fr=12,it=12
"""

import argparse
import heapq
import json
import string
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
ASCII_PUNCT = set(string.punctuation)  # == Rust char::is_ascii_punctuation
MAX_WORD_CHARS = 100  # engine-side guard; longer words are [UNK] at runtime


def pre_tokenize(text: str):
    """Mirror of the engine's pre_tokenize — keep in lockstep."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    out, cur = [], []
    for c in normalized:
        if c.isspace():
            if cur:
                out.append("".join(cur))
                cur = []
        elif c in ASCII_PUNCT:
            if cur:
                out.append("".join(cur))
                cur = []
            out.append(c)
        else:
            cur.append(c)
    if cur:
        out.append("".join(cur))
    return out


# ── BPE training (frequency-scored, WordPiece output form) ───────────────────


# ── Non-Latin [UNK]-protection (operator 2026-07-02, adapted smart_trim idea) ──
# WordPiece char-level FALLBACK: a word it can't split into subwords is broken to
# single chars; if those chars are in vocab it is NOT [UNK], just a longer piece
# sequence. If even ONE char of an alphabet is missing, the WHOLE word → [UNK] and
# its entropy is zeroed. So we UNCONDITIONALLY seed the base letters of Cyrillic +
# Arabic (cheap, ~300 tokens — guarantees no-total-[UNK] for ru/ar even with zero
# training data), and take a frequency QUOTA of CJK chars from the corpus (their
# semantics arrive once Mistral zh/ja data is added). Rules are the operator's;
# the source is our CORPUS frequency (not a 256k teacher vocab we don't have).
CYRILLIC = range(0x0400, 0x0460)  # main Cyrillic letters (Ё…я + Ukrainian/etc.)
ARABIC = range(0x0620, 0x0650)    # Arabic base letters
CJK_RANGES = ((0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF))  # Han·Kana·Hangul


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in CJK_RANGES)


def nonlatin_extra_alphabet(word_freq: Counter, cjk_quota: int) -> set:
    """Base chars to force into the vocab (both `c` and `##c` forms) so char-level
    fallback never total-[UNK]s a non-Latin word. Cyrillic+Arabic = full letter
    blocks unconditionally; CJK = the `cjk_quota` most frequent in the corpus."""
    extra = set()
    for cp in list(CYRILLIC) + list(ARABIC):
        c = chr(cp)
        extra.add(c)
        extra.add("##" + c)
    cjk_freq = Counter()
    for word, f in word_freq.items():
        for ch in word:
            if _is_cjk(ch):
                cjk_freq[ch] += f
    for ch, _ in cjk_freq.most_common(cjk_quota):
        extra.add(ch)
        extra.add("##" + ch)
    return extra


def train(word_freq: Counter, size: int, min_char_freq: int, extra_alphabet=None):
    """Iterative pair merges over the weighted word-frequency model.

    Words are piece-sequences; pieces after the first carry the `##` prefix,
    so emitted merge tokens are directly WordPiece-vocab entries.
    `extra_alphabet` = non-Latin base chars force-seeded for [UNK] protection.
    """
    # Alphabet: first-position chars + ## continuation chars above threshold.
    char_freq = Counter()
    for word, f in word_freq.items():
        chars = list(word)
        char_freq[chars[0]] += f
        for c in chars[1:]:
            char_freq["##" + c] += f
    alphabet = {tok for tok, f in char_freq.items() if f >= min_char_freq}
    # Force-seed the non-Latin base chars (counted against the merge budget).
    alphabet |= (extra_alphabet or set())

    # Words whose chars survived the threshold, as mutable piece lists.
    words, freqs = [], []
    for word, f in word_freq.items():
        pieces = [word[0]] + ["##" + c for c in word[1:]]
        if all(p in alphabet for p in pieces):
            words.append(pieces)
            freqs.append(f)

    # pair → weighted count, and pair → word ids that contain it.
    pair_count = Counter()
    pair_words = defaultdict(set)
    for wid, pieces in enumerate(words):
        f = freqs[wid]
        for a, b in zip(pieces, pieces[1:]):
            pair_count[(a, b)] += f
            pair_words[(a, b)].add(wid)

    heap = [(-c, pair) for pair, c in pair_count.items()]
    heapq.heapify(heap)

    merged_tokens = []
    budget = size - len(SPECIALS) - len(alphabet)
    while heap and len(merged_tokens) < budget:
        neg, pair = heapq.heappop(heap)
        if pair_count.get(pair, 0) != -neg or -neg < 2:
            continue  # stale heap entry / too rare to merge
        a, b = pair
        new_piece = a + b[2:]  # strip the ## off the right side
        merged_tokens.append(new_piece)
        touched = Counter()
        for wid in list(pair_words[pair]):
            pieces, f = words[wid], freqs[wid]
            i = 0
            while i < len(pieces) - 1:
                if pieces[i] == a and pieces[i + 1] == b:
                    if i > 0:
                        touched[(pieces[i - 1], a)] -= f
                        touched[(pieces[i - 1], new_piece)] += f
                        pair_words[(pieces[i - 1], new_piece)].add(wid)
                    if i + 2 < len(pieces):
                        touched[(b, pieces[i + 2])] -= f
                        touched[(new_piece, pieces[i + 2])] += f
                        pair_words[(new_piece, pieces[i + 2])].add(wid)
                    touched[(a, b)] -= f
                    pieces[i : i + 2] = [new_piece]
                else:
                    i += 1
        for p, delta in touched.items():
            if delta:
                pair_count[p] = pair_count.get(p, 0) + delta
                if pair_count[p] > 0:
                    heapq.heappush(heap, (-pair_count[p], p))
    return alphabet, merged_tokens


# ── greedy longest-match (mirror of the engine's encode, for coverage) ──────


def encode_words(words, vocab_set):
    """Yield per-word piece counts; None = [UNK] (mirrors engine semantics)."""
    for word in words:
        if len(word) > MAX_WORD_CHARS:
            yield None
            continue
        n, start, pieces = len(word), 0, 0
        ok = True
        while start < n:
            end = n
            while end > start:
                cand = word[start:end] if start == 0 else "##" + word[start:end]
                if cand in vocab_set:
                    pieces += 1
                    break
                end -= 1
            else:
                ok = False
                break
            start = end
        yield pieces if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(Path(__file__).parent / "corpus_work" / "corpus.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "corpus_work" / "vocab.txt"))
    ap.add_argument("--size", type=int, default=11000, help="total vocab entries (contract: 10-12k)")
    ap.add_argument("--min-char-freq", type=int, default=3,
                    help="weighted min frequency for alphabet chars (rarer → word becomes [UNK])")
    ap.add_argument("--cjk-quota", type=int, default=2500,
                    help="max CJK chars force-seeded (by corpus frequency) for [UNK] "
                         "protection; Cyrillic+Arabic base blocks are always seeded")
    ap.add_argument("--lang-weights", default="de=4,it=12",
                    help="per-language row upweights; unlisted languages weigh 1 "
                         "(defaults tuned to the 150k corpus: fr/en/es ham now arrive "
                         "at natural volume; de/it still need a boost)")
    args = ap.parse_args()

    weights = {}
    for part in args.lang_weights.split(","):
        lang, w = part.split("=")
        weights[lang.strip()] = int(w)

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"{corpus} not found — run import_corpus.py first (G1)")

    word_freq = Counter()
    rows = []  # (language, words) kept for the coverage report
    for line in corpus.open(encoding="utf-8"):
        rec = json.loads(line)
        words = pre_tokenize(rec["text"])
        rows.append((rec["language"], words))
        w = weights.get(rec["language"], 1)
        for token in words:
            word_freq[token] += w
    print(f"corpus: {len(rows):,} rows, {len(word_freq):,} unique words "
          f"(weights: {weights}, rest=1)")

    extra = nonlatin_extra_alphabet(word_freq, args.cjk_quota)
    cjk_seeded = sum(1 for t in extra if not t.startswith("##") and _is_cjk(t))
    alphabet, merged = train(word_freq, args.size, args.min_char_freq, extra_alphabet=extra)
    vocab = SPECIALS + sorted(alphabet, key=lambda t: (-word_freq.get(t, 0), t)) + merged
    print(f"vocab: {len(SPECIALS)} specials + {len(alphabet):,} alphabet "
          f"({len(extra):,} non-Latin force-seeded: Cyrillic+Arabic + {cjk_seeded} CJK) + "
          f"{len(merged):,} merges = {len(vocab):,}")

    out = Path(args.out)
    out.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    # ── coverage report (engine-equivalent greedy longest-match) ────────────
    vocab_set = set(vocab)
    by_lang = defaultdict(lambda: [0, 0, 0, 0])  # words, unk_words, msgs, msgs_fit64
    for lang, words in rows:
        tokens = 0
        unk = 0
        for pieces in encode_words(words, vocab_set):
            if pieces is None:
                unk += 1
                tokens += 1
            else:
                tokens += pieces
        st = by_lang[lang]
        st[0] += len(words)
        st[1] += unk
        st[2] += 1
        st[3] += tokens <= 126  # [CLS] + 126 + [SEP] = SEQ_LEN 128 (v7)

    report = out.parent / "vocab_report.md"
    with report.open("w", encoding="utf-8") as r:
        r.write("# G4 vocab report\n\n")
        r.write(f"`vocab.txt`: **{len(vocab):,} entries** (specials {len(SPECIALS)}, "
                f"alphabet {len(alphabet):,}, merges {len(merged):,}); "
                f"lang-weights {weights}, min-char-freq {args.min_char_freq}.\n\n")
        r.write("| language | msgs | [UNK] word-rate | fits SEQ_LEN 128 |\n|---|---|---|---|\n")
        total = [0, 0, 0, 0]
        order = sorted(by_lang.items(), key=lambda kv: -kv[1][2])
        for lang, st in order:
            for i in range(4):
                total[i] += st[i]
            if st[2] >= 30:  # skip micro-languages in the table
                r.write(f"| {lang} | {st[2]:,} | {st[1] / max(st[0], 1):.2%} "
                        f"| {st[3] / st[2]:.2%} |\n")
        r.write(f"| **TOTAL** | {total[2]:,} | {total[1] / max(total[0], 1):.2%} "
                f"| {total[3] / total[2]:.2%} |\n")
        r.write("\nPre-tokenization mirrors the engine's `pre_tokenize`; coverage uses the "
                "same greedy longest-match + whole-word-[UNK] semantics as the engine.\n")
    print(f"vocab → {out}\nreport → {report}")
    for lang in ("de", "fr", "it", "en", "tr"):
        if lang in by_lang:
            st = by_lang[lang]
            print(f"  {lang}: unk {st[1] / max(st[0], 1):.2%}, fit64 {st[3] / st[2]:.2%}")


if __name__ == "__main__":
    main()
