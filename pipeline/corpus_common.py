"""mmbert-tiny-inject corpus — shared schema writer / validator / dedup (stdlib-only).

Every ingest adapter imports this module and emits rows through `Row` +
`CorpusWriter` so the normalized schema in docs/PIPELINE.md stays the single source
of truth. No pyarrow/pandas/datasets dependency — mirrors the stdlib-only convention of
the SMS classifier project.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from typing import Iterable, Iterator

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus_work")
CORPUS_PATH = os.path.join(CORPUS_DIR, "inject.jsonl")

LABELS = ("benign", "injection")
CHANNELS = ("url_text", "pdf", "image", "email", "tool_result", "sms", "chat")

# Cheap language hint (best-effort; the mmBERT teacher is the real judge).
# Presence of a script/diacritic set nudges the guess; default "en".
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_CJK = re.compile(r"[　-鿿]")
_ARABIC = re.compile(r"[؀-ۿ]")
# Diacritics/script are the PRIMARY signal. Function-word hints are restricted
# to language-specific words that do NOT also occur in English — e.g. "instructions"
# is English-shared and must never be a French hint, and "hat"/"die"/"pour" are
# English words and must never be German/French hints. TR chars are the
# UNIQUELY Turkish ones (ğ ı ş İ) — ö/ü/ç are shared with German/French and
# claiming them for Turkish stole real German rows.
_DE_CHARS = re.compile(r"[äöüßÄÖÜ]")
_DE_WORDS = re.compile(
    r"\b(und|nicht|ignoriere|Anweisungen?|vorherigen|deine|bitte|ich|sind|"
    r"welche|wird|werden|gibt|gegen|dagegen|unsere|möchte|keine|schon|jetzt|"
    r"warum|wenn|haben|für|wie)\b", re.IGNORECASE)
_FR_CHARS = re.compile(r"[àâçéèêëîïôûù]")
_FR_WORDS = re.compile(
    r"\b(ignorez|précédentes|veuillez|vôtre|vous|avec|cette|sont|aussi|mais|"
    r"les|des|est|une|c'est|s'il\s+vous\s+plaît)\b", re.IGNORECASE)
_TR_CHARS = re.compile(r"[ğışĞİŞ]")
_TR_WORDS = re.compile(
    r"\b(talimat|unut|önceki|yoksay|için|değil|daha|nasıl|neden|lütfen|"
    r"hesap|mesaj|bir|ve)\b", re.IGNORECASE)


_LETTER = re.compile(r"[^\W\d_]")

# A language owns the row only when its signal DOMINATES — never on mere
# presence. Adaptive attacks (LLMail) smuggle short foreign-script/-language
# snippets inside English text; presence-based detection mislabeled 1,322
# English emails as "zh" (median CJK ratio 3%), and a single French diacritic
# inside an English email claimed "fr" — poisoning the per-language eval.
# Real Chinese rows sit at ~87% CJK ratio, so 0.30 separates cleanly; Latin
# langs must show repeated hint evidence, not one smuggled snippet.
_SCRIPT_DOMINANCE = 0.30


def guess_lang(text: str) -> str:
    letters = _LETTER.findall(text)
    n = len(letters) or 1
    if len(_CJK.findall(text)) / n >= _SCRIPT_DOMINANCE:
        return "zh"
    if len(_CYRILLIC.findall(text)) / n >= _SCRIPT_DOMINANCE:
        return "ru"
    if len(_ARABIC.findall(text)) / n >= _SCRIPT_DOMINANCE:
        return "ar"
    # Latin langs: score = unique chars (weighted) + function words, take the
    # argmax — ordered first-match let Turkish claim German rows via shared
    # ö/ü. The floor scales with text length: a whole smuggled French SENTENCE
    # inside a long English email yields a handful of hits, but a genuinely
    # French text of that length yields dozens — absolute thresholds can't
    # separate the two.
    scores = {
        "tr": 2 * len(_TR_CHARS.findall(text)) + len(_TR_WORDS.findall(text)),
        "de": len(_DE_CHARS.findall(text)) + len(_DE_WORDS.findall(text)),
        "fr": len(_FR_CHARS.findall(text)) + len(_FR_WORDS.findall(text)),
    }
    words = len(text.split()) or 1
    need = max(2 if len(text) < 80 else 3, int(0.04 * words))
    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= need:
        return best
    return "en"


# Injection-shaped / credential vocabulary that makes a BENIGN row a hard
# negative (FP-hard). SINGLE source of truth — both the sms-ham miner and
# eval_oracle's fp_hard tagging import this, so "what we mine" and "what we
# measure" can never drift apart again. Multilingual on purpose: the ham
# pool is fr/bs/tr-heavy and the old EN-only list found just 114 rows.
FP_HARD_WORDS = (
    # en — override / credential / AI-meta
    "ignore", "disregard", "forget", "override", "instruction", "instructions",
    "system prompt", "verify", "confirm", "password", "api key", "api-key",
    "token", "credential", "reveal", "print your", "repeat the", "as an ai",
    "you are now", "prompt", "jailbreak", "username", "verification code",
    "login",
    # de
    "ignoriere", "passwort", "kennwort", "bestätigen", "bestätige",
    "anweisung", "zugangsdaten", "überprüfen", "bestätigungscode",
    # fr
    "ignorez", "ignorer", "mot de passe", "confirmez", "vérifiez",
    "identifiant", "consigne", "code de vérification",
    # tr
    "şifre", "parola", "doğrula", "doğrulama", "onayla", "talimat",
    "yoksay", "kullanıcı adı",
    # bs/hr/sr-latin
    "lozinka", "potvrdi", "zanemari", "uputstvo", "prijava",
    # es / it / pt / nl / pl
    "contraseña", "credenciales", "istruzioni", "credenziali", "senha",
    "instruções", "wachtwoord", "negeer", "hasło", "zignoruj", "potwierdź",
)


def is_fp_hard(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in FP_HARD_WORDS)


def norm_text(text: str) -> str:
    """Light normalization for storage: NFC + collapse whitespace + trim.

    Deliberately does NOT strip zero-width / homoglyphs — those are signal
    for the concealment layers and may be useful adversarial training text.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedup_key(text: str) -> str:
    """Case/space-insensitive fingerprint for cross-source dedup."""
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(collapsed.encode("utf-8")).hexdigest()


@dataclass
class Row:
    text: str
    label: str
    source: str
    license: str
    lang: str = ""
    channel: str = "url_text"
    is_translated: bool = False
    is_synthetic: bool = False
    weak_seed: bool = False
    group: str = ""   # shared key for rows that must not straddle train/eval
                      # (e.g. an assembled source's poisoned + clean twins).
                      # Empty = the row is its own group.
    notes: str = ""

    def validate(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"bad label {self.label!r} (want {LABELS})")
        if self.channel not in CHANNELS:
            raise ValueError(f"bad channel {self.channel!r} (want {CHANNELS})")
        if not self.text or not self.text.strip():
            raise ValueError("empty text")
        if not self.source or not self.license:
            raise ValueError("source and license are required")

    def finalize(self) -> "Row":
        self.text = norm_text(self.text)
        if not self.lang:
            self.lang = guess_lang(self.text)
        self.validate()
        return self


class CorpusWriter:
    """Append-mode JSONL writer with in-run + cross-run dedup.

    Loads existing dedup keys from the target file so re-running an adapter
    is idempotent (won't duplicate rows already written by a prior run).
    """

    def __init__(self, path: str = CORPUS_PATH, min_len: int = 3, max_len: int = 4000):
        self.path = path
        self.min_len = min_len
        self.max_len = max_len
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._seen: set[str] = set()
        self.written = 0
        self.skipped_dup = 0
        self.skipped_len = 0
        self.skipped_bad = 0
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        self._seen.add(dedup_key(json.loads(line)["text"]))
                    except Exception:
                        continue

    def add(self, row: Row) -> bool:
        try:
            row = row.finalize()
        except ValueError:
            self.skipped_bad += 1
            return False
        n = len(row.text)
        if n < self.min_len or n > self.max_len:
            self.skipped_len += 1
            return False
        key = dedup_key(row.text)
        if key in self._seen:
            self.skipped_dup += 1
            return False
        self._seen.add(key)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        self.written += 1
        return True

    def report(self, name: str) -> None:
        print(
            f"[{name}] wrote {self.written} | dup {self.skipped_dup} | "
            f"len {self.skipped_len} | bad {self.skipped_bad} -> {self.path}"
        )


def iter_rows(path: str = CORPUS_PATH) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
