#!/usr/bin/env python3
"""One-shot corpus repair: re-tag script-mislabeled lang fields in place.

Why: guess_lang() used to claim "zh"/"ru"/"ar" on the presence of a single
non-Latin char. LLMail adaptive attacks smuggle short CJK/Cyrillic snippets
inside English emails, so 1,322 English rows were tagged "zh" (median CJK
ratio 3%) — the per-language eval then "measured" Chinese recall on English
text. guess_lang is now dominance-based (corpus_common._SCRIPT_DOMINANCE).

This script re-runs the fixed guess_lang over rows whose lang was guessed at
ingest time (sources that don't carry an authoritative language field) and
currently claim a non-Latin script. Every change is recorded in the row's
`notes` (no silent row surgery). Idempotent — a second run changes nothing.

Deliberately not touched: polyguard / sms-ham / synthetic-ml (lang comes from
the dataset metadata or is authored), and Latin-lang guesses (the hint regexes
were already safe).
"""
from __future__ import annotations

import json
import os

from corpus_common import CORPUS_PATH, guess_lang

import re

GUESSED_SOURCES = {
    "llmail", "jackhhao", "deepset", "bipia", "bipia-ctx", "nemotron",
 "synthetic", "email-benign",
}
# All langs guess_lang can emit besides "en" — Latin hints were also
# presence-based (a smuggled French snippet inside an English email claimed
# "fr"), so fr/tr/de guesses are revisited too, not just the script langs.
SCRIPT_LANGS = {"zh", "ru", "ar", "tr", "de", "fr"}

_FIX_NOTE = re.compile(r"lang-fix:(\w+)->(\w+)")


def main() -> int:
    rows = []
    changed = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            notes = r.get("notes", "")
            prior_fix = _FIX_NOTE.search(notes)
            if r.get("source") in GUESSED_SOURCES and (
                r.get("lang") in SCRIPT_LANGS or prior_fix
            ):
                # Original ingest-time tag: from the first fix note if this row
                # was repaired before (keeps the ledger a single orig->final
                # entry instead of an append chain), else the current lang.
                orig = prior_fix.group(1) if prior_fix else r["lang"]
                base_notes = _FIX_NOTE.sub("", notes).strip(" |").strip()
                new = guess_lang(r["text"])
                if new != r["lang"]:
                    key = (r["lang"], new, r["source"])
                    changed[key] = changed.get(key, 0) + 1
                if new != orig:
                    note = f"lang-fix:{orig}->{new}"
                    r["notes"] = f"{base_notes} | {note}" if base_notes else note
                else:
                    r["notes"] = base_notes
                r["lang"] = new
            rows.append(r)

    if not changed:
        print("nothing to repair — corpus lang tags already consistent")
        return 0

    tmp = CORPUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, CORPUS_PATH)

    total = sum(changed.values())
    print(f"repaired {total} rows in {CORPUS_PATH}")
    for (old, new, src), n in sorted(changed.items(), key=lambda kv: -kv[1]):
        print(f"  {old} -> {new}  {src:14s} {n:5d}")
    print("\nnext: python3 eval_oracle.py build   (re-split with honest lang tags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
