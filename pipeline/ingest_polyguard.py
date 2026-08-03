#!/usr/bin/env python3
"""Stream D — PolyGuardMix (ToxicityPrompts/PolyGuardMix, 17 langs, 1.9M rows).

Value here is NOT indirect-injection (PolyGuard is a safety/jailbreak+harm set) —
it is REAL multilingual coverage we otherwise only had synthetically:
  - `prompt_harm_label == "no"`  → REAL multilingual BENIGN negatives.
  - `prompt_harm_label == "yes"` → harmful/jailbreak-class positives, tagged
    channel=chat + notes "jailbreak-class" (jailbreak ≠ indirect; SAME convention
    as ingest_jackhhao so the trainer can down-weight — memory: jailbreak-class).

⚠ FLOOD-BOUND: 1.9M rows. Hard PER-LANG × PER-LABEL cap. Latin-script focus
(our student vocab is Latin; non-Latin positives UNK — §E/K5). A few non-Latin
BENIGN kept for negative diversity only.

License: card (skipped per operator — public safety-research data); provenance
in SOURCES.md. Stdlib-only, HF datasets-server rows API. Network required.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

from corpus_common import CorpusWriter, Row

DATASET = "ToxicityPrompts/PolyGuardMix"
CONFIG = "default"
SPLIT = "train"
PAGE = 100
BASE = "https://datasets-server.huggingface.co/rows"
LICENSE = "polyguard-card"
PER_LANG_LABEL_CAP = 200     # cap per (lang, label). Raised so the group-aware
                             # eval split lands >=30/lang for Latin langs → per-lang
                             # transfer becomes MEASURABLE for the first time
                             # (eval_oracle marks n<30 "unmeasurable").
STEPS = 200                  # evenly-spaced sampling windows (diverse langs)
FETCH_PAUSE = 0.4            # polite pacing between pages (HF 429 guard)

# PolyGuard `metadata.language` is a full name → our lang code.
LANG = {
    "English": "en", "German": "de", "French": "fr", "Spanish": "es",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Polish": "pl",
    "Swedish": "sv", "Czech": "cs",
    # non-Latin: benign-only (negative diversity), positives would UNK
    "Russian": "ru", "Chinese": "zh", "Arabic": "ar", "Hindi": "hi",
    "Japanese": "ja", "Korean": "ko", "Thai": "th",
}
LATIN = {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "cs"}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-tools/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                print(f"  skip page after 6 tries ({exc})", file=sys.stderr)
                return {}   # resilient: skip this page, DON'T crash the run
            print(f"  retry {attempt+1} ({exc})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))   # 3,6,9,12,15s — patient on 429
    return {}


def main() -> int:
    writer = CorpusWriter()
    first = _get(f"{BASE}?dataset={urllib.parse.quote(DATASET)}&config={CONFIG}"
                 f"&split={SPLIT}&offset=0&length={PAGE}")
    total = int(first.get("num_rows_total", 0))
    step = max(PAGE, total // STEPS) if total else PAGE
    counts: dict[tuple[str, str], int] = {}
    scanned = 0
    for off in range(0, total or PAGE, step):
        if off != 0:
            time.sleep(FETCH_PAUSE)   # polite pacing → avoid HF 429
        data = first if off == 0 else _get(
            f"{BASE}?dataset={urllib.parse.quote(DATASET)}&config={CONFIG}"
            f"&split={SPLIT}&offset={off}&length={PAGE}")
        for item in data.get("rows", []):
            r = item["row"]
            scanned += 1
            prompt = (r.get("prompt") or "").strip()
            harm = (r.get("prompt_harm_label") or "").strip().lower()
            if not prompt or harm not in ("yes", "no"):
                continue
            md = r.get("metadata") or {}
            if isinstance(md, str):
                try:
                    md = json.loads(md.replace("'", '"'))
                except Exception:  # noqa: BLE001
                    md = {}
            lang = LANG.get(str(md.get("language", "")).strip(), "")
            if not lang:
                continue
            label = "injection" if harm == "yes" else "benign"
            # non-Latin: keep benign only (positives would UNK in Latin vocab)
            if lang not in LATIN and label == "injection":
                continue
            key = (lang, label)
            if counts.get(key, 0) >= PER_LANG_LABEL_CAP:
                continue
            note = "jailbreak-class" if label == "injection" else "multilingual-benign"
            if writer.add(Row(text=prompt, label=label, source="polyguard",
                              license=LICENSE, lang=lang, channel="chat",
                              notes=note)):
                counts[key] = counts.get(key, 0) + 1
    writer.report("polyguard")
    top = sorted(counts.items(), key=lambda x: -x[1])[:12]
    print(f"[polyguard] scanned {scanned} | per(lang,label): {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
