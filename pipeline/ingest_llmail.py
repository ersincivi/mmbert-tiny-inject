#!/usr/bin/env python3
"""Stream B — Microsoft LLMail-Inject (SaTML 2025 challenge submissions).

The single highest-value addition: real, adaptive, indirect prompt-injection.
Participants attacked a simulated M365-Copilot email assistant knowing the
defenses, so `subject`+`body` are genuine attacker-crafted email payloads —
exactly the indirect-email channel BIPIA-context could only synthesize.

Every row is an attack submission → label=injection, channel=email. (There is
no benign class here; benign email prose comes from other streams.)

Flood-bound (remembering the Mistral blanket-flood regression). The challenge log is
~461K rows (Phase1 370,724 + Phase2 90,916) of near-duplicate retry variants.
Ingesting all would make "email = injection" and drown every other channel.
So: spread-sample across the full range (diverse submission-times/teams),
CorpusWriter dedups, hard cap `MAX_WRITE`.

License: MIT (repo). Provenance kept per SOURCES.md; not a gate.
Stdlib-only, HF datasets-server rows API. Network required.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

from corpus_common import CorpusWriter, Row

DATASET = "microsoft/llmail-inject-challenge"
CONFIG = "default"
SPLITS = ("Phase1", "Phase2")
PAGE = 100
BASE = "https://datasets-server.huggingface.co/rows"
LICENSE = "mit"
MAX_WRITE = 4000          # hard cap on rows written (post-dedup) — flood guard.
                          # 3500-4000 is the quality sweet-spot; above it
                          # near-dup retry variants dilute without adding signal.
STEPS_PER_SPLIT = 130     # evenly-spaced sampling windows across each phase
                          # (deeper sampling → ~4000 distinct after ~22% dedup).


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-tools/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                print(f"  skip page after 6 tries ({exc})", file=sys.stderr)
                return {}   # resilient: skip this page, don't crash the run
            print(f"  retry {attempt+1} ({exc})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))   # 3,6,9,12,15s — patient on 429
    return {}


def fetch_page(split: str, offset: int) -> dict:
    q = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": split,
         "offset": offset, "length": PAGE})
    return _get(f"{BASE}?{q}")


def main() -> int:
    writer = CorpusWriter()
    scanned = 0
    for split in SPLITS:
        first = fetch_page(split, 0)
        total = int(first.get("num_rows_total", 0))
        if total <= 0:
            continue
        # Evenly-spaced offsets → diverse teams/time-windows, not one cluster.
        step = max(PAGE, total // STEPS_PER_SPLIT)
        offsets = list(range(0, total, step))
        for off in offsets:
            if writer.written >= MAX_WRITE:
                break
            data = first if off == 0 else fetch_page(split, off)
            for item in data.get("rows", []):
                r = item["row"]
                subject = (r.get("subject") or "").strip()
                body = (r.get("body") or "").strip()
                text = (subject + "\n" + body).strip() if subject else body
                if not text:
                    continue
                writer.add(Row(
                    text=text,
                    label="injection",
                    source="llmail",
                    license=LICENSE,
                    channel="email",
                    notes=f"satml2025 {split} {r.get('scenario', '')}".strip(),
                ))
                scanned += 1
        if writer.written >= MAX_WRITE:
            break
    writer.report("llmail")
    print(f"[llmail] scanned {scanned} source rows | cap {MAX_WRITE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
