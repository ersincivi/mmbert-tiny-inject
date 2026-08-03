#!/usr/bin/env python3
"""Stream B — nvidia/Nemotron Agentic Indirect Prompt Injection (CC-BY-4.0).

High-value indirect + agentic injection: realistic malicious instructions
embedded in tool results / documents (healthcare/finance domains, exfiltration
& unauthorized-action categories). Per-row `license: CC BY 4.0` — training-safe
with attribution. 1272 rows (train split only).

Matches our threat model closely (an AI acting on the user's behalf is hijacked
by content it processes). English-only, so it boosts the indirect/agentic
category, not multilingual coverage — the multilingual gap is addressed by the
mmBERT transfer strategy (see SOURCES.md Stream-D decision), not by this source.

Extracts `injection.injection_text` (the realistic embedded payload) with a
fallback to `injection.goal`. channel = tool_result.

Stdlib-only: HF datasets-server rows API. Network required.
"""
from __future__ import annotations

import ast
import json
import sys
import time
import urllib.parse
import urllib.request

from corpus_common import CorpusWriter, Row

DATASET = "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"
PAGE = 100
BASE = "https://datasets-server.huggingface.co/rows"
LICENSE = "cc-by-4.0"


def fetch_page(offset: int) -> dict:
    q = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train",
         "offset": offset, "length": PAGE}
    )
    req = urllib.request.Request(f"{BASE}?{q}",
                                 headers={"User-Agent": "corpus-tools/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {attempt+1} ({exc})", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    return {}


def parse_injection(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(raw)
                if isinstance(v, dict):
                    return v
            except Exception:  # noqa: BLE001
                continue
    return {}


def main() -> int:
    writer = CorpusWriter()
    scanned = 0
    offset = 0
    while True:
        data = fetch_page(offset)
        rows = data.get("rows", [])
        if not rows:
            break
        for item in rows:
            r = item["row"]
            inj = parse_injection(r.get("injection"))
            text = (inj.get("injection_text") or inj.get("goal") or "").strip()
            if not text:
                continue
            domain = r.get("domain", "?")
            category = r.get("attack_category", "?")
            writer.add(Row(text=text, label="injection", source="nemotron",
                           license=LICENSE, channel="tool_result",
                           notes=f"agentic/{domain}/{category}"))
            scanned += 1
        offset += PAGE
        if offset >= data.get("num_rows_total", 0):
            break
    writer.report("nemotron")
    print(f"[nemotron] scanned {scanned} source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
