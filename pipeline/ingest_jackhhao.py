#!/usr/bin/env python3
"""Stream B — jackhhao/jailbreak-classification (Apache-2.0).

Named in the ProtectAI-22 blueprint (Apache-2.0, verified on the dataset's own
card). ~1306 rows (1044 train / 262 test), roughly 50/50 benign / jailbreak.

⚠ HONEST NOTE: jailbreak != indirect prompt injection. Jailbreak = coercing a
model to violate its own safety policy (direct chat). It shares the injection
*surface* (role override, instruction override, "you are now …") so it is
useful positive signal, but it is direct-channel — tagged `channel: chat` and
`notes: jailbreak-class` so downstream training can weight it distinctly from
indirect (BIPIA) positives.

Stdlib-only: HF datasets-server rows API. Network required.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

from corpus_common import CorpusWriter, Row

DATASET = "jackhhao/jailbreak-classification"
SPLITS = ("train", "test")
PAGE = 100
BASE = "https://datasets-server.huggingface.co/rows"
LICENSE = "apache-2.0"


def fetch_page(split: str, offset: int) -> dict:
    q = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": split,
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


def main() -> int:
    writer = CorpusWriter()
    scanned = 0
    for split in SPLITS:
        offset = 0
        while True:
            data = fetch_page(split, offset)
            rows = data.get("rows", [])
            if not rows:
                break
            for item in rows:
                r = item["row"]
                text = r.get("prompt", "")
                typ = (r.get("type") or "").strip().lower()
                if typ == "jailbreak":
                    label, note = "injection", "jailbreak-class"
                elif typ == "benign":
                    label, note = "benign", "jailbreak-ds-benign"
                else:
                    continue  # unknown type — skip rather than mislabel
                writer.add(Row(text=text, label=label, source="jackhhao",
                               license=LICENSE, channel="chat", notes=note))
                scanned += 1
            offset += PAGE
            if offset >= data.get("num_rows_total", 0):
                break
    writer.report("jackhhao")
    print(f"[jackhhao] scanned {scanned} source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
