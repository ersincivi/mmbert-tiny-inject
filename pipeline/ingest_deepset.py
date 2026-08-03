#!/usr/bin/env python3
"""Stream B anchor — deepset/prompt-injections (HF datasets-server rows API).

Real labeled injection text (label 1) + benign (label 0), EN/DE/FR. This is
the mmbert-tiny-inject corpus anchor: the only source of genuine, human-labeled injection
payloads we currently have. 662 rows (546 train / 116 test).

License: see SOURCES.md (HF card is contradictory apache-2.0 / cc-by-4.0;
both training-safe with attribution). Rows tagged `license: apache-2.0`.

Stdlib-only: pulls JSON via the datasets-server `rows` endpoint (no parquet,
no `datasets` package). Network required.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

from corpus_common import CorpusWriter, Row

DATASET = "deepset/prompt-injections"
CONFIG = "default"
SPLITS = ("train", "test")
PAGE = 100
BASE = "https://datasets-server.huggingface.co/rows"
LICENSE = "apache-2.0"


def fetch_page(split: str, offset: int) -> dict:
    q = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": split,
         "offset": offset, "length": PAGE}
    )
    url = f"{BASE}?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-tools/1.0"})
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
    total = 0
    for split in SPLITS:
        offset = 0
        while True:
            data = fetch_page(split, offset)
            rows = data.get("rows", [])
            if not rows:
                break
            for item in rows:
                r = item["row"]
                text = r.get("text", "")
                label = "injection" if int(r.get("label", 0)) == 1 else "benign"
                writer.add(Row(
                    text=text,
                    label=label,
                    source="deepset",
                    license=LICENSE,
                    channel="chat",  # deepset is direct-chat, not indirect
                    notes=f"split={split}",
                ))
                total += 1
            offset += PAGE
            if offset >= data.get("num_rows_total", 0):
                break
    writer.report("deepset")
    print(f"[deepset] scanned {total} source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
