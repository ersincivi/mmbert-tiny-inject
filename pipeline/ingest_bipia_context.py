#!/usr/bin/env python3
"""Stream B — BIPIA CONTEXT-ASSEMBLY (true indirect injection signal). MIT.

`ingest_bipia.py` ingests the isolated injected instruction. Its full value,
though, is the instruction *embedded in otherwise-benign content* — that is the
real indirect-injection signal our threat model cares about (a normal-looking
email that carries a hidden malicious instruction the AI obeys).

This adapter mirrors BIPIA's own assembly (bipia/data/utils.py insert_start /
insert_end / insert_middle): it takes each real email context and splices a
malicious instruction into it, producing:
  - INJECTION: benign email body + spliced malicious instruction (channel email)
  - BENIGN   : the same email body, unpoisoned

Paired same-context positive/negative teaches the classifier to spot the
injected line *within* legitimate content — not just recognise a bare attack
string. Position + attack are chosen deterministically (text hash, no RNG) so
runs are reproducible.

Stdlib-only: raw.githubusercontent.com. Network required.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request

from corpus_common import CorpusWriter, Row
from ingest_bipia import TEXT_MALICIOUS

RAW = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark"
LICENSE = "mit"
ATTACKS_PER_CONTEXT = 4  # keep the positive:context ratio bounded, avoid near-dupes


def fetch_json(name: str):
    req = urllib.request.Request(f"{RAW}/{name}",
                                 headers={"User-Agent": "corpus-tools/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_jsonl(name: str):
    req = urllib.request.Request(f"{RAW}/{name}",
                                 headers={"User-Agent": "corpus-tools/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp.read().decode("utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def load_malicious_attacks() -> list[str]:
    """Union of malicious text-attack strings across train + test files."""
    out: list[str] = []
    seen: set[str] = set()
    for fname in ("text_attack_train.json", "text_attack_test.json"):
        for category, items in fetch_json(fname).items():
            if category not in TEXT_MALICIOUS:
                continue
            for a in items:
                if a not in seen:
                    seen.add(a)
                    out.append(a)
    return out


def splice(context: str, attack: str, pos: str) -> str:
    if pos == "start":
        return f"{attack}\n{context}"
    if pos == "end":
        return f"{context}\n{attack}"
    # middle: insert at a sentence-ish boundary (deterministic pick).
    bounds = [m.end() for m in re.finditer(r"[.!?]\s+", context)]
    if not bounds:
        return f"{context}\n{attack}"
    idx = int(hashlib.sha1((context + attack).encode()).hexdigest()[:8], 16) % len(bounds)
    cut = bounds[idx]
    return f"{context[:cut]}{attack}\n{context[cut:]}"


POSITIONS = ("start", "end", "middle")


def main() -> int:
    attacks = load_malicious_attacks()
    if not attacks:
        print("[bipia-ctx] no malicious attacks loaded")
        return 1

    writer = CorpusWriter()
    contexts: list[str] = []
    for fname in ("email/train.jsonl", "email/test.jsonl"):
        for row in fetch_jsonl(fname):
            ctx = (row.get("context") or "").strip()
            if ctx:
                contexts.append(ctx)

    for ci, ctx in enumerate(contexts):
        # Shared group key so poisoned + clean twins never straddle train/eval.
        group = "ctx-" + hashlib.sha1(ctx.encode()).hexdigest()[:12]
        # benign: the clean context
        writer.add(Row(text=ctx, label="benign", source="bipia-ctx",
                       license=LICENSE, channel="email", group=group,
                       notes="clean-context"))
        # injection: splice ATTACKS_PER_CONTEXT distinct attacks (rotated by ci)
        for j in range(ATTACKS_PER_CONTEXT):
            attack = attacks[(ci * ATTACKS_PER_CONTEXT + j) % len(attacks)]
            pos = POSITIONS[int(hashlib.sha1(f"{ci}:{j}".encode()).hexdigest()[:8], 16) % len(POSITIONS)]
            poisoned = splice(ctx, attack, pos)
            writer.add(Row(text=poisoned, label="injection", source="bipia-ctx",
                           license=LICENSE, channel="email", group=group,
                           notes=f"indirect/{pos}"))

    writer.report("bipia-ctx")
    print(f"[bipia-ctx] {len(contexts)} contexts × {ATTACKS_PER_CONTEXT} attacks "
          f"(pool {len(attacks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
