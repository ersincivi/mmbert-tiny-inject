#!/usr/bin/env python3
"""Stream B — Microsoft BIPIA indirect-injection benchmark (MIT).

BIPIA is the one source here that matches our threat model exactly: indirect
prompt injection — malicious instructions embedded in external content (email,
web, table, code) that the AI processes on the user's behalf. arXiv 2312.14197.

Structure: `benchmark/{text,code}_attack_{train,test}.json`, each a dict keyed
by attack category → list of injected-instruction strings.

Label discipline (the same class of bug as the SMS-phishing leak): the text file
mixes truly-manipulative categories with benign-task categories. In isolation,
"What is the capital of Brazil?" is not injection — labeling it positive would
poison the classifier. So we split explicitly:
  - Manipulative text categories  -> injection  (channel url_text)
  - benign-task text categories    -> benign     (channel url_text)
  - all code categories            -> injection  (channel tool_result; every
    code_attack category is malicious: Cookie Theft, Memory Scanning, …)
Any category not in the known maps is skipped + warned (fail loud, not silent).

Honest limit: we ingest the isolated injected instruction. BIPIA's full value
is the instruction embedded in context (indirect signal). Context-assembly
(email/qa body + injected line) is a documented follow-up (SOURCES.md).

Stdlib-only: raw.githubusercontent.com JSON. Network required.
"""
from __future__ import annotations

import json
import sys
import urllib.request

from corpus_common import CorpusWriter, Row

RAW = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark"
LICENSE = "mit"

# Text-attack categories that carry manipulative/malicious intent in the
# instruction string itself. Union of the train and test files (their category
# sets differ; each was classified by reading its content, not guessed).
TEXT_MALICIOUS = {
    # output obfuscation (make the response hard to read / smuggle data)
    "Alphanumeric Substitution", "Homophonic Substitution",
    "Misspelling Intentionally", "Anagramming", "Space Removal & Grouping",
    "Substitution Ciphers", "Base Encoding", "Reverse Text",
    "Emoji Substitution",
    # inject unsolicited / harmful content into the response
    "Instruction", "Social Interaction", "Persuasion", "Clickbait",
    "Malware Distribution", "Information Dissemination",
    "Marketing & Advertising", "Entertainment", "Scams & Fraud",
    "Misinformation & Propaganda",
}
# Text-attack categories that are benign tasks (legit queries) — negatives.
TEXT_BENIGN = {
    "Information Retrieval", "Content Creation", "Learning and Tutoring",
    "Language Translation", "Programming Help",
    "Task Automation", "Business Intelligence", "Conversational Agent",
    "Research Assistance", "Sentiment Analysis",
}


def fetch_json(name: str):
    req = urllib.request.Request(f"{RAW}/{name}",
                                 headers={"User-Agent": "corpus-tools/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ingest_text(writer: CorpusWriter, fname: str, split: str) -> None:
    data = fetch_json(fname)
    for category, items in data.items():
        if category in TEXT_MALICIOUS:
            label = "injection"
        elif category in TEXT_BENIGN:
            label = "benign"
        else:
            print(f"  ⚠ unknown text category {category!r} in {fname} — SKIPPED",
                  file=sys.stderr)
            continue
        for text in items:
            writer.add(Row(text=text, label=label, source="bipia",
                           license=LICENSE, channel="url_text",
                           notes=f"text/{category}/{split}"))


def ingest_code(writer: CorpusWriter, fname: str, split: str) -> None:
    data = fetch_json(fname)
    # Every code_attack category is a malicious code-injection technique.
    for category, items in data.items():
        for text in items:
            writer.add(Row(text=text, label="injection", source="bipia",
                           license=LICENSE, channel="tool_result",
                           notes=f"code/{category}/{split}"))


def main() -> int:
    writer = CorpusWriter()
    for split, fname in (("train", "text_attack_train.json"),
                         ("test", "text_attack_test.json")):
        ingest_text(writer, fname, split)
    for split, fname in (("train", "code_attack_train.json"),
                         ("test", "code_attack_test.json")):
        ingest_code(writer, fname, split)
    writer.report("bipia")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
