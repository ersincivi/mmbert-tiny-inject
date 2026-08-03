# `demo_inject_500.jsonl` — provenance and licences

A **500-row demonstration sample**, not a training set. 250 injection + 250
benign; English base (300) plus German (80) and a 19-language tail (120), so the
cross-lingual behaviour is inspectable. All 500 texts unique, median length 190
characters. Every row carries its own `source` and `license` field, and uses the
schema the pipeline reads (`text`, `label`, `lang`, `source`, `license`,
`channel`, `is_synthetic`, `is_translated`, `weak_seed`, `group`).

Licence identifiers are normalised to SPDX form; rows generated for this project
carry `CC0-1.0`. Rows are redistributed **unmodified** from their upstream datasets. Where a row
is ours, it is marked as such.

| Rows | Source | Licence | Where it comes from |
|---:|---|---|---|
| 118 | `llmail` | MIT | [microsoft/llmail-inject-challenge](https://github.com/microsoft/llmail-inject-challenge) |
| 73 | `deepset` | Apache-2.0 | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) |
| 71 | `jackhhao` | Apache-2.0 | [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) |
| 57 | `synthetic-ml` | CC0-1.0 | Generated for this project (multilingual positives/negatives) |
| 44 | `email-benign` | CC0-1.0 | Generated for this project (benign business e-mail) |
| 23 | `bipia`, `bipia-ctx` | MIT | [microsoft/BIPIA](https://github.com/microsoft/BIPIA) |
| 21 | `synthetic` | CC0-1.0 | Generated for this project |
| 15 | `nemotron` | CC-BY-4.0 | NVIDIA Nemotron agentic indirect-prompt-injection set |
| 78 | `sms-ham` | CC-BY-4.0 / CC0-1.0 | Benign SMS text reused as negatives (see the companion SMS project for full attribution) |

## How it was selected

1. licence on the redistributable allow-list (MIT / Apache-2.0 / CC-BY-4.0 /
   CC0 variants);
2. **`polyguard` excluded** — PolyGuardMix is distributed under a gated dataset
   card rather than a redistributable SPDX licence, so none of its 5,400 rows
   are here;
3. **an in-house attack taxonomy excluded** — our own seed fragments stay
   unpublished, along with the 152 rows they produced;
4. length 20–4000 characters;
5. de-duplicated on case-folded, whitespace-collapsed text;
6. balanced round-robin across languages so no single language swamps the tail.

## The e-mail addresses inside these texts

Many injection samples are shaped like e-mails and contain addresses. These are
**scenario placeholders authored by the upstream research datasets**, not
personal data: across the whole clean pool there are 210 unique addresses over
112 domains, dominated by fictional ones (`contact.com`, `evil.example`,
`zenithcorp.com`, `you@gmail.com`). They are kept intact because altering the
texts would break the integrity of datasets redistributed under their original
licences.

## On publishing attack samples

Every injection row comes from an **already-public** research dataset —
Microsoft, NVIDIA, deepset, jackhhao. Republishing a 250-row sample adds no
capability that is not already downloadable from the originals. Our own attack
catalogue is not included.

## What is not here

The full training corpus (16,072 documents), the trained weights, the
vocabulary and the teacher logits.
