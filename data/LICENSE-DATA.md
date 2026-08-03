# Licence of the demo dataset

The MIT licence in the repository root covers the **code and documentation**. It
does **not** cover `demo_inject_500.jsonl`. Those rows are not ours to
relicense — they are redistributed from third-party research datasets, and each
row carries its own `source` and `license` field.

| Rows | Licence | What you must do |
|---|---|---|
| `llmail`, `bipia`, `bipia-ctx` | **MIT** | Keep the copyright notice of the original project. |
| `deepset`, `jackhhao` | **Apache-2.0** | Keep the notice; state changes if you modify the rows. |
| `nemotron`, part of `sms-ham` | **CC-BY-4.0** | Keep attribution — the citation list is in [`DATA_LICENSES.md`](DATA_LICENSES.md). |
| `synthetic*`, `email-benign` (our own generated rows) | **CC0-1.0** | Nothing. Public domain dedication. |

## If you reuse this file

1. **Keep the `source` and `license` fields on every row.** They are what makes
   the mixed licensing tractable downstream.
2. **Keep [`DATA_LICENSES.md`](DATA_LICENSES.md) with the data.** It carries the
   attributions the licences require, the selection filters, and the note on why
   the e-mail addresses inside the injection samples are scenario placeholders
   rather than personal data.
3. **Do not assume the whole file is one licence.** It is a mixture by design.
