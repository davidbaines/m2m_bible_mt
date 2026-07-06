# Results: single-family (Indo-European) run `ie_base`

First substantial local run on the RTX 3090, 2026-07-06. See spec.md,
"Single-family (Indo-European) run", for the design.

## Setup

- Source: composite Koine Greek (Brenton LXX + Tischendorf), one-to-many into
  34 Indo-European languages, best-covered translation per language.
- Held out: the whole Old Testament of English (`engbsb`), German (`deuelbbk`)
  and Hindi (`hin2017`); each trained on its New Testament plus the full Bibles
  of the other languages.
- Model: transformer-base, 60.7M parameters, SentencePiece BPE 32k, bf16.
- Training: 60,000 steps (7.6 epochs) in about 2.1 hours, final label-smoothed
  eval loss 2.92.
- Generation: beam 5, hard length cap 192, no truncations.

## Scores

Verse-weighted over each language's generated Old Testament (39 books, 20,833
scored verses each). The source-copy baseline returns the Greek source verse
unchanged.

| Language | chrF3 | spBLEU | BLEU | source-copy chrF3 |
|---|---|---|---|---|
| English | 40.73 | 19.22 | 18.68 | 0.34 |
| German | 40.51 | 16.20 | 15.40 | 0.33 |
| Hindi | 38.08 | 19.81 | 15.47 | 0.25 |

Every one of the 108 held-out books beats the source-copy baseline on chrF3.
The smallest margin is German Psalms at chrF3 28.64 against 0.33, which is
expected because the Psalms use poetic language and a different versification.

## Qualitative sample (English Genesis, held out entirely)

- GEN 1:1 generated: "In the beginning, God created heavens and earth."
  Reference: "In the beginning God created the heavens and the earth."
- GEN 1:3 generated: "Then God said, 'Be a light.' And it was a light."
  Reference: "And God said, 'Let there be light,' and there was light."

Full generated books, per-book metrics and side-by-side sample sheets are under
`checkpoints/ie_base/generated/`.

## Status

This run reproduces the Sami Liedes closed-text effect with a transformer: a
model trained on the New Testament of a language, plus full Bibles of its
relatives, drafts a recognisable Old Testament for that language.

The run is research and comparison only. Its selection includes translations
whose licences forbid derivative works (`deuelbbk` by-nc-nd, `polubg` by-nd,
`swef` Unknown), so it cannot be published. The publishable twin is
`ie_base_shareable`, trained on a licence-clean 32-language selection with the
same design.
