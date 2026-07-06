# Results: publishable single-family run `ie_base_shareable`

Licence-clean twin of `ie_base` (see `ie-base-results.md`), trained on the 3090,
2026-07-06, and published to the Hub. Same design; the selection is restricted
to translations whose licences permit sharing a derived model (spec.md,
"Publishing").

## Setup

- Selection: `experiments/selection-ie-shareable.csv`, 32 Indo-European
  languages, all with derivative-permitting licences. German uses the Public
  Domain `deutkw` edition in place of the by-nc-nd `deuelbbk`.
- Held out: the whole Old Testament of English (`engbsb`), German (`deutkw`)
  and Hindi (`hin2017`); each trained on its New Testament plus the full Bibles
  of the other languages.
- Model: transformer-base, 60.7M parameters, BPE 32k, bf16.
- Training: 60,000 steps in ~2.1 h, final eval loss 2.99, 684,605 training pairs.
- Generation: beam 5, hard length cap 192, no truncations.

## Scores

Verse-weighted over each language's generated Old Testament (36 books,
~20,830 scored verses each).

| Language | chrF3 | spBLEU | BLEU | best other-language | source-copy |
|---|---|---|---|---|---|
| English | 41.00 | 19.46 | 18.72 | 19.57 (French) | 0.34 |
| Hindi | 38.72 | 20.54 | 16.04 | 32.79 (Urdu) | 0.25 |
| German | 32.69 | 8.67 | 7.86 | 21.22 (Dutch) | 0.32 |

Every language beats both baselines, so the publish quality gate passes. German
scores lower here than in `ie_base` (32.7 vs 40.5), because the Public Domain
`deutkw` is a more archaic German text and a harder reference to match; the
Dutch relative-copy baseline is also lower for it. The Hindi and English scores
are essentially unchanged from `ie_base`.

## Publication

Published to `DavidCBaines/ebible_m2m-ie-base-shareable` (private for review),
licence `cc-by-sa-4.0` (a by-sa source is present). The repository holds the
model, a `from_pretrained`-loadable tokeniser, all 108 generated draft books,
sample sheets, per-book metrics and a model card listing every source and its
licence. Loading from the Hub and generating a verse was verified.

Per-book metrics and sample sheets: `checkpoints/ie_base_shareable/generated/`.
