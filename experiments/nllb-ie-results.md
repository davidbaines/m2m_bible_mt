# Results: NLLB-200-600M fine-tune (pretrained comparison track)

Modest local run on the 3090, 2026-07-07 (spec.md phase 7). This is the
*pretrained* track: NLLB already knows these languages, so it is reported
separately from the from-scratch results and is not a like-for-like comparison
(see caveats).

## Setup

- Base model: `facebook/nllb-200-distilled-600M`, fine-tuned.
- Data: Indo-European selection (`selection-ie.csv`), many-to-many pairs among
  the languages NLLB covers (Koine Greek excluded). 3,330,550 training pairs.
- Held out: whole OT of English (`engbsb`), German (`deuelbbk`), Hindi
  (`hin2017`), the same as `ie_base`.
- Fine-tune: 6,000 steps, lr 3e-5, bf16, gradient checkpointing, batch 16 x 2.
  This is only ~0.06 of an epoch over 3.3M pairs, so the model is barely
  adapted; the scores are close to NLLB's out-of-the-box ability on Bible text.
- Held-out generation: translate each held-out OT verse from the **Spanish
  pivot** (`spablm`), since NLLB has no Koine Greek. Beam 5.

## Scores (verse-weighted per language)

| Language | chrF3 | spBLEU | BLEU | best other-language | from-scratch `ie_base` chrF3 |
|---|---|---|---|---|---|
| German | 52.82 | 30.16 | 27.32 | 20.62 (Dutch) | 40.51 |
| English | 51.14 | 27.46 | 24.11 | 19.49 (French) | 40.73 |
| Hindi | 42.73 | 21.63 | 17.62 | 32.79 (Urdu) | 38.08 |

## Reading the numbers

NLLB scores well above the from-scratch model (about +12 chrF3 for German, +10
for English, +5 for Hindi), and far above the relative-copy baseline. This is
the expected outcome for the practical drafting track, but two things must be
kept in mind before concluding "NLLB wins":

1. **Easier task.** NLLB generates from Spanish, a high-resource pair it already
   translates well. The from-scratch model generates from Koine Greek, a
   language it only ever saw inside the Bible. Much of NLLB's advantage is the
   easier source, not just a better model.
2. **Outside knowledge.** NLLB brings fluency in all these languages from its
   pretraining. The from-scratch experiment deliberately starts from nothing;
   its point is what can be learned from the closed text alone.

Hindi gains least, consistent with spa->hin being a weaker NLLB direction and
Devanagari being harder; its relative-copy baseline (Urdu, 32.8) is also the
highest, so there is less headroom.

## Takeaways

- A pretrained model is by far the fastest route to high absolute draft quality,
  even with almost no fine-tuning. For real drafting work this is the strong
  option.
- For a fair pretrained-vs-from-scratch comparison, a later run should give NLLB
  the same source the from-scratch model uses (or give both a common pivot), and
  fine-tune NLLB for longer. Both are follow-ups, not part of this quick test.

Artefacts: `checkpoints/nllb_ie/generated/` (books, metrics, per-book table).
