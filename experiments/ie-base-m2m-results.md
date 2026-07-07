# Results: many-to-many vs one-to-many at base scale (`ie_base_m2m`)

Controlled comparison on the 3090, 2026-07-07. Identical to `ie_base` in every
respect (same 34-language IE selection, same English/German/Hindi whole-OT
holdouts, same transformer-base model, same 60,000-step budget, same seed). The
only change: many-to-many pairing (K=4 sampled sources per target verse, plus
the Greek composite) instead of one-to-many from Greek.

## Result

Held-out OT, verse-weighted chrF3 (and spBLEU):

| Language | one-to-many `ie_base` | many-to-many `ie_base_m2m` | Δ chrF3 |
|---|---|---|---|
| English | 40.73 (spBLEU 19.22) | 31.86 (11.21) | −8.87 |
| German | 40.51 (16.20) | 32.61 (9.36) | −7.90 |
| Hindi | 38.08 (19.81) | 28.30 (11.20) | −9.78 |

Many-to-many is clearly **worse** here, by about 8 to 10 chrF3, across all three
holdouts. Training used 3,553,023 pairs (K=4 expansion of 809k target verses);
final eval loss 3.62.

## Why (the important part)

This is not evidence that many-to-many is a bad idea; it is evidence that this
particular comparison is unfavourable to it, for two compounding reasons:

1. **The test uses a single fixed source (Greek), and many-to-many dilutes it.**
   Both models are evaluated by generating the held-out book from the Greek
   source. One-to-many trains on Greek->target for *every* verse; many-to-many
   picks Greek as the source only about 1 in 9 times (4 samples from ~35
   candidates), so it sees far fewer Greek->target examples at the same step
   budget. The metric measures exactly the skill many-to-many practised least.
2. **Equal steps under-trains the harder task.** Many-to-many has to learn
   34-plus source directions from the same 60k updates, so each direction gets a
   fraction of the capacity and exposure that one-to-many devotes to the single
   Greek direction.

In other words, we measured many-to-many on one-to-many's home turf. Its actual
strength, translating a held-out book from a *related* source, was never tested
here.

## Implications for the roadmap

- Do **not** treat many-to-many as a free win at fixed compute when evaluation
  is from one fixed source. Before spending H100 time on a large many-to-many
  run, change the comparison so it is fair to the method:
  - evaluate many-to-many by generating from the held-out language's best
    available relatives, not only from Greek; and/or
  - oversample the pivot/Greek source during many-to-many training; and/or
  - give many-to-many a larger step budget.
- This also reframes the pretrained many-to-one experiments (which generate from
  a relative, not a fixed foreign source): those test the direction where
  multi-source training should actually help, so they are a better-matched use
  of the idea.

Artefacts: `checkpoints/ie_base_m2m/generated/`.
