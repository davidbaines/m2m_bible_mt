# NLLB many-to-one matrix: 15 runs, token-init × target

Teaches NLLB-600M an unseen target language from four same-family sources,
NT-only (whole OT withheld), then generates and scores Ruth, Jonah and
Genesis 1. 3090, 2026-07-07, all runs at **lr 3e-4** (see
`m2o-bracketing.md` for why 3e-5 was void), max 8000 steps, patience 3,
min-delta 0.2, eval every 1000. Raw rows: `m2o-matrix-results.csv`.

## Best validation chrF3 (250-verse NT set) per run

| target | same-script sources | relative | scratch | same_script | existing |
|---|---|---|---|---|---|
| Ilocano (control, in NLLB) | 4/4 | 60.53 | 60.83 | — | 60.63 |
| Tongan | 4/4 | 58.03 | 59.00 | 58.29 | — |
| N. Ndebele | 4/4 | 54.62 | 55.12 | 54.80 | — |
| Romani | 0/4 | 44.42 | 44.64 | 44.09 | — |
| Male | 1/4 | 10.17 | 10.16 | 10.17 | — |

Test books (best source per book) follow the same ordering: Ilocano 61-62,
Tongan 46-49, Ndebele 53-56, Romani 30-37, Male 6-7 chrF3.

## Finding 1: the token-init method does not matter

Within every target the three inits sit inside a ≤1.0 chrF3 band, with no
consistent winner (scratch is nominally first in 3 of 5). The control makes
the point sharpest: using Ilocano's *real, pretrained* NLLB token (60.63)
buys nothing over adding a brand-new scratch token (60.83). At a healthy lr
the embedding trains to wherever it needs to be almost immediately, erasing
its initialisation. (At the original too-cold lr the init *did* matter — the
Hindi-init Romani run generated the wrong script — but that whole regime is
degenerate; see `m2o-bracketing.md`.)

Caveat: for Male, `relative` and `same_script` are both `amh_Ethi`, so those
two runs are bit-identical duplicates, not independent evidence.

## Finding 2: what matters is the target's real distance to its sources

- **Close relatives, same script** (Ilocano, Tongan among Austronesian;
  Ndebele among Bantu): 55-61 val chrF3 — strong drafts.
- **Real relatives, different script** (Romani among Indic): 44 — the model
  crosses the script barrier fine once the lr allows it.
- **No close relatives** (Male, Omotic, from Semitic/Cushitic
  Afro-Asiatic cousins): ~10. The curve rises (7.9 → 10.2) then plateaus —
  genuine learning to a low ceiling, not an optimisation failure. Its best
  test source is Arabic at only ~6.8 chrF3. Data verified fine (full
  31k-verse Ethiopic-script Bible). The "family" here is much looser than
  the others, and transfer is correspondingly thin.

## Finding 3: knowing the target beforehand is worth surprisingly little

The control (Ilocano, which NLLB-600M already knows) beats never-seen Tongan
by only ~2 val chrF3 at matched sources and budget. In-domain fine-tuning on
7.7k NT verses dominates whatever the pretrained model already knew about the
target language.

## Timing

Runtimes 16-38 min per run with early stopping; whole matrix ≈ 7 h wall on
the 3090.

## Follow-ups suggested by the table

- Male: try genuinely closer sources if any exist in the corpus (other
  Omotic/Gonga languages), or accept it as the "no relatives" floor.
- The per-book best sources for Romani flip between Hindi/Gujarati/Marathi —
  consistent with the alignment-factor work
  (`m2o-alignment-comparison.md`); worth a joint read once scores are in one
  table.
