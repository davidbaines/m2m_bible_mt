# Many-to-one bracketing runs: convergence and timing

Two runs to size the step budget for the full matrix, with generation-based
early stopping (chrF3 on the fixed 250-verse NT validation set, beam 5, every
1000 steps). 3090, 2026-07-07.

## Easy case: Tongan, relative init (completed)

Validation chrF3 by step:

| step | 1000 | 2000 | 3000 | 5000 | 10000 | 15000 |
|---|---|---|---|---|---|---|
| val chrF3 | 20.95 | 20.90 | 21.01 | 21.21 | **21.31** | 21.08 |

It is **converged by step 1000** (roughly half an epoch) and then flat: the
whole curve sits within a ~0.4 chrF3 band. The best (21.31 at step 10000) is
noise above the step-1000 value. Test scores (best source per book): Ruth 21.92
(Maori), Jonah 20.11 (Tagalog), Genesis 1 19.79 (Maori).

Wall-clock: 3687 s (~61 min) for 15000 steps, i.e. ~0.2 s/step training plus
about 45 s per beam-5 validation probe. **Most of that hour was wasted**: the
useful budget for a relative-init run is ~1000-2000 steps.

Two consequences, both now applied to the driver:
- Early stopping with patience 5 and a raw chrF3 metric is too lenient here:
  ~0.3 chrF3 noise kept resetting patience, so the run went to 15000 steps for
  nothing. Added `--min-delta` (default 0.2 chrF3 required to count as
  improvement) and lowered default patience to 3.
- Relative / same-script / existing inits need only a small budget (~2000-3000
  steps). A max of ~4000 with the tighter stopping should suffice.

## Hard case: Romani, scratch init (run 2026-07-07, after the GPU fix)

(First attempt died to the NVIDIA driver mismatch; rerun after the reboot +
DKMS rebuild.) Max 20000 steps, patience 3, min-delta 0.2.

| step | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|
| val chrF3 | 5.71 | 5.68 | 5.64 | **5.76** |

Early-stopped at step 4000 in 875 s (~15 min). The curve is **flat at ~5.7 and
never takes off**; eval_loss is equally flat (7.522 -> 7.520). Test scores are
near the noise floor: Ruth 7.41 (hin), Jonah 6.43 (hin), Gen 1 7.46 (asm) —
versus ~20-22 for Tongan-relative.

This is not "scratch needs more steps than relative"; at 4k steps it is not
learning at all. Two follow-ups separated the confounds:

1. **Romani-relative** (init from `hin_Deva`, max 6000): *worse* — flat at
   ~1.0 val chrF3 (test 0.3-1.4). The Hindi-initialised target token likely
   steers generation into Devanagari, which scores ~0 against Latin references.
2. **Romani-scratch, early stopping disabled, full 20000 steps**: flat at
   ~5.55-5.78 for all 20 evals (73 min). No slow ramp; it is stuck.

So *both* inits fail on Romani while Tongan-relative reached ~21 — the failure
is systemic, not the init.

## Root cause: the learning rate, not the language or the init

Diagnosis (2026-07-07): training loss in the failed runs dropped 8.5 -> 7.6 in
the first epoch, then stayed flat for 9 more epochs with grad norms ~2 — the
gradients exist, the updates are just too small. The inverse-sqrt schedule
(warmup 100) decays lr = 3e-5 to ~9e-6 by step 1000 and ~2e-6 by 20000.

A manual 500-step run at **lr 3e-4** (10x) took the loss from 8.2 to 2.1 and
already generates near-perfect Romani (JHN 3:16 sample differed from the
reference only in minor word order/inflection). Data and pairs were verified
correct along the way (`rmc` is a full 31k-verse Latin-script Bible; pairs and
tokenisation sane; base-model loss 5.6 on rmc vs 3.5 on ton).

Interpretation: a cross-script target with a fresh language token needs
sustained lr to pull the decoder into the new orthography; Tongan (Latin
script, same-script sources, close to NLLB's prior) converged inside the
warmup honeymoon, which masked the problem. **Tongan's ~21 chrF3 is itself
probably underpowered.**

Consequence for the whole m2o series: lr had to be re-bracketed before the
matrix (results in `experiments/m2o-bracketing-lr.csv`; the run script now
takes `--lr` and records it).

## The lr re-bracket (2026-07-07)

| run | lr | val chrF3 curve | best | test (RUT / JON / GEN 1) |
|---|---|---|---|---|
| rmc scratch | 3e-4 | 41.6 → 44.6, converged ~4-5k | **44.64** @ 5000 | 30.4 / 33.6 / 37.4 |
| rmc scratch | 1e-4 | flat ~13.2, stopped @ 4000 | 13.27 @ 2000 | 12.9 / 12.9 / 12.2 |
| ton relative | 3e-4 | 55.7 → 57.8, converged ~3k | **57.77** @ 4000 | 46.3 / 46.7 / 48.4 |

Conclusions:

- **lr 3e-4 is the matrix lr.** 1e-4 is still too cold for the cross-script
  scratch case (13 vs 45); 3e-5 was hopeless (5.7).
- **The old Tongan result was indeed underpowered**: ~21 chrF3 at lr 3e-5
  became ~58 val / 46-48 test at 3e-4 — more than doubled, same data and
  budget class.
- Convergence: scratch ~4-5k steps, relative ~3k. **Matrix budget: max 8000
  steps for every run**, patience 3, min-delta 0.2, eval every 1000.
- Timing: 8000 steps ≈ 29 min, early-stopped runs ~15-20 min → the 15-run
  matrix ≈ 4-6 h on the 3090.

## Matrix launched (2026-07-07)

All 15 runs (ton/nde/mdy/rmc × relative/scratch/same_script + control_ilo ×
existing/relative/scratch) queued sequentially at lr 3e-4, max 8000, results
appending to `experiments/m2o-matrix-results.csv` (the old `m2o-results.csv`
contains one pre-lr-fix run and is void).

## Aside: does the eflomal "closest source" predict transfer?

For Tongan, eflomal ranked Maori closest (IBM-1 ranked Tagalog). Maori was the
best test source for Ruth and Genesis 1; Tagalog won Jonah. A single noisy data
point, but eflomal's structural pick has some downstream support here.
