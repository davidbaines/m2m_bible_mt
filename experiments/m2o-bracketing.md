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

## Hard case: Romani, scratch init (did not run)

It failed before training: partway through the session an NVIDIA driver package
was upgraded (userspace NVML 580.159 vs the still-loaded kernel module
580.126.09), giving CUDA error 804 / "Driver/library version mismatch". The GPU
is unusable until the box is rebooted or the nvidia kernel module reloaded. So
**scratch-init convergence is still unmeasured** and must be bracketed after the
GPU is back; scratch is expected to need materially more steps than relative.

## Preliminary budget recommendation (pending the scratch measurement)

- relative / same-script / existing: max ~4000 steps, patience 3, min-delta 0.2.
- scratch: re-bracket after reboot; likely a larger max (e.g. 10000) is needed.

## Aside: does the eflomal "closest source" predict transfer?

For Tongan, eflomal ranked Maori closest (IBM-1 ranked Tagalog). Maori was the
best test source for Ruth and Genesis 1; Tagalog won Jonah. A single noisy data
point, but eflomal's structural pick has some downstream support here.
