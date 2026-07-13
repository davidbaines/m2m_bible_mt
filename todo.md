# Todo

Working list for the vref-source series. `spec-vref.md` is the stable "why
and what" for this series (`spec.md` for shared policy); this file is the
living "where we are". Keep the Current status block current and tick tasks
`[x]` as they are completed.

## Current status

- **Done**: planning interview + `spec-vref.md` (2026-07-10). **Phase 0
  plumbing complete and verified** (2026-07-10): all three encodings build,
  pair set is byte-identical to ie_base, probe/early-stop/best-checkpoint
  work, overfit passes per encoding, tiny end-to-end produces every artefact.
- **Done**: all three Phase-1 encoding runs (2026-07-10). **Negative result
  at base scale** — see `experiments/vref-encoding-results.md`. All early-
  stopped at 18–28k steps, peak macro chrF3 13–16 (vtok best) vs ie_base 40;
  the model produces fluent wrong-content scripture and fails even on *seen*
  training verses (eval_loss 4.36 vs ie_base 2.92). Pipeline verified sound.
- **Done (2026-07-11)**: diagnostic `vref_ie_vtok_long` (180k, no early stop,
  seen probe) complete. **H1 REJECTED**: peak at step 7,000 (chrF3 13.8), then
  decline + collapses to ~0 by 180k. The seen curve tracked held-out
  throughout (peak 14.7) — the model never memorises even trained verses at
  82M. See `experiments/vref-encoding-results.md`. Base scale is a dead end;
  the only remaining lever in the current design is capacity (transformer-big,
  H2), with real risk it too is insufficient for whole-Bible recall.
- **Report written (2026-07-11)**: `experiments/vref-encoding-results.md`,
  carefully scoped to what the runs prove and prose-checked. David chose to
  write up and pause rather than launch the next run.
- **Published (2026-07-13)**: negative-result **HF dataset** (score artifacts
  only — no scripture text/model output, per `deuelbbk` by-nc-nd):
  `DavidCBaines/ebible_m2m-vref-negative-result` — uploaded PRIVATE for
  review; **David to check and flip to public**. Card = the report; contains
  probe curves/CSVs, generation metrics, configs, selection list.
  - [ ] Commit + push to GitHub the report, `LICENSE` (Apache-2.0, added) and
        the vref code/configs, so the dataset card's GitHub link resolves.
  - [ ] Flip the HF dataset to public after review.
- **Open direction (not yet chosen)**: transformer-big capacity test on the
  A100; a reduced-scope diagnostic; a design change (source signal / retrieval
  / pretrained start); or pause. Listed in the report's "Possible next steps".
- **Blocked**: nothing. DNS restored (WireGuard split-DNS fix: scope tunnel
  resolver to `psonet.languagetechnology.org` only).

### Code changes for the diagnostic (2026-07-11)
- `ProbeConfig.early_stop` (bool) and `.seen_verses_per_language`.
- `ProbeStopper` takes multiple named probe sets (`""` held-out drives
  stop+best-checkpoint; `"seen_"` logged only); honours `early_stop`.
- `build_probe_set` gains a `translations=` filter for the seen probe.
- Plot shows held-out (solid) vs seen (dotted) chrF3.

## Phase 0 — plumbing (COMPLETE)

- [x] `data.source: vref` mode with `vref_encoding: struct | vtok | text`;
      source built from the vref column only, masked to Greek coverage so the
      pair set is unchanged from ie_base. (`src/samileides/vref.py`,
      `data_pipeline.py`.)
  - [x] Unit tests: exact source token sequences for known vrefs, all three
        encodings; atomicity of every special symbol. (`tests/test_vref.py`.)
  - [x] Pair-set identity vs ie_base: checksum test + pre-training assertion
        in `train.py`. Verified identical — 809,477 pairs, checksum committed
        to `experiments/vref-train-manifest.txt`.
- [x] Shared tokeniser: one BPE base on training-split targets + all vref
      strings; per-encoding symbols appended atomically via
      `vref.extend_tokenizer`; base cached under `checkpoints/vref_shared/`.
      Round-trip + base-segmentation-preserved tests pass.
- [x] Probe evaluator (`src/samileides/probe.py`): seeded per-language
      sample, greedy chrF3/BLEU every N steps to `probe.csv` + `probe.png`.
  - [x] Determinism test (order- and shuffle-independent).
  - [x] Early-stopping callback + best-checkpoint save; stop rule unit-tested
        on synthetic curves (uses best-so-far, waits a full window).
  - [x] Overhead measured at smoke scale: ~0.1–0.4 s/probe, ~6% of wall time
        (40 verses, tiny model). Real-scale (750 verses, 65M) overhead still
        to be confirmed on the first Phase-1 probes — carried forward below.
- [x] Three configs `configs/experiments/vref/vref_ie_{struct,vtok,text}.yaml`
      (transformer-base, max_ratio 0, probe block, 180k ceiling) + a
      `vref_smoke.yaml`.
- [x] 100-pair overfit per encoding: struct 0.37, text 0.36, vtok 0.0096
      (vtok needs more steps for its 42k-row embedding table — expected).
- [x] Tiny end-to-end (`vref_smoke`): probe curve, best checkpoint, generated
      Jonah, metrics, sample sheet all produced; output is recognisable
      English scripture vocabulary (degenerate at toy scale, as expected).

### Carried into Phase 1

- [x] Confirmed probe overhead on `vref_ie_struct`: ~9% (11 s/probe vs
      ~125 s/1000 steps). Cadence of 1000 steps kept.
- [x] Online runs populated the IE verse caches; DNS restored via the
      WireGuard split-DNS fix. (`data/cache/columns.txt` not needed while the
      per-selection caches are present; offline fallback covers it.)

## Phase 1 — the encoding comparison (3090, sequential)

Probe overhead confirmed on struct: ~11 s/probe vs ~125 s/1000 steps ≈ **9%**
(well under the 25% threshold), so the 1000-step cadence stands. Training runs
at ~8 steps/s (source is ~3 tokens), so the 180k ceiling is ~6 h/run at most;
early stopping usually sooner. struct chrF3_macro rising (5.5 @1k → 8.0 @2k).

- [x] Train `vref_ie_struct` (early-stopped 26k) — peak held-out macro 13.3.
- [x] Train `vref_ie_vtok` (early-stopped 28k) — peak 15.8 (highest of three).
- [x] Train `vref_ie_text` (early-stopped 26k) — peak 13.6.
- [x] Generate + score all three held-out OTs (best checkpoint); sample sheets.
- [x] Diagnostic `vref_ie_vtok_long` (180k, no early stop, +seen probe):
      peak @ step 7k, then decline/collapse; seen tracked held-out. H1
      rejected.
- [x] Results doc `experiments/vref-encoding-results.md` written and
      prose-checked, scoped to what the runs prove.
- [ ] David decides direction (see "Open direction" above / report's
      "Possible next steps"). Series paused here.

## Phase 2 — scale-up (only if the direction decision calls for it)

- [ ] `vref_ie_big`: transformer-big (~210M) on the A100 as a capacity test,
      with a decaying LR schedule for late-training stability. NOTE: base
      scale never fit the vref→content mapping even on seen verses, so this is
      a genuine test, not an expected win.
- [ ] Generate + score; extend the results doc; compare against base scale.

## Recorded follow-ups (unscheduled)

- Full-coverage variant (verses the LXX lacks).
- Multi-translation-per-language.
- Decoder-only architecture.
- Diverse ~50 selection; all-179 scale-up.
