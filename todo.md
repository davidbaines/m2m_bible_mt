# Todo

Working list for the Liedes reproduction. `spec.md` is the stable "why and
what"; this file is the living "where we are". Keep the Current status block
current and tick tasks `[x]` as they are completed. Maintenance routine:
`spec.md`, "Maintaining these documents".

## Current status (2026-07-06)

- **Done**: repo and data foundations; the full pipeline end to end (selection,
  splits, composite Greek source, preprocessing, tokeniser, training,
  generation, evaluation); the single-family Indo-European research run
  `ie_base` (held-out OT chrF3 40.7 English / 40.5 German / 38.1 Hindi); its
  licence-clean twin `ie_base_shareable`; and the Hugging Face publishing
  pipeline with quality and licence gates.
- **Live on the Hub**: `DavidCBaines/ebible_m2m-ie-base-shareable` (private,
  `cc-by-sa-4.0`), verified loadable with `from_pretrained`.
- **Next action**: the flagship transformer-big many-to-many run on
  `jobs_backlog`, once the ClearML agent env is fixed (below).
- **Many-to-many sampler**: built and verified on the 3090
  (`samileides.manytomany`, smoke_m2m).
- **NLLB fine-tune**: done. `nllb_ie` (600M, IE many-to-many, from a Spanish
  pivot) drafted held-out OTs at chrF3 52.8 German / 51.1 English / 42.7 Hindi,
  above from-scratch `ie_base` (40.5/40.7/38.1) but on an easier task; see
  `experiments/nllb-ie-results.md`.
- **ie_base_m2m** done: many-to-many scored *below* one-to-many by ~8-10 chrF3
  (31.9/32.6/28.3 vs 40.7/40.5/38.1), because both were evaluated from the fixed
  Greek source, which many-to-many trains on far less. Not a fair test of m2m's
  real strength (generating from a relative). See
  `experiments/ie-base-m2m-results.md`.
- **NLLB many-to-one series**: now has generation-based early stopping (chrF3 on
  a fixed 250-verse NT validation set every 1000 steps, best model kept). Two
  bracketing runs are training now (Tongan-relative = easy, Romani-scratch =
  hard, max 20k steps, patience 5) to measure convergence and per-run time
  before committing the full matrix.
- **Alignment factor** done: eflomal (uv-managed-Python build) vs in-repo IBM-1;
  they disagree on the closest source for 3/5 targets
  (`experiments/m2o-alignment-comparison.md`).
- **ClearML**: connected (queue `jobs_backlog`, 8 workers). Remote launch code
  is in place (`train --remote-queue`, artifact upload, `samileides.fetch`) and
  the git remote (`github.com/davidbaines/m2m_bible_mt`, public) works, but the
  plumbing test **failed on the agent side**: workers run each task in a fresh
  `python:3.12-bullseye` container where `clearml-agent 2.0.4`'s own bootstrap
  crashes with `ModuleNotFoundError: No module named 'pkg_resources'` (it
  installs setuptools 83, which no longer ships pkg_resources). Needs an SIL
  admin fix; nothing to change on our side.
- **Blocked on David**: SIL to fix the ClearML agent env (pin `setuptools<81`
  in the worker/container, upgrade clearml-agent, or set a working default
  docker image); choice of the next run; whether to make the published HF repo
  public.
- **GPU restored (2026-07-07)**: the reboot cleared the driver/library
  mismatch but booted a newer kernel (6.17.0-35) with no headers installed, so
  DKMS had no nvidia module for it. Fixed by installing
  `linux-headers-6.17.0-35-generic` (DKMS rebuilt 580.159.03) + `modprobe
  nvidia`. torch 2.12.1+cu130 verified on the 3090 (bf16 matmul ok).
- Bracketing: Tongan-relative converged by ~step 1000 (val chrF3 ~21, flat),
  61 min for a wasteful 15k steps -> tightened early stopping (patience 3,
  min-delta 0.2). Romani-scratch never ran (GPU died first);
  `experiments/m2o-bracketing.md`.
- **m2o root cause found and fixed**: the Romani brackets failed flat because
  **lr 3e-5 was too cold** for a cross-script target with a fresh token. The
  lr re-bracket settled it: at lr 3e-4, rmc-scratch went 5.7 → 44.6 val chrF3
  and ton-relative ~21 → 57.8 (the old Tongan number was underpowered too);
  1e-4 is still too cold (13.2). `train_nllb_m2o` gained `--lr` and records
  lr in results. Full story: `experiments/m2o-bracketing.md`.
- **15-run matrix launched** (2026-07-07): ton/nde/mdy/rmc ×
  relative/scratch/same_script + control_ilo × existing/relative/scratch, all
  at lr 3e-4, max 8000 steps, patience 3/min-delta 0.2. ~4-6 h on the 3090.
  Results → `experiments/m2o-matrix-results.csv` (old `m2o-results.csv` is
  void, pre-lr-fix).

## Active - next up

- [x] Many-to-many pair sampler (`samileides.manytomany`, +tests); wired through
      train/generate; verified end to end on the 3090 (smoke_m2m)
- [~] ClearML plumbing: remote launch + artifact upload + `samileides.fetch`
      built; enqueue works; blocked by the agent-side `pkg_resources` crash
- [ ] Re-run the plumbing test (smoke on `jobs_backlog`) once SIL fixes the agent
- [ ] Flagship run: transformer-big (~210M) many-to-many on `jobs_backlog`
      (needs a committed transformer-big m2m config sized for the remote GPU)
- [x] Local 3090, NLLB-600M IE many-to-many fine-tune (`nllb_ie`) → strong
      draft quality; `experiments/nllb-ie-results.md`
- [x] Local 3090, `ie_base_m2m` (base-scale many-to-many vs one-to-many): m2m
      scored below o2m at equal compute when evaluated from Greek; the fixed
      source dilution explains it. `experiments/ie-base-m2m-results.md`
- [ ] Fairer m2m test before any big run: evaluate from relatives and/or
      oversample the pivot source and/or give m2m more steps
- [~] NLLB many-to-one series (15 runs): unknown target from same-family
      sources, 3 token-init methods + control; NT-only, generate withheld
      Ruth/Jonah/Genesis-1 (`samileides.train_nllb_m2o`, running)
- [x] Alignment scores as a source factor: eflomal (built via a uv-managed
      Python) and an in-repo IBM-1, compared. The two disagree on the closest
      source for 3/5 targets, agreeing on same-script Latin cases and inverting
      on cross-script ones (`experiments/m2o-alignment-comparison.md`). Notable:
      for Male, eflomal ranks Oromo above same-script Amharic.
- [x] Generation-based early stopping for the m2o runs (fixed 250-verse NT
      validation set, chrF3 every 1000 steps, best model kept) — default method
- [x] Bracketing (easy): Tongan-relative converges by ~step 1000; tightened
      early stopping (patience 3, min-delta 0.2). `experiments/m2o-bracketing.md`
- [x] Reboot the box; fix the missing DKMS build for the new kernel; GPU
      verified working again (torch cu130, bf16)
- [x] Re-bracket the hard case (Romani-scratch): failed flat under both inits;
      diagnosed to a too-cold lr (3e-5 inverse-sqrt), not the init or language
- [x] lr re-bracket (rmc scratch 3e-4/1e-4, ton relative 3e-4): lr 3e-4 wins
      decisively; matrix budget max 8000 steps
- [~] Full 15-run matrix at lr 3e-4 (running, ~4-6 h) →
      `experiments/m2o-matrix-results.csv`
- [ ] Make `ebible_m2m-ie-base-shareable` public once reviewed.

## Blocked on David

- [ ] SIL to fix the ClearML agent bootstrap on the workers (pin `setuptools<81`,
      upgrade clearml-agent, or set a working default docker image).
- [ ] Approve making the published HF repo public.

## Roadmap (strategic direction, not a strict order)

Liedes' original phase plan, kept as direction. We went straight to the
single-family experiment and publishing rather than a numbered pilot, so treat
these as a backlog to pull from.

- **Ablations** (his open questions): 3x holdout-language oversampling on/off;
  beam 120 vs beam 5 on one checkpoint; write up whether his two suspicions
  hold.
- **Scaled one-to-many**: all ~179 full Bibles plus diverse partials; BPE 64k;
  compare with the family runs.
- **Many-to-many** (see Active): plus a pivot-source variant (small fixed source
  set) versus K-random.
- **Tokeniser variants**: unigram, then byte-level, on the best config so far.
- **English-source variant**: one-to-many from `engbsb` (English leaves the
  holdout set); evaluate remaining holdouts.
- **Pretrained comparison**: NLLB-200-600M fine-tune done for a modest IE
  many-to-many run (`experiments/nllb-ie-results.md`); strong draft quality.
  Follow-ups: fair-source comparison and a longer fine-tune (spec.md track 6).
- **Other single families**: an all-Bantu or all-Austronesian run, and
  family-versus-diverse at H100 scale.
- **Pilot (superseded)**: `configs/experiments/pilot.yaml` (transformer-big, 50
  languages) was never trained; the IE runs served the same "prove it" purpose.
  Run later only if a direct pilot-versus-family comparison is wanted.
- **Report and publish**: `report/` write-up (setup, deviations table, metric
  tables per phase, qualitative samples, answers to his open questions); push
  the best models to the Hub; final review with David.

## Done

### Foundations
- [x] git + uv project, `src/samileides/` package, pytest, `.gitignore`
- [x] data access (`ebible_corpus` main/metadata, cached column-subset reads)
- [x] alignment integration tests (known verses, coverage vs metadata)
- [x] selection script (criteria + YAML overrides) emitting committed lists
- [x] composite Greek source + coverage report (2,400 of 31,170 canon vrefs,
      7.7%, lack a Greek source; NEH/EST/DAN absent from the LXX column)
- [x] split builder (book-level holdouts by `vref`, validation sample,
      checksums) + leakage tests
- [x] extended-holdout proposal (superseded by the IE whole-OT holdout choice)

### Pipeline (built and verified on the 3090)
- [x] preprocessing (NFC, empty/`<range>` drop, length filters, `<2xxx>` tags),
      +tests
- [x] SentencePiece BPE 32k (atomic tags, byte fallback), +tests
- [x] evaluation: chrF3/chrF3+/chrF3++, spBLEU, BLEU; source-copy and
      best-other-language baselines, +tests
- [x] sample-sheet generator (configurable passages), +tests
- [x] training (`samileides.train`, Seq2SeqTrainer, bf16, ClearML opt-in) with
      overfit sanity (eval_loss 0.006) and a tiny end-to-end smoke run;
      config/dataset/model/data_pipeline modules, +tests
- [x] generation (`samileides.generate`, held-out + template mode, beam, hard
      length caps, truncation logging) and `samileides.rescore`
- [x] dev environment verified (Ubuntu 24.04 + RTX 3090, torch cu130, bf16;
      `experiments/dev-3090-setup.md`)

### Single-family Indo-European experiment
- [x] curated IE code->branch map; family selection builder (34 languages) plus
      shareable variant (32 languages, automatic Public Domain substitutions)
- [x] whole-OT holdout configs for English/German/Hindi
- [x] `ie_base` trained, generated, scored -> `experiments/ie-base-results.md`
      (chrF3 40.7/40.5/38.1; every book beats both baselines)
- [x] `ie_base_shareable` trained, generated, scored ->
      `experiments/ie-base-shareable-results.md` (41.0/38.7/32.7)

### Hugging Face publishing
- [x] licence policy (`samileides.licensing`), HF packaging + model card
      (`samileides.hf_export`), publish command with quality + licence gates
      (`samileides.publish`), repository README, all +tests
- [x] first publish -> `DavidCBaines/ebible_m2m-ie-base-shareable` (private),
      verified loadable from the Hub

## Deferred / open

- [ ] machine.py / silnlp metric cross-check on a shared fixture (needs a
      silnlp install)
- [ ] rclone artefact sync for remote runs
- [ ] family-versus-diverse comparison on shared holdouts (needs a diverse run
      at matching scale)
