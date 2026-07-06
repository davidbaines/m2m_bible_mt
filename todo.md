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
- **Next action**: bring many-to-many forward, and/or scale to transformer-big
  on the A100 (see spec.md "Roadmap to stronger results"); wire up ClearML for
  the H100s.
- **ClearML**: connected. Authenticated to `api.sil.hosted.allegro.ai`; queue
  `jobs_backlog` exists with 8 GPU workers listening (`aqua-gpu-dallas:gpu0-7`).
- **Git remote**: `github.com/davidbaines/m2m_bible_mt`, `main` up to date, so
  the agents can clone the code. Ready to package a ClearML task.
- **Blocked on David**: choice of the first remote run (recommendation:
  transformer-big many-to-many); confirm the GitHub repo is cloneable by the
  agents (public, or a token/deploy key if private); whether to make the
  published HF repo public.
- Nothing is running in the background right now.

## Active - next up

- [ ] ClearML plumbing test: repeat the IE base run on `jobs_backlog`, then pull
      the checkpoint and artefacts back locally (so the local publish pipeline
      can use them). Validate the whole remote loop before the big run.
- [ ] Many-to-many pair sampler: K random sources per target verse per epoch
      (K default 4); source + target tags. Build and test on the 3090.
- [ ] Then the flagship run: transformer-big (~210M) many-to-many on
      `jobs_backlog`; large batch, longer schedule.
- [ ] Make `ebible_m2m-ie-base-shareable` public once reviewed.

## Blocked on David

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
- **Pretrained comparison**: fine-tune NLLB-200 on identical data and holdouts;
  report separately (spec.md "Roadmap", track 6).
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
