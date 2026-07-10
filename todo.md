# Todo

Working list for the Liedes reproduction. `spec.md` is the stable "why and
what"; this file is the living "where we are". Keep the Current status block
current and tick tasks `[x]` as they are completed. Maintenance routine:
`spec.md`, "Maintaining these documents".

## Current status (2026-07-10)

- **The NLLB many-to-one chapter is complete** — pipeline, lr root-cause,
  15-run matrix, code-reviewed publish path, and four private Hub repos.
  Nothing is running; the 3090 is free. Next experiment is David's choice
  (candidates under Roadmap: fairer m2m evaluation before the flagship
  transformer-big run; scaled one-to-many; another single family; ClearML
  smoke once SIL fixes the agent).
- **m2o headline findings** (`experiments/m2o-matrix-results.md`): (1) the
  token-init method doesn't matter (≤1 chrF3 spread; even the control's real
  pretrained Ilocano token buys nothing over scratch); (2) real source
  proximity is everything: Ilocano 61 / Tongan 59 / Ndebele 55 (close
  relatives, same script) > Romani 44 (relatives, cross-script) >> Male 10
  (no true relatives); (3) NLLB already knowing the target is worth only
  ~2 chrF3. **Critical caveat for all NLLB fine-tunes: lr 3e-5 is fatally
  cold for new-language targets — use 3e-4** (`experiments/m2o-bracketing.md`).
- **Live on the Hub, private, awaiting David's review to go public**:
  `DavidCBaines/ebible_m2m-ie-base-shareable` (cc-by-sa-4.0) and the four m2o
  repos — `ebible_m2o-nllb600m-ton` / `-nde` / `-control-ilo`
  (cc-by-nc-sa-4.0 models) + `ebible_m2o-nllb-results` (dataset). All
  verified loadable. David hand-edited the model-card acknowledgements on
  the Hub (2026-07-10); the wording is folded into `build_model_card`, so
  re-publishing preserves it.
- **Local model artefacts**: `runs/m2o_winners/` (gitignored) holds the
  published checkpoints plus Romani-scratch (unpublishable — `rmc` is
  **by-nd** — but the best cross-script demo). `runs/staging/` mirrors the
  Hub repos. Everything else in the session /tmp scratchpad is disposable.
- **Earlier from-scratch results** (all in `experiments/`): `ie_base` OT
  chrF3 40.7/40.5/38.1 (eng/deu/hin); `ie_base_shareable` 41.0/38.7/32.7;
  `ie_base_m2m` scored below one-to-many from the fixed Greek source (not a
  fair test of m2m; see roadmap); `nllb_ie` 52.8/51.1/42.7 from a Spanish
  pivot. NOTE: `nllb_ie` used lr 3e-5 — the m2o lr finding suggests it too
  was underpowered; a rerun at a higher lr may lift it substantially.
- **Alignment factor** done: eflomal vs in-repo IBM-1 disagree on the closest
  source for 3/5 targets (`experiments/m2o-alignment-comparison.md`); eflomal's
  Maori-for-Tongan pick got small downstream support in the bracketing.
- **ClearML**: connected (queue `jobs_backlog`, 8 workers), remote launch code
  ready, but the agent-side bootstrap crash (`pkg_resources` missing in the
  worker container) still needs an SIL admin fix. Re-run the smoke test when
  that lands.
- **Blocked on David**: SIL ClearML fix; review + make-public decision for
  the five Hub repos; choice of the next experiment.
- GPU healthy (driver 580.159.03 rebuilt via DKMS after the 2026-07-07
  kernel-upgrade incident; recovery recipe in the gpu-driver-mismatch memory
  and `experiments/m2o-bracketing.md`).

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
- [x] NLLB many-to-one series (15 runs): unknown target from same-family
      sources, 3 token-init methods + control; NT-only, generate withheld
      Ruth/Jonah/Genesis-1 → `experiments/m2o-matrix-results.md`
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
- [x] Full 15-run matrix at lr 3e-4 → `experiments/m2o-matrix-results.md`
      (init method irrelevant; source proximity is everything; Male is the
      no-relatives floor at ~10)
- [x] NLLB publish path (`samileides.publish_nllb`, +tests): source-copy
      quality gate, licence gate with the NLLB CC-BY-NC-4.0 base forcing NC,
      deterministic tokenizer reconstruction, model card. Four repos staged
      **private** on the Hub (2026-07-08), all verified loadable:
      `ebible_m2o-nllb600m-ton` / `-nde` / `-control-ilo` (cc-by-nc-sa-4.0)
      and the results dataset `ebible_m2o-nllb-results` (methodology,
      write-ups, CSVs, configs). Winning checkpoints secured from /tmp to
      `runs/m2o_winners/` (gitignored). Ilocano publishes the existing-init
      variant (60.63, standard `ilo_Latn` token, friendlier than scratch's
      `ilo_Latn_new` for a 0.2 noise-level difference).
- [x] Post-publish code review (10 findings) applied: schema-guarded results
      append; shared `target_token_for` in `nllb_m2o`; base-model licence
      propagation moved into `licensing.model_licence_for`; gate fails (not
      crashes/passes) on missing baselines; `matrix_rows` demands an
      unambiguous lr; `<range>` filtered from scoring; card reads real
      lr/steps/source-count; configs updated to lr 3e-4 / 8000 steps;
      publish helpers imported from `publish` instead of copied. Incidental
      finding: the Romani translation (`rmc`) is **by-nd** — an rmc model can
      never be published; research-only.
- [ ] Make `ebible_m2m-ie-base-shareable` public once reviewed.

## Blocked on David

- [ ] SIL to fix the ClearML agent bootstrap on the workers (pin `setuptools<81`,
      upgrade clearml-agent, or set a working default docker image).
- [ ] Approve making the published HF repo public.
- [ ] Review the four private m2o repos (3 models + results dataset) and
      approve making them public.

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
