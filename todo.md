# Todo

Working list for the Liedes reproduction. Consult `spec.md` before every
change; tick tasks `[x]` as they are completed.

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
- **Blocked on David**: ClearML queue name and a git remote the agents can
  clone; choice of the next run; whether to make the published repo public.
- Nothing is running in the background right now.

## Phase 0 - repo and data foundations

- [x] git init; uv project (`pyproject.toml`, Python ≥ 3.11); `src/samileides/`
      package skeleton; pytest wiring; `.gitignore` (data, checkpoints)
- [x] Data access module: download/cache `main.parquet` + `metadata.parquet`
      from `DavidCBaines/ebible_corpus`; load wide table keyed by `vref`
      (column-subset reads over `hf://`, cached under `data/cache/`)
- [x] Alignment unit tests (known verses, coverage counts vs metadata) —
      integration tests, run with `uv run pytest -m integration`
- [x] Selection script: criteria from spec (one translation per language by
      coverage, family/script diversity) + YAML include/exclude overrides;
      emits committed translation-ID lists (`samileides.selection`)
- [x] Composite Greek source builder (`grcbrent` OT + `grc-tisch` NT); report
      which vrefs lack Greek source → `experiments/greek-source-coverage.md`.
      Finding: NEH, EST, DAN wholly absent from the LXX column; PSA missing
      823 vrefs, JER 247 (LXX versification); 2,400/31,170 canon vrefs (7.7%)
      have no Greek source
- [x] Extended-holdout selector (~5 diverse full-Bible languages) — proposal
      written to `experiments/extended-holdouts-proposal.md` (lin, pon, hin,
      azb, vie). **Approval gate still open: David must confirm/edit the list
      into `configs/holdouts.yaml` before Phase 1 training**
- [x] Split builder: book-level holdouts by `vref` prefix (English OT, German
      Genesis, Finnish Matthew, extended Genesis); random ~5k-pair validation
      sample; split checksums (`samileides.splits`)
- [x] Leakage tests: zero train/test overlap per holdout (synthetic +
      real-data integration test); tokeniser-corpus leakage check lands with
      the tokeniser in Phase 1

## Phase 1 — pilot (one-to-many, ~50 translations)

- [x] Pilot selection list (~50, echoing Liedes' languages where possible) —
      `samileides.pilot` → `experiments/selection-pilot.csv` (50 languages,
      all 8 holdouts, diverse top-ups)
- [x] Preprocessing pipeline: NFC, drop empty/`<range>`, length filters,
      `<2xxx>` target tags (`samileides.preprocess`, +tests)
- [x] SentencePiece BPE 32k (byte fallback, tags as user-defined symbols) +
      atomic-tag and round-trip tests (`samileides.tokenizer`, +tests;
      `add_dummy_prefix=False` so the tag is token 0)
- [x] Evaluation module: chrF3/chrF3+/chrF3++, spBLEU, BLEU (silnlp/sacreBLEU
      conventions); identity=100 / noise≈0 checks; source-copy baseline
      (`samileides.evaluate`, +tests, spBLEU verified)
- [x] Sample-sheet generator with configurable passage list (defaults per spec,
      `configs/passages.yaml`) (`samileides.sheets`, +tests)
- [x] Transformer-big experiment config committed
      (`configs/experiments/pilot.yaml`)
- [x] Training script (HF Seq2SeqTrainer, bf16, reads pilot.yaml, ClearML
      logging) — GPU/Linux; develop on the 3090 (`samileides.train`,
      `config.py`/`dataset.py`/`model.py`/`data_pipeline.py`, +tests;
      `--clearml` opt-in). Verified on Ubuntu 24.04 + RTX 3090, torch cu130,
      bf16 (`experiments/dev-3090-setup.md`)
- [x] Overfit-100-pairs sanity check — GPU/Linux (`train --overfit 100`;
      label smoothing disabled in overfit mode so loss can reach ~0;
      eval_loss 0.006 → PASS)
- [x] Tiny end-to-end run on Linux 3090 (small model, 3 translations) —
      produced checkpoint, tokeniser, generated Jonah, metrics.csv/.md and a
      populated sample sheet (`configs/experiments/smoke.yaml`,
      `configs/holdouts-smoke.yaml`, `experiments/selection-smoke.csv`;
      held-out chrF3 17.82 vs source-copy 0.38)
- [x] Generation utility: held-out book generation + template mode (any book,
      any language), beam 5, hard length caps, truncation logging
      (`samileides.generate`) — GPU/Linux
- [ ] machine.py cross-check of the metrics on a shared fixture
- [ ] ClearML job packaging for the remote H100s; rclone artefact sync
      (training already emits ClearML scalars via `--clearml`; remote job
      packaging + rclone sync still to do)
- [ ] **Pilot H100 run** (< 12 h) → metrics tables, generated books, sheets
- [ ] Review pilot results with David; record conclusions in `experiments/`

## Local experiment — single-family (Indo-European), whole-OT holdouts

Substantial 3090 run while ClearML/H100 access is pending (spec.md,
"Single-family (Indo-European) run"). Tests whether one-family training helps
held-out languages, and whether the model can draft an entire withheld OT.

- [x] Curated Indo-European code→branch map
      (`configs/families/indo_european.csv`)
- [x] Family selection builder (`samileides.family`, all IE languages, best
      translation each, holdouts forced in) → `experiments/selection-ie.csv`
      (34 languages: Germanic 7, Slavic 8, Romance 6, Indo-Aryan 10, Iranian 2,
      Baltic 1)
- [x] Whole-OT holdout config for English/German/Hindi (`configs/holdouts-ie.yaml`)
- [x] transformer-base experiment config (`configs/experiments/ie_base.yaml`,
      ≈61M params, sized for a few hours on the 3090)
- [x] Throughput/memory probe on the 3090 (≈5 steps/s, 759k train pairs, fits
      in 24 GB)
- [x] Full training run on the 3090 (60k steps, ~2.1 h, eval_loss 2.92)
- [x] Generate the withheld OTs (English/German/Hindi), score, sample sheets —
      chrF3 40.7/40.5/38.1, every book beats the source-copy baseline; results
      in `experiments/ie-base-results.md`
- [ ] Compare vs diverse pilot on shared holdouts (English OT directly;
      German/Hindi on the Genesis subset) — pending a pilot run

## Phase 2 — ablations (pilot config)

- [ ] 3× holdout-language oversampling vs none
- [ ] Beam 120 vs beam 5 (same checkpoint)
- [ ] Write up: do his two suspicions hold?

## Phase 3 — scaled one-to-many

- [ ] Scaled selection list (all 179 full Bibles + diverse partials)
- [ ] BPE 64k tokeniser; re-run tag/leakage tests
- [ ] Scaled H100 run (< 12 h; adjust steps/data if needed) → full evaluation
- [ ] Compare pilot vs scaled: does more data help held-out books?

## Phase 4 — many-to-many

- [ ] Pair sampler: K random sources per target verse per epoch (K default 4);
      source + target tags
- [ ] Many-to-many H100 run → full evaluation vs Phase 3
- [ ] Pivot-source variant (small fixed source set) → compare with K-random

## Phase 5 — tokeniser variants (best config so far)

- [ ] SentencePiece unigram run → does Finnish/morphology improve?
- [ ] Byte-level run (expect longer sequences; may need step budget changes)

## Phase 6 — English-source variant

- [ ] One-to-many from `engbsb` (English leaves the holdout set); evaluate
      remaining holdouts vs Greek-source results

## Phase 7 — pretrained comparison (optional)

- [ ] Fine-tune a pretrained multilingual model (e.g. NLLB-200) on identical
      data/holdouts; report separately from the from-scratch results

## Hugging Face publishing pipeline

Gated sharing of successful models (spec.md, "Publishing"). Agreed with David
2026-07-06.

- [x] Licence policy module: classify each translation's licence, flag
      non-derivative-permitting/unknown sources, derive the model licence
      (`samileides.licensing`, +tests). Finding: `Redistributable` is `True`
      for all; the real signal is `licence_Licence_Type`
- [x] HF packaging: SentencePiece → loadable `MarianTokenizer`; model-card
      builder (`samileides.hf_export`, +tests)
- [x] Publish command with quality gate (chrF3 beats baseline) + licence gate,
      staging assembly, `--dry-run`, interactive confirm (`samileides.publish`,
      +tests). Verified end-to-end on the smoke run: dry-run staged a full repo
      and `from_pretrained` loaded and generated from it
- [x] Repository README (`README.md`)
- [x] Shareable-licence selection: `samileides.family --shareable` filters to
      derivative-permitting licences (auto-substitutes Public Domain editions,
      e.g. German `deutkw` for `deuelbbk`) → `experiments/selection-ie-shareable.csv`
      (32 languages, all shareable, model licence `cc-by-sa-4.0`), +test
- [x] Publishable experiment ready: `configs/experiments/ie_base_shareable.yaml`
      + `configs/holdouts-ie-shareable.yaml` (same design as ie_base, licence-clean)
- [x] Train `ie_base_shareable` (licence-clean, 32 languages), eval_loss 2.99
- [x] Generate + score `ie_base_shareable`: chrF3 41.0 (eng), 38.7 (hin),
      32.7 (deu, archaic Public Domain `deutkw`); all beat both baselines
- [x] HF authentication on this box (`hf auth login`, DavidCBaines, write)
- [x] First real publish through the gate → `DavidCBaines/ebible_m2m-ie-base-shareable`
      (private), verified on the Hub and loadable via `from_pretrained`
- [x] Best-other-language baseline (`evaluate.best_reference_baseline`,
      `samileides.rescore`); the publish gate and model card both use it
- [ ] Make the repo public when David is satisfied with the review

## Phase 8 — report and publish

- [ ] Push checkpoints, tokenisers, generated books, sample sheets to HF Hub
      (via the publishing pipeline above)
- [ ] `report/` write-up: setup, deviations table, metric tables per phase,
      qualitative samples, answers to Liedes' open questions
- [ ] Final review pass with David
