# Todo

Working list for the Liedes reproduction. Consult `spec.md` before every
change; tick tasks `[x]` as they are completed.

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

## Phase 8 — report and publish

- [ ] Push checkpoints, tokenisers, generated books, sample sheets to HF Hub
- [ ] `report/` write-up: setup, deviations table, metric tables per phase,
      qualitative samples, answers to Liedes' open questions
- [ ] Final review pass with David
