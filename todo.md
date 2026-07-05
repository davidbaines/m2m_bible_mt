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

- [ ] Pilot selection list (~50, echoing Liedes' languages where possible)
- [ ] Preprocessing pipeline: NFC, drop empty/`<range>`, length filters,
      `<2xxx>` target tags
- [ ] SentencePiece BPE 32k (byte fallback, tags as user-defined symbols) +
      atomic-tag and round-trip tests
- [ ] Transformer-big config in HF Transformers (from scratch) + training
      script (Seq2SeqTrainer, bf16, YAML config, ClearML logging)
- [ ] Overfit-100-pairs sanity check
- [ ] Tiny end-to-end run on Linux 3090 (small model, 3 translations) —
      must produce checkpoint, generated book, metrics, sample sheet
- [ ] Generation utility: held-out book generation + template mode (any book,
      any language), beam 5, hard length caps, truncation logging
- [ ] Evaluation module: chrF3/chrF3+/chrF3++, spBLEU, BLEU (silnlp
      conventions); fixture cross-check against machine.py; trivial baselines
- [ ] Sample-sheet generator with configurable passage list (defaults per spec)
- [ ] ClearML job packaging for the remote H100s; rclone artefact sync
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
