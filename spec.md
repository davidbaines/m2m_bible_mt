# Spec: reproducing the Sami Liedes closed-text Bible translation experiment

## Goal

Reproduce and extend Sami Liedes' 2018 experiment ("Machine translating the
Bible into new languages", samiliedes.wordpress.com, 7 March 2018): train a
multilingual model on Bible translations in many languages, withholding certain
books from certain languages, then generate those held-out books - testing
whether a model that "knows" the Bible from many languages can translate it
into a language it has only partially seen. The reproduction replaces his
convolutional seq2seq with a modern transformer, replaces the 53 Sword-project
translations with the eBible corpus (1,253 translations, 997 languages), and
adds quantitative evaluation, which he never had.

## The original experiment (reference points)

- One-to-many: fixed source = Koine Greek (`2TGreek`: LXX + Tischendorf NT,
  romanised), target language indicated by a `TGT_xxx` tag prepended to the
  source line (Johnson et al. 2016).
- 53 translations, verse-aligned via Sword OSIS IDs. Everything lowercased;
  non-Latin scripts flattened with unidecode.
- Moses tokenisation, joint subword-nmt BPE, 30k merges, tags kept atomic via
  glossaries; length filters: ratio 1.5, max 175 tokens.
- Architecture `fconv_wmt_en_ro` (20+20 conv layers, 512 dim), NAG lr 0.25,
  clip-norm 0.1, dropout 0.2, max-tokens 3000, label smoothing 0.1. ~7 days on
  a GTX 1080 Ti.
- Holdouts: English `ESV2011` trained on NT only (whole OT generated); Finnish
  `FinPR` and German `GerNeUe` trained without Genesis. Test languages
  oversampled 3× (he suspected this was unnecessary). Decoding: beam 120.
- Evaluation: qualitative samples only.
- Code reference: https://github.com/sliedes/fairseq-py (branch `master`,
  scripts `data/prepare_bible.py`, `train.sh`, `batch_translate.py`).

## Data

**Source**: Hugging Face dataset `DavidCBaines/ebible_corpus`.
`main.parquet` is a wide table: one row per verse of the standard 41,899-line
`vref.txt` list (`vref` column, e.g. `GEN 1:1`), one column per translation.
Empty string = verse absent; `<range>` = verse merged into the preceding
range. `metadata.parquet` gives per-translation language code, coverage counts
(OT/NT/DC books, chapters, verses), script, licence.

**Corpus facts that shape the design** (verified 2026-07-05):

- No single Greek translation covers the whole Bible. Composite source =
  `grcbrent` (Brenton LXX, 36 OT books, 22,561 verses) for the OT +
  `grc-tisch` (Tischendorf NT) for the NT - the same recipe as his `2TGreek`,
  but kept in Greek script. OT books/verses absent from the LXX column have no
  Greek source and are excluded from one-to-many training; the data-prep step
  must report exactly which vrefs these are.
- The only Finnish translation (`fin`) is NT-only: **his Finnish-Genesis
  holdout cannot be replicated**. Decision: keep Finnish but hold out one
  Gospel (default Matthew, configurable) - an NT-holdout variant of the same
  test. Estonian and Hungarian are also NT-only, so there is no full-Bible
  Uralic substitute.
- German: `GerNeUe` is not in the corpus. Use the highest-OT-coverage German
  full Bible (`deuelbbk`, Elberfelder) with Genesis held out.
- English: use `engbsb` (Berean Standard Bible), trained on NT only, whole OT
  generated - matching his English protocol.
- 179 translations across 107 languages have (near-)full Bibles. Partial
  (mostly NT-only) translations **are** included in training - they contribute
  whatever verses they have, as in the original; only holdout languages need
  full coverage of the held-out material so references exist for scoring.

**Translation selection** is done by a reproducible script driven by
`metadata.parquet`, with a config file for manual includes/excludes. Criteria:
one translation per language (prefer highest verse coverage), then maximise
diversity of language family and script. Selected translation-ID lists are
committed to the repo per experiment, so every run is exactly reproducible.

- **Pilot**: ~50 translations, echoing Liedes' scale and (where the corpus
  allows) his languages.
- **Scaled**: a curated few hundred translations (all 179 full Bibles, plus a
  diverse selection of partial ones).

**Holdout design** (the test set):

| Language | Translation | Held out | Trained on |
|---|---|---|---|
| English | `engbsb` | whole OT | NT only |
| German | `deuelbbk` | Genesis | everything else |
| Finnish | `fin` | Matthew (configurable) | rest of NT |
| + ~5 extended | criteria-based | Genesis | everything else |

Extended holdouts are chosen automatically: full-Bible coverage, distinct
language families (target roughly one each of Bantu, Austronesian, Indo-Aryan,
Turkic, and one other), at least one non-Latin script - **David must approve
the list before any training run**. Holdouts are book-level splits
on the `vref` book code, never random verses.

**Preprocessing**:

- Native scripts, case preserved. Unicode NFC normalisation only. (Deviation
  from his romanised-lowercase pipeline, deliberately.)
- Verse-by-verse examples: one verse in, one verse out. Drop pairs where
  either side is empty or `<range>`.
- Length filters (configurable defaults): max 192 subword tokens per side,
  length ratio cap 2.0.
- Language tags as atomic tokens: target tag (e.g. `<2deu>`) prepended to the
  source; the many-to-many phase adds a source tag as well.

## Tokenisation

Three tokeniser experiments, in this order, on otherwise-identical configs:

1. **SentencePiece BPE** (primary): byte fallback on, language tags as
   `user_defined_symbols`. 32k vocab for the pilot (comparable to his 30k),
   64k for the scaled run.
2. **SentencePiece unigram**: same sizes - tests his hypothesis that BPE
   segmentation hurt morphologically rich targets (his *poi+kala+psi*
   complaint about Finnish).
3. **Byte-level** (no learned vocab): script-agnostic; expect ~4× longer
   sequences and correspondingly slower training.

The tokeniser is trained on the training split only - held-out book text is
excluded to avoid leakage.

## Model

From-scratch encoder-decoder transformer defined in Hugging Face Transformers
(MarianMT/mBART-style config, randomly initialised - no pretrained weights).
**Transformer-big scale throughout**: 6 encoder + 6 decoder layers,
d_model 1024, 16 heads, FFN 4096, ≈210M parameters (varies with vocab size).
Dropout 0.1, label smoothing 0.1.

Recorded alternatives if HF Transformers proves limiting: (a) fairseq2 -
strong NMT pedigree, Linux-only, heavier setup; (b) a custom PyTorch loop -
maximum control, more code. Revisit only if a concrete blocker appears.

## Training

- HF `Seq2SeqTrainer`; AdamW; warmup then inverse-sqrt (or cosine) decay;
  bf16; gradient accumulation sized to the 40 GB H100; gradient clipping 1.0.
- Batch by token count; exact hyperparameters live in committed YAML configs,
  not in this spec.
- Validation: ~5k randomly sampled (verse, language) pairs excluded from
  training for loss/early-stopping, plus a periodic generation probe on a few
  held-out passages logged to ClearML. Final test = the full held-out books
  (never used for any training decision beyond the probe's eyeball value).
- **Compute cap: every full training run under ~12 h on one H100 (40 GB).**
  Smoke tests and dev runs on the local Linux 3090. If a config cannot finish
  in ~12 h, shrink data or steps, not the evaluation.
- No oversampling of holdout languages in main runs (deviation from his 3×;
  tested in the ablation phase).

## Inference

- Generate held-out books verse-by-verse: source verse + target-language tag,
  batched beam search, beam 5, length penalty 1.0, max length matched to the
  length filter. (His beam-120 choice is tested in the ablation phase.)
- A "template" generation utility mirrors his `TGT_TEMPLATE` trick: generate
  any book into any training language on demand.
- Watch for runaway/degenerate output on out-of-distribution books (his
  maxlen-enforcement cherry-pick suggests he hit this); enforce hard length
  caps and log truncation counts.

## Evaluation

Metrics per held-out book per language, following sillsdev/silnlp and
machine.py conventions:

- **chrF3, chrF3+, chrF3++** (sacreBLEU chrF, β=3, word order 0/1/2) - chrF3
  is the headline metric.
- **spBLEU** (sacreBLEU with the Flores-200 SentencePiece tokeniser) and
  **BLEU** as secondary scores.

Artefacts:

- Full generated books as text files (whole English OT, each held-out
  Genesis, the Finnish Gospel), verse-referenced, as he published on kapsi.fi.
- Side-by-side generated-vs-reference sample sheets per holdout. The passage
  list is user-configurable (YAML); defaults mirror his blog: Gen 1:1–10,
  Gen 3:1–7, Num 6:22–26, Ps 23, plus Matt 5:1–12 for NT holdouts.
- Trivial baselines for context: copying the source verse, and the
  best-scoring training language's reference - generated output must clearly
  beat both for the reproduction to count as successful.

## Experiment phases

1. **Pilot (one-to-many)**: ~50 translations, composite Greek source, BPE 32k,
   transformer-big. Proves the whole pipeline and gives first scores.
2. **Ablations** (pilot config): 3× holdout-language oversampling on/off;
   beam 120 vs beam 5 - answering his own open questions.
3. **Scaled one-to-many**: few hundred translations, BPE 64k.
4. **Many-to-many**: source and target tags; per epoch, each target verse is
   paired with K (default 4, configurable) randomly sampled source
   translations containing that verse. Later comparison: a small fixed pivot
   source set (e.g. Greek, Hebrew, English, Spanish, Swahili, Chinese)
   instead of random sources.
5. **Tokeniser variants**: unigram, then byte-level, on the best config so far.
6. **English-source variant**: one-to-many from `engbsb`. (English then cannot
   be a holdout; holdout scoring shifts to the other languages.)
7. **Pretrained comparison (optional, later)**: fine-tune a pretrained
   multilingual model (e.g. NLLB-200) on the same data and holdouts - a
   deliberately different experiment (the model brings outside knowledge),
   kept separate in reporting.
8. **Report and publish.**

## Single-family (Indo-European) run

A substantial *local* experiment run on the 3090 while ClearML access to the
H100s is pending. Two questions motivate it:

1. **Does restricting training to one language family help held-out
   languages?** Hypothesis: a held-out language surrounded by close relatives
   (German among Dutch/Danish/Swedish; Hindi among Gujarati/Marathi/Nepali/
   Bengali/…) transfers morphology and lexicon better than one embedded in a
   maximally diverse mix. The content knowledge Liedes relied on is
   language-independent, but the *decoding into* a held-out language should
   benefit from siblings. This run trains on **Indo-European only**; its
   held-out scores are compared against the diverse pilot on the shared
   holdouts to isolate the family-restriction effect.
2. **Can the model draft a whole OT that does not yet exist?** Rather than the
   pilot's book-level Genesis holdouts, the **entire OT is withheld** from each
   test language (train on their NT + the full Bibles of the sibling
   languages), then generated end-to-end — the practical "no OT exists yet"
   scenario this project ultimately serves.

Design:

- **Selection**: every Indo-European language in the corpus with ≥5k verses,
  best-covered translation per language — 34 languages across Germanic (7),
  Slavic (8), Romance (6), Indo-Aryan (10), Iranian (2), Baltic (1). Curated
  code→branch map: `configs/families/indo_european.csv`; reproducible builder:
  `samileides.family` → `experiments/selection-ie.csv`. Partial (NT-only) IE
  translations are kept, as elsewhere. The Greek source (`grc`) and ancient
  Hebrew (`hbo`) are excluded as targets.
- **Holdouts** (`configs/holdouts-ie.yaml`): whole OT withheld from English
  (`engbsb`), German (`deuelbbk`) and Hindi (`hin2017`); each keeps close
  siblings in training and has a full OT reference for scoring. English OT is
  directly comparable to the pilot's English holdout; German and Hindi differ
  from the pilot (whole OT here vs Genesis-only there), so cross-run comparison
  for those two is done on the Genesis subset of the generated OT.
- **Model/config** (`configs/experiments/ie_base.yaml`): transformer-**base**
  (d_model 512, 6+6 layers, 8 heads, FFN 2048, ≈61M params), BPE 32k, bf16,
  sized so the run finishes in a few hours on one RTX 3090 (measured ≈5
  optimiser steps/s; 60k steps ≈ 3–4 h). This is a dev-box run; transformer-big
  remains the H100 target.
- **Evaluation**: the standard metric set per held-out OT per language, plus a
  Genesis-subset score for pilot comparability, and the source-copy baseline.

If family-restriction helps, later phases can test *other* single families
(e.g. an all-Bantu or all-Austronesian run) and family-vs-diverse at H100
scale.

## Infrastructure

- **Dev**: this repo, developed on Windows; data prep and analysis must run on
  Windows and Linux; training targets Linux (local 3090 for smoke runs,
  remote H100s for real runs).
- **Jobs and tracking**: ClearML for both scheduling and experiment tracking
  (scalars, configs, artefacts); the remote H100s are already ClearML-managed.
  File transfer via the existing rclone + WireGuard setup.
- **Publishing**: trained checkpoints, tokenisers, generated books and sample
  sheets pushed to the Hugging Face Hub under David's account; the written
  report lives in this repo.

## Repository layout

- git repo in this folder; uv-managed Python ≥ 3.11; `pyproject.toml`;
  `src/samileides/` package; `configs/` (YAML per experiment); `experiments/`
  (committed selection lists and results tables); `report/`; `tests/` (pytest).
- The `webpages/` folder and `project-brief.md` stay as project background.

## Deviations from the original (to be tabled in the report)

| Aspect | Liedes 2018 | This reproduction |
|---|---|---|
| Architecture | fconv 20+20×512 | transformer-big ~210M |
| Corpus | 53 Sword modules | eBible corpus, staged 50 → hundreds |
| Scripts/case | romanised, lowercase | native scripts, case kept |
| Tokenisation | Moses + subword-nmt 30k | SentencePiece (BPE/unigram/byte), 32k–64k |
| Greek source | `2TGreek` romanised | `grcbrent`+`grc-tisch`, Greek script |
| Finnish holdout | Genesis | one Gospel (no Finnish OT exists in corpus) |
| German text | GerNeUe | `deuelbbk` |
| English text | ESV2011 | `engbsb` |
| Oversampling | 3× test languages | none (ablated) |
| Beam | 120 | 5 (ablated) |
| Evaluation | qualitative only | chrF3(/+/++), spBLEU, BLEU + samples |

## Verification

How each piece is proven to work before it is relied on:

1. **Alignment**: unit tests assert known verses match expected text (e.g.
   `GEN 1:1` in `engbsb` begins "In the beginning God created"); row counts
   match coverage metadata for a sample of translations.
2. **Holdout integrity**: automated leakage test - for every holdout language,
   assert zero overlap between training pairs and held-out-book vrefs, and
   that the tokeniser corpus contains no held-out text. Runs in CI/pytest and
   again as a pre-training assertion inside the training script.
3. **Tokeniser**: tests assert language tags survive as single atomic tokens
   and round-trip encode/decode is lossless on samples from every script in
   the selection.
4. **Model plumbing**: overfit a 100-pair subset to near-zero loss (proves the
   train loop learns); a tiny-config end-to-end run on the 3090 (small model,
   3 translations) must produce every artefact - checkpoint, generated book,
   metric table, sample sheet - before any H100 time is spent.
5. **Metrics**: scoring a reference against itself yields chrF3/BLEU ≈ 100;
   scoring shuffled text yields near 0; values cross-checked against
   machine.py/silnlp output on a shared fixture.
6. **End-to-end success criterion**: pilot-generated German Genesis and
   English OT score clearly above the trivial baselines (source-copy and
   other-language-reference), and the sample sheets read as recognisable
   translations - the qualitative bar his blog post set.
7. **Reproducibility**: every run records its selection list, config, git
   commit and seed in ClearML; re-running a config regenerates identical
   train/valid/test splits (asserted by checksum).
