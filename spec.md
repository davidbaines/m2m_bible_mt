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

## Roadmap to stronger results

The `ie_base` scores (chrF3 in the high 30s to low 40s for a whole held-out OT)
confirm the effect but sit below what larger, better-resourced systems reach.
Priorities agreed 2026-07-06, in expected order of impact:

1. **Many-to-many** (bring phase 4 forward). Pair each target verse with K
   sampled source languages per epoch instead of Greek alone. NOTE (2026-07-07):
   the base-scale test `ie_base_m2m` scored *below* one-to-many when both were
   evaluated from the fixed Greek source, because many-to-many trains on
   Greek->target far less often at equal compute (see
   `experiments/ie-base-m2m-results.md`). Before a large run, make the test fair
   to the method: evaluate from the held-out language's relatives, oversample
   the pivot source, and/or increase the step budget. Do not assume it is a free
   win at fixed compute with single-source evaluation.
2. **Scale up** to transformer-big (~210M) on the A100 (80 GB), with a large
   batch and a longer schedule.
3. **Broader data**: hundreds of translations, keeping a held-out language's
   close relatives while adding wide multilingual coverage.
4. **Multi-source inference ensembling**: at generation, translate a held-out
   verse from several source languages and combine or rerank. Exploits the
   closed-text redundancy that open-text systems lack.
5. **Iterative backtranslation**: feed the model's own drafts back as synthetic
   training data.
6. **Pretrained fine-tune track (phase 7), reported separately.** Fine-tune a
   strong multilingual model (e.g. NLLB-200 at 1.3B) on the A100 for the best
   absolute drafting quality. This brings outside knowledge, so it is a
   different experiment, kept apart from the from-scratch results.

## Baselines and evaluation notes

Two baselines are reported per held-out book (spec verification #6):

- **source-copy**: the untagged Greek source verse. Near zero, because the
  source is in a different script; a floor against degenerate output only.
- **best other-language reference**: chrF3 of the single most similar training
  language's own text for the same verses. A far stronger floor, since a close
  relative shares script and vocabulary. `samileides.generate` computes it for
  new runs; `samileides.rescore` adds it to runs generated before it existed.

The publish quality gate requires beating source-copy on every book **and**
beating the other-language baseline per language on a verse-weighted average.

## Infrastructure

Compute available (updated 2026-07-06):

- **Local Linux 3090 (24 GB)**: the primary development box. Runs the test
  suite, data prep, smoke tests, and, in practice, full substantial runs at
  transformer-base scale (the `ie_base` and `ie_base_shareable` runs each
  trained 60k steps in ~2.1 h here). Setup notes in
  `experiments/dev-3090-setup.md`. Note: `uv` is installed at
  `~/.local/bin/uv` and invoked by absolute path (a VS Code snap sandbox put
  the installer's PATH entry out of reach).
- **A100 (80 GB), research**: headroom for transformer-big from-scratch runs
  and for fine-tuning a pretrained model (e.g. NLLB-200). Not yet wired in.
- **Remote H100s (40 GB) via ClearML**: for scheduled real runs; ClearML is
  also the experiment tracker (scalars, configs, artefacts). Access is being
  set up (credentials, queue name, and a git remote the agents can clone are
  the outstanding pieces). File transfer via the existing rclone + WireGuard
  setup.

The original single-H100-40 GB constraint in `project-brief.md` is superseded
by this wider pool; the ~12 h per-full-run cap remains the working guideline
for H100 runs.

Data prep and analysis run on both Windows and Linux; earlier development was
on Windows, current development is on the Linux 3090.

- **Publishing**: trained checkpoints, tokenisers, generated books and sample
  sheets pushed to the Hugging Face Hub under David's account; the written
  report lives in this repo. Full policy in "Publishing" below.

## Publishing

Successful models are shared on the Hugging Face Hub by a single gated command,
`samileides.publish` (run after a human review, not automatically). Decisions
agreed with David on 2026-07-06:

- **Quality gate**: publishing is refused unless every generated held-out book
  beats the source-copy baseline on chrF3. The command re-checks this from the
  run's `generated/metrics.csv`, so `samileides.generate` must have been run.
- **Licence gate**: publishing is refused unless every training translation
  carries a licence that permits sharing a *derived* model. The eBible
  `Redistributable` flag is `True` for all translations and only covers the
  source text, so the real signal is `licence_Licence_Type`. A trained model is
  treated as a derivative work. Shareable licences: `Public Domain`, `by`,
  `by-sa` (commercial use allowed); `by-nc` is shareable only non-commercially
  (opt-in with `--allow-nc`). Never shareable as a derivative: `by-nd`,
  `by-nc-nd`, `Unknown`, and anything else. Implication: a publishable run must
  train on a licence-filtered selection; runs that include, e.g., `by-nc-nd`
  German (`deuelbbk`) are research/comparison only. ShareAlike propagates, so
  any `by-sa` source forces the model licence to `cc-by-sa-4.0`; otherwise
  `cc-by-4.0` (any `by`) or `cc0-1.0` (all public domain). The model card lists
  every source translation and its licence.
- **Repository layout**: one repository per run under `DavidCBaines`, named
  `ebible_m2m-<experiment>` (e.g. `ebible_m2m-ie-base`). The command can push
  public or `--private`; only redistributable data reaches a published model,
  so public is safe, but in practice the first upload of a run goes private for
  review and is made public once checked. The best models may later be mirrored
  to the `bible-nlp` organisation. Liedes' name is not
  used in any repository name.
- **Loadability**: the raw SentencePiece model is packaged as a
  `MarianTokenizer`, so `MarianMTModel.from_pretrained` plus
  `MarianTokenizer.from_pretrained` work directly.
- **Contents**: model weights and config, the loadable tokeniser, the generated
  books, metrics and sample sheets, the committed config and selection list
  (for provenance), and a model card recording architecture, held-out books,
  metrics, every source licence, and the git commit and seed for reproduction.

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
8. **Publishing gates**: unit tests assert the quality gate rejects any run
   whose held-out chrF3 does not beat the baseline, and the licence gate flags
   every non-derivative-permitting or unknown-licence source; the packaged
   tokeniser round-trips and the staged repository loads with `from_pretrained`.
   A `--dry-run` builds the full repository (card, tokeniser, artefacts) without
   pushing, for inspection before any upload.

## Decisions log

Dated record of choices made in planning, so a later reader (or a restarted
session) does not relitigate them. The rationale for each lives in the relevant
section above.

- **2026-07-05**: architecture, staged data, one-to-many-first, holdout design,
  tokeniser order, evaluation metrics, infra (the "Settled decisions" in
  `project-brief.md`).
- **2026-07-06, single-family experiment**: run an Indo-European-only,
  one-to-many experiment on the 3090; whole OT withheld from English, German
  and Hindi (approved for this run); transformer-base scale.
- **2026-07-06, publishing policy**: publish under `DavidCBaines`, one repo per
  run named `ebible_m2m-<experiment>`; Liedes' name not used in repo names.
  Quality gate = beat source-copy per book and the other-language baseline per
  language; licence gate = train only on derivative-permitting licences
  (Public Domain / by / by-sa), ShareAlike propagating to the model licence.
  Fully `from_pretrained`-loadable. The best models may later be mirrored to the
  `bible-nlp` organisation.
- **2026-07-06, first publish**: `ie_base_shareable` published **private** first
  (`DavidCBaines/ebible_m2m-ie-base-shareable`) for review before going public,
  even though only redistributable data is included.
- **2026-07-06, roadmap**: prioritise many-to-many, then transformer-big on the
  A100, with the NLLB fine-tune as a separate track (see "Roadmap" above).
- **2026-07-08, NLLB m2o publishing**: the three useful m2o fine-tunes
  (Tongan, N. Ndebele, Ilocano control) plus a results/methodology dataset
  repo staged **private** under `DavidCBaines` via a new gated command
  (`samileides.publish_nllb`). Licence rule extended: the base model's
  licence propagates too — NLLB-200 is CC-BY-NC-4.0, so every NLLB fine-tune
  is non-commercial (`cc-by-nc-sa-4.0` here) even when the data is all
  by-sa/PD. Quality gate for m2o = beat the source-copy baseline (the best
  source's own text vs the target reference) on every test book. Repo naming:
  `ebible_m2o-nllb600m-<target>`; results dataset `ebible_m2o-nllb-results`
  (cc-by-sa-4.0). Checkpoints not in the repos stay local only.
- **2026-07-06, local experiments while ClearML is down**: the H100 agents are
  blocked by an agent-side bug (`experiments/clearml-agent-issue.md`), so the
  3090 is used for: (a) the many-to-many pair sampler, built and verified
  (`samileides.manytomany`); (b) a modest NLLB-200-600M many-to-many fine-tune
  (`samileides.train_nllb`, `configs/experiments/nllb_ie.yaml`), generating the
  held-out OTs from a Spanish pivot since NLLB has no Koine Greek. NLLB smoke on
  3 languages drafted held-out Jonah at chrF3 46 after only 150 steps. The full
  `nllb_ie` run scored chrF3 52.8/51.1/42.7 (deu/eng/hin), above from-scratch
  `ie_base` (40.5/40.7/38.1) but on an easier task (Spanish source, outside
  knowledge); write-up in `experiments/nllb-ie-results.md`. Also queued:
  `ie_base_m2m`, a controlled many-to-many-vs-one-to-many test at base scale.

## Maintaining these documents

`spec.md` is the stable "why and what"; `todo.md` (with its "Current status"
block) is the living "where we are". Lightweight routine, at the end of each
working session:

1. Update the "Current status" block in `todo.md` (done, running, next, blocked).
2. Tick completed tasks; add newly discovered ones.
3. Record any new decision in the Decisions log above.
4. Drop run results into `experiments/` and link them from `todo.md`.

Keep it brief; the point is restartability, not ceremony.
