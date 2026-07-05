# Project brief

## Outline
Sami Liedes carried out a successful experiment in machine translation of a
closed text - the text of the Bible. The aim of this project is to reproduce
that experiment with a transformer model rather than the sequence-to-sequence
model he used, and with more data available.

## Constraints
- The experiment must run on a single H100 with 40 GB of RAM.

## Background research
1. The surrounding context is on the Sami Liedes website (downloaded as
   "Sami Liedes – Data and general life geekery.html").
2. The original source code for all the scripts is at
   https://github.com/sliedes/fairseq-py — useful for ideas and guidance.

## Approach
Implement a multilingual closed-text translation model with up-to-date machine
learning techniques.

## Data source
Training data comes from this Hugging Face dataset:
https://huggingface.co/datasets/DavidCBaines/ebible_corpus

The dataset was built partly to reproduce Sami Liedes' experiment, described on
his blog: https://samiliedes.wordpress.com/author/samiliedes/ — He trained a
model on the Bible in ~50 languages, omitting Genesis (or the whole Old
Testament) from three of them. The model then "knows" the content of the Bible
from having seen it in many languages, and can translate Genesis into the
languages it was never trained on for Genesis. This data is being prepared to
repeat variations of that experiment.

## Settled decisions
Agreed in the planning interview on 2026-07-05 (details in `spec.md`):

- From-scratch encoder-decoder transformer in HF Transformers, transformer-big
  scale throughout; a pretrained fine-tune (e.g. NLLB) is a later, separate
  comparison. fairseq2 / custom loop noted as fallbacks.
- Staged data: ~50-translation pilot, then a curated few hundred; partial
  (NT-only) translations included in training; selection by a criteria-driven
  script with manual overrides, lists committed.
- One-to-many first from a composite Greek source (`grcbrent` LXX +
  `grc-tisch` NT, native Greek script), then many-to-many (K random sources
  per target verse, K=4 default; pivot-set variant compared later). English
  source variant uses `engbsb` (BSB).
- Holdouts: English OT (`engbsb`, NT-only training), German Genesis
  (`deuelbbk`), Finnish one Gospel (no Finnish OT exists in the corpus), plus
  ~5 criteria-chosen diverse languages with Genesis held out (list approved
  before training). Book-level splits only.
- Native scripts, case kept, NFC only; verse-by-verse examples.
- Tokenisers, in order: SentencePiece BPE (32k pilot / 64k scaled), then
  unigram, then byte-level; tags atomic; trained on training data only.
- Modern decoding defaults (beam 5, no holdout oversampling); his 3×
  oversampling and beam 120 tested as pilot-stage ablations.
- Evaluation: chrF3, chrF3+, chrF3++, spBLEU, BLEU (silnlp/machine.py
  conventions); full generated books plus configurable side-by-side sample
  sheets; trivial baselines for context.
- Infra: local Linux 3090 for dev/smoke runs; ClearML-managed remote H100s
  (40 GB) for real runs, ClearML also the experiment tracker; rclone +
  WireGuard transfer; models and outputs published to the HF Hub. Every full
  run capped at ~12 h.
- This folder is the repo: git + uv, Python ≥ 3.11, `src/` layout, YAML
  configs, pytest.

## Design summary
Train a transformer-big encoder-decoder from scratch on verse-aligned Bible
translations from the eBible corpus, target-language tags prepended to a fixed
Greek source (later many-to-many), with whole books withheld from designated
languages. Generate the withheld books and score them with chrF3-family and
BLEU-family metrics against the real translations — a quantitative,
larger-scale rerun of Liedes' qualitative 2018 experiment. Full details and
verification plan: `spec.md`; running task list: `todo.md`.
