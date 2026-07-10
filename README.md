# eBible many-to-many Bible translation

This repository trains multilingual machine translation models on the text of
the Bible. It reproduces and extends an experiment first carried out by Sami
Liedes in 2018, in which a single model was trained on the Bible in many
languages while certain books were withheld from certain languages. The model
then generated the withheld books, in effect drafting scripture in a language
for which that part of the Bible had never been seen during training. The
reproduction replaces his convolutional sequence to sequence model with a
modern transformer, uses the much larger eBible corpus in place of his 53
translations, and adds the quantitative evaluation that his original work
lacked.

The practical purpose behind the project is Bible translation drafting. A model
that has read the Bible in hundreds of languages holds the content of the text
independently of any single language. Given a language for which only part of
the Bible exists, the model can propose a draft of the missing books, which a
human translator can then review and correct.

## Background

Sami Liedes described his experiment in a 2018 blog post, "Machine translating
the Bible into new languages". He trained a model on the Bible in roughly fifty
languages, using Koine Greek as a fixed source and a language tag to select the
target. He withheld Genesis, or the whole Old Testament, from a small number of
languages, and generated those books afterwards. His evaluation was
qualitative. The full design of this reproduction, the decisions taken and the
deviations from his original method are recorded in `spec.md`.

Training data comes from the eBible corpus, published on the Hugging Face Hub as
the dataset `DavidCBaines/ebible_corpus`. It provides 1,253 translations across
997 languages, aligned verse by verse against a standard list of verse
references.

## What the pipeline does

The steps below are each a small, testable module in `src/samileides`.

1. Select a set of translations by reproducible criteria, and write the chosen
   list to a committed file.
2. Build a composite Koine Greek source from the Brenton Septuagint for the Old
   Testament and Tischendorf for the New Testament.
3. Split the data so that whole books are withheld from designated languages,
   never individual verses, and confirm that no withheld text can reach
   training.
4. Train a SentencePiece tokeniser on the training split only.
5. Preprocess verse pairs, applying Unicode normalisation, length filters and a
   target language tag prepended to each source verse.
6. Train a transformer from scratch with the Hugging Face trainer.
7. Generate the withheld books, score them against the real translations, and
   produce side by side sample sheets.
8. Publish successful models to the Hugging Face Hub, together with their
   generated books, metrics and a model card.

## Repository layout

- `src/samileides/`: the Python package.
- `configs/`: experiment configurations, holdout definitions, passage lists and
  the curated language family files, all in YAML or CSV.
- `experiments/`: committed selection lists, coverage reports and results, so
  that every run is reproducible.
- `tests/`: the pytest suite.
- `spec.md`: the full specification, including the verification plan.
- `todo.md`: the running task list.

## Installation

The project uses uv and requires Python 3.11 or newer.

```bash
uv sync                 # data preparation, evaluation and the test suite
uv sync --extra train   # adds the training stack (PyTorch, Transformers)
```

The training stack expects an NVIDIA GPU. Development and smoke runs are done on
a local RTX 3090; full runs are intended for larger GPUs. The verified local
setup is documented in `experiments/dev-3090-setup.md`.

## Running the pipeline

Build a selection and inspect coverage:

```bash
uv run python -m samileides.pilot     # a roughly 50 language pilot selection
uv run python -m samileides.family    # all Indo-European languages in the corpus
uv run python -m samileides.greek     # composite Greek source coverage report
```

Train a model, then generate and score the withheld books:

```bash
uv run python -m samileides.train    --config configs/experiments/ie_base.yaml \
    --output-dir checkpoints/ie_base
uv run python -m samileides.generate --run checkpoints/ie_base
```

Generate any book into any training language on demand:

```bash
uv run python -m samileides.generate --run checkpoints/ie_base \
    --book GEN --lang spa --out checkpoints/ie_base/template
```

## Experiments

The project proceeds in phases, described in full in `spec.md`. In outline these
are a pilot at roughly fifty translations, a set of ablations answering
questions Liedes left open, a scaled run at several hundred translations, a many
to many phase in which sources as well as targets vary, tokeniser variants, an
English source variant, an optional comparison against a pretrained model, and a
final written report.

### Local single-family experiment

While access to larger GPUs is pending, a substantial run is carried out on the
local 3090. It trains only on Indo-European languages, on the hypothesis that a
withheld language surrounded by close relatives will be drafted more accurately
than one embedded in a diverse mix. The whole Old Testament is withheld from
English, German and Hindi, so that the model must draft an entire testament,
which is the practical case the project ultimately serves. The configuration is
`configs/experiments/ie_base.yaml` and the selection is
`experiments/selection-ie.csv`.

## Sharing models on the Hugging Face Hub

Successful models are published with the publish command, which is deliberately
cautious and refuses to run unless two conditions hold.

First, the generated held out books must beat a trivial source copy baseline on
chrF3 for every holdout, so that weak or broken runs are not published.

Second, every translation used in training must carry a licence that permits
sharing a derived model. The eBible metadata records a licence type for each
translation. Public domain, attribution and attribution with share alike
licences permit derivatives and are treated as safe to publish a model from.
Licences that forbid derivatives, such as the no derivatives and no derivatives
with non-commercial variants, are not, and any translation whose licence is
unknown is also excluded. Where a share alike source is used, that condition
propagates to the licence of the published model.

```bash
uv run python -m samileides.publish --run checkpoints/ie_base --dry-run
```

A dry run assembles the repository contents and reports the licence and metric
checks without pushing anything. Removing the flag pushes to the Hub after a
final confirmation. Each run is published to its own repository under the
account `DavidCBaines`, named with the prefix `ebible_m2m`. The published
repository contains the model weights, a tokeniser that loads with
`from_pretrained`, the generated books, the metrics, the sample sheets and a
model card that lists every source translation and its licence.

```python
from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained("DavidCBaines/ebible_m2m-ie-base")
tokenizer = MarianTokenizer.from_pretrained("DavidCBaines/ebible_m2m-ie-base")
```

NLLB fine-tunes (the many-to-one series, prefix `ebible_m2o`) have their own
publish command with the same gates, plus the rule that the base model's
CC-BY-NC-4.0 licence propagates to every fine-tune:

```bash
uv run python -m samileides.publish_nllb --config configs/experiments/m2o/ton.yaml \
    --init scratch --checkpoint runs/m2o_winners/m2o_ton_scratch --dry-run
```

### Updating a published repository

Every repository on the Hub is a git repository, so an edit made in the web
editor creates a commit in the repo's history, exactly as a push would. There
are four equivalent ways to update a published repo:

1. **The web editor** on huggingface.co — open the file in the repo's *Files*
   tab and edit in place. Perfectly good for model-card (README) edits.
2. **The API from local files** — what the publish commands use: regenerate or
   edit the local staging folder (`runs/staging/<name>`), then
   `HfApi().upload_folder(folder_path=..., repo_id=..., repo_type="model")`.
   Only changed files are transferred.
3. **git clone** — e.g.
   `git clone https://huggingface.co/DavidCBaines/ebible_m2o-nllb600m-ton`,
   then edit, commit and push like any git repo. Works, but cloning pulls the
   full model weights (~2.4 GB), so it is the clunkiest option for card tweaks.
4. **`huggingface-cli upload <repo> <file>`** — one-off file pushes from the
   terminal.

One caution: the model cards are *generated* by the publish commands, so a
hand edit made on the Hub (methods 1, 3 or 4) will be overwritten by the next
re-publish unless the wording is also folded into the card template
(`build_model_card` in `src/samileides/publish_nllb.py` or
`src/samileides/hf_export.py`).

## Testing

```bash
uv run pytest                    # unit tests, no network access
uv run pytest -m integration     # tests that read from the Hugging Face Hub
```

The verification plan in `spec.md` sets out how each part of the pipeline is
proven before it is relied upon, including alignment against known verses,
leakage checks on the holdouts, tokeniser round tripping, an overfit sanity
check on the training loop, and metric checks against known values.

## Licensing

The code in this repository is available for reuse. The Bible translations in
the eBible corpus carry their own licences, which vary by translation, and those
licences govern any use of the text and of models derived from it. The publish
command enforces the licence policy described above for any model shared on the
Hub.

## Acknowledgement

This project reproduces the closed text Bible translation experiment described
by Sami Liedes.
