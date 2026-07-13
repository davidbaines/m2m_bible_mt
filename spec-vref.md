# Spec: vref-source experiments

The spec for the new series agreed 2026-07-10. The Liedes-reproduction spec
(`spec.md`) remains the reference for everything shared: the corpus, the
holdout definitions, the evaluation conventions, infrastructure, and the
publishing gates. This document covers only what is new.

## Goal

Test whether a from-scratch model whose *source side is only a verse
reference and a target-language tag* can hold the meaning of each verse
internally and express it in a language for which only part of the Bible was
seen in training. The model never sees a source text: training pairs look
like

```
source:  <2deu> <GEN> 1 : 1
target:  Im Anfang schuf Gott den Himmel und die Erde.
```

Compared with Google-Translate-style open translation, the task is inverted:
the model only ever produces verses it has seen many times in other
languages, but it must learn a new language purely by comparing the part of
the Bible provided in that language with its own internal representation of
the meaning.

The benchmark is `ie_base` (Greek-source one-to-many, transformer-base, 60k
steps on the 3090): held-out whole-OT chrF3 of **40.7 / 40.5 / 38.1** for
English / German / Hindi (`experiments/ie-base-results.md`).

## Questions, in order

1. **Does it work at all?** Is the output recognisably scripture in the
   right language for held-out (verse, language) pairs?
2. **Which vref encoding works best?** Three candidates, below; nobody knows
   which will win, so all three are run.
3. **Where does each encoding saturate?** This task is memorisation-heavy;
   a fixed 60k-step budget may misrank the encodings. Every run therefore
   trains to convergence (early stopping, 180k-step ceiling) with a
   chrF3/BLEU curve recorded throughout — the curves answer this directly.
4. **How close does the best variant come to Greek-source at matched
   scale?** No score bar is imposed — the comparison is reported, not gated.
5. (Eventually) **Can the method scale to a SOTA closed-text model?**
   Transformer-big on the A100 is the first step; the scaled selections come
   after this series.

## The three vref encodings

Run on otherwise-identical configs, differing only in how the source line is
built:

1. **Structured tokens** (`vref_ie_struct`): atomic book, chapter and verse
   tokens: `<2deu> <GEN> <c1> <v1>`. Neighbouring verses share most of their
   key, so structure is shared across chapters and verses. (Implementation
   note: chapter/verse are atomic `<cN>`/`<vN>` symbols, not raw digits.
   Making digit characters user-defined would have changed target-side
   segmentation and broken the one-shared-tokeniser guarantee below; atomic
   position symbols keep the target vocabulary untouched. The symbol set
   spans every book plus chapters 1..max and verses 1..max seen in the full
   vref list, so held-out references are always encodable.)
2. **One token per verse** (`vref_ie_vtok`): a single atomic symbol per line
   of vref.txt: `<2deu> <GEN_1:1>`. Adds 41,899 symbols (~21M extra
   embedding parameters at d_model 512 with tied embeddings — the parameter
   count is inherently larger; this is part of what the encoding *is*, not a
   confound to remove). Every verse key is independent — the purest "meaning
   slot" design.
3. **Plain text** (`vref_ie_text`): the literal string through the ordinary
   tokeniser: `<2deu> GEN 1:1`. Adds nothing; the key structure is whatever
   BPE produces.

## Data: identical to `ie_base`

Everything the vref runs train on is exactly the pair set `ie_base` used —
only the source side is replaced.

- Selection: `experiments/selection-ie.csv` (34 IE languages, one
  translation per language).
- Holdouts: `configs/holdouts-ie.yaml` — whole OT withheld from English
  (`engbsb`), German (`deuelbbk`) and Hindi (`hin2017`); each trains on its
  NT plus the sibling languages' full Bibles.
- **Greek-coverage restriction kept**: although vref sourcing needs no Greek
  text, training is restricted to the same (vref, translation) pairs
  `ie_base` trained on, so the comparison is on identical data. The verses
  the LXX lacks are a genuine advantage of the vref method and are measured
  later as a separate follow-up, not mixed into the comparison.
- Length/ratio filters apply to the target side as before; the source side
  is a handful of tokens and never filters anything.

## Tokeniser

Target-side segmentation is held constant across all three encodings: **one
shared SentencePiece BPE 32k model** trained on the training-split target
text plus the 41,899 vref strings (so the plain-text encoding gets sensible
segmentation of `GEN 1:1` rather than byte-fallback debris). Per-encoding
symbols (`<2xxx>` tags, the 66 book tokens, the 41,899 verse tokens, digits
as atomic pieces for the structured encoding) are added as special tokens on
top of the shared model. Trained on the training split only; held-out book
text excluded, as always.

## Model and training

`configs/experiments/vref/` holds one YAML per run, cloned from
`configs/experiments/ie_base.yaml`: MarianMT-style transformer-base (6+6,
d_model 512, 8 heads, FFN 2048), AdamW lr 5e-4, inverse-sqrt, 4000 warmup,
bf16, seed 13, on the 3090. The config deltas are `data.source: vref`,
`data.vref_encoding: struct | vtok | text`, and the train-to-convergence
regime below.

**Train to convergence, not to a fixed budget** (agreed 2026-07-10):

- **Probe evaluation every 1000 steps**: generate a fixed, seeded random
  sample of 250 held-out OT verses per holdout language (750 total, the
  *same* verses for every probe and every run), greedy decoding. Score
  chrF3 (stopping metric, macro-averaged over eng/deu/hin) and BLEU, per
  language and macro. Written to a per-run CSV in the run directory (these
  are local 3090 runs — no ClearML involved); a chrF3/BLEU-vs-steps graph
  (per language + macro) is produced per run and overlaid across the three
  runs in the results doc. Probe generation costs
  roughly 30–60 s against ~3.3 min of training per 1000 steps (~15–25%
  overhead).
- **Early stopping on probe chrF3**: stop unless macro-average probe chrF3
  has improved by at least **1.0 point within the last 20,000 steps**;
  `max_steps: 180000` as the ceiling. Worst case ~10–12 h per run on the
  3090; early stopping will usually cut that. The three runs go
  sequentially — budget a few days.
- **Checkpointing**: keep the best-probe-chrF3 checkpoint and the last;
  final evaluation and any downstream use take the best.

Encoder-decoder is used throughout this series for pipeline reuse and
comparability. A **decoder-only variant** (prompt = vref + tag, continuation
= verse) is arguably the more natural shape for "memorise and express"; it
is a recorded follow-up after the confirmation runs, not part of the
comparison.

## Generation and evaluation

Standard pipeline (`samileides.generate`) from the **best checkpoint**:
beam 5, length penalty 1.0, the three held-out OTs generated verse by
verse, chrF3 headline plus the usual metric set, Genesis-subset scores, and
sample sheets on the standard passages.

- **No baselines for this series.** Source-copy is meaningless (the source
  is a reference, not text), and the other-language baseline is not needed:
  the comparison that matters is against `ie_base`'s scores on the identical
  holdouts. Every results table shows the vref runs beside ie_base.
- **Matched-budget row**: ie_base trained a blind fixed 60k steps, so
  best-checkpoint scores flatter the vref runs slightly. The probe curve
  gives the 60k-step probe score for free; results report both the
  best-checkpoint whole-OT scores and the 60k-step probe scores, clearly
  labelled.
- **Recorded deviation from `spec.md`**: the held-out books now inform two
  training decisions — when to stop and which checkpoint is best. This is
  ordinary model selection, applied identically to all three encodings (so
  the comparison between them is fair), but it is tuning on the test
  distribution and is stated plainly wherever scores are reported.
- Watch for degenerate output: with an information-free source, a broken
  model may emit fluent scripture for the *wrong verse*. The sample sheets
  are the check — David reads them; chrF3 would also collapse.

## Decision rules

- **Gate to scale-up (lenient)**: any encoding whose sample sheets read as
  recognisably coherent scripture in the right language proceeds. The
  converged whole-OT chrF3 (best checkpoint) picks the winner; the probe
  curves show whether the ranking was stable or a late crossover.
- **Scale-up — transformer-big on A100** (`vref_ie_big`): the winning
  encoding at ~210M parameters, same probe/early-stopping regime with a
  ceiling sized to the A100 budget — the best-scores attempt of the series.
  (The previously planned separate "extended 3090 confirmation run" is
  gone: the main runs now train to convergence themselves.)

## Publishing

Deferred — no publishing commitments in this series. The standing gates in
`spec.md` apply to any future publish, with one recorded caveat: the quality
gate's source-copy criterion is inapplicable to vref models, so a
replacement gate must be decided before anything is pushed. (As before, any
run including `deuelbbk` (by-nc-nd) is research-only; a licence-filtered
shareable twin would be trained at that point, as was done for `ie_base`.)

## Recorded follow-ups (not scheduled)

- Full-coverage variant: train on the verses the Greek source lacks.
- Multiple translations per language (same vref+tag, different targets).
- Decoder-only architecture variant.
- Diverse ~50-language selection; then the all-179-full-Bibles scale-up.
- Whether vref-source combines with text-source training (multi-task) —
  noted only; out of scope.

## Verification

1. **Source construction**: unit tests per encoding assert the exact token
   sequence for known vrefs (e.g. `GEN 1:1`, a 3-digit chapter such as
   `PSA 119:176`, a book with a short code), and that every special symbol
   survives tokenisation as a single atomic token.
2. **Pair-set identity**: the (vref, translation) training-pair set is
   asserted identical to `ie_base`'s by checksum — the comparison claim
   depends on it.
3. **Holdout integrity**: the existing leakage tests run unchanged (zero
   overlap between training pairs and held-out vrefs; no held-out text in
   the tokeniser corpus).
4. **Plumbing**: overfit a 100-pair subset to near-zero loss for each
   encoding; one tiny end-to-end run (small model, 3 languages) must produce
   checkpoint, generated book, metrics and sample sheet before the real
   series starts.
5. **Metrics**: same scorers as before (sacreBLEU chrF3 etc., silnlp
   conventions); results tables always include the ie_base row.
5b. **Probe determinism**: a test asserts the probe verse sample is
   identical (same vrefs, same order) across encodings and across restarts
   — the curves are only comparable if every run probes the same verses.
6. **Coherence check**: sample sheets for all three runs go to David; the
   lenient gate is his judgement, recorded in the results doc.
7. **Reproducibility**: every run records selection list, config, git
   commit and seed; splits regenerate identically (checksum-asserted).

## Decisions log

- 2026-07-10 — Series scoped in the planning interview. Vref encoding is an
  open experimental question: all three candidates run at full ie_base scale
  (not screened at reduced scale). Encoder-decoder first, decoder-only as a
  follow-up. Data, holdouts and pair set matched to ie_base exactly; one
  translation per language; LXX-gap verses excluded for comparability.
  Lenient gate (coherent output) to the confirmation stage; confirmation is
  extended-3090 then A100 transformer-big, in that order. No baselines —
  report against ie_base only. Publishing deferred. The nllb_ie lr rerun
  stays out of this series (removed from the brief).
- 2026-07-10 (later) — Train-to-convergence replaces fixed budgets: probe
  eval (250 held-out verses/language, greedy) every 1000 steps producing
  chrF3/BLEU curves; early stop when macro probe chrF3 gains <1.0 over
  20,000 steps; max_steps 180000; best checkpoint kept and used for final
  eval. The separate extended-3090 confirmation run is deleted (the main
  runs answer it); A100 transformer-big moves up to Phase 2. Deviation
  accepted and recorded: held-out books now drive stopping and checkpoint
  selection, identically for all encodings; results also report 60k-step
  probe scores for a matched-budget comparison with ie_base.

- 2026-07-10 (Phase 0 build) — Implementation decisions made while building
  the plumbing:
  - **`struct` uses atomic `<cN>`/`<vN>` position tokens, not digits.** Making
    digits user-defined would have altered target-side segmentation and broken
    the shared-tokeniser guarantee. Symbols span every book + chapters/verses
    1..max over the *full* vref list, so held-out refs are always encodable.
  - **One shared BPE base per selection**, cached under
    `checkpoints/vref_shared/` keyed by a corpus+tags digest; per-encoding
    symbols are appended atomically on top (`vref.extend_tokenizer`). The
    three encodings of a series therefore share byte-identical target
    segmentation.
  - **`_leading_tags` tightened** to require a lowercase language code, so
    numbered *book* codes (`<1CH>`, `<2CO>`) and vtok verse symbols
    (`<1CH_10:10>`) are no longer misread as language tags and fed to the base
    tokeniser. (Regression test added.)
  - **`max_ratio: 0` disables the length-ratio filter** for vref runs — a
    3-token source would fail every ratio check otherwise.
  - **Offline resilience**: `data.translation_columns()` now caches the column
    list to `data/cache/columns.txt` and, if the Hub is unreachable, falls
    back to the union of cached verse-parquet schemas (correct for exact-id
    resolution). The dev 3090 has intermittent DNS; runs whose verse data is
    already cached no longer need the network.
  - **Pair-set identity confirmed**: vref (struct) and ie_base (Greek) train
    manifests are byte-identical (809,477 pairs); checksum committed to
    `experiments/vref-train-manifest.txt` and asserted before each vref run.

## Maintaining these documents

Same routine as `spec.md`: `todo.md` holds the living status; results land
in `experiments/` and are linked from `todo.md`; decisions are dated here.
