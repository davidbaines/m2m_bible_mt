# vref-source encoding experiments — Phase 1 (base scale)

Runs on the local RTX 3090, 2026-07-10/11. This document records what was run
and what was observed, and distinguishes observation from interpretation.

A licence-safe mirror of these results — scores, training curves, configs and
the selection list, with no scripture text — is published as a Hugging Face
dataset:
<https://huggingface.co/datasets/DavidCBaines/ebible_m2m-vref-negative-result>.

## What was tested

Whether a from-scratch encoder-decoder model whose source side carries only a
target-language tag and an encoded verse reference — no source text — can
produce the held-out books of a language for which only part of the Bible was
seen in training. Three encodings of the reference were compared:

- `struct`: atomic book, chapter and verse tokens (`<GEN> <c1> <v1>`).
- `vtok`: one atomic symbol per verse (`<GEN_1:1>`).
- `text`: the literal reference string (`GEN 1:1`) through ordinary BPE.

All three used transformer-base (`struct`/`text` 60.9M parameters, `vtok`
82.2M because of its ~42k extra verse-symbol embeddings), and were trained on
the same data and holdouts as the earlier Greek-source run `ie_base`: 34
Indo-European languages, the whole Old Testament withheld from English
(`engbsb`), German (`deuelbbk`) and Hindi (`hin2017`). The training pair set
was asserted byte-identical to `ie_base`'s (809,477 pairs), so the only
deliberate difference from `ie_base` is the source side.

Each run evaluated a fixed 750-verse held-out probe (250 per test language,
greedy decoding) every 1,000 steps, and stopped when the macro held-out chrF3
failed to gain 1.0 point over 20,000 steps, up to a 180,000-step ceiling.

The reference point is `ie_base` (Greek source, identical data): held-out
whole-OT chrF3 of 40.7 / 40.5 / 38.1 for English / German / Hindi, validation
loss 2.92.

## Observations

### The three encoding runs

All three stopped early — between 18,000 and 28,000 steps — because the
held-out probe chrF3 stopped rising.

| Run | stopped at | peak held-out probe chrF3 (deu / eng / hin) | macro | val loss |
|---|---|---|---|---|
| `vref_ie_struct` | 26k | 17.1 / 12.9 / 9.8 | 13.3 | 4.35 |
| `vref_ie_vtok` | 28k | 18.0 / 16.4 / 13.0 | 15.8 | 4.37 |
| `vref_ie_text` | 26k | 15.7 / 13.5 / 11.7 | 13.6 | 4.37 |
| `ie_base` (Greek) | — | 40.7 / 40.5 / 38.1 (whole-OT) | — | 2.92 |

Whole-OT beam-5 generation from each run's best checkpoint gave per-book chrF3
of roughly 9 to 14, consistent with the probe figures. `vtok` had
the highest peak of the three, by a few chrF3 points; each run used a single
seed, so the size of that gap should not be over-read.

Generated text is fluent and in the correct target language, uses scripture
vocabulary, is frequently repetitive, and does not match the content of the
requested verse. For example, English Genesis 1:1 (held out) from `struct`:

> "Which is the sons of the sons, the sons of the sons, the sons of the
> sons…"

### Diagnostic re-run: `vref_ie_vtok_long`

To test whether the early stop was hiding later improvement, `vtok` was
re-run to the full 180,000 steps with early stopping disabled, adding a second
probe over 750 *seen* verses — the test languages' New Testament, which was in
training — to observe reproduction of trained material alongside held-out
generation.

| step | held-out macro chrF3 | seen macro chrF3 |
|---|---|---|
| 2,000 | 9.8 | 9.8 |
| 7,000 (peak) | 13.8 | 14.7 |
| 30,000 | 5.5 | 6.1 |
| 90,000 | 5.3 | 5.9 |
| 130,000 | 6.5 | 7.0 |
| 170,000 | 0.3 | 0.4 |
| 180,000 | 0.04 | 0.04 |

- The highest score was reached at about step 7,000; scores did not rise above
  that level over the remaining ~173,000 steps.
- After the peak the curve fell and became erratic, including several probes
  near zero at 100k, 120k, 170k and 180k.
- The seen-verse curve stayed within about one chrF3 point of the held-out
  curve throughout, peaking at 14.7.
- Validation loss finished at 4.34.

A qualitative spot check on a handful of trained New Testament verses (English
Matthew 1:1 and John 3:16, German John 3:16) returned fluent but
content-incorrect output rather than the trained verse.

### Pipeline checks

The training pair set matched `ie_base`'s by checksum; the 100-pair overfit
check reaches near-zero loss for each encoding; the holdout leakage tests
pass; the probe (greedy) and final (beam-5) scores agree; generation uses the
same code path that produced `ie_base`'s 40 chrF3. Together these make a
defect in the data, training loop or scorer an unlikely cause of the low
scores.

## Interpretation

The following are inferences from the observations above, not established
facts.

- In `ie_base` the verse's meaning is supplied at inference by the Greek
  source text; in the vref runs it is not, so the model would have to supply
  verse content from its parameters. Under the configurations tested, the
  base-scale models did not learn to do this: their output is a fluent
  per-language prior rather than the specific verse.
- Because the seen and held-out curves stayed close, the shortfall appears to
  lie in fitting the reference-to-content mapping in general, rather than
  specifically in transferring content to a withheld language. This is
  inferred from two tracking curves in a single run and would need repetition
  to confirm.
- Training far longer did not raise the scores in the single long run;
  the decline and the near-zero probes late in that run are consistent with
  optimisation instability under a learning-rate schedule that does not decay
  toward zero, but the cause was not isolated (no schedule or seed was varied).

## What these runs do not establish

- Whether a larger model (transformer-big, ~210M) would succeed. Only base
  scale was tested.
- Whether the approach can work at all with a different learning-rate
  schedule, optimiser, seed, or a smaller closed set. Each factor was held
  fixed.
- That `vtok` is the best encoding in general. It had the highest peak, but on
  one seed per encoding and by a small margin.
- Any claim that the method is impossible. The runs show that this
  configuration did not work, not that no configuration can.

## Limitations

One seed per configuration; a single long run (vtok only); one learning-rate
schedule; base scale only. Held-out probe figures are a 750-verse sample;
whole-OT generation figures are separate and noted as such. The seen-verse
reproduction check is a small qualitative spot check, not a systematic
evaluation.

## Possible next steps

Not recommendations, and none is implied to succeed:

- A capacity test at transformer-big scale on the A100, with a decaying
  learning-rate schedule for late-training stability.
- A reduced-scope run (fewer languages, less withheld) to see whether the
  reference-to-content mapping is learned when the closed set is small.
- Design changes that reduce the amount the model must store (for example a
  source signal or a retrieval component), or starting from a pretrained
  multilingual model.
- Recording the base-scale result and pausing pending a direction decision.
