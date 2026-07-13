# Project brief: next series of experiments

We're working on a machine translation system for a closed text such as the Bible.
That's available in many languages and there are verse references that can be used to align the 
meaning across translations in different languages. 

The goal of the next series of experiment are to test the idea that a model can be trained in multiple 
languages and contain the 'meaning' of each verse within the model. Then the model needs to learn 
a language by comparing a portion of the Bible in the new language in such a way that it learns how
to express the remaining part of the text in that language.

As Sami Leides said in his blog, Google translate has a very difficult task since it must be able
to translate any arbitary sentence that it has never seen before. This model only has to translate
verses it has seen many times in various other languages. It has a different difficult task; to 
learn the language simply by comparing the New Testament in the language with it's own understanding
of the meaning. Then it must be able to translate verses it knows well into that language.

## Outline

<!-- What should this series find out or produce? One or two sentences. -->

The question I want to answer is: Can we give verse references on the source side to a model with a <2iso> tag that tells the model
which language it is seeing on the target side. Can such a model learn to translate from the vref into a target 
language for which only part of the Bible was provided in training? Can we create a SOTA model with this method.
This will be a model from scratch and not use NLLB.

## Where the last series left off

The Liedes reproduction is complete: full from-scratch pipeline; the
Indo-European runs (`ie_base` 40.7/40.5/38.1 held-out-OT chrF3 for
eng/deu/hin, shareable twin published); the NLLB many-to-one series (15-run
matrix; token-init irrelevant, source proximity decisive, lr 3e-4 essential);
five repos on the Hub awaiting review. Details: `spec.md` (decisions log),
`todo.md` (current status), `experiments/*.md` (results).

Two findings that should be remembered if they are relevant for this series::

1. **Learning rate dominates everything else** for NLLB fine-tunes into new
   languages: 3e-5 looks like fundamental failure, 3e-4 works. Any earlier
   NLLB result trained at 3e-5 (notably `nllb_ie`, 52.8/51.1/42.7) is
   probably underpowered and cheap to improve.
2. **Real linguistic proximity of the sources is the ceiling**: close
   relatives 55-61 chrF3, cross-script relatives 44, no true relatives ~10.
   Source selection matters more than model tricks.

## Candidate directions (from the roadmap)

- **Fairer many-to-many test**, then the flagship transformer-big (~210M)
  many-to-many run (blocked on the ClearML agent fix for remote H100s;
  runnable at reduced scale on the A100/3090).
- **Scaled one-to-many**: all ~179 full Bibles plus diverse partials, BPE 64k.
- **Other single families** (all-Bantu, all-Austronesian) and
  family-versus-diverse at matched scale.
- **Multi-source inference ensembling** and **iterative backtranslation**
  (spec.md, "Roadmap to stronger results").
- **Tokeniser variants** (unigram, byte-level) on the best config.
- **Report and publish**: the written report in `report/` remains unstarted.

## Constraints

- Local RTX 3090 (24 GB) always available; A100 (80 GB) for research runs;
  remote H100s via ClearML once SIL fixes the agent bootstrap.
- Publishing gates as established: quality floor, licence policy (base-model
  licences propagate; `by-nd`/unknown sources are never publishable).

## Data source

Hugging Face dataset `DavidCBaines/ebible_corpus`, as before.

## Approach

Run the planning interview (`/interview-and-plan`) against this brief once
the Outline is filled in; it will produce a fresh `spec.md` section (or a new
spec) and a task list.
