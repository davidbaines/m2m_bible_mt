# Project brief: next series of experiments

*(Draft — David to state the goal; the previous, completed brief is
`project-brief-liedes-reproduction.md`.)*

## Outline

<!-- What should this series find out or produce? One or two sentences. -->

To be decided. The candidate directions below carry over from the completed
Liedes-reproduction series; pick, combine, or replace them.

## Where the last series left off

The Liedes reproduction is complete: full from-scratch pipeline; the
Indo-European runs (`ie_base` 40.7/40.5/38.1 held-out-OT chrF3 for
eng/deu/hin, shareable twin published); the NLLB many-to-one series (15-run
matrix; token-init irrelevant, source proximity decisive, lr 3e-4 essential);
five repos on the Hub awaiting review. Details: `spec.md` (decisions log),
`todo.md` (current status), `experiments/*.md` (results).

Two findings that should shape whatever comes next:

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
- **Rerun `nllb_ie` at the corrected lr** — likely the cheapest large gain.
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
