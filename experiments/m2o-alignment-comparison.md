# Alignment scores as a source-selection factor: eflomal vs IBM-1

For each many-to-one experiment we scored how alignable each candidate source
is to the (unknown) target, on their shared New-Testament verses, with two
aligners: eflomal (lower = more alignable) and an in-repo IBM Model 1 (higher =
more alignable). Method and setup: `experiments/alignment-setup.md`. Raw scores:
`experiments/m2o-alignment-eflomal.csv`, `experiments/m2o-alignment.csv`.

## Closest source to each target (best first)

| Target | Script | Same-script sources | IBM-1 ranking | eflomal ranking | Top-1 agree? |
|---|---|---|---|---|---|
| Tongan `ton` | Latn | 4/4 | tgl, ceb, mri, ind | mri, tgl, ceb, ind | no |
| N. Ndebele `nde` | Latn | 4/4 | sna, swh, lin, tsn | sna, lin, swh, tsn | yes (Shona) |
| Male `mdy` | Ethi | 1/4 | amh, arb, heb, gaz | gaz, amh, heb, arb | no |
| Romani `rmc` | Latn | 0/4 | guj, mar, asm, hin | hin, guj, asm, mar | no |
| Ilocano `ilo` (control) | Latn | 4/4 | tgl, ceb, ind, mri | tgl, ceb, mri, ind | yes (Tagalog) |

The two methods pick the **same** closest source only for N. Ndebele and the
Ilocano control; they disagree for Tongan, Male and Romani.

## Where they agree and where they diverge

Per-target rank correlation between the methods (Pearson, oriented so positive =
agreement):

| Target | correlation |
|---|---|
| Tongan | +0.75 |
| N. Ndebele | +0.80 |
| Ilocano (control) | +0.80 |
| Romani | −0.60 |
| Male | −0.79 |

The pattern is striking: the methods **agree on the same-script Latin cases**
(Tongan, Ndebele, Ilocano, all positive) and **disagree, even invert, on the
cross-script cases** (Male, Ethiopic with mixed-script sources; Romani, Latin
with Indic-script sources, both negative).

- **Male**: IBM-1 prefers Amharic (same Ethiopic script, shared surface forms);
  eflomal prefers Oromo (a Latin-script Cushitic neighbour). Contact/geography
  vs script/shared-characters.
- **Romani**: IBM-1 prefers Gujarati; eflomal prefers Hindi.
- **Tongan**: IBM-1 prefers Tagalog (Philippine, lexically closer in surface);
  eflomal prefers Maori (Polynesian, genetically closest).

A plausible reading: IBM-1's per-token likelihood rewards shared vocabulary and
shared script/characters, so it leans toward surface similarity; eflomal's
alignment model captures structural correspondence and can favour a
genetically/positionally closer language even across scripts. Which is a better
predictor of transfer is an empirical question.

## Why this matters, and the tie-in

The "closest source" is **aligner-dependent**, so alignment scores are a useful
but not decisive selection factor. Usefully, the many-to-one experiments will
adjudicate: each generates the withheld books from every source and reports the
best, so we can check **which aligner's closest source actually gives the best
downstream generation**. That turns this disagreement into a test rather than a
guess. All four candidate sources are kept in every experiment regardless, so
this analysis does not change the source sets; it sharpens interpretation.
