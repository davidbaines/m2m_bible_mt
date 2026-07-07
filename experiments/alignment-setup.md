# Alignment scoring for source selection

We score how alignable two languages are on their shared verses, as a factor for
choosing many-to-one source languages (spec.md phase 7). Two aligners are used
and compared: eflomal (the intended tool) and a compact in-repo IBM Model 1.

## eflomal (intended tool)

eflomal does not build against the system Python here (no `python3-dev` headers,
no sudo). The workaround is a uv-managed Python, which ships headers:

```bash
uv python install 3.11
uv venv --python 3.11 <eflomal-venv>
uv pip install --python <eflomal-venv>/bin/python numpy cython eflomal
```

`scripts/alignment/eflomal_score.py` runs under that venv: given two
whitespace-tokenised parallel text files it aligns them and prints a symmetric
**per-token alignment score** (mean of forward and reverse eflomal sentence
scores, normalised by token count). **Lower is more alignable.**

`scripts/alignment/run_eflomal.py` (run under the main project env; it shells
out to the eflomal venv) computes source-to-target and source-to-source scores
for every many-to-one experiment and writes `experiments/m2o-alignment-eflomal.csv`.
Set the eflomal venv path near the top of that script for your machine.

## IBM Model 1 (in-repo fallback)

`samileides.align_score` is a small pure-Python IBM-1 (no build needed): it
EM-trains lexical translation probabilities and reports the corpus mean
log-likelihood per target token. **Higher (less negative) is more alignable.**
`scripts/alignment/run_ibm1.py` writes `experiments/m2o-alignment.csv`. It is
coarser than eflomal (no positional model) but gives the same kind of signal.

## Reading the scores

Both produce a source-to-target vector (which source aligns best to the unknown
target) and a source-to-source matrix (how alignable the sources are to each
other). Compare the two aligners' "closest source" per target to see whether the
choice is robust to the method. Report alongside the same-script count for each
pair.
