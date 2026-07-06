# Local dev box: Ubuntu 24.04 + RTX 3090 (24 GB)

Verified working on 2026-07-06 for smoke runs and the sanity checks in
spec.md's verification plan. This is the "local Linux 3090 for dev/smoke runs"
box from the infra plan; real runs still target the ClearML-managed H100s.

## Environment

- GPU: NVIDIA GeForce RTX 3090, 24 GB, driver 580.126.09, CUDA 13.0,
  compute capability 8.6. bf16 is supported and used.
- Python 3.12, `uv` 0.11.
- `uv sync --extra train` installs the training stack (torch 2.12 + cu130,
  transformers 5.13, accelerate, clearml). The base install (no extra) is
  enough for data prep, tokeniser, evaluation and the pytest suite.

## Commands

```bash
uv sync --extra train                       # one-time; pulls the CUDA wheels
uv run pytest                                # 42 unit tests
uv run pytest -m integration                 # 6 tests, hit the HF Hub

# Overfit sanity check (spec.md verification #4): loss must fall near zero.
uv run python -m samileides.train \
    --config configs/experiments/smoke.yaml --overfit 100 --max-steps 800

# Tiny end-to-end run: train, then generate + score + sheets.
uv run python -m samileides.train    --config configs/experiments/smoke.yaml \
    --output-dir checkpoints/smoke
uv run python -m samileides.generate --run checkpoints/smoke

# Template mode: generate any book into any training language.
uv run python -m samileides.generate --run checkpoints/smoke \
    --book OBA --lang spa --out checkpoints/smoke/template
```

## Results observed (smoke config, 3 translations, ~1500 steps, 3.7M params)

- Overfit-100: eval_loss 0.006 (label smoothing disabled in overfit mode so
  the near-zero target is reachable) — the train loop learns.
- End-to-end: produced checkpoint, tokeniser, `engbsb-JON.txt`,
  `metrics.csv`/`.md`, and a populated sample sheet. Held-out Jonah scored
  chrF3 17.82 vs source-copy baseline 0.38 (tiny model, so output is
  repetitive, but the whole artefact chain works and clears the baseline).
- Template mode produced recognisable Spanish for Obadiah.

The pilot (transformer-big, 32k vocab, ~210M params) is an H100 job, not a
3090 job; the 3090 is only for these smoke and sanity runs.
