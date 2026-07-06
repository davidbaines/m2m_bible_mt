"""Re-score an already-generated run without running the model again.

    uv run python -m samileides.rescore --run checkpoints/ie_base

Reads the generated books from ``<run>/generated/*.txt`` and recomputes the
metrics table, including both the source-copy baseline and the stronger
best-other-language baseline (spec.md verification #6). Use this to enrich runs
generated before the other-language baseline existed; new runs already include
it. Overwrites ``metrics.csv`` and ``metrics.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .canon import book_of
from .config import ExperimentConfig
from .data_pipeline import prepare
from .generate import score_book


def read_generated_book(path: Path) -> tuple[list[str], list[str]]:
    vrefs, hyps = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        vref, _, hyp = line.partition("\t")
        vrefs.append(vref)
        hyps.append(hyp)
    return vrefs, hyps


def enrich(run_dir: Path) -> pd.DataFrame:
    cfg = ExperimentConfig.load(run_dir / "config.yaml")
    data = prepare(cfg)
    gen_dir = run_dir / "generated"

    rows = []
    for translation in data.holdouts:
        for path in sorted(gen_dir.glob(f"{translation}-*.txt")):
            book = path.stem.split("-", 1)[1]
            vrefs, hyps = read_generated_book(path)
            # keep only verses that are in the scored test set, in file order
            valid = set(
                data.test_pairs[data.test_pairs["translation"] == translation]["vref"]
            )
            pairs = [(v, h) for v, h in zip(vrefs, hyps) if v in valid]
            if not pairs:
                continue
            vrefs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
            row = score_book(
                translation=translation, book=book, vrefs=vrefs, hyps=hyps,
                data=data, truncated=0,
            )
            rows.append(row)
            print(f"{translation}/{book}: chrF3={row['chrF3']} "
                  f"copy={row['copy_chrF3']} other={row['other_chrF3']} "
                  f"[{row['other_lang']}]")

    table = pd.DataFrame(rows)
    table.to_csv(gen_dir / "metrics.csv", index=False)
    (gen_dir / "metrics.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    return table


def main() -> None:
    p = argparse.ArgumentParser(description="Re-score a generated run")
    p.add_argument("--run", required=True)
    table = enrich(Path(p.parse_args().run))
    print(f"\nRescored {len(table)} books.")


if __name__ == "__main__":
    main()
