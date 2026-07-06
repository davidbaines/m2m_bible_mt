"""Single-family selection: all corpus languages of one language family.

For the Indo-European run (spec.md, "Single-family (Indo-European) run"): take
every eBible language whose ISO 639-3 code is listed in a curated family file
(``configs/families/<family>.csv``), keep the best-covered translation per
language, and force the holdout editions in. Unlike the diverse pilot there is
no diversity cap — the point is to saturate a single family so held-out
languages are surrounded by close relatives. Run:

    python -m samileides.family                      # Indo-European default
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .data import load_metadata, repo_root
from .selection import SelectionConfig, load_families, select_translations, write_selection

DEFAULT_FAMILY = "indo_european"
MIN_VERSES = 5000


def load_family_codes(family: str = DEFAULT_FAMILY) -> dict[str, str]:
    """Read the curated ISO 639-3 -> branch map for a family."""
    path = repo_root() / "configs" / "families" / f"{family}.csv"
    df = pd.read_csv(path, comment="#")
    return dict(zip(df["languageCode"], df["branch"]))


def load_holdout_ids(holdouts_file: str) -> list[str]:
    raw = yaml.safe_load((repo_root() / holdouts_file).read_text(encoding="utf-8"))
    return list(raw["holdouts"])


def build_family_selection(
    family: str = DEFAULT_FAMILY,
    holdouts_file: str = "configs/holdouts-ie.yaml",
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    meta = load_metadata() if metadata is None else metadata
    branch_of = load_family_codes(family)
    in_family = meta[meta["languageCode"].isin(branch_of)]

    holdout_ids = load_holdout_ids(holdouts_file)
    config = SelectionConfig(
        target_size=None,          # take every language in the family
        include=holdout_ids,       # force the exact holdout editions in
        min_verses=MIN_VERSES,
        one_per_language=True,     # best-covered translation per language
    )
    selection = select_translations(config, in_family, load_families())
    selection["branch"] = selection["languageCode"].map(branch_of)
    return selection


def main() -> None:
    selection = build_family_selection()
    # write_selection keeps the standard columns; branch is added alongside.
    out = write_selection(selection, "ie")
    # re-write including the branch column for readability
    cols = [
        "translationId", "languageCode", "languageNameInEnglish", "branch",
        "script", "OTverses", "NTverses", "DCverses", "totalVerses",
    ]
    selection[[c for c in cols if c in selection.columns]].to_csv(
        Path(out), index=False
    )
    n = selection["languageCode"].nunique()
    print(f"Wrote {out}: {len(selection)} translations, {n} languages")
    print(selection["branch"].value_counts().to_string())


if __name__ == "__main__":
    main()
