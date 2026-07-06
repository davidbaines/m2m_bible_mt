"""NLLB fine-tune data path (spec.md phase 7, pretrained comparison).

Unlike the from-scratch model, NLLB uses its own SentencePiece tokeniser and
FLORES-200 language tokens rather than our `<1xxx>`/`<2xxx>` tags. This module
maps our corpus language codes to FLORES codes, samples many-to-many pairs among
the languages NLLB knows (Koine Greek is not one of them, so it is excluded),
and builds NLLB-format input_ids/labels: ``[src_lang] tokens [eos]`` on the
source side, ``[tgt_lang] tokens [eos]`` on the target side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .data import VREF_COLUMN, repo_root
from .manytomany import _present_by_vref
from .preprocess import normalise


def load_flores_map() -> dict[str, str]:
    df = pd.read_csv(repo_root() / "configs" / "nllb_lang_map.csv", comment="#")
    return dict(zip(df["languageCode"], df["flores"]))


def build_nllb_pairs(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    verses: pd.DataFrame,
    language_of: dict[str, str],
    flores: dict[str, str],
    k: int = 4,
    seed: int = 13,
) -> pd.DataFrame:
    """Many-to-many pairs as raw text + FLORES codes (no Greek, no tags)."""
    present = _present_by_vref(train, valid)
    rng = np.random.default_rng(seed)
    rows = []
    for v, tgt in zip(train[VREF_COLUMN], train["translation"]):
        if language_of[tgt] not in flores:
            continue
        cands = [t for t in present[v]
                 if t != tgt and language_of[t] in flores]
        if not cands:
            continue
        n = min(k, len(cands))
        picks = [cands[i] for i in rng.choice(len(cands), size=n, replace=False)]
        tgt_text = normalise(verses.at[v, tgt])
        for src in picks:
            rows.append({
                "src_text": normalise(verses.at[v, src]),
                "src_code": flores[language_of[src]],
                "tgt_text": tgt_text,
                "tgt_code": flores[language_of[tgt]],
            })
    return pd.DataFrame(rows, columns=["src_text", "src_code", "tgt_text", "tgt_code"])


class NllbDataset(Dataset):
    """Tokenised NLLB pairs: [src_lang] src [eos] -> [tgt_lang] tgt [eos]."""

    def __init__(self, pairs: pd.DataFrame, tokenizer, max_len: int = 128):
        self.src_text = pairs["src_text"].tolist()
        self.tgt_text = pairs["tgt_text"].tolist()
        self.src_code = pairs["src_code"].tolist()
        self.tgt_code = pairs["tgt_code"].tolist()
        self.tok = tokenizer
        self.max_len = max_len
        self.eos = tokenizer.eos_token_id
        self._lang_id = {c: tokenizer.convert_tokens_to_ids(c)
                         for c in set(self.src_code) | set(self.tgt_code)}

    def __len__(self) -> int:
        return len(self.src_text)

    def _core(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids[: self.max_len - 2]

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        src = [self._lang_id[self.src_code[idx]]] + self._core(self.src_text[idx]) + [self.eos]
        tgt = [self._lang_id[self.tgt_code[idx]]] + self._core(self.tgt_text[idx]) + [self.eos]
        return {"input_ids": src, "labels": tgt}
