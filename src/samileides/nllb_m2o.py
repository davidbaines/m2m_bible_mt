"""Many-to-one NLLB fine-tuning to an unknown target language (spec.md phase 7).

Teaches NLLB a target language it has never seen, from K source languages in the
same family that it does know. Training is New-Testament-only (the whole OT is
withheld), so we can test whether the model drafts withheld OT books (Ruth,
Jonah, Genesis 1) of a language it only saw in the NT.

The unknown target gets a new language token added to NLLB, whose embedding is
initialised one of three ways (spec.md decisions log): from its closest known
relative, from scratch, or from an existing NLLB token of the same script. A
control run uses a target NLLB already knows, adding the "existing" init.
"""

from __future__ import annotations

import pandas as pd
import torch

from .canon import NT_BOOKS
from .data import VREF_COLUMN
from .nllb import NllbDataset
from .preprocess import normalise


def add_target_token(tokenizer, model, new_token: str, init_from: str | None) -> int:
    """Add a new target language token and initialise its embedding.

    ``init_from`` is an existing token (a relative's or a same-script anchor's
    FLORES code) to copy the embedding from, or None for scratch (random). The
    input and output embeddings are tied in NLLB, so one copy covers both.
    Returns the new token id.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": [new_token]})
    model.resize_token_embeddings(len(tokenizer))
    new_id = tokenizer.convert_tokens_to_ids(new_token)
    if init_from is not None:
        src_id = tokenizer.convert_tokens_to_ids(init_from)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[new_id] = emb[src_id].clone()
    return new_id


def build_m2o_pairs(target_tid, sources, verses, target_new_token,
                    exclude_vrefs=None) -> pd.DataFrame:
    """Source-NT -> target-NT pairs. ``sources`` is a list of (tid, flores).

    Only New-Testament verses the target has are used (whole OT withheld from
    training). ``exclude_vrefs`` (the validation set) is held out of training.
    The target side is labelled with ``target_new_token``.
    """
    nt = set(NT_BOOKS)
    exclude = set(exclude_vrefs or ())
    idx = [v for v in verses.index
           if v.split(" ", 1)[0] in nt and verses.at[v, target_tid] and v not in exclude]
    rows = []
    for v in idx:
        tgt = normalise(verses.at[v, target_tid])
        for tid, flores in sources:
            s = verses.at[v, tid]
            if s:
                rows.append({"src_text": normalise(s), "src_code": flores,
                             "tgt_text": tgt, "tgt_code": target_new_token})
    return pd.DataFrame(rows, columns=["src_text", "src_code", "tgt_text", "tgt_code"])


def build_valid_pairs(target_tid, sources, verses, target_new_token, val_vrefs) -> pd.DataFrame:
    """One pair per validation vref, round-robin over the sources that have it.

    A single balanced generation set (250 verses) used as the early-stopping
    signal: chrF3 of the generated target against the reference, every N steps.
    """
    rows = []
    k = 0
    for v in val_vrefs:
        if v not in verses.index or not verses.at[v, target_tid]:
            continue
        avail = [(tid, fl) for tid, fl in sources if verses.at[v, tid]]
        if not avail:
            continue
        tid, flores = avail[k % len(avail)]
        k += 1
        rows.append({"src_text": normalise(verses.at[v, tid]), "src_code": flores,
                     "tgt_text": normalise(verses.at[v, target_tid]), "tgt_code": target_new_token})
    return pd.DataFrame(rows, columns=["src_text", "src_code", "tgt_text", "tgt_code"])


def test_vrefs(verses, spec) -> dict[str, list[str]]:
    """Resolve the generation spec to vrefs. ``spec`` has books and chapters.

    books: whole-book codes (e.g. RUT, JON). chapters: vref prefixes with a
    trailing colon (e.g. 'GEN 1:') to take a single chapter.
    """
    out = {}
    for book in spec.get("books", []):
        out[book] = [v for v in verses.index if v.split(" ", 1)[0] == book]
    for prefix in spec.get("chapters", []):
        out[prefix.strip()] = [v for v in verses.index if v.startswith(prefix)]
    return out


@torch.no_grad()
def generate_book(model, tokenizer, device, src_texts, src_flores, tgt_token_id,
                  beam, max_length, batch_size=16):
    tokenizer.src_lang = src_flores
    hyps = []
    for i in range(0, len(src_texts), batch_size):
        batch = tokenizer(src_texts[i:i + batch_size], return_tensors="pt",
                          padding=True, truncation=True, max_length=max_length).to(device)
        out = model.generate(**batch, forced_bos_token_id=tgt_token_id,
                            num_beams=beam, max_length=max_length)
        hyps.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return hyps
