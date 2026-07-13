"""Vref-only source construction (spec-vref.md).

The vref-source experiments replace the Greek source text with an encoding of
the verse reference itself, so a training pair looks like
``<2deu> <GEN> <c1> <v1>  ->  Im Anfang schuf Gott ...``. Three encodings are
compared on otherwise-identical configs:

- ``struct``: atomic book, chapter and verse tokens (``<GEN> <c1> <v1>``).
  Chapter/verse symbols are shared across books, so neighbouring verses get
  related keys. (The spec first sketched digits here; atomic ``<cN>``/``<vN>``
  tokens are used instead so no digit pieces need to be made user-defined,
  which would have changed target-side segmentation and broken the
  shared-tokeniser guarantee.)
- ``vtok``: one atomic symbol per verse (``<GEN_1:1>``); every key independent.
- ``text``: the literal vref string (``GEN 1:1``) through ordinary BPE.

All three restrict training to the vrefs the composite Greek source covers,
so the (vref, translation) pair set is identical to ie_base's and the
comparison isolates the sourcing change (spec-vref.md, "Data").
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

ENCODINGS = ("struct", "vtok", "text")


def parse_vref(vref: str) -> tuple[str, int, int]:
    """Split ``'GEN 1:1'`` into ``('GEN', 1, 1)``."""
    book, rest = vref.split(" ", 1)
    chapter, verse = rest.split(":", 1)
    return book, int(chapter), int(verse)


def encode_vref(vref: str, encoding: str) -> str:
    """The source-side string (before the language tag) for one vref."""
    if encoding == "struct":
        book, chapter, verse = parse_vref(vref)
        return f"<{book}> <c{chapter}> <v{verse}>"
    if encoding == "vtok":
        return f"<{vref.replace(' ', '_')}>"
    if encoding == "text":
        return vref
    raise ValueError(f"Unknown vref encoding {encoding!r} (want one of {ENCODINGS})")


def build_vref_source(encoding: str, availability: pd.Series) -> pd.Series:
    """A vref-indexed source Series of encoded vrefs.

    ``availability`` is the composite Greek source (or any vref-indexed
    Series); vrefs where it is empty stay empty here too, so ``build_pairs``
    drops exactly the pairs ie_base dropped and the pair sets stay identical.
    """
    values = [
        encode_vref(v, encoding) if text else ""
        for v, text in zip(availability.index, availability.to_numpy())
    ]
    return pd.Series(values, index=availability.index, name="source")


def struct_symbols(vrefs: Iterable[str]) -> list[str]:
    """Atomic symbols the ``struct`` encoding needs for these vrefs."""
    books: set[str] = set()
    max_chapter = 0
    max_verse = 0
    for v in vrefs:
        book, chapter, verse = parse_vref(v)
        books.add(book)
        max_chapter = max(max_chapter, chapter)
        max_verse = max(max_verse, verse)
    return (
        [f"<{b}>" for b in sorted(books)]
        + [f"<c{n}>" for n in range(1, max_chapter + 1)]
        + [f"<v{n}>" for n in range(1, max_verse + 1)]
    )


def vtok_symbols(vrefs: Iterable[str]) -> list[str]:
    """One atomic symbol per vref, in vref-list order."""
    return [encode_vref(v, "vtok") for v in vrefs]


def source_symbols(encoding: str, vrefs: Iterable[str]) -> list[str]:
    """The extra tokeniser symbols an encoding needs (``text`` needs none).

    Derived from the FULL vref list, never the training split only: held-out
    vrefs must be encodable at generation time.
    """
    if encoding == "struct":
        return struct_symbols(vrefs)
    if encoding == "vtok":
        return vtok_symbols(vrefs)
    if encoding == "text":
        return []
    raise ValueError(f"Unknown vref encoding {encoding!r} (want one of {ENCODINGS})")


def extend_tokenizer(base_model: Path, symbols: Sequence[str], out_model: Path) -> Path:
    """Append user-defined symbols to a trained SentencePiece model.

    The base model's pieces (and therefore target-side segmentation) are
    untouched; new symbols are appended at the end of the vocabulary as
    atomic USER_DEFINED pieces. This is how one shared base tokeniser serves
    all three encodings (spec-vref.md, "Tokeniser").
    """
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2

    proto = sp_pb2.ModelProto()
    proto.ParseFromString(Path(base_model).read_bytes())
    existing = {p.piece for p in proto.pieces}
    for symbol in symbols:
        if symbol in existing:
            continue
        piece = proto.pieces.add()
        piece.piece = symbol
        piece.score = 0.0
        piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_bytes(proto.SerializeToString())
    return out_model
