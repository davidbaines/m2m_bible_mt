import pandas as pd
import pytest

from samileides.tokenizer import load_tokenizer, train_tokenizer
from samileides.vref import (
    ENCODINGS,
    build_vref_source,
    encode_vref,
    extend_tokenizer,
    source_symbols,
)

# A vref list spanning: short and long book codes, a 3-digit chapter and a
# 3-digit verse (Psalm 119:176), single digits, and NT.
VREFS = ["GEN 1:1", "GEN 1:2", "GEN 3:1", "PSA 119:176", "MAT 5:1", "REV 22:21"]


def test_encode_vref_struct():
    assert encode_vref("GEN 1:1", "struct") == "<GEN> <c1> <v1>"
    assert encode_vref("PSA 119:176", "struct") == "<PSA> <c119> <v176>"


def test_encode_vref_vtok():
    assert encode_vref("GEN 1:1", "vtok") == "<GEN_1:1>"
    assert encode_vref("PSA 119:176", "vtok") == "<PSA_119:176>"


def test_encode_vref_text():
    assert encode_vref("GEN 1:1", "text") == "GEN 1:1"


def test_encode_vref_rejects_unknown():
    with pytest.raises(ValueError):
        encode_vref("GEN 1:1", "nope")


def test_build_vref_source_masks_to_availability():
    avail = pd.Series(["g", "", "g", "g", "g", "g"], index=VREFS)
    src = build_vref_source("struct", avail)
    # GEN 1:2 has no Greek source -> stays empty -> the pair is dropped later,
    # keeping the pair set identical to ie_base's.
    assert src["GEN 1:2"] == ""
    assert src["GEN 1:1"] == "<GEN> <c1> <v1>"
    assert list(src.index) == VREFS


def test_struct_symbols_cover_max_chapter_and_verse():
    syms = source_symbols("struct", VREFS)
    assert "<GEN>" in syms and "<PSA>" in syms and "<REV>" in syms
    # chapters run 1..119, verses 1..176 (from Psalm 119:176)
    assert "<c119>" in syms and "<v176>" in syms
    assert "<c1>" in syms and "<v1>" in syms
    # no gaps below the max
    assert "<c60>" in syms and "<v100>" in syms


def test_vtok_symbols_one_per_vref():
    syms = source_symbols("vtok", VREFS)
    assert syms == [f"<{v.replace(' ', '_')}>" for v in VREFS]
    assert len(syms) == len(VREFS)


def test_text_encoding_needs_no_symbols():
    assert source_symbols("text", VREFS) == []


def _base_tokenizer(tmp_path):
    corpus = (
        ["Im Anfang schuf Gott den Himmel und die Erde"] * 40
        + ["In the beginning God created the heavens and the earth"] * 40
        + [encode_vref(v, "text") for v in VREFS] * 5
    )
    model = train_tokenizer(corpus, tmp_path / "base", tags=["<2deu>", "<2eng>"],
                            vocab_size=400)
    return model


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_extended_tokenizer_symbols_are_atomic(tmp_path, encoding):
    base = _base_tokenizer(tmp_path)
    syms = source_symbols(encoding, VREFS)
    out = extend_tokenizer(base, syms, tmp_path / f"ext_{encoding}.model")
    sp = load_tokenizer(out)
    for vref in VREFS:
        src = f"<2deu> {encode_vref(vref, encoding)}"
        pieces = sp.encode(src, out_type=str)
        assert pieces[0] == "<2deu>", pieces
        # every special symbol of this encoding survives as one piece
        for sym in encode_vref(vref, encoding).split(" "):
            if encoding != "text":
                assert sym in pieces, (sym, pieces)
        # round-trip is lossless
        assert sp.decode(sp.encode(src)) == src


def test_extend_tokenizer_preserves_base_pieces(tmp_path):
    base = _base_tokenizer(tmp_path)
    base_sp = load_tokenizer(base)
    base_size = base_sp.get_piece_size()
    # A target sentence must segment identically before and after extension,
    # so target-side segmentation is shared across encodings.
    target = "In the beginning God created the heavens and the earth"
    before = base_sp.encode(target, out_type=str)

    out = extend_tokenizer(base, source_symbols("vtok", VREFS),
                           tmp_path / "ext.model")
    ext_sp = load_tokenizer(out)
    assert ext_sp.get_piece_size() == base_size + len(VREFS)
    assert ext_sp.encode(target, out_type=str) == before


def test_extend_tokenizer_idempotent_on_duplicate_symbols(tmp_path):
    base = _base_tokenizer(tmp_path)
    syms = source_symbols("vtok", VREFS)
    out1 = extend_tokenizer(base, syms, tmp_path / "e1.model")
    size1 = load_tokenizer(out1).get_piece_size()
    # feeding the already-added symbols again adds nothing
    out2 = extend_tokenizer(out1, syms, tmp_path / "e2.model")
    assert load_tokenizer(out2).get_piece_size() == size1


def test_leading_tags_ignores_numbered_book_and_verse_symbols():
    from samileides.train import _leading_tags

    # struct: only the language tag is a tag; <1CH> is a book symbol
    assert _leading_tags("<2eng> <1CH> <c10> <v10>") == ["<2eng>"]
    # vtok: <1CH_10:10> is a verse symbol, not a tag
    assert _leading_tags("<2deu> <1CH_10:10>") == ["<2deu>"]
    # text: plain vref, single tag
    assert _leading_tags("<2spa> 1CH 10:10") == ["<2spa>"]
    # a normal Greek-source line still yields the tag
    assert _leading_tags("<2hin> εν αρχη") == ["<2hin>"]
    # many-to-many: source + target tags both survive
    assert _leading_tags("<2eng> <1deu> text") == ["<2eng>", "<1deu>"]


def test_vref_pairs_match_greek_pairs(synthetic_verses):
    """The vref source must yield exactly the (vref, translation) pairs the
    Greek source does (spec-vref.md #2): same coverage mask, different src."""
    from samileides.greek import build_composite_source
    from samileides.preprocess import build_pairs
    from samileides.splits import build_splits, manifest_checksum

    language_of = {"alpha": "aaa", "beta": "bbb",
                   "grcbrent": "grc", "grc-tisch": "grc"}
    splits = build_splits(synthetic_verses, {"alpha": ["GEN"]}, valid_size=1, seed=0)
    greek = build_composite_source(synthetic_verses)
    vref_src = build_vref_source("struct", greek)

    greek_pairs = build_pairs(splits.train, synthetic_verses, greek, language_of)
    vref_pairs = build_pairs(splits.train, synthetic_verses, vref_src, language_of)

    assert manifest_checksum(greek_pairs) == manifest_checksum(vref_pairs)
    # but the source text differs: vref pairs carry the encoded reference
    row = vref_pairs[vref_pairs["vref"] == "EXO 1:1"].iloc[0]
    assert row["src"] == "<2aaa> <EXO> <c1> <v1>"
