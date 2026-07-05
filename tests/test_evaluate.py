import pytest

from samileides.evaluate import METRIC_NAMES, score

REFS = [
    "In the beginning God created the heavens and the earth.",
    "Now the earth was formless and void.",
    "And God said, Let there be light, and there was light.",
]
NOISE = ["zzz qqq xxx", "vvv www yyy", "rrr sss ttt"]

def test_identity_scores_100():
    from sacrebleu.metrics import BLEU, CHRF

    # identical hypothesis/reference must max out the local metrics
    for beta_order in ((3, 0), (3, 1), (3, 2)):
        chrf = CHRF(char_order=6, word_order=beta_order[1], beta=beta_order[0])
        assert chrf.corpus_score(REFS, [REFS]).score == pytest.approx(100.0)
    assert BLEU().corpus_score(REFS, [REFS]).score == pytest.approx(100.0)


def test_noise_scores_near_zero():
    from sacrebleu.metrics import CHRF

    chrf = CHRF(char_order=6, word_order=0, beta=3)
    assert chrf.corpus_score(NOISE, [REFS]).score < 5.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score(REFS[:2], REFS)


@pytest.mark.integration
def test_full_metric_set_including_spbleu():
    results = score(REFS, REFS)
    assert set(results) == set(METRIC_NAMES)
    assert results["chrF3"] == pytest.approx(100.0)
    assert results["spBLEU"] == pytest.approx(100.0)
