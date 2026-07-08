import pandas as pd
import pytest

from samileides.publish_nllb import (
    check_gate,
    default_repo_id,
    nllb_model_licence,
    target_token_for,
)


def test_model_licence_forces_nc():
    # pure by-sa data still cannot escape the NLLB base's NC clause
    assert nllb_model_licence(["by-sa", "by-sa"]) == "cc-by-nc-sa-4.0"
    assert nllb_model_licence(["by", "Public Domain"]) == "cc-by-nc-4.0"
    # all-Public-Domain data inherits the base licence itself
    assert nllb_model_licence(["Public Domain"]) == "cc-by-nc-4.0"
    # nc data adds nothing new
    assert nllb_model_licence(["by-nc", "by-sa"]) == "cc-by-nc-sa-4.0"


def test_target_token_mirrors_training():
    cfg = {"target": {"new_token": "ton_Latn"}, "existing_token": "ilo_Latn"}
    # unknown code: used as-is
    assert target_token_for(cfg, "scratch", {"eng_Latn"}) == "ton_Latn"
    # code NLLB already has: suffixed, exactly as train_nllb_m2o does
    assert target_token_for(cfg, "scratch", {"ton_Latn"}) == "ton_Latn_new"
    # existing init: the real NLLB token
    assert target_token_for(cfg, "existing", {"ilo_Latn"}) == "ilo_Latn"


def _rows(best):
    return pd.DataFrame([
        {"book": "RUT", "best_chrF3": best, "best_source": "sna"},
    ])


def test_gate_beats_strongest_source_copy():
    baselines = {"RUT": {"sna": 20.0, "tsn": 15.0}}
    ok, problems = check_gate(_rows(46.6), baselines)
    assert ok and not problems
    ok, problems = check_gate(_rows(19.0), baselines)
    assert not ok
    assert "RUT" in problems[0] and "sna" in problems[0]


def test_gate_missing_baseline_does_not_crash():
    ok, _ = check_gate(_rows(10.0), {})
    assert ok  # floor defaults to 0; absence of a floor never blocks silently


def test_repo_id_convention():
    assert default_repo_id("m2o_ton") == "DavidCBaines/ebible_m2o-nllb600m-ton"
    assert default_repo_id("m2o_control_ilo") == "DavidCBaines/ebible_m2o-nllb600m-control-ilo"
