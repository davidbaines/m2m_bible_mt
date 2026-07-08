import pandas as pd
import pytest

from samileides.licensing import model_licence_for
from samileides.nllb_m2o import target_token_for, usable
from samileides.publish_nllb import (
    NLLB_BASE_LICENCE,
    check_gate,
    default_repo_id,
    matrix_rows,
    source_floor,
)
from samileides.train_nllb_m2o import append_results


def test_base_model_licence_forces_nc():
    # pure by-sa data still cannot escape the NLLB base's NC clause
    assert model_licence_for(["by-sa"], base_model_licence=NLLB_BASE_LICENCE) == "cc-by-nc-sa-4.0"
    assert model_licence_for(["by", "Public Domain"], base_model_licence=NLLB_BASE_LICENCE) == "cc-by-nc-4.0"
    # all-Public-Domain data inherits the base licence itself
    assert model_licence_for(["Public Domain"], base_model_licence=NLLB_BASE_LICENCE) == "cc-by-nc-4.0"
    # nc data adds nothing new
    assert model_licence_for(["by-nc", "by-sa"], allow_nc=True,
                             base_model_licence=NLLB_BASE_LICENCE) == "cc-by-nc-sa-4.0"
    # and without a base model, behaviour is unchanged
    assert model_licence_for(["by-sa"]) == "cc-by-sa-4.0"
    assert model_licence_for(["Public Domain"]) == "cc0-1.0"


def test_target_token_mirrors_training():
    cfg = {"target": {"new_token": "ton_Latn"}, "existing_token": "ilo_Latn"}
    # unknown code: used as-is
    assert target_token_for(cfg, "scratch", {"eng_Latn"}) == "ton_Latn"
    # code NLLB already has: suffixed, exactly as train_nllb_m2o does
    assert target_token_for(cfg, "scratch", {"ton_Latn"}) == "ton_Latn_new"
    # existing init: the real NLLB token
    assert target_token_for(cfg, "existing", {"ilo_Latn"}) == "ilo_Latn"


def test_usable_filters_markers():
    assert usable("In the beginning")
    assert not usable("")
    assert not usable("<range>")


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


def test_gate_fails_without_a_baseline():
    # a book whose floor cannot be computed must FAIL the gate, not crash
    # (empty per-source dict) nor silently pass (book missing entirely)
    for baselines in ({"RUT": {}}, {}):
        ok, problems = check_gate(_rows(10.0), baselines)
        assert not ok
        assert "no source-copy baseline" in problems[0]


def test_source_floor():
    assert source_floor({"RUT": {"sna": 20.0, "tsn": 15.0}}, "RUT") == ("sna", 20.0)
    assert source_floor({"RUT": {}}, "RUT") is None
    assert source_floor({}, "RUT") is None


def test_matrix_rows_requires_unambiguous_lr(tmp_path):
    csv = tmp_path / "results.csv"
    pd.DataFrame([
        {"target": "rmc", "init": "scratch", "lr": 0.0003, "book": "RUT", "best_chrF3": 30.4},
        {"target": "rmc", "init": "scratch", "lr": 0.0001, "book": "RUT", "best_chrF3": 12.9},
    ]).to_csv(csv, index=False)
    with pytest.raises(SystemExit, match="several learning rates"):
        matrix_rows(csv, "rmc", "scratch")
    rows = matrix_rows(csv, "rmc", "scratch", lr=0.0003)
    assert len(rows) == 1 and rows.at[0, "best_chrF3"] == 30.4


def test_append_results_refuses_schema_mismatch(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text("target,init,book\nton,relative,RUT\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="misalign"):
        append_results(csv, [{"target": "ton", "init": "relative", "lr": 3e-4, "book": "RUT"}])
    # untouched on refusal, appends cleanly when the schema matches
    assert csv.read_text(encoding="utf-8").count("\n") == 2
    append_results(csv, [{"target": "ton", "init": "scratch", "book": "JON"}])
    assert "ton,scratch,JON" in csv.read_text(encoding="utf-8")


def test_repo_id_convention():
    assert default_repo_id("m2o_ton") == "DavidCBaines/ebible_m2o-nllb600m-ton"
    assert default_repo_id("m2o_control_ilo") == "DavidCBaines/ebible_m2o-nllb600m-control-ilo"
