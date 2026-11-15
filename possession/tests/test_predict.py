"""Tests for possession/predict.py: the kickoff rule (hand-built Smart Tagging entries) and the
blend of LightGBM and the GRU (tiny models trained on a synthetic match)."""

import json

import lightgbm as lgb
import numpy as np
import pytest

from possession import features, predict
from possession.common import A, D, H, X

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")  # all-NaN slices are expected
DECODER ={"team_cost": 2.0, "dead_cost": 64.0, "min_len": 5, "prior_power": 0.0}


def _match(T=600, seed=0):
    rng = np.random.default_rng(seed)
    home = np.cumsum(rng.normal(0, 0.1, (T, 3, 2)), axis=0).astype(np.float32) - 10
    away = np.cumsum(rng.normal(0, 0.1, (T, 3, 2)), axis=0).astype(np.float32) + 10
    ball = np.cumsum(rng.normal(0, 0.3, (T, 2)), axis=0).astype(np.float32)
    labels = np.select([np.arange(T) < 200, np.arange(T) < 300], [H, D], A).astype(np.int8)
    return {"home": home, "away": away, "ball": ball, "halves": [(0, T)], "labels": labels,
            "pitch": (105.0, 68.0), "name": "synthetic"}


def _model_dir(folder, match, weight=None):
    """A tiny LightGBM, and with a weight a tiny GRU, trained on the synthetic match."""
    f, names, mask = features.build(match)
    keep = mask & (match["labels"] != X)
    params = dict(objective="multiclass", num_class=3, num_leaves=4, min_data_in_leaf=5,
                  verbose=-1, seed=0, deterministic=True, force_row_wise=True, num_threads=1)
    lgb.train(params, lgb.Dataset(f[keep], match["labels"][keep]),
              num_boost_round=3).save_model(str(folder / "lgbm.txt"))
    cfg = {"decoder": DECODER, "prior": [1 / 3] * 3, "extra_features": 0, "features": names}
    if weight is not None:
        from possession import seq
        cfg["lgbm_weight"] = weight
        model, mean, std = seq.fit([(f, match["labels"], match["halves"], keep)], seed=0,
                                   epochs=1, hidden=4, window=20, batch=4)
        seq.save(str(folder), model, mean, std, {"features": names})
    with open(folder / "decoder.json", "w") as handle:
        json.dump(cfg, handle)
    return f


def test_predict_without_a_gru_uses_lightgbm_alone(tmp_path):
    m = _match()
    f = _model_dir(tmp_path, m)
    codes, proba = predict.predict(m, str(tmp_path))
    assert np.array_equal(proba, lgb.Booster(model_file=str(tmp_path / "lgbm.txt")).predict(
        f, num_threads=1))
    assert len(codes) == len(f)


def test_predict_blends_lightgbm_and_the_gru(tmp_path):
    pytest.importorskip("torch")
    from possession import seq
    m = _match()
    f = _model_dir(tmp_path, m, weight=0.35)
    _, proba = predict.predict(m, str(tmp_path))
    lgb_p = lgb.Booster(model_file=str(tmp_path / "lgbm.txt")).predict(f, num_threads=1)
    model, mean, std, _ = seq.load(str(tmp_path))
    gru_p = seq.predict_match(model.double(), seq.scale(f, mean, std), m["halves"])
    assert proba.dtype == np.float64
    assert np.allclose(proba, 0.35 * lgb_p + 0.65 * gru_p, rtol=0, atol=1e-12)
    assert not np.allclose(proba, lgb_p, atol=1e-3)


def test_predict_refuses_a_gru_trained_on_other_features(tmp_path):
    pytest.importorskip("torch")
    from possession import seq
    m = _match()
    _model_dir(tmp_path, m, weight=0.35)
    model, mean, std, info = seq.load(str(tmp_path))
    seq.save(str(tmp_path), model, mean, std, {"features": info["features"][::-1]})
    with pytest.raises(AssertionError):
        predict.predict(m, str(tmp_path))


def _kickoff(half, start):
    return {"code": "SET PIECES", "start_frame": start, "end_frame": start + 110,
            "tags": {"Type": "Kick Off (Start)", "Half": half, "Team": "Barcelona"}}


def test_pre_kickoff_marks_frames_before_the_tagged_kickoff():
    mask = predict.pre_kickoff([_kickoff("2nd Half", 150)], [(0, 100), (120, 400)], 400)
    assert mask.nonzero()[0].tolist() == list(range(120, 150))


def test_pre_kickoff_ignores_a_kickoff_tagged_at_or_before_the_half_start():
    mask = predict.pre_kickoff([_kickoff("1st Half", 0)], [(10, 100), (120, 400)], 400)
    assert not mask.any()


def test_pre_kickoff_without_a_kickoff_tag_changes_nothing():
    other = {"code": "SET PIECES", "start_frame": 150, "end_frame": 200,
             "tags": {"Type": "Corner Kick", "Half": "2nd Half"}}
    assert not predict.pre_kickoff([other], [(0, 100), (120, 400)], 400).any()
