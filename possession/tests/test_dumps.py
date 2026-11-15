"""Tests for possession/dumps.py: exam dumps and blends. Synthetic arrays only."""

import numpy as np
import pytest

from possession import dumps

T = 50


def _exams(seed):
    """Two fake exam copies: a ball with gaps, graded and ungraded labels, two halves."""
    out = {}
    for k, name in enumerate(("m1", "m2")):
        rng = np.random.default_rng(seed + k)
        ball = rng.normal(size=(T, 2)).astype(np.float32)
        ball[rng.random(T) < 0.3] = np.nan
        out[name] = {"ball": ball, "labels": rng.integers(0, 4, T).astype(np.int8),
                     "halves": [(0, 20), (25, T)]}
    return out


def _dump(folder, tag, seed, exam_seed=0):
    exams = _exams(exam_seed)
    rng = np.random.default_rng(seed)
    probas = {i: rng.dirichlet(np.ones(3), T) for i in exams}
    priors = {i: np.array([0.4, 0.4, 0.2]) for i in exams}
    dumps.save_dump(tag, exams, probas, priors, folder=str(folder))
    return exams, probas, dumps.load_dump(tag, folder=str(folder))


def test_dump_round_trip(tmp_path):
    exams, probas, d = _dump(tmp_path, "t", 1)
    assert list(d) == ["m1", "m2"]
    for i in exams:
        assert d[i]["proba"].dtype == np.float32
        assert np.allclose(d[i]["proba"], probas[i], atol=1e-7)
        assert np.array_equal(d[i]["labels"], exams[i]["labels"])
        assert np.array_equal(d[i]["seen"], ~np.isnan(exams[i]["ball"][:, 0]))
        assert d[i]["halves"].tolist() == [[0, 20], [25, T]]
        assert np.allclose(d[i]["prior"], [0.4, 0.4, 0.2])


def test_blend_is_a_weighted_average(tmp_path):
    _, _, a = _dump(tmp_path, "a", 1)
    _, _, b = _dump(tmp_path, "b", 2)
    out = dumps.blend(a, b, 0.25)
    for i in a:
        assert np.allclose(out[i]["proba"], 0.25 * a[i]["proba"] + 0.75 * b[i]["proba"], atol=1e-6)
        assert np.array_equal(out[i]["labels"], a[i]["labels"])
        assert np.array_equal(out[i]["halves"], a[i]["halves"])


def test_blend_refuses_runs_that_sat_other_exams(tmp_path):
    _, _, a = _dump(tmp_path, "a", 1)
    _, _, b = _dump(tmp_path, "b", 2)
    changed = b["m1"]["labels"].copy()
    changed[3] = (changed[3] + 1) % 4
    with pytest.raises(ValueError):
        dumps.blend(a, dict(b, m1=dict(b["m1"], labels=changed)), 0.5)
    # The answers do not depend on how the exam copy was corrupted; the ball mask does.
    with pytest.raises(ValueError):
        dumps.blend(a, dict(b, m2=dict(b["m2"], seen=~b["m2"]["seen"])), 0.5)
    with pytest.raises(ValueError):
        dumps.blend(a, {"m1": b["m1"]}, 0.5)
