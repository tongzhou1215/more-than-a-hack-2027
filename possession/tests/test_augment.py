"""Tests for possession/augment.py's dead-aware ball gaps. Synthetic matches only."""

import numpy as np
import pytest

from possession import augment
from possession.common import A, D, H

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")  # all-NaN slices are expected
RUNS = (np.array([300]), np.array([100]))  # ball_runs for the random mode


def _match(labels):
    """Players scattered round their halves, the ball on the centre spot and always seen."""
    T = len(labels)
    rng = np.random.default_rng(0)
    return {"home": (rng.normal(0, 5, (T, 11, 2)) - 10).astype(np.float32),
            "away": (rng.normal(0, 5, (T, 11, 2)) + 10).astype(np.float32),
            "ball": np.zeros((T, 2), np.float32), "halves": [(0, T)],
            "labels": np.asarray(labels, np.int8), "pitch": (105.0, 68.0), "name": "synthetic"}


def _seen(match, gaps, seed):
    return ~np.isnan(augment.corrupt(match, RUNS, np.random.default_rng(seed),
                                     dead_gaps=gaps)["ball"][:, 0])


def test_without_dead_gaps_the_corruption_is_unchanged():
    m = _match([H] * 2000)
    a = augment.corrupt(m, RUNS, np.random.default_rng(1))
    b = augment.corrupt(m, RUNS, np.random.default_rng(1), dead_gaps=None)
    for k in ("ball", "home", "away"):
        assert np.array_equal(a[k], b[k], equal_nan=True)


def test_dead_periods_are_hidden_until_just_before_the_restart():
    m = _match([H] * 1000 + [D] * 500 + [A] * 1000)
    seen = _seen(m, augment.DeadGaps(1.0, 0.0, np.array([50])), 2)
    assert seen[:1000].all() and seen[1500:].all()
    assert not seen[1000:1500 - augment.REAPPEAR_MAX].any()


def test_dead_hide_is_the_share_of_dead_periods_hidden():
    labels = np.array(([H] * 300 + [D] * 200) * 400)
    seen = _seen(_match(labels), augment.DeadGaps(0.5, 0.0, np.array([50])), 3)
    assert 0.4 < np.mean(~seen[np.arange(300, len(labels), 500)]) < 0.6
    assert seen[labels != D].all()


def test_live_hide_is_the_share_of_live_frames_hidden():
    seen = _seen(_match([H] * 200000), augment.DeadGaps(1.0, 0.1, np.array([50, 150])), 4)
    assert 0.08 < np.mean(~seen) < 0.12


def test_dead_gaps_are_measured_from_the_targets_inputs():
    # A target whose ball is lost exactly while its players stand still, so it has no fast gaps.
    T = 30000
    rng = np.random.default_rng(5)
    still = (np.arange(T) // 1500) % 2 == 1
    step = np.where(still, 0.0, 0.3)[:, None, None] * rng.normal(size=(T, 11, 2))
    target = {"home": (np.cumsum(step, axis=0) - 10).astype(np.float32),
              "away": (np.cumsum(step[:, ::-1], axis=0) + 10).astype(np.float32),
              "ball": np.where(still[:, None], np.nan, np.zeros((T, 2))).astype(np.float32),
              "halves": [(0, T)], "labels": None, "pitch": (105.0, 68.0), "name": "target"}
    g = augment.dead_gaps(target)
    assert g.dead_hide > 0.9
    assert g.live_hide == 0.0 and len(g.live_gaps) == 0


def test_dead_mode_needs_labels():
    m = dict(_match([H] * 100), labels=None)
    with pytest.raises(ValueError):
        augment.corrupt(m, RUNS, np.random.default_rng(6), dead_gaps=augment.DeadGaps(1, 0, []))
