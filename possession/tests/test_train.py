"""Tests for the training helpers in possession/train.py. Synthetic matches only."""

import numpy as np
import pytest

from possession import features, train
from possession.common import A, H, X

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")  # all-NaN slices are expected


def _labelled_match(T=600, seed=0):
    rng = np.random.default_rng(seed)
    home = np.cumsum(rng.normal(0, 0.1, (T, 3, 2)), axis=0).astype(np.float32) - 10
    away = np.cumsum(rng.normal(0, 0.1, (T, 3, 2)), axis=0).astype(np.float32) + 10
    ball = np.cumsum(rng.normal(0, 0.3, (T, 2)), axis=0).astype(np.float32)
    labels = np.where(np.arange(T) < T // 2, H, A).astype(np.int8)
    return {"home": home, "away": away, "ball": ball, "halves": [(0, T)], "labels": labels,
            "pitch": (105.0, 68.0), "name": "synthetic"}


def test_frame_table_with_extra_features_keeps_the_base_columns():
    m = _labelled_match()
    base, y0, names0 = train.frame_table([m], 1)
    more, y1, names1 = train.frame_table([m], 1, extra=1)
    assert names1[:len(names0)] == names0 and len(names1) == len(names0) + 25
    assert np.array_equal(more[:, :len(names0)], base, equal_nan=True)
    assert np.array_equal(y0, y1)


def test_training_views_copy_zero_is_what_train_copies_gives():
    m = _labelled_match()
    runs = (np.array([40]), np.array([20]))
    views = train.training_views(m, runs, 3, copies=2)
    plain = train.copies(m, runs, train.SEED + 3)
    assert len(views) == 4
    assert np.array_equal(views[0]["ball"], plain[0]["ball"], equal_nan=True)
    assert np.array_equal(views[1]["labels"], plain[1]["labels"])


def test_lightgbm_table_is_what_frame_table_gives():
    views = [_labelled_match(seed=s) for s in (0, 1)]
    views[1]["labels"][:50] = X
    built, names = train.build_views(views, extra=1)
    Xb, yb = train.lightgbm_table(built, 3)
    Xf, yf, names_f = train.frame_table(views, 3, extra=1)
    assert names == names_f
    assert np.array_equal(Xb, Xf, equal_nan=True) and np.array_equal(yb, yf)


def test_build_views_keeps_what_the_gru_needs():
    m = _labelled_match()
    m["labels"][:40] = X
    built, _ = train.build_views([m])
    f, y, halves, graded = built[0]
    f0, _, mask = features.build(m)
    assert np.array_equal(f, f0, equal_nan=True) and halves == m["halves"]
    assert np.array_equal(graded, mask & (m["labels"] != X))
    assert np.array_equal(y, m["labels"])
