"""Tests for possession/labels.py:controlled. Hand-built label sequences only."""

import numpy as np

from possession.common import A, D, H, X
from possession import labels as L

MIN = 50  # 2 s at 25 fps


def seq(runs):
    """[(label, length), ...] -> int8 array."""
    return np.concatenate([np.full(n, v, dtype=np.int8) for v, n in runs])


def test_short_sandwiched_spell_is_merged():
    out = L.controlled(seq([(H, 100), (A, 10), (H, 100)]), MIN)
    assert out.tolist() == seq([(H, 210)]).tolist()


def test_long_spell_is_a_change():
    lab = seq([(H, 100), (A, 60), (H, 100)])
    assert L.controlled(lab, MIN).tolist() == lab.tolist()


def test_short_touch_before_dead_ball_is_merged():
    out = L.controlled(seq([(H, 100), (A, 10), (D, 30)]), MIN)
    assert out.tolist() == seq([(H, 110), (D, 30)]).tolist()


def test_first_spell_after_restart_is_kept():
    lab = seq([(D, 30), (A, 10), (H, 100)])
    assert L.controlled(lab, MIN).tolist() == lab.tolist()


def test_chain_of_short_spells_stays_with_possessor():
    out = L.controlled(seq([(H, 100), (A, 10), (H, 5), (A, 10), (H, 100)]), MIN)
    assert out.tolist() == seq([(H, 225)]).tolist()


def test_chain_then_real_change():
    out = L.controlled(seq([(H, 100), (A, 10), (H, 5), (A, 100)]), MIN)
    assert out.tolist() == seq([(H, 115), (A, 100)]).tolist()


def test_spell_of_exactly_min_frames_takes_over():
    lab = seq([(H, 100), (A, 50), (H, 100)])
    assert L.controlled(lab, MIN).tolist() == lab.tolist()


def test_x_resets_the_possessor():
    lab = seq([(H, 100), (X, 1), (A, 10), (H, 100)])
    assert L.controlled(lab, MIN).tolist() == lab.tolist()


def test_input_is_not_mutated():
    lab = seq([(H, 100), (A, 10), (H, 100)])
    before = lab.copy()
    out = L.controlled(lab, MIN)
    assert lab.tolist() == before.tolist()
    assert out is not lab
