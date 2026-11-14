"""Tests for the helpers in possession/cv.py. Synthetic inputs only: no DFL or competition data."""

import numpy as np
import pytest

from possession import cv
from possession.common import X

H, A, D = 0, 1, 2
NAN = float("nan")


def _match(ball_x, labels=None):
    """One home player at (0, 0) and one away player at (10, 0); the ball at (ball_x[t], 0), NaN = unseen."""
    T = len(ball_x)
    home = np.zeros((T, 1, 2), dtype=np.float32)
    away = np.zeros((T, 1, 2), dtype=np.float32)
    away[:, 0, 0] = 10.0
    ball = np.zeros((T, 2), dtype=np.float32)
    ball[:, 0] = ball_x
    ball[np.isnan(ball[:, 0])] = np.nan
    labels = np.full(T, H, dtype=np.int8) if labels is None else np.asarray(labels, dtype=np.int8)
    return {"home": home, "away": away, "ball": ball, "labels": labels,
            "halves": [(0, T)], "pitch": (105.0, 68.0), "name": "synthetic"}


def test_folds_hold_out_each_match_once():
    assert cv.folds(["a", "b", "c"]) == [("a", ["b", "c"]), ("b", ["a", "c"]), ("c", ["a", "b"])]


def test_paired_summary_counts_wins():
    s = cv.paired_summary({"m1": 0.5, "m2": 0.4, "m3": 0.6}, {"m1": 0.4, "m2": 0.45, "m3": 0.5})
    assert s["n"] == 3 and s["wins"] == 2 and s["losses"] == 1
    assert abs(s["mean_delta"] - 0.05) < 1e-9


def test_paired_summary_tie_is_neither_win_nor_loss():
    s = cv.paired_summary({"m": 0.3}, {"m": 0.3})
    assert (s["n"], s["wins"], s["losses"]) == (1, 0, 0)
    assert s["mean_delta"] == 0.0


def test_paired_summary_uses_only_matches_in_both_runs():
    s = cv.paired_summary({"a": 0.5, "b": 0.9}, {"a": 0.4, "c": 0.1})
    assert (s["n"], s["wins"], s["losses"]) == (1, 1, 0)
    assert abs(s["mean_delta"] - 0.1) < 1e-9


def test_baseline_codes_nearest_team_and_dead_when_unseen():
    m = _match([1, 1, 1, 9, 9, 9, NAN, NAN])
    assert cv.baseline_codes(m, window=0).tolist() == [H, H, H, A, A, A, D, D]


def test_baseline_codes_vote_removes_a_one_frame_flicker():
    m = _match([1, 1, 1, 1, 9, 1, 1, 1, 1])
    assert cv.baseline_codes(m, window=0).tolist()[4] == A
    assert cv.baseline_codes(m, window=2).tolist() == [H] * 9


def test_baseline_codes_marks_ungraded_frames_x():
    m = _match([1, 1, 1, 1], labels=[H, H, X, X])
    assert cv.baseline_codes(m, window=0).tolist() == [H, H, X, X]


def test_mean_and_se_uses_the_sample_standard_deviation():
    mean, se = cv.mean_and_se([0.4, 0.6])
    assert abs(mean - 0.5) < 1e-12
    assert abs(se - 0.1) < 1e-12  # sd (ddof=1) 0.1414 / sqrt(2)


def test_mean_and_se_of_one_value_has_no_spread():
    assert cv.mean_and_se([0.7]) == (0.7, 0.0)


def test_exam_fingerprint_changes_when_the_exam_changes():
    assert cv.exam_fingerprint(_match([1, 1, NAN])) == cv.exam_fingerprint(_match([1, 1, NAN]))
    assert cv.exam_fingerprint(_match([1, 2, NAN])) != cv.exam_fingerprint(_match([1, 1, NAN]))


def _result(fingerprint, spell_f1):
    return {"config": {"exam_fingerprints": {"a": fingerprint}},
            "matches": {"a": {"model": {"spell_f1": spell_f1}}}}


def test_print_against_warns_only_when_the_exams_differ(capsys):
    cv.print_against(_result("same", 0.5), _result("same", 0.4), "old")
    assert "WARNING" not in capsys.readouterr().out
    cv.print_against(_result("new", 0.5), _result("old", 0.4), "old")
    assert "WARNING" in capsys.readouterr().out


def test_main_without_dfl_data_names_the_download_command(tmp_path):
    with pytest.raises(SystemExit) as stop:
        cv.main(["--dfl-dir", str(tmp_path), "--cv-dir", str(tmp_path)])
    assert "download_dfl" in str(stop.value.code)


def test_main_against_unknown_tag_lists_the_saved_tags(tmp_path):
    (tmp_path / "v1.json").write_text("{}")
    with pytest.raises(SystemExit) as stop:
        cv.main(["--against", "nope", "--cv-dir", str(tmp_path), "--dfl-dir", str(tmp_path)])
    message = str(stop.value.code)
    assert "nope" in message and "v1" in message
