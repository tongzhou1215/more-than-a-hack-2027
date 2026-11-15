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


def test_exam_copy_changes_only_the_labels():
    labels = [H] * 100 + [A] * 10 + [H] * 290  # a 0.4 s away touch inside a home possession
    m = _match([1.0] * 400, labels=labels)
    ball_runs = (np.array([40]), np.array([20]))  # visible 40 frames, then lost for 20
    raw = cv.exam_copy(m, ball_runs, 0)
    read = cv.exam_copy(m, ball_runs, 0, exam_frames=50)
    for key in ("home", "away", "ball"):
        assert np.array_equal(raw[key], read[key], equal_nan=True)
    assert raw["labels"].tolist() == labels
    assert read["labels"].tolist() == [H] * 400


def _probas(classes):
    """Near one-hot (T, 3) probabilities for a list of class codes (0 H, 1 A, 2 D)."""
    p = np.full((len(classes), 3), 0.01)
    p[np.arange(len(classes)), classes] = 0.98
    return p


def test_parse_decoder_reads_team_dead_and_min_len():
    assert cv.parse_decoder("1,64,25") == {"team_cost": 1.0, "dead_cost": 64.0, "min_len": 25,
                                           "prior_power": 0.0}


def test_parse_assignment_types_the_value():
    assert cv.parse_assignment("num_leaves=127") == ("num_leaves", 127)
    assert cv.parse_assignment("learning_rate=0.03") == ("learning_rate", 0.03)
    assert cv.parse_assignment("extra_trees=true") == ("extra_trees", True)
    assert cv.parse_assignment("boosting=dart") == ("boosting", "dart")


def test_training_views_copy_zero_is_the_original_and_more_copies_differ():
    m = _match([1.0] * 400, labels=[H] * 200 + [A] * 200)
    ball_runs = (np.array([40]), np.array([20]))
    one = cv.training_views(m, ball_runs, 0, copies=1)
    two = cv.training_views(m, ball_runs, 0, copies=2)
    assert len(one) == 2 and len(two) == 4
    assert np.array_equal(two[0]["ball"], one[0]["ball"], equal_nan=True)
    assert np.array_equal(two[1]["labels"], one[1]["labels"])
    assert not np.array_equal(two[2]["ball"], two[0]["ball"], equal_nan=True)


def test_parse_decoder_reads_an_optional_dead_ball_minimum():
    assert cv.parse_decoder("2,64,75,125") == {"team_cost": 2.0, "dead_cost": 64.0, "min_len": 75,
                                               "min_len_dead": 125, "prior_power": 0.0}


def test_decode_match_uses_its_own_minimum_for_dead_ball():
    labels = [H] * 100 + [D] * 40 + [H] * 100
    exam = _match([1.0] * 240, labels=labels)
    flat = np.full(3, 1 / 3)
    base = {"team_cost": 0.0, "dead_cost": 0.0, "min_len": 10, "prior_power": 0.0}
    short = cv.train.decode_match(_probas(labels), exam, dict(base, min_len_dead=10), flat)
    long = cv.train.decode_match(_probas(labels), exam, dict(base, min_len_dead=50), flat)
    assert int(np.sum(short == D)) == 40
    assert int(np.sum(long == D)) == 50


def test_parse_number_lists():
    assert cv.parse_floats("1,2.5") == [1.0, 2.5]
    assert cv.parse_ints("75,100") == [75, 100]


def test_grid_settings_cover_every_given_value_in_order():
    grid = {"team_cost": (1.0,), "dead_cost": (32.0, 64.0), "min_len": (100,)}
    assert cv.grid_settings(grid) == [
        {"team_cost": 1.0, "dead_cost": 32.0, "min_len": 100, "prior_power": 0.0},
        {"team_cost": 1.0, "dead_cost": 64.0, "min_len": 100, "prior_power": 0.0}]


def test_score_decoder_is_perfect_on_clean_probabilities():
    labels = [H] * 100 + [A] * 100
    exam = _match([1.0] * 200, labels=labels)
    decoder = {"team_cost": 2.0, "dead_cost": 32.0, "min_len": 10, "prior_power": 0.0}
    scores = cv.score_decoder({"m": exam}, {"m": _probas(labels)}, {"m": np.full(3, 1 / 3)}, decoder)
    assert scores["m"]["spell_f1"] == 1.0


def test_evaluate_grid_puts_the_better_setting_first():
    exam = _match([1.0] * 200, labels=[H] * 200)
    flicker = [H] * 100 + [A] * 5 + [H] * 95  # a 5-frame false alarm inside one possession
    jumpy = {"team_cost": 0.0, "dead_cost": 32.0, "min_len": 1, "prior_power": 0.0}
    steady = {"team_cost": 0.0, "dead_cost": 32.0, "min_len": 25, "prior_power": 0.0}
    rows = cv.evaluate_grid({"m": exam}, {"m": _probas(flicker)}, {"m": np.full(3, 1 / 3)},
                            [jumpy, steady])
    assert [r[1] for r in rows] == [steady, jumpy]
    assert rows[0][0] == 1.0 and rows[1][0] < 1.0


def test_evaluate_grid_keeps_grid_order_on_ties():
    exam = _match([1.0] * 200, labels=[H] * 200)
    a = {"team_cost": 1.0, "dead_cost": 16.0, "min_len": 25, "prior_power": 0.0}
    b = {"team_cost": 1.0, "dead_cost": 64.0, "min_len": 25, "prior_power": 0.0}  # no dead ball here
    rows = cv.evaluate_grid({"m": exam}, {"m": _probas([H] * 200)}, {"m": np.full(3, 1 / 3)}, [a, b])
    assert [r[1] for r in rows] == [a, b]


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
