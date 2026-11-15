"""Tests for the optional extra features in possession/features.py. Synthetic matches only."""

import numpy as np
import pytest

from possession import features

# features.py ignores RuntimeWarning on purpose (all-NaN slices are expected); pytest resets
# warning filters per test, so say it again here.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _walk_match(T=600, seed=0):
    """3 v 3 players and a ball wandering on a 105 x 68 pitch, one half, nothing missing."""
    rng = np.random.default_rng(seed)
    home = np.cumsum(rng.normal(0, 0.1, (T, 3, 2)), axis=0).astype(np.float32) - 10
    away = np.cumsum(rng.normal(0, 0.1, (T, 3, 2)), axis=0).astype(np.float32) + 10
    ball = np.cumsum(rng.normal(0, 0.3, (T, 2)), axis=0).astype(np.float32)
    return {"home": home, "away": away, "ball": ball, "halves": [(0, T)],
            "labels": None, "pitch": (105.0, 68.0), "name": "synthetic"}


def test_extra_features_keep_the_base_columns():
    m = _walk_match()
    base, base_names, base_mask = features.build(m)
    more, more_names, more_mask = features.build(m, extra=True)
    assert more_names[:len(base_names)] == base_names
    assert np.array_equal(more[:, :len(base_names)], base, equal_nan=True)
    assert len(more_names) == len(base_names) + 25
    assert np.array_equal(base_mask, more_mask)


def test_extra_level_two_keeps_level_one_and_adds_more_sided_windows():
    m = _walk_match()
    one, one_names, _ = features.build(m, extra=1)
    two, two_names, _ = features.build(m, extra=2)
    assert two_names[:len(one_names)] == one_names
    assert np.array_equal(two[:, :len(one_names)], one, equal_nan=True)
    assert len(two_names) == len(one_names) + 36


def test_named_groups_reproduce_level_one():
    m = _walk_match()
    one, one_names, _ = features.build(m, extra=1)
    named, named_names, _ = features.build(m, extra="long,sided,close")
    assert named_names == one_names
    assert np.array_equal(named, one, equal_nan=True)


def test_ball_direction_group_adds_five_columns_after_level_one():
    m = _walk_match()
    one, one_names, _ = features.build(m, extra=1)
    vx, vx_names, _ = features.build(m, extra="long,sided,close,ballvx")
    assert vx_names[:len(one_names)] == one_names
    assert np.array_equal(vx[:, :len(one_names)], one, equal_nan=True)
    assert vx_names[len(one_names):] == ["b_vx", "b_vx_m25", "b_vx_m75", "b_vx_past2", "b_vx_next2"]


def test_ball_moving_towards_the_away_goal_has_positive_forward_speed():
    T = 300
    m = _walk_match(T)
    m["home"][:, :, 0] -= 20.0   # home stays in its own half (it defends -x)
    m["away"][:, :, 0] += 20.0
    m["ball"][:, 0] = np.linspace(-30, 30, T)  # the ball travels towards +x, the away goal
    m["ball"][:, 1] = 0.0
    f, names, _ = features.build(m, extra="ballvx")
    assert np.nanmedian(f[:, names.index("b_vx")]) > 0


def test_orientation_follows_the_teams_not_the_halfway_line():
    # A dominant home team camps past the halfway line (mean x +5) while defending -x: the away
    # team sits deeper still (+15). That half must not be turned round. DFL J03WR9's first half.
    T = 100
    home = np.zeros((T, 3, 2), dtype=np.float32)
    away = np.zeros((T, 3, 2), dtype=np.float32)
    home[..., 0], away[..., 0] = 5.0, 15.0
    ball = np.zeros((T, 2), dtype=np.float32)
    h, a, _ = features._orient(home, away, ball)
    assert np.all(h[..., 0] == 5.0) and np.all(a[..., 0] == 15.0)
    # Mirrored: home defends +x, so the half is turned and home ends up below away again.
    h, a, _ = features._orient(-home, -away, ball)
    assert np.all(h[..., 0] == 5.0) and np.all(a[..., 0] == 15.0)


TOUCH_COLUMNS = ("since_h", "since_a", "until_h", "until_a", "last", "next")


def _touch_match(T=600):
    """3 v 3 players far from a still ball on the centre spot; a home player next to the ball on
    frames 100-109 and an away player next to it on frames 300-309."""
    ys = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    home = np.zeros((T, 3, 2), dtype=np.float32)
    home[:, :, 0], home[:, :, 1] = -30.0, ys
    away = np.zeros((T, 3, 2), dtype=np.float32)
    away[:, :, 0], away[:, :, 1] = 30.0, ys
    home[100:110, 0] = (0.5, 0.0)
    away[300:310, 0] = (0.5, 0.0)
    return {"home": home, "away": away, "ball": np.zeros((T, 2), dtype=np.float32),
            "halves": [(0, T)], "labels": None, "pitch": (105.0, 68.0), "name": "touch"}


def _touch(m, r):
    f, names, _ = features.build(m, extra="touch")
    return {k: f[:, names.index("touch{}_{}".format(r, k))] for k in TOUCH_COLUMNS}


def test_touch_group_adds_twelve_columns_after_level_one():
    m = _walk_match()
    one, one_names, _ = features.build(m, extra=1)
    t, t_names, _ = features.build(m, extra="long,sided,close,touch")
    assert t_names[:len(one_names)] == one_names
    assert np.array_equal(t[:, :len(one_names)], one, equal_nan=True)
    assert t_names[len(one_names):] == ["touch{}_{}".format(r, k)
                                        for r in ("1", "2") for k in TOUCH_COLUMNS]


@pytest.mark.parametrize("r", ["1", "2"])
def test_touch_cues_follow_who_had_the_ball(r):
    c = _touch(_touch_match(), r)
    cap = features.CAP
    assert (c["since_h"][105], c["since_h"][150], c["since_h"][50]) == (0, 41, cap)
    assert (c["until_h"][50], c["until_a"][150], c["until_a"][400]) == (50, 150, cap)
    assert c["since_a"][150] == cap
    assert (c["last"][50], c["last"][150], c["last"][350]) == (0, 1, -1)
    assert (c["next"][50], c["next"][150], c["next"][350]) == (1, -1, 0)


def test_touch_needs_the_ball_seen():
    m = _touch_match()
    m["ball"][100:110] = np.nan  # the home player stands where the ball would be, unseen
    c = _touch(m, "1")
    assert c["since_h"][150] == features.CAP
    assert c["last"][150] == 0


def test_touch_needs_no_rival_close():
    m = _touch_match()
    m["away"][100:110, 1] = (-0.8, 0.0)  # contested: an away player is within 1 m as well
    for r in ("1", "2"):
        c = _touch(m, r)
        assert c["since_h"][150] == features.CAP
        assert c["since_a"][150] == features.CAP


def test_touch_counts_when_the_rival_team_is_unseen():
    m = _touch_match()
    m["away"][100:110] = np.nan  # no away player tracked on those frames
    assert _touch(m, "1")["since_h"][150] == 41


def test_touch_cues_reset_at_halftime():
    m = _touch_match()
    m["halves"] = [(0, 300), (300, 600)]
    m["away"][300:310, 0] = (30.0, -10.0)  # no away touch at all
    m["home"][400:410, 0] = (0.5, 0.0)     # a second home touch, in the second half
    c = _touch(m, "1")
    assert c["since_h"][310] == features.CAP  # the first-half touch does not carry over
    assert c["until_h"][250] == features.CAP  # nor does the second-half one reach back
    assert (c["until_h"][50], c["since_h"][450]) == (50, 41)


def test_touch_cues_swap_with_the_teams():
    m = _touch_match()
    swapped = dict(m, home=m["away"], away=m["home"])
    for r in ("1", "2"):
        c, s = _touch(m, r), _touch(swapped, r)
        for mine, theirs in (("since_h", "since_a"), ("until_h", "until_a")):
            assert np.array_equal(c[mine], s[theirs])
            assert np.array_equal(c[theirs], s[mine])
        assert np.array_equal(c["last"], -s["last"])
        assert np.array_equal(c["next"], -s["next"])


def test_approach_speed_of_players_running_at_the_ball():
    T = 200
    t = np.arange(T) / 25.0
    ball = np.zeros((T, 2), dtype=np.float32)
    pos = np.zeros((T, 3, 2), dtype=np.float32)
    pos[:, :, 0] = (30.0 - 2.0 * t)[:, None]  # all three run at 2 m/s straight at the ball
    pos[:, 1, 1] = 1.0
    pos[:, 2, 1] = -1.0
    closing = features._approach(pos, ball)
    assert abs(np.nanmedian(closing) - 2.0) < 0.01
