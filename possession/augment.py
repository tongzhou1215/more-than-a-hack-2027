"""Make clean DFL tracking look like the competition's automatic tracking.

The noise model is measured from the competition ATD *inputs* only (never its labels): the ball
goes missing in long contiguous stretches -- median ~10 s, some over 3 minutes -- rather than
frame by frame, and about 20.5 of 22 players are visible on average.

Where the ball goes missing matters as much as for how long. By default the gaps fall at random.
With dead_gaps (see dead_gaps()), they follow the game the way the competition ATD's do: nearly the
whole stoppage when play is dead, short gaps now and then when it is live. possession/gap.py shows
why: the ATD misses the ball 90% of the time in the slowest tenth of players' speed and 10% in the
fastest, which is the shape of DFL's own dead-ball share (94% and 2%).
"""

import warnings
from collections import namedtuple

import numpy as np

from .common import A, D, H

OFF_PITCH_MARGIN = 1.0  # metres beyond a line before the simulated tracker loses the ball
REAPPEAR_MAX = 50  # frames: a hidden stoppage's ball may be back up to 2 s before the restart
SPEED_TAIL = 0.1   # dead_gaps reads the slowest and the fastest tenth of players' speed

DeadGaps = namedtuple("DeadGaps", "dead_hide live_hide live_gaps")
DeadGaps.__doc__ = """How the target loses its ball: the share of stoppages hidden (dead_hide), the
share of live frames hidden (live_hide) and the gap lengths in frames used in live play."""


def ball_run_lengths(match):
    """Empirical (visible_runs, gap_runs) in frames, from a match's own ball track."""
    vis_runs, gap_runs = [], []
    for start, stop in match["halves"]:
        seen = ~np.isnan(match["ball"][start:stop, 0])
        change = np.flatnonzero(seen[1:] != seen[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seen)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            (vis_runs if seen[a] else gap_runs).append(b - a)
    return np.array(vis_runs), np.array(gap_runs)


def _alternating_mask(n, on_runs, off_runs, rng):
    """True where kept. Alternates runs drawn from the two empirical distributions."""
    keep = np.ones(n, dtype=bool)
    t = -int(rng.integers(0, max(1, on_runs.mean())))  # random phase
    on = bool(rng.random() < on_runs.sum() / (on_runs.sum() + off_runs.sum()))
    while t < n:
        length = int(rng.choice(on_runs if on else off_runs))
        if not on:
            keep[max(t, 0):max(t + length, 0)] = False
        t += length
        on = not on
    return keep


def dead_gaps(target):
    """DeadGaps measured from the target's inputs only. dead_hide is the share of frames with the
    ball missing among the slowest SPEED_TAIL of players' speed (v_all), where play is nearly
    always dead -- 94% in DFL; live_hide the same among the fastest, nearly always live -- 98%;
    live_gaps the lengths of the target's gaps whose players move faster than the half's median.
    A target without such gaps gets live_hide 0."""
    from . import features  # features does not import this module; kept local to stay light
    f, names, mask = features.build(target)
    v = f[:, names.index("v_all")]
    seen = ~np.isnan(target["ball"][:, 0])
    ok = mask & ~np.isnan(v)
    lo, hi = np.quantile(v[ok], [SPEED_TAIL, 1 - SPEED_TAIL])
    live = []
    for start, stop in target["halves"]:
        s = seen[start:stop]
        bounds = np.concatenate(([0], np.flatnonzero(s[1:] != s[:-1]) + 1, [len(s)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a gap without any speed
                fast = np.nanmean(v[start + a:start + b]) >= 1.0
            if not s[a] and fast:
                live.append(b - a)
    live_hide = float(np.mean(~seen[ok & (v >= hi)])) if live else 0.0
    return DeadGaps(float(np.mean(~seen[ok & (v <= lo)])), live_hide, np.array(live, dtype=int))


def _dead_mask(labels, gaps, rng):
    """True where the ball stays visible. Each stoppage (a run of D) is hidden with probability
    gaps.dead_hide, from its first frame until 0..REAPPEAR_MAX frames before the restart. On top,
    an alternating process hides gaps.live_hide of the frames in runs drawn from gaps.live_gaps."""
    n = len(labels)
    keep = np.ones(n, dtype=bool)
    dead = labels == D
    bounds = np.concatenate(([0], np.flatnonzero(dead[1:] != dead[:-1]) + 1, [n]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if dead[a] and rng.random() < gaps.dead_hide:
            keep[a:max(a, b - int(rng.integers(0, REAPPEAR_MAX + 1)))] = False
    if gaps.live_hide > 0 and len(gaps.live_gaps):
        off = np.asarray(gaps.live_gaps)
        on_mean = off.mean() * (1 - gaps.live_hide) / gaps.live_hide
        on = np.maximum(1, rng.exponential(on_mean, 1024)).astype(int)
        keep &= _alternating_mask(n, on, off, rng)
    return keep


def corrupt(match, ball_runs, rng, player_drop=0.07, mean_drop_frames=75,
            pos_noise=0.15, ball_noise=0.3, dead_gaps=None):
    """A noisy copy of a match. ball_runs is (visible_runs, gap_runs) from the target ATD. With
    dead_gaps (a DeadGaps), the ball gaps follow the match's own dead and live play instead of
    falling at random, which needs the match's labels."""
    if dead_gaps is not None and match["labels"] is None:
        raise ValueError("dead-aware ball gaps need the match's labels to find its stoppages")
    home, away, ball = match["home"].copy(), match["away"].copy(), match["ball"].copy()
    vis_runs, gap_runs = ball_runs
    # DFL keeps tracking the ball after it leaves the pitch (20% of its ball frames are off it);
    # the competition ATD almost never does (4%). Off the pitch, the ball is simply lost.
    length, width = match["pitch"]
    with np.errstate(invalid="ignore"):
        off = (np.abs(ball[:, 0]) > length / 2 + OFF_PITCH_MARGIN) | \
              (np.abs(ball[:, 1]) > width / 2 + OFF_PITCH_MARGIN)
    ball[off] = np.nan
    for start, stop in match["halves"]:
        if dead_gaps is None:
            keep = _alternating_mask(stop - start, vis_runs, gap_runs, rng)
        else:
            keep = _dead_mask(match["labels"][start:stop], dead_gaps, rng)
        ball[start:stop][~keep] = np.nan

    # Players: each slot drops out in runs so ~player_drop of its frames are unseen.
    on_mean = mean_drop_frames * (1 - player_drop) / player_drop
    for arr in (home, away):
        T, P, _ = arr.shape
        for p in range(P):
            on = np.maximum(1, rng.exponential(on_mean, 64)).astype(int)
            off = np.maximum(1, rng.exponential(mean_drop_frames, 64)).astype(int)
            arr[~_alternating_mask(T, on, off, rng), p] = np.nan
        arr += rng.normal(0, pos_noise, arr.shape).astype(np.float32)
    ball += rng.normal(0, ball_noise, ball.shape).astype(np.float32)
    out = dict(match)
    out.update(home=home, away=away, ball=ball)
    return out


def swap_teams(match):
    """Home and away exchanged, labels H<->A. Orientation is re-derived in features."""
    out = dict(match)
    out["home"], out["away"] = match["away"], match["home"]
    if match["labels"] is not None:
        lab = match["labels"].copy()
        lab[match["labels"] == H] = A
        lab[match["labels"] == A] = H
        out["labels"] = lab
    return out
