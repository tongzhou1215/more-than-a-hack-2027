"""Make clean DFL tracking look like the competition's automatic tracking.

The noise model is measured from the competition ATD *inputs* only (never its labels): the ball
goes missing in long contiguous stretches -- median ~10 s, some over 3 minutes -- rather than
frame by frame, and about 20.5 of 22 players are visible on average.
"""

import numpy as np

from .common import A, H

OFF_PITCH_MARGIN = 1.0  # metres beyond a line before the simulated tracker loses the ball


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


def corrupt(match, ball_runs, rng, player_drop=0.07, mean_drop_frames=75,
            pos_noise=0.15, ball_noise=0.3):
    """A noisy copy of a match. ball_runs is (visible_runs, gap_runs) from the target ATD."""
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
        keep = _alternating_mask(stop - start, vis_runs, gap_runs, rng)
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
