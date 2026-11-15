"""Per-frame features from the common match representation.

Everything is permutation invariant over player slots (the target match has tracklets, not
players) and computed inside each half, so nothing leaks across halftime. Pitch orientation is
normalised per half so the home team always defends the negative-x goal; together with the
home/away swap augmentation this makes the model team-symmetric.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected here

MAX_PLAYER_SPEED = 12.0  # m/s; faster than this between two frames is an identity switch
SPEED_LAG = 5            # frames between the two positions a speed is taken from
CONTEXT = (25, 75, 250)  # half-widths in frames for centred rolling means (1 s, 3 s, 10 s)
CAP = 750                # frames; "time since ball seen" saturates at 30 s
EXTRA_CONTEXT = 500      # extra features: +/- 20 s centred means
SIDED = 125              # extra features: 5 s past-only and future-only means
CLOSING_PLAYERS = 3      # extra features: how many of a team's nearest players close in
TOUCH_RADII = (1.0, 2.0)  # m; touch features: a player this close to the seen ball, no rival

# Optional feature groups, appended in this order after the 84 base features.
EXTRA_GROUPS = ("long", "sided", "close", "sided2", "ballvx", "touch")
EXTRA_LEVELS = {0: (), 1: ("long", "sided", "close"), 2: ("long", "sided", "close", "sided2")}


def extra_groups(extra):
    """The groups an `extra` argument asks for: a level (0, 1 or True, 2), or group names
    separated by commas, e.g. "long,sided,close,ballvx"."""
    if isinstance(extra, str):
        groups = {g.strip() for g in extra.split(",") if g.strip()}
        unknown = groups - set(EXTRA_GROUPS)
        if unknown:
            raise ValueError("unknown extra feature groups: {}".format(sorted(unknown)))
        return groups
    return set(EXTRA_LEVELS[int(extra)])


def _orient(home, away, ball):
    """Flip x and y so home defends -x: afterwards home's mean x over the half lies below away's.

    The two teams are compared with each other, not home with the halfway line: a dominant team
    can spend a whole half past the halfway line while the other team sits deeper still (DFL
    J03WR9's first half, which the halfway-line rule turned the wrong way round).
    """
    if np.nanmean(home[..., 0]) > np.nanmean(away[..., 0]):
        return home * -1, away * -1, ball * -1
    return home, away, ball


def _interp_ball(ball):
    """Linear interpolation over gaps, held flat at the ends. Plus gap bookkeeping."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    idx = np.arange(n)
    out = np.zeros_like(ball)
    if seen.sum() < 2:
        return out, seen, np.full(n, CAP, float), np.full(n, CAP, float), np.full(n, CAP, float)
    for k in range(2):
        out[:, k] = np.interp(idx, idx[seen], ball[seen, k])
    last = np.maximum.accumulate(np.where(seen, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(seen, idx, 10 ** 9)[::-1])[::-1]
    since = np.minimum(idx - last, CAP).astype(float)
    until = np.minimum(nxt - idx, CAP).astype(float)
    gap = np.minimum(since + until, CAP)
    gap[seen] = 0
    return out, seen, since, until, gap


def _speed(pos):
    """Per-slot speed in m/s from positions SPEED_LAG frames apart, NaN where unreliable."""
    v = np.full(pos.shape[:-1], np.nan, dtype=np.float32)
    d = np.linalg.norm(pos[SPEED_LAG:] - pos[:-SPEED_LAG], axis=-1) * (25.0 / SPEED_LAG)
    d[d > MAX_PLAYER_SPEED] = np.nan
    v[SPEED_LAG // 2: SPEED_LAG // 2 + len(d)] = d
    return v


def _approach(pos, ball, k=CLOSING_PLAYERS):
    """Mean speed (m/s) at which a team's k players nearest the ball close in on it, (T,).

    Positive = running at the ball (pressing or going to receive), negative = moving away.
    Measured as the change of player-to-ball distance over SPEED_LAG frames, so the ball's own
    motion counts too; implausible jumps (identity switches) are dropped.
    """
    T, P, _ = pos.shape
    out = np.full(T, np.nan)
    if P == 0 or T <= SPEED_LAG:
        return out
    dist = np.linalg.norm(pos - ball[:, None, :], axis=2)
    closing = np.full((T, P), np.nan)
    closing[SPEED_LAG // 2: SPEED_LAG // 2 + T - SPEED_LAG] = \
        (dist[:-SPEED_LAG] - dist[SPEED_LAG:]) * (25.0 / SPEED_LAG)
    closing[np.abs(closing) > MAX_PLAYER_SPEED] = np.nan
    order = np.argsort(np.where(np.isnan(dist), np.inf, dist), axis=1)[:, :k]
    return np.nanmean(np.take_along_axis(closing, order, axis=1), axis=1)


def _rolling_mean(x, w):
    """Centred NaN-aware mean over +/- w frames, x shape (T,) or (T, F)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    ok = ~np.isnan(x)
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(np.where(ok, x, 0), 0)])
    n = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(ok, 0)])
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / (n[hi] - n[lo])
    return out[:, 0] if squeeze else out


def _shifted_mean(x, w, forward):
    """Mean over the w frames strictly after (forward) or before the current one."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    c = np.concatenate([[0], np.cumsum(np.where(ok, x, 0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(x))
    if forward:
        lo, hi = np.minimum(len(x), i + 1), np.minimum(len(x), i + 1 + w)
    else:
        lo, hi = np.maximum(0, i - w), i
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c[hi] - c[lo]) / (n[hi] - n[lo])


def _frames_since(event):
    """Frames since the latest True at or before each frame, -1 where there is none yet."""
    idx = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, idx, -1))
    return np.where(last >= 0, idx - last, -1)


def _whose(h, a):
    """+1 where home's event is the nearer one in time, -1 where away's is, 0 where neither team
    has one. h, a: frame distances from _frames_since, -1 for none."""
    h = np.where(h < 0, np.inf, h)
    a = np.where(a < 0, np.inf, a)
    return np.where(h < a, 1.0, np.where(a < h, -1.0, 0.0))


def _half_features(home, away, ball, pitch, extra=False):
    length, width = pitch
    home, away, ball = _orient(home, away, ball)
    bi, seen, since, until, gap = _interp_ball(ball)

    f = {}
    f["ball_seen"] = seen.astype(float)
    f["ball_since"] = since
    f["ball_until"] = until
    f["ball_gap"] = gap
    f["bx"], f["by"] = bi[:, 0], bi[:, 1]
    f["b_touch"] = width / 2 - np.abs(bi[:, 1])
    f["b_goal"] = length / 2 - np.abs(bi[:, 0])
    f["b_out"] = ((f["b_touch"] < 0) | (f["b_goal"] < 0)).astype(float)

    bv = np.full(len(bi), np.nan)
    both = seen[SPEED_LAG:] & seen[:-SPEED_LAG]
    sp = np.linalg.norm(bi[SPEED_LAG:] - bi[:-SPEED_LAG], axis=1) * (25.0 / SPEED_LAG)
    sp[~both] = np.nan
    bv[SPEED_LAG // 2: SPEED_LAG // 2 + len(sp)] = sp
    f["b_speed"] = bv

    for side, pos in (("h", home), ("a", away)):
        d = np.linalg.norm(pos - bi[:, None, :], axis=2)
        ds = np.sort(d, axis=1)  # NaNs sort last
        f["d1_" + side] = ds[:, 0]
        f["d2_" + side] = ds[:, 1]
        f["n5_" + side] = np.sum(d < 5, axis=1).astype(float)
        f["n15_" + side] = np.sum(d < 15, axis=1).astype(float)
        f["cx_" + side] = np.nanmean(pos[..., 0], axis=1)
        f["cy_" + side] = np.nanmean(pos[..., 1], axis=1)
        f["sx_" + side] = np.nanstd(pos[..., 0], axis=1)
        f["sy_" + side] = np.nanstd(pos[..., 1], axis=1)
        f["xmin_" + side] = np.nanmin(pos[..., 0], axis=1)
        f["xmax_" + side] = np.nanmax(pos[..., 0], axis=1)
        f["nvis_" + side] = np.sum(~np.isnan(pos[..., 0]), axis=1).astype(float)
        spd = _speed(pos)
        f["v_" + side] = np.nanmean(spd, axis=1)
        f["vmax_" + side] = np.nanmax(spd, axis=1)
        cx = f["cx_" + side]
        f["cvx_" + side] = _shifted_mean(cx, 25, True) - _shifted_mean(cx, 25, False)

    # Speeds are divided by their median over the half. The competition ATD measures speeds
    # about 30% lower than DFL (median 1.5 vs 2.1 m/s for all players), and "everyone slowed
    # down" is the strongest dead-ball cue, so raw speeds would read as dead ball on the target.
    # Only the relative change within a match is comparable across tracking providers.
    v_ref = np.nanmedian(np.concatenate([f["v_h"], f["v_a"]]))
    for k in ("v_h", "v_a", "vmax_h", "vmax_a"):
        f[k] = f[k] / v_ref
    f["b_speed"] = f["b_speed"] / np.nanmedian(f["b_speed"])

    f["d1_diff"] = f["d1_h"] - f["d1_a"]
    f["n5_diff"] = f["n5_h"] - f["n5_a"]
    f["v_all"] = np.nanmean(np.concatenate([f["v_h"][:, None], f["v_a"][:, None]], 1), 1)
    f["cx_diff"] = f["cx_h"] - f["cx_a"]
    f["b_rel_h"] = f["bx"] - f["cx_h"]
    f["b_rel_a"] = f["bx"] - f["cx_a"]

    # Temporal context: who has been nearer the ball, how fast everyone is moving, how long
    # the ball has been missing -- averaged over 1 s, 3 s and 10 s, and the change across now.
    base = ["d1_diff", "n5_diff", "v_all", "ball_seen", "b_speed", "cx_diff", "b_out",
            "cvx_h", "cvx_a", "b_goal", "b_touch"]
    near = (f["d1_h"] < f["d1_a"]).astype(float)
    near[np.isnan(f["d1_h"]) | np.isnan(f["d1_a"]) | ~seen] = np.nan
    f["near_h"] = near
    base.append("near_h")
    for w in CONTEXT:
        stack = np.column_stack([f[k] for k in base])
        rm = _rolling_mean(stack, w)
        for j, k in enumerate(base):
            f["{}_m{}".format(k, w)] = rm[:, j]
    for k in ("near_h", "v_all", "d1_diff"):
        f[k + "_delta"] = _shifted_mean(f[k], 50, True) - _shifted_mean(f[k], 50, False)

    groups = extra_groups(extra)
    if "long" in groups:
        # Longer memory: the same signals averaged over +/- 20 s.
        rm = _rolling_mean(np.column_stack([f[k] for k in base]), EXTRA_CONTEXT)
        for j, k in enumerate(base):
            f["{}_m{}".format(k, EXTRA_CONTEXT)] = rm[:, j]
    if "sided" in groups:
        # Before vs after: a change of possession flips who is near the ball and which way each
        # team's block moves, so the 5 s before and the 5 s after are given separately.
        for k in ("near_h", "d1_diff", "cx_diff", "cvx_h", "cvx_a"):
            f[k + "_past5"] = _shifted_mean(f[k], SIDED, False)
            f[k + "_next5"] = _shifted_mean(f[k], SIDED, True)
    if "close" in groups:
        # Who closes in on the ball (the pressing side, or the receiver), relative to the
        # half's median speed like every other speed here.
        for side, pos in (("h", home), ("a", away)):
            f["close_" + side] = _rolling_mean(_approach(pos, bi), 25) / v_ref
        f["close_diff"] = f["close_h"] - f["close_a"]

    if "sided2" in groups:
        # More "before vs after" (the kind of extra feature the model leans on most): 2 s and
        # 10 s sided means, and 5 s ones for the dead-ball signals -- everyone slowing down,
        # the ball leaving the pitch or reappearing.
        for w in (50, 250):
            for k in ("near_h", "cvx_h", "cvx_a", "cx_diff", "v_all", "b_speed", "b_out"):
                f["{}_past{}".format(k, w // 25)] = _shifted_mean(f[k], w, False)
                f["{}_next{}".format(k, w // 25)] = _shifted_mean(f[k], w, True)
        for k in ("v_all", "b_speed", "b_out", "ball_seen"):
            f[k + "_past5"] = _shifted_mean(f[k], SIDED, False)
            f[k + "_next5"] = _shifted_mean(f[k], SIDED, True)

    if "ballvx" in groups:
        # Which way the ball travels. With home defending -x, a ball moving towards +x is usually
        # being played forward by home, towards -x by away. Scaled by the half's median ball
        # speed, like b_speed, so tracking providers stay comparable.
        vx = np.full(len(bi), np.nan)
        dx = (bi[SPEED_LAG:, 0] - bi[:-SPEED_LAG, 0]) * (25.0 / SPEED_LAG)
        dx[~both] = np.nan
        vx[SPEED_LAG // 2: SPEED_LAG // 2 + len(dx)] = dx
        f["b_vx"] = vx / np.nanmedian(sp)
        f["b_vx_m25"] = _rolling_mean(f["b_vx"], 25)
        f["b_vx_m75"] = _rolling_mean(f["b_vx"], 75)
        f["b_vx_past2"] = _shifted_mean(f["b_vx"], 50, False)
        f["b_vx_next2"] = _shifted_mean(f["b_vx"], 50, True)

    if "touch" in groups:
        # Who last had the ball at their feet, and who has it next. A team touches the ball on a
        # frame where the ball is seen, one of its players is within r of it and nobody from the
        # other team is (a missing distance counts as nobody close).
        for r in TOUCH_RADII:
            tag = "touch{:g}_".format(r)
            near_h, near_a = f["d1_h"] <= r, f["d1_a"] <= r
            hits = {"h": seen & near_h & ~near_a, "a": seen & near_a & ~near_h}
            past = {s: _frames_since(hits[s]) for s in hits}
            future = {s: _frames_since(hits[s][::-1])[::-1] for s in hits}
            for name, gaps in (("since", past), ("until", future)):
                for s in ("h", "a"):
                    g = gaps[s]
                    f["{}{}_{}".format(tag, name, s)] = np.where(
                        g < 0, CAP, np.minimum(g, CAP)).astype(float)
            f[tag + "last"] = _whose(past["h"], past["a"])
            f[tag + "next"] = _whose(future["h"], future["a"])
    return f


def build(match, extra=False):
    """(features float32 (T, F), names, mask) where mask marks frames inside a half.

    extra picks optional feature groups (see extra_groups): 1 (or True) = long + sided + close,
    25 columns; 2 adds sided2, 36 more; names such as "long,sided,close,ballvx" pick groups
    directly ("touch" adds 12: who touched the seen ball last and next, and how long ago / until
    then, for each of TOUCH_RADII). Groups are appended in EXTRA_GROUPS order, so the base 84
    columns and every earlier group keep their order and values.
    """
    T = len(match["ball"])
    names, out = None, None
    mask = np.zeros(T, dtype=bool)
    for start, stop in match["halves"]:
        f = _half_features(match["home"][start:stop], match["away"][start:stop],
                           match["ball"][start:stop], match["pitch"], extra)
        if names is None:
            names = list(f)
            out = np.full((T, len(names)), np.nan, dtype=np.float32)
        out[start:stop] = np.column_stack([f[k] for k in names])
        mask[start:stop] = True
    return out, names, mask
