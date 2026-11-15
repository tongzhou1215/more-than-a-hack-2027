"""How far the corrupted DFL training views are from the competition match, from inputs only.

    python -m possession.gap

The models learn on DFL matches corrupted to look like the competition's automatic tracking, and
the exams (cv.py, seq_cv.py) are sat on such copies. Whatever the corruption fails to imitate is a
difference the exam cannot see. This measures it without a single competition label:

1. A LightGBM classifier learns to tell corrupted-DFL frames from BAR-ATH frames, on the features
   the models use. Its out-of-fold AUC is the gap: 0.5 means it cannot tell them apart, 1.0 means
   every frame gives itself away. The features it leans on are where to look.
2. The single features that differ most (one-feature AUC).
3. A reference: one DFL match against the other six. Same tracking and corruption, so what is
   left is how different one real match looks anyway (teams, tactics) -- a floor for 1.
4. How often the ball is missing by the players' speed, BAR-ATH against DFL, with DFL's own
   dead-ball share per speed decile to read it.

Both sides come with their team-swapped twins, so which team dominated is not a difference the
classifier can use. One frame per second inside the halves; folds are contiguous time blocks and a
frame's twin shares its fold, so neighbouring seconds never sit on both sides. BAR-ATH's labels
are never read. Prints aggregate numbers only.
"""

import argparse

import lightgbm as lgb
import numpy as np

from . import atd, augment, dfl, features, train
from .common import D, X

EVERY = 25          # one frame per second
N_FOLDS = 5
ROUNDS = 200
TOP = 12
SPEED = "v_all"     # all players' mean speed over the half's median
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
              seed=train.SEED, deterministic=True, force_row_wise=True, num_threads=4)


def auc(scores, labels):
    """Area under the ROC curve: how often a positive scores above a negative, ties counting
    half."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    order = np.argsort(s, kind="mergesort")
    _, start, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def separability(xa, xb):
    """Per feature, how well its value alone tells the two sets apart: the one-feature AUC folded
    into [0.5, 1], where 0.5 means the same distribution. A missing value (NaN) counts as the
    lowest value, so a feature that is missing more often on one side tells them apart too."""
    x = np.vstack([xa, xb]).astype(np.float64)
    x = np.where(np.isnan(x), -np.inf, x)
    y = np.r_[np.zeros(len(xa), dtype=bool), np.ones(len(xb), dtype=bool)]
    out = np.array([auc(x[:, j], y) for j in range(x.shape[1])])
    return np.maximum(out, 1.0 - out)


def block_folds(n, k):
    """Fold ids for n consecutive samples cut into k contiguous, near-equal blocks."""
    return np.repeat(np.arange(k), [len(c) for c in np.array_split(np.arange(n), k)])


def domain_auc(xa, xb, folds_a, folds_b, rounds=ROUNDS):
    """Out-of-fold AUC of a LightGBM classifier telling xb (label 1) from xa (label 0), and each
    feature's share of the gain. Fold f is predicted by a model trained on all other folds of both
    sets."""
    x = np.vstack([xa, xb])
    y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    folds = np.r_[folds_a, folds_b]
    params = dict(PARAMS, scale_pos_weight=len(xa) / len(xb))
    score = np.zeros(len(y))
    gain = np.zeros(x.shape[1])
    for f in np.unique(folds):
        test = folds == f
        booster = lgb.train(params, lgb.Dataset(x[~test], y[~test]), num_boost_round=rounds)
        score[test] = booster.predict(x[test], num_threads=1)
        gain += booster.feature_importance("gain")
    return auc(score, y), gain / max(gain.sum(), 1e-12)


def rate_by_bins(values, flags, edges):
    """The mean of flags among the samples whose value falls in each bin (np.digitize on edges:
    below edges[0], ..., at or above edges[-1]); samples without a value are left out."""
    v = np.asarray(values, dtype=np.float64)
    f = np.asarray(flags, dtype=np.float64)
    ok = ~np.isnan(v)
    bins = np.digitize(v[ok], edges)
    return np.array([f[ok][bins == b].mean() if np.any(bins == b) else np.nan
                     for b in range(len(edges) + 1)])


def _samples(view, extra, every, with_labels=True):
    """(features, ball seen, labels or None, names) of every `every`-th frame inside the halves."""
    f, names, mask = features.build(view, extra=extra)
    idx = np.flatnonzero(mask)[::every]
    seen = ~np.isnan(view["ball"][idx, 0])
    labels = view["labels"][idx] if with_labels and view["labels"] is not None else None
    return f[idx], seen, labels, names


def _stack(groups, n_folds=N_FOLDS):
    """[(original samples, twin samples)] -> (features, folds); a twin shares its original's
    fold, and the folds are contiguous blocks over the originals in order."""
    sizes = [len(o[0]) for o, _ in groups]
    folds = block_folds(sum(sizes), n_folds)
    x = np.vstack([o[0] for o, _ in groups] + [t[0] for _, t in groups])
    return x, np.r_[folds, folds]


def _row(label, values):
    return "  {:<12}".format(label) + "".join("{:>6.2f}".format(v) for v in values)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--every", type=int, default=EVERY,
                   help="keep every Nth frame inside the halves (default 25: one per second)")
    p.add_argument("--gaps", choices=("random", "dead"), default="random",
                   help="how the DFL views lose their ball: at random (default) or by dead and "
                        "live play (augment.dead_gaps)")
    a = p.parse_args(argv)
    extra = train.EXTRA_FEATURES

    target = atd.load()
    runs = augment.ball_run_lengths(target)
    dg = augment.dead_gaps(target) if a.gaps == "dead" else None
    tgt = (_samples(target, extra, a.every, with_labels=False),
           _samples(augment.swap_teams(target), extra, a.every, with_labels=False))
    names = tgt[0][3]
    matches = []
    for k, i in enumerate(dfl.match_ids()):
        noisy, twin = train.training_views(dfl.load(i), runs, k, 1, dg)
        matches.append((_samples(noisy, extra, a.every), _samples(twin, extra, a.every)))

    xs, fs = _stack(matches)
    xt, ft = _stack([tgt])
    gap, share = domain_auc(xs, xt, fs, ft)
    sep = separability(xs, xt)
    refs = []
    for m in range(len(matches)):
        xo, fo = _stack([g for j, g in enumerate(matches) if j != m])
        xm, fm = _stack([matches[m]])
        refs.append(domain_auc(xo, xm, fo, fm)[0])

    print("domain gap: corrupted DFL ({} matches, copy 0 + team swap, {} ball gaps) against "
          "BAR-ATH (+ team swap), one frame in {}".format(len(matches), a.gaps, a.every))
    if dg is not None:
        print("  dead_gaps from BAR-ATH's inputs: stoppages hidden {:.2f}, live frames hidden {:.2f}"
              ", {} fast gaps (median {:.1f} s)".format(
                  dg.dead_hide, dg.live_hide, len(dg.live_gaps),
                  np.median(dg.live_gaps) / 25 if len(dg.live_gaps) else float("nan")))
    print("  frames: DFL {:,}  BAR-ATH {:,}  features {}".format(len(xs), len(xt), len(names)))
    print("  classifier AUC, {} time-block folds: {:.3f}   (0.5 = cannot tell them apart, "
          "1.0 = every frame gives itself away)".format(N_FOLDS, gap))
    print("  reference, one DFL match against the other six: mean {:.3f} (range {:.3f}-{:.3f})"
          .format(np.mean(refs), np.min(refs), np.max(refs)))
    print("  what the classifier leans on (share of gain, then one-feature AUC):")
    for j in np.argsort(-share)[:TOP]:
        print("    {:<16} {:.3f}   {:.3f}".format(names[j], share[j], sep[j]))
    print("  features that differ most one at a time (one-feature AUC):")
    for j in np.argsort(-sep)[:TOP]:
        print("    {:<16} {:.3f}".format(names[j], sep[j]))

    j = names.index(SPEED)
    t_v, t_seen = tgt[0][0][:, j], tgt[0][1]
    s_v = np.concatenate([o[0][:, j] for o, _ in matches])
    s_seen = np.concatenate([o[1] for o, _ in matches])
    s_lab = np.concatenate([o[2] for o, _ in matches])
    graded = s_lab != X
    deciles = np.arange(0.1, 1.0, 0.1)
    print("ball missing by players' speed ({} deciles of each side, slowest first); ball missing "
          "overall: BAR-ATH {:.2f}, DFL {:.2f}".format(SPEED, 1 - t_seen.mean(), 1 - s_seen.mean()))
    print("  {:<12}".format("decile") + "".join("{:>6}".format(d) for d in range(1, 11)))
    print(_row("BAR-ATH", rate_by_bins(t_v, ~t_seen, np.nanquantile(t_v, deciles))))
    s_edges = np.nanquantile(s_v, deciles)
    print(_row("DFL", rate_by_bins(s_v, ~s_seen, s_edges)))
    print(_row("DFL dead", rate_by_bins(s_v[graded], s_lab[graded] == D, s_edges)))
    print("  (DFL dead = DFL's own answers: the share of dead-ball frames in each speed decile)")


if __name__ == "__main__":
    main()
