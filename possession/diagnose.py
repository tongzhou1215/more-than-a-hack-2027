"""Unsupervised domain-shift check between the competition ATD and the corrupted DFL data.

    python -m possession.diagnose

Reads only the competition *inputs*, never its labels. Prints feature percentiles side by side and
how much dead ball the trained model predicts with and without the ball in view. A large gap in
either is a sign that the augmentation or the feature normalisation does not match the target.
"""

import json

import numpy as np
import lightgbm as lgb

from . import atd, augment, dfl, features
from .train import MODEL_DIR, N_VAL, SEED

WATCH = ["v_all", "v_all_m75", "vmax_h", "b_speed", "b_out", "b_out_m25", "b_touch", "b_goal",
         "sy_h", "nvis_h", "d1_h", "ball_seen"]


def main():
    with open(MODEL_DIR + "/decoder.json") as handle:
        extra = json.load(handle).get("extra_features", 0)  # the features the model was trained on
    target = atd.load()
    runs = augment.ball_run_lengths(target)
    ft, names, mt = features.build(target, extra=extra)
    ft = ft[mt]

    parts = []
    for n, i in enumerate(dfl.match_ids()[-N_VAL:]):
        m = augment.corrupt(dfl.load(i), runs, np.random.default_rng(SEED + 100 + n))
        f, _, mk = features.build(m, extra=extra)
        parts.append((f[mk], m["labels"][mk]))
    fd = np.vstack([a for a, _ in parts])
    yd = np.concatenate([b for _, b in parts])

    def q(x):
        return np.round(np.nanpercentile(x, [10, 50, 90]), 2)

    print("{:<14} {:>24} {:>24}".format("feature", "ATD p10/p50/p90", "DFL p10/p50/p90"))
    for k in WATCH:
        j = names.index(k)
        print("{:<14} {:>24} {:>24}".format(k, str(q(ft[:, j])), str(q(fd[:, j]))))

    seen_t = ft[:, names.index("ball_seen")] == 1
    seen_d = fd[:, names.index("ball_seen")] == 1
    out = names.index("b_out")
    print("ball off the pitch when seen: ATD {:.3f}  DFL {:.3f}".format(
        ft[seen_t, out].mean(), fd[seen_d, out].mean()))

    booster = lgb.Booster(model_file=MODEL_DIR + "/lgbm.txt")
    pt, pd_ = booster.predict(ft), booster.predict(fd)
    print("mean P(D): ATD {:.3f}  DFL {:.3f} (true DFL D share {:.3f})".format(
        pt[:, 2].mean(), pd_[:, 2].mean(), (yd == 2).mean()))
    for name, p, s in (("ATD", pt, seen_t), ("DFL", pd_, seen_d)):
        print("{}  P(D) ball seen {:.3f} | ball missing {:.3f}".format(
            name, p[s, 2].mean(), p[~s, 2].mean()))


if __name__ == "__main__":
    main()
