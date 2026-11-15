"""Train the final frame scorers on all DFL matches, with the configuration the 7-match exam chose.

    python -m possession.train                 # the GRU trains on CUDA when there is a GPU
    python -m possession.train --device cpu

No label from the competition match is read here. The only thing taken from it is the shape of
its ball-track gaps (an input statistic), which the augmentation copies onto the DFL data.

The configuration below was chosen on the leave-one-match-out exams of possession/cv.py
(LightGBM) and possession/seq_cv.py (the GRU), which set the same exams:
- each of the 7 DFL matches is held out once;
- the answers are read as controlled possession with a 2 s threshold (possession/labels.py);
- every exam is scored with the official spellF1.

Nothing about the competition match took part in choosing it.

Steps
  1. Corrupt each DFL match COPIES times to look like the competition ATD, each copy with a
     home/away-swapped twin.
  2. Compute the base features plus the extra groups (features.build(extra=EXTRA_FEATURES)) on
     every frame of every view, once.
  3. Train LightGBM on every SUBSAMPLE-th graded frame for ROUNDS rounds.
  4. Train the bidirectional GRU (possession/seq.py) on every frame of the same views for
     seq.EPOCHS epochs, seed SEED.
  5. Save lgbm.txt, gru.npy, gru.json and decoder.json. decoder.json holds the decoder settings
     and the blend: predict.py scores LGBM_WEIGHT x LightGBM + (1 - LGBM_WEIGHT) x GRU.
"""

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np

from . import atd, augment, dfl, features, viterbi
from .common import ROOT, X

MODEL_DIR = os.path.join(ROOT, "possession", "model")
SEED = 2027
SUBSAMPLE = 5
N_VAL = 2  # diagnose.py compares the last N_VAL DFL matches with the competition ATD
LGB_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, seed=SEED, deterministic=True, force_row_wise=True,
    num_threads=4,
)
COPY_SEED_STRIDE = 1000  # corrupted copy c of match k is seeded SEED + k + 1000 * c

# Chosen on the 7-match exams (possession/cv.py, possession/seq_cv.py), 2026-11-14.
ROUNDS = 126
COPIES = 2
EXTRA_FEATURES = 1
DECODER = {"team_cost": 2.0, "dead_cost": 64.0, "min_len": 75, "prior_power": 0.0}
LGBM_WEIGHT = 0.35  # the submission scores 0.35 x LightGBM + 0.65 x GRU
SELECTION = ("possession/cv.py --exam-controlled 2 --decoder 2,64,75 --extra-features 1 "
             "--copies 2 (LightGBM alone: 0.6592) and possession/seq_cv.py --epochs 6 (GRU alone: "
             "0.7012), blended 0.35 x LightGBM + 0.65 x GRU: mean spellF1 0.7122 over the 7 "
             "held-out DFL matches, against 0.1949 for the official baseline on the same exams")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def corrupted_pair(match, ball_runs, seed):
    """One DFL match made to look like the competition ATD, and its home/away-swapped twin."""
    noisy = augment.corrupt(match, ball_runs, np.random.default_rng(seed))
    return [noisy, augment.swap_teams(noisy)]


copies = corrupted_pair  # the name earlier versions of this module used


def training_views(teach, ball_runs, k, copies=1):
    """Match k's training views: `copies` corrupted pairs, pair c seeded SEED + k + 1000 * c."""
    views = []
    for c in range(copies):
        views += corrupted_pair(teach, ball_runs, SEED + k + COPY_SEED_STRIDE * c)
    return views


def build_views(views, extra=0):
    """features.build on every view, once: ([(features, labels, halves, graded mask)], names).
    LightGBM's table and the GRU's sequences are both cut from it."""
    built, names = [], None
    for m in views:
        f, names, mask = features.build(m, extra=extra)
        built.append((f, m["labels"], m["halves"], mask & (m["labels"] != X)))
    return built, names


def lightgbm_table(built, subsample):
    """(X, y) from every subsample-th graded frame of the built views."""
    xs, ys = [], []
    for f, y, _, graded in built:
        idx = np.flatnonzero(graded)[::subsample]
        xs.append(f[idx])
        ys.append(y[idx])
    return np.vstack(xs), np.concatenate(ys)


def frame_table(matches, subsample, extra=0):
    """(X, y, feature names) from every subsample-th graded frame of the given matches."""
    built, names = build_views(matches, extra)
    xs, ys = lightgbm_table(built, subsample)
    return xs, ys, names


def decode_match(proba, match, params, prior):
    out = np.full(len(proba), X, dtype=np.int8)
    for start, stop in match["halves"]:
        min_len = params["min_len"]
        out[start:stop] = viterbi.decode(
            proba[start:stop], params["team_cost"], params["dead_cost"],
            (min_len, min_len, params.get("min_len_dead", min_len)), prior, params["prior_power"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", help="where the GRU trains: cuda or cpu (default: cuda when a GPU "
                                    "is available)")
    a = p.parse_args(argv)
    from . import seq  # PyTorch is needed only for the GRU, so cv.py and diagnose.py run without it
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    target = atd.load()
    ball_runs = augment.ball_run_lengths(target)
    log("target ball runs: visible median {} / gap median {} frames".format(
        int(np.median(ball_runs[0])), int(np.median(ball_runs[1]))))

    ids = dfl.match_ids()
    log("DFL matches:", ids)
    views = []
    for k, i in enumerate(ids):
        views += training_views(dfl.load(i), ball_runs, k, COPIES)
    built, names = build_views(views, EXTRA_FEATURES)
    Xall, yall = lightgbm_table(built, SUBSAMPLE)
    log("train rows", Xall.shape, "class share", np.bincount(yall) / len(yall))

    final = lgb.train(LGB_PARAMS, lgb.Dataset(Xall, yall, feature_name=names),
                      num_boost_round=ROUNDS)
    prior = (np.bincount(yall, minlength=3) / len(yall)).tolist()
    del Xall
    imp = sorted(zip(final.feature_importance("gain"), names), reverse=True)[:15]
    log("LightGBM trained; top features:", ", ".join(n for _, n in imp))

    t0 = time.time()
    log("GRU: {} views, {} epochs, seed {}, on {}".format(len(built), seq.EPOCHS, SEED, device))
    gru, mean, std = seq.fit(built, seed=SEED, device=device, log=log)
    log("GRU trained in {:.0f} s".format(time.time() - t0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(MODEL_DIR, "lgbm.txt"))
    trained_on = ("cuda: " + torch.cuda.get_device_name()) if device.startswith("cuda") else device
    seq.save(MODEL_DIR, gru, mean, std, {
        "features": names, "epochs": seq.EPOCHS, "seed": SEED, "batch": seq.BATCH, "lr": seq.LR,
        "weight_decay": seq.WEIGHT_DECAY, "train_ids": ids, "trained_on": trained_on,
        "torch": torch.__version__})
    with open(os.path.join(MODEL_DIR, "decoder.json"), "w", newline="\n") as handle:
        json.dump({"decoder": DECODER, "prior": prior, "lgbm_weight": LGBM_WEIGHT,
                   "rounds": ROUNDS, "copies": COPIES, "extra_features": EXTRA_FEATURES,
                   "features": names, "train_ids": ids, "selected_by": SELECTION},
                  handle, indent=2)
    log("saved", MODEL_DIR)


if __name__ == "__main__":
    main()
