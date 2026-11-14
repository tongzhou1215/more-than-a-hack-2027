"""Train the frame scorer on DFL matches and pick decoder settings on held-out DFL matches.

    python -m possession.train

No label from the competition match is read here. The only thing taken from it is the shape of
its ball-track gaps (an input statistic), which the augmentation copies onto the DFL data.

Steps
  1. Corrupt each DFL match to look like the competition ATD, plus a home/away-swapped copy.
  2. LightGBM on frames from the training matches (every SUBSAMPLE-th frame).
  3. Viterbi grid search on the validation matches, scored with the official spellF1.
  4. Refit on all matches with the chosen number of rounds; save model + decoder settings.
"""

import argparse
import itertools
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

from . import atd, augment, dfl, features, viterbi
from .common import CACHE_DIR, ROOT, X, ensure_cache

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "possession", "model")
SEED = 2027
SUBSAMPLE = 5
N_VAL = 2
LGB_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, seed=SEED, deterministic=True, force_row_wise=True,
    num_threads=4,
)
GRID = dict(
    team_cost=(2.0, 4.0, 8.0),
    dead_cost=(8.0, 16.0, 32.0),
    min_len=(25, 50, 75),
    prior_power=(0.0, 0.5),
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def copies(match, ball_runs, seed):
    """The training views of one DFL match: corrupted, and corrupted + team-swapped."""
    noisy = augment.corrupt(match, ball_runs, np.random.default_rng(seed))
    return [noisy, augment.swap_teams(noisy)]


def frame_table(matches, subsample):
    xs, ys = [], []
    names = None
    for m in matches:
        f, names, mask = features.build(m)
        keep = mask & (m["labels"] != X)
        idx = np.flatnonzero(keep)[::subsample]
        xs.append(f[idx])
        ys.append(m["labels"][idx])
    return np.vstack(xs), np.concatenate(ys), names


def decode_match(proba, match, params, prior):
    out = np.full(len(proba), X, dtype=np.int8)
    for start, stop in match["halves"]:
        out[start:stop] = viterbi.decode(
            proba[start:stop], params["team_cost"], params["dead_cost"],
            (params["min_len"],) * 3, prior, params["prior_power"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small grid, for a smoke test")
    args = parser.parse_args()

    target = atd.load()
    ball_runs = augment.ball_run_lengths(target)
    log("target ball runs: visible median {} / gap median {} frames".format(
        int(np.median(ball_runs[0])), int(np.median(ball_runs[1]))))

    ids = dfl.match_ids()
    log("DFL matches:", ids)
    raw = {i: dfl.load(i) for i in ids}
    train_ids, val_ids = ids[:-N_VAL], ids[-N_VAL:]

    views = []
    for n, i in enumerate(train_ids):
        views += copies(raw[i], ball_runs, SEED + n)
    Xtr, ytr, names = frame_table(views, SUBSAMPLE)
    log("train rows", Xtr.shape, "class share", np.bincount(ytr) / len(ytr))
    prior = np.bincount(ytr, minlength=3) / len(ytr)

    val_views = [augment.corrupt(raw[i], ball_runs, np.random.default_rng(SEED + 100 + n))
                 for n, i in enumerate(val_ids)]
    Xva, yva, _ = frame_table(val_views, SUBSAMPLE)

    dtrain = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=False)
    dval = lgb.Dataset(Xva, yva, reference=dtrain)
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    rounds = booster.best_iteration
    log("best rounds", rounds, "val logloss", booster.best_score["valid_0"]["multi_logloss"])

    # Frame-level check, then decoder search on full-length validation matches.
    val_proba = []
    for m in val_views:
        f, _, _ = features.build(m)
        val_proba.append(booster.predict(f, num_iteration=rounds))
    for m, p in zip(val_views, val_proba):
        g = m["labels"]
        raw_pred = np.where(g == X, X, p.argmax(1)).astype(np.int8)
        r = pm.evaluate(g, raw_pred)
        log("{} argmax: macroF1 {:.4f} spellF1 {:.4f} poss {} vs {}".format(
            m["name"], r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))

    grid = GRID if not args.quick else dict(team_cost=(8.0,), dead_cost=(8.0,), min_len=(25,),
                                             prior_power=(0.0,))
    results = []
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        scores = []
        for m, p in zip(val_views, val_proba):
            pred = decode_match(p, m, params, prior)
            r = pm.evaluate(m["labels"], pred)
            scores.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
        s = np.array(scores)
        results.append((float(s[:, 0].mean()), float(s[:, 1].mean()), params, scores))
    results.sort(key=lambda r: (-r[0], -r[1]))
    for spell, macro, params, scores in results[:8]:
        log("spellF1 {:.4f} macroF1 {:.4f} {} poss {}".format(
            spell, macro, params, [(int(a), int(b)) for _, _, a, b in scores]))
    best = results[0][2]

    # Refit on every DFL match, validation ones included, with the chosen round count.
    all_views = list(views)
    for n, i in enumerate(val_ids):
        all_views += copies(raw[i], ball_runs, SEED + len(train_ids) + n)
    Xall, yall, _ = frame_table(all_views, SUBSAMPLE)
    final = lgb.train(LGB_PARAMS, lgb.Dataset(Xall, yall, feature_name=names), num_boost_round=rounds)
    prior_all = (np.bincount(yall, minlength=3) / len(yall)).tolist()

    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(MODEL_DIR, "lgbm.txt"))
    with open(os.path.join(MODEL_DIR, "decoder.json"), "w") as handle:
        json.dump({"decoder": best, "prior": prior_all, "rounds": rounds, "features": names,
                   "val_spell_f1": results[0][0], "val_macro_f1": results[0][1],
                   "train_ids": train_ids, "val_ids": val_ids}, handle, indent=2)
    imp = sorted(zip(final.feature_importance("gain"), names), reverse=True)[:15]
    log("top features:", ", ".join(n for _, n in imp))
    log("saved", MODEL_DIR)


if __name__ == "__main__":
    main()
