"""Leave-one-match-out exam on the DFL matches: the scoreboard every change is measured on.

    python -m possession.cv                          # score the current pipeline (~3-4 min)
    python -m possession.cv --tag v2                 # ... and save it as data/cache/cv/v2.json
    python -m possession.cv --tag v3 --against v2    # ... and compare with v2, match by match
    python -m possession.cv --folds 2 --rounds 20    # quick smoke test

Each DFL match sits the exam once. Its model is trained on the other matches exactly the way
possession/train.py trains: corrupted and team-swapped copies, the same seeds, the same LightGBM
parameters. It is trained for a fixed number of rounds, so nothing about the exam match shapes
its own model. Each match's exam copy is corrupted once, with a seed of its own, so every run sits
the same exams and two versions can be compared match by match. The scores come from the official
possession_metrics.evaluate. The official baseline (nearest player, then the +/-50-frame vote from
starter/baseline.py) sits the same exams, for reference.

No competition label is read. As in train.py, the only thing taken from the competition match is
the shape of its ball-gap runs.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

from . import atd, augment, dfl, features, train
from .common import CACHE_DIR, DFL_DIR, ROOT, X

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the organisers' scorer)
from starter.baseline import smooth  # noqa: E402  (the official baseline's own vote)

CV_DIR = os.path.join(CACHE_DIR, "cv")  # data/ is gitignored
BASELINE_WINDOW = 50  # +/- frames, the published floor baseline_smooth50
EXAM_SEED_OFFSET = 100  # exam copy of match k is corrupted with SEED + 100 + k
METRICS = ("spell_f1", "macro_f1", "accuracy")


def folds(ids):
    """(exam_id, train_ids) for every id: each match sits the exam once, the others train."""
    return [(i, [j for j in ids if j != i]) for i in ids]


def paired_summary(new, old):
    """Match-by-match comparison of two {match: score} dicts, over the matches both runs have."""
    shared = [k for k in new if k in old]
    delta = np.array([new[k] - old[k] for k in shared], dtype=float)
    return {
        "n": len(shared),
        "mean_delta": float(delta.mean()) if len(shared) else 0.0,
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
    }


def mean_and_se(values):
    """Mean and standard error (sample standard deviation / sqrt(n)); no spread for one value."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def exam_fingerprint(match):
    """Short hash of everything an exam copy is made of. Two runs sat the same exam on a match
    only if its fingerprint agrees -- a change to augment.py changes the exams, not just training."""
    digest = hashlib.sha256()
    for key in ("home", "away", "ball", "labels"):
        digest.update(np.ascontiguousarray(match[key]).tobytes())
    return digest.hexdigest()[:16]


def baseline_codes(match, window=BASELINE_WINDOW):
    """The official baseline on one match, as int8 codes (0 H, 1 A, 2 D, X where ungraded).

    Step 1 gives the team of the nearest player wherever the ball is seen, nobody elsewhere;
    step 2 is starter/baseline.py's centred majority vote, D where nobody had the ball.
    """
    ball = match["ball"]
    dist_h = np.linalg.norm(match["home"] - ball[:, None, :], axis=2)
    dist_a = np.linalg.norm(match["away"] - ball[:, None, :], axis=2)
    near_h = np.where(np.isnan(dist_h), np.inf, dist_h).min(axis=1)
    near_a = np.where(np.isnan(dist_a), np.inf, dist_a).min(axis=1)

    codes = np.full(len(ball), -1, dtype=np.int8)  # -1 = nobody, as in starter/baseline.py
    seen = ~np.isnan(ball[:, 0]) & (np.isfinite(near_h) | np.isfinite(near_a))
    codes[seen] = np.where(near_h[seen] <= near_a[seen], 0, 1)

    letters = smooth(codes, window)
    out = np.where(letters == "H", 0, np.where(letters == "A", 1, 2)).astype(np.int8)
    if match["labels"] is not None:
        out[match["labels"] == X] = X
    return out


def _scores(gt, pred):
    r = pm.evaluate(gt, pred)
    return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
            "accuracy": float(r["accuracy"]), "n_pred": int(r["n_pred_spells"]),
            "n_gt": int(r["n_gt_spells"])}


def run_exam(rounds, decoder, n_folds=None, log=print, dfl_dir=DFL_DIR):
    """Sit every exam (or the first n_folds). Returns {"config": ..., "matches": {id: scores}}."""
    ids = dfl.match_ids(dfl_dir)
    ball_runs = augment.ball_run_lengths(atd.load())
    tables, exams, names = {}, {}, None
    for k, i in enumerate(ids):
        raw = dfl.load(i, dfl_dir)
        Xk, yk, names = train.frame_table(train.copies(raw, ball_runs, train.SEED + k),
                                          train.SUBSAMPLE)
        tables[i] = (Xk, yk)
        exams[i] = augment.corrupt(raw, ball_runs,
                                   np.random.default_rng(train.SEED + EXAM_SEED_OFFSET + k))
    log("built training tables and exam copies for {} matches".format(len(ids)))

    turns = folds(ids)[:n_folds] if n_folds else folds(ids)
    matches = {}
    for exam_id, train_ids in turns:
        assert exam_id not in train_ids, "exam match leaked into its own training set"
        t0 = time.time()
        Xtr = np.vstack([tables[j][0] for j in train_ids])
        ytr = np.concatenate([tables[j][1] for j in train_ids])
        booster = lgb.train(train.LGB_PARAMS, lgb.Dataset(Xtr, ytr, feature_name=names),
                            num_boost_round=rounds)
        prior = np.bincount(ytr, minlength=3) / len(ytr)
        exam = exams[exam_id]
        proba = booster.predict(features.build(exam)[0])
        pred = train.decode_match(proba, exam, decoder, prior)
        matches[exam_id] = {"model": _scores(exam["labels"], pred),
                            "baseline": _scores(exam["labels"], baseline_codes(exam))}
        log("{}: spellF1 {:.4f} (baseline {:.4f}), {:.0f} s".format(
            exam_id, matches[exam_id]["model"]["spell_f1"],
            matches[exam_id]["baseline"]["spell_f1"], time.time() - t0))

    config = {"ids": ids, "exams": [e for e, _ in turns], "rounds": rounds, "decoder": decoder,
              "lgb_params": train.LGB_PARAMS, "subsample": train.SUBSAMPLE, "seed": train.SEED,
              "exam_seed_offset": EXAM_SEED_OFFSET, "baseline_window": BASELINE_WINDOW,
              "exam_fingerprints": {i: exam_fingerprint(exams[i]) for i in ids}}
    return {"config": config, "matches": matches}


def print_table(result):
    cfg, matches = result["config"], result["matches"]
    d = cfg["decoder"]
    print("\nexam: {} of {} DFL matches, each held out once | LightGBM {} rounds | decoder "
          "team_cost={} dead_cost={} min_len={} prior_power={}".format(
              len(matches), len(cfg["ids"]), cfg["rounds"], d["team_cost"], d["dead_cost"],
              d["min_len"], d["prior_power"]))
    print("{:<18}{:>34}   {:>30}".format("", "------------ our model ------------",
                                         "--- official baseline (+/-50) ---"))
    print("{:<18}{:>9}{:>9}{:>9}{:>11}   {:>10}{:>10}{:>10}".format(
        "match", "spellF1", "macroF1", "accuracy", "poss", "spellF1", "macroF1", "accuracy"))
    for mid, s in matches.items():
        m, b = s["model"], s["baseline"]
        print("{:<18}{:>9.4f}{:>9.4f}{:>9.4f}{:>11}   {:>10.4f}{:>10.4f}{:>10.4f}".format(
            mid, m["spell_f1"], m["macro_f1"], m["accuracy"],
            "{}/{}".format(m["n_pred"], m["n_gt"]), b["spell_f1"], b["macro_f1"], b["accuracy"]))
    for label, pick in (("mean", 0), ("standard error", 1)):
        model = [mean_and_se([s["model"][k] for s in matches.values()])[pick] for k in METRICS]
        base = [mean_and_se([s["baseline"][k] for s in matches.values()])[pick] for k in METRICS]
        print("{:<18}{:>9.4f}{:>9.4f}{:>9.4f}{:>11}   {:>10.4f}{:>10.4f}{:>10.4f}".format(
            label, *model, "", *base))
    print("(poss = possessions predicted / true)")


def print_against(result, old, tag):
    new_s = {k: v["model"]["spell_f1"] for k, v in result["matches"].items()}
    old_s = {k: v["model"]["spell_f1"] for k, v in old["matches"].items()}
    new_fp = result["config"].get("exam_fingerprints", {})
    old_fp = old["config"].get("exam_fingerprints", {})
    changed = [k for k in new_s if k in old_s and new_fp.get(k) != old_fp.get(k)]
    print("\nagainst {} (our model's spellF1, match by match){}".format(
        tag, "" if not changed else
        "  WARNING: the exams differ on {} -- not comparable".format(", ".join(changed))))
    for k in new_s:
        if k in old_s:
            print("  {:<18}{:.4f} -> {:.4f}   {:+.4f}".format(k, old_s[k], new_s[k],
                                                          new_s[k] - old_s[k]))
    s = paired_summary(new_s, old_s)
    print("  mean change {:+.4f} | better on {} of {}, worse on {}".format(
        s["mean_delta"], s["wins"], s["n"], s["losses"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Leave-one-match-out exam on the DFL matches.")
    parser.add_argument("--tag", help="save the result as data/cache/cv/<tag>.json")
    parser.add_argument("--against", help="compare with a saved result, match by match")
    parser.add_argument("--folds", type=int, help="sit only the first N exams (smoke test)")
    parser.add_argument("--rounds", type=int, help="LightGBM rounds (default: decoder.json)")
    parser.add_argument("--dfl-dir", default=DFL_DIR, help="DFL data folder")
    parser.add_argument("--cv-dir", default=CV_DIR, help="where saved results live")
    args = parser.parse_args(argv)

    old = None
    if args.against:
        path = os.path.join(args.cv_dir, args.against + ".json")
        if not os.path.isfile(path):
            saved = sorted(f[:-5] for f in os.listdir(args.cv_dir)
                           if f.endswith(".json")) if os.path.isdir(args.cv_dir) else []
            sys.exit("no saved run '{}' in {}; saved runs: {}".format(
                args.against, args.cv_dir, ", ".join(saved) or "none"))
        with open(path) as handle:
            old = json.load(handle)
    if not dfl.match_ids(args.dfl_dir):
        sys.exit("no DFL matches in {}; download them with: python -m possession.download_dfl"
                 .format(args.dfl_dir))

    with open(os.path.join(train.MODEL_DIR, "decoder.json")) as handle:
        cfg = json.load(handle)
    t0 = time.time()
    result = run_exam(args.rounds or cfg["rounds"], cfg["decoder"], args.folds,
                      dfl_dir=args.dfl_dir)
    print_table(result)
    if old is not None:
        print_against(result, old, args.against)
    if args.tag:
        os.makedirs(args.cv_dir, exist_ok=True)
        path = os.path.join(args.cv_dir, args.tag + ".json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print("saved", path)
    print("total {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
