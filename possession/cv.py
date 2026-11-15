"""Leave-one-match-out exam on the DFL matches: the scoreboard every change is measured on.

    python -m possession.cv                          # score the current pipeline (~3-4 min)
    python -m possession.cv --tag v2                 # ... and save it as data/cache/cv/v2.json
    python -m possession.cv --tag v3 --against v2    # ... and compare with v2, match by match
    python -m possession.cv --folds 2 --rounds 20    # quick smoke test
    python -m possession.cv --exam-controlled 2 --train-controlled 2   # answers read as
                                                     # controlled possession (possession/labels.py)
    python -m possession.cv --grid --tag g           # score all 36 decoder settings of GRID
    python -m possession.cv --decoder 1,64,25        # sit the exam with one given setting

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

from . import atd, augment, dfl, dumps, features, train
from .common import CACHE_DIR, DFL_DIR, FPS, ROOT, X
from .labels import controlled

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


def _frames(seconds):
    return None if seconds is None else int(round(seconds * FPS))


def exam_copy(raw, ball_runs, k, exam_frames=None, dead_gaps=None):
    """Match k's exam: corrupted with a seed of its own (its ball gaps by dead and live play when
    dead_gaps is given), answers read as controlled possession when exam_frames is set. The inputs
    never depend on exam_frames -- only the labels do."""
    exam = augment.corrupt(raw, ball_runs,
                           np.random.default_rng(train.SEED + EXAM_SEED_OFFSET + k),
                           dead_gaps=dead_gaps)
    if exam_frames is not None:
        exam["labels"] = controlled(exam["labels"], exam_frames)
    return exam


GAP_MODES = ("random", "dead")            # how an exam copy loses its ball
TRAIN_GAP_MODES = GAP_MODES + ("mixed",)  # mixed: copy 0 random, copy 1 dead, copy 2 random, ...


def gap_settings(target, train_gaps="random", exam_gaps="random"):
    """(the training views' dead_gaps, the exam copies' dead_gaps, a summary for the config). A
    side with "random" keeps today's random ball gaps (None); "dead" uses augment.dead_gaps(target),
    measured once from the target's inputs; "mixed" (training only) alternates the two by copy."""
    if train_gaps not in TRAIN_GAP_MODES:
        raise ValueError("training ball gaps must be one of {}, not {!r}".format(
            TRAIN_GAP_MODES, train_gaps))
    if exam_gaps not in GAP_MODES:
        raise ValueError("exam ball gaps must be one of {}, not {!r}".format(GAP_MODES, exam_gaps))
    measured = (augment.dead_gaps(target) if train_gaps != "random" or exam_gaps == "dead"
                else None)
    summary = None if measured is None else {
        "dead_hide": measured.dead_hide, "live_hide": measured.live_hide,
        "live_gaps": int(len(measured.live_gaps)),
        "live_gap_median_s": (float(np.median(measured.live_gaps)) / FPS
                              if len(measured.live_gaps) else None)}
    train_dg = {"random": None, "dead": measured, "mixed": [None, measured]}[train_gaps]
    return (train_dg, measured if exam_gaps == "dead" else None,
            {"train_gaps": train_gaps, "exam_gaps": exam_gaps, "dead_gaps": summary})


def _scores(gt, pred):
    r = pm.evaluate(gt, pred)
    return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
            "accuracy": float(r["accuracy"]), "n_pred": int(r["n_pred_spells"]),
            "n_gt": int(r["n_gt_spells"])}


GRID = {"team_cost": (0.5, 1.0, 2.0, 4.0), "dead_cost": (16.0, 32.0, 64.0), "min_len": (25, 50, 75)}


def parse_decoder(text):
    """'team_cost,dead_cost,min_len[,min_len_dead]' -> a decoder dict, prior_power 0."""
    values = text.split(",")
    decoder = {"team_cost": float(values[0]), "dead_cost": float(values[1]),
               "min_len": int(values[2]), "prior_power": 0.0}
    if len(values) > 3:
        decoder["min_len_dead"] = int(values[3])
    return decoder


COPY_SEED_STRIDE = train.COPY_SEED_STRIDE  # corrupted copy c of match k: SEED + k + 1000 * c


def parse_assignment(text):
    """'key=value' -> (key, value), the value read as int, float, bool or else kept as text."""
    key, value = text.split("=", 1)
    for cast in (int, float):
        try:
            return key, cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    return key, value


training_views = train.training_views  # the exam trains exactly the way train.py does


def _frame_table(views, subsample, extra):
    return train.frame_table(views, subsample, extra)


def parse_floats(text):
    return [float(v) for v in text.split(",")]


def parse_ints(text):
    return [int(v) for v in text.split(",")]


def grid_settings(grid=GRID):
    """Every combination, in order. A 'min_len_dead' entry gives dead ball its own minimum."""
    out = []
    for t in grid["team_cost"]:
        for d in grid["dead_cost"]:
            for m in grid["min_len"]:
                for md in grid.get("min_len_dead") or (None,):
                    s = {"team_cost": t, "dead_cost": d, "min_len": m, "prior_power": 0.0}
                    if md is not None:
                        s["min_len_dead"] = md
                    out.append(s)
    return out


def score_decoder(exams, probas, priors, decoder):
    """Per-match scores of one decoder setting on out-of-fold probabilities."""
    return {i: _scores(exams[i]["labels"],
                       train.decode_match(probas[i], exams[i], decoder, priors[i]))
            for i in probas}


def evaluate_grid(exams, probas, priors, settings, log=None):
    """[(mean spellF1, setting, per-match scores)], best first; ties keep the grid order."""
    rows = []
    for n, setting in enumerate(settings):
        per = score_decoder(exams, probas, priors, setting)
        mean = float(np.mean([s["spell_f1"] for s in per.values()]))
        rows.append((mean, setting, per))
        if log:
            log("setting {}/{} {}: mean spellF1 {:.4f}".format(n + 1, len(settings), setting, mean))
    return sorted(rows, key=lambda r: -r[0])


def out_of_fold(rounds, n_folds=None, log=print, dfl_dir=DFL_DIR,
                exam_controlled_s=None, train_controlled_s=None,
                lgb_overrides=None, seeds=1, copies=1, extra_features=False,
                train_gaps="random", exam_gaps="random"):
    """Train once per fold. Returns (config, exams, probas, priors) for the exams sat.

    exam_controlled_s / train_controlled_s read the exam / training answers as controlled
    possession with that many seconds as the threshold (possession/labels.py); None = raw DFL.
    lgb_overrides change LightGBM parameters; seeds > 1 averages that many models per fold (seed
    SEED + s); copies > 1 adds corrupted training copies; extra_features adds the 25 extra
    features; train_gaps / exam_gaps ("random" or "dead", see gap_settings) choose how the
    training views and the exam copies lose their ball. The defaults reproduce train.py's model
    exactly.
    """
    params = dict(train.LGB_PARAMS, **(lgb_overrides or {}))
    ids = dfl.match_ids(dfl_dir)
    target = atd.load()
    ball_runs = augment.ball_run_lengths(target)
    train_dg, exam_dg, gap_info = gap_settings(target, train_gaps, exam_gaps)
    exam_frames, train_frames = _frames(exam_controlled_s), _frames(train_controlled_s)
    tables, exams, names = {}, {}, None
    for k, i in enumerate(ids):
        raw = dfl.load(i, dfl_dir)
        teach = raw if train_frames is None else dict(
            raw, labels=controlled(raw["labels"], train_frames))
        Xk, yk, names = _frame_table(training_views(teach, ball_runs, k, copies, train_dg),
                                     train.SUBSAMPLE, extra_features)
        tables[i] = (Xk, yk)
        exams[i] = exam_copy(raw, ball_runs, k, exam_frames, exam_dg)
    log("built training tables and exam copies for {} matches".format(len(ids)))

    turns = folds(ids)[:n_folds] if n_folds else folds(ids)
    probas, priors = {}, {}
    for exam_id, train_ids in turns:
        assert exam_id not in train_ids, "exam match leaked into its own training set"
        t0 = time.time()
        Xtr = np.vstack([tables[j][0] for j in train_ids])
        ytr = np.concatenate([tables[j][1] for j in train_ids])
        dtrain = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=False)
        exam_x = features.build(exams[exam_id], extra=extra_features)[0]
        total = 0.0
        for s in range(seeds):
            booster = lgb.train(dict(params, seed=params["seed"] + s), dtrain,
                                num_boost_round=rounds)
            total = total + booster.predict(exam_x)
        priors[exam_id] = np.bincount(ytr, minlength=3) / len(ytr)
        probas[exam_id] = total / seeds
        log("{}: {} model(s) trained, exam predicted, {:.0f} s".format(
            exam_id, seeds, time.time() - t0))

    config = {"ids": ids, "exams": [e for e, _ in turns], "rounds": rounds,
              "lgb_params": params, "subsample": train.SUBSAMPLE, "seed": train.SEED,
              "exam_seed_offset": EXAM_SEED_OFFSET, "baseline_window": BASELINE_WINDOW,
              "exam_controlled_s": exam_controlled_s, "train_controlled_s": train_controlled_s,
              "lgb_overrides": dict(lgb_overrides or {}), "seeds": seeds, "copies": copies,
              "extra_features": extra_features, "n_features": len(names), **gap_info,
              "exam_fingerprints": {i: exam_fingerprint(exams[i]) for i in ids}}
    return config, exams, probas, priors


def run_exam(rounds, decoder, n_folds=None, log=print, dfl_dir=DFL_DIR,
             exam_controlled_s=None, train_controlled_s=None, dump=None, **knobs):
    """Sit every exam (or the first n_folds). Returns {"config": ..., "matches": {id: scores}}.
    knobs go to out_of_fold (lgb_overrides, seeds, copies, extra_features, train_gaps,
    exam_gaps); dump saves the out-of-fold probabilities as data/cache/diag/<dump>.npz."""
    config, exams, probas, priors = out_of_fold(rounds, n_folds, log, dfl_dir,
                                                exam_controlled_s, train_controlled_s, **knobs)
    if dump:
        log("saved", dumps.save_dump(dump, exams, probas, priors))
    model = score_decoder(exams, probas, priors, decoder)
    matches = {i: {"model": model[i],
                   "baseline": _scores(exams[i]["labels"], baseline_codes(exams[i]))}
               for i in probas}
    config["decoder"] = decoder
    return {"config": config, "matches": matches}


def _reading(seconds):
    return "raw DFL" if seconds is None else "controlled, {:g} s".format(seconds)


def print_grid(rows, config, current, top=10):
    """The best settings, and where the committed decoder setting ranks."""
    print("\ndecoder grid: {} settings on {} exams | LightGBM {} rounds | answers: exam {} | "
          "training {}".format(len(rows), len(config["exams"]), config["rounds"],
                               _reading(config.get("exam_controlled_s")),
                               _reading(config.get("train_controlled_s"))))
    print("{:>4}  {:>12}  {:>9} {:>9} {:>7} {:>8}   {}".format(
        "rank", "mean spellF1", "team_cost", "dead_cost", "min_len", "min_dead", "spellF1 per exam"))
    for rank, (mean, s, per) in enumerate(rows, 1):
        mark = "  <- committed setting" if s == current else ""
        if rank <= top or s == current:
            print("{:>4}  {:>12.4f}  {:>9g} {:>9g} {:>7} {:>8}   {}{}".format(
                rank, mean, s["team_cost"], s["dead_cost"], s["min_len"],
                s.get("min_len_dead", s["min_len"]),
                " ".join("{:.3f}".format(v["spell_f1"]) for v in per.values()), mark))


def print_table(result):
    cfg, matches = result["config"], result["matches"]
    d = cfg["decoder"]
    print("\nexam: {} of {} DFL matches, each held out once | LightGBM {} rounds | decoder "
          "team_cost={} dead_cost={} min_len={} min_len_dead={} prior_power={}".format(
              len(matches), len(cfg["ids"]), cfg["rounds"], d["team_cost"], d["dead_cost"],
              d["min_len"], d.get("min_len_dead", d["min_len"]), d["prior_power"]))

    print("answers: exam {} | training {}".format(_reading(cfg.get("exam_controlled_s")),
                                                  _reading(cfg.get("train_controlled_s"))))
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
    parser.add_argument("--exam-controlled", type=float, metavar="S",
                        help="read the exam answers as controlled possession, threshold S seconds")
    parser.add_argument("--train-controlled", type=float, metavar="S",
                        help="read the training answers as controlled possession, S seconds")
    parser.add_argument("--decoder", type=parse_decoder, metavar="T,D,M",
                        help="sit the exam with team_cost T, dead_cost D, min_len M "
                             "(default: decoder.json)")
    parser.add_argument("--grid", action="store_true",
                        help="score the {} decoder settings of GRID instead of one".format(
                            len(grid_settings())))
    parser.add_argument("--lgb", type=parse_assignment, action="append", metavar="KEY=VALUE",
                        help="override a LightGBM parameter (repeatable), e.g. num_leaves=127")
    parser.add_argument("--seeds", type=int, default=1,
                        help="average this many LightGBM models per exam (seeds SEED, SEED+1, ...)")
    parser.add_argument("--copies", type=int, default=1,
                        help="corrupted training copies per match (each with its team swap)")
    parser.add_argument("--extra-features", nargs="?", const=1, default=0, metavar="LEVEL|GROUPS",
                        type=lambda t: int(t) if t.isdigit() else t,
                        help="features.build's extra groups: a level (1, 2) or names, e.g. "
                             "long,sided,close,ballvx")
    parser.add_argument("--grid-team", type=parse_floats, metavar="T1,T2,...",
                        help="team_cost values for --grid (default: GRID)")
    parser.add_argument("--grid-dead", type=parse_floats, metavar="D1,D2,...",
                        help="dead_cost values for --grid (default: GRID)")
    parser.add_argument("--grid-min-len-dead", type=parse_ints, metavar="M1,M2,...",
                        help="dead-ball min_len values (frames) for --grid (default: = min_len)")
    parser.add_argument("--grid-min-len", type=parse_ints, metavar="M1,M2,...",
                        help="min_len values (frames) for --grid (default: GRID)")
    parser.add_argument("--train-gaps", choices=TRAIN_GAP_MODES, default="random",
                        help="how the training views lose their ball: at random (default), by "
                             "dead and live play measured from the competition inputs, or mixed "
                             "(copy 0 random, copy 1 by dead and live play)")
    parser.add_argument("--exam-gaps", choices=GAP_MODES, default="random",
                        help="the same for the exam copies")
    parser.add_argument("--dump", metavar="TAG",
                        help="save the out-of-fold probabilities as data/cache/diag/<TAG>.npz")
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
    rounds = args.rounds or cfg["rounds"]
    knobs = {"lgb_overrides": dict(args.lgb or []), "seeds": args.seeds, "copies": args.copies,
             "extra_features": args.extra_features, "train_gaps": args.train_gaps,
             "exam_gaps": args.exam_gaps}
    t0 = time.time()
    if args.grid:
        config, exams, probas, priors = out_of_fold(
            rounds, args.folds, dfl_dir=args.dfl_dir, exam_controlled_s=args.exam_controlled,
            train_controlled_s=args.train_controlled, **knobs)
        grid = {"team_cost": tuple(args.grid_team or GRID["team_cost"]),
                "dead_cost": tuple(args.grid_dead or GRID["dead_cost"]),
                "min_len": tuple(args.grid_min_len or GRID["min_len"]),
                "min_len_dead": tuple(args.grid_min_len_dead or ())}
        if args.dump:
            print("saved", dumps.save_dump(args.dump, exams, probas, priors))
        rows = evaluate_grid(exams, probas, priors, grid_settings(grid), log=print)
        print_grid(rows, config, cfg["decoder"])
        result = {"config": config, "grid": [{"mean_spell_f1": m, "decoder": s, "matches": per}
                                             for m, s, per in rows]}
    else:
        result = run_exam(rounds, args.decoder or cfg["decoder"], args.folds,
                          dfl_dir=args.dfl_dir, exam_controlled_s=args.exam_controlled,
                          train_controlled_s=args.train_controlled, dump=args.dump, **knobs)
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
