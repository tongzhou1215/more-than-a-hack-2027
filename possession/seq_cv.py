"""The GRU frame scorer on the 7-match DFL exam, and blends of two exam runs.

    python -m possession.seq_cv --tag g3 --against v3o       # the GRU sits the exam (~8 min, GPU)
    python -m possession.seq_cv --tag g0 --folds 1 --epochs 2  # quick smoke test
    python -m possession.seq_cv --blend v3o g3 --grid         # w * v3o + (1 - w) * g3

The GRU sits exactly the exams possession/cv.py sets. DFL match k's exam copy is corrupted with
seed SEED + 100 + k and its answers are read as controlled possession (2 s). Its model is trained
on the other matches' training views: the same corrupted copies and team swaps LightGBM trains
on, with raw DFL answers. Each fold is trained by seq.fit, the recipe train.py uses for the final
model, so its feature scaler is fitted on that fold's training views only.

Out-of-fold probabilities are saved as data/cache/diag/<tag>.npz, in the format of the
scratchpad's oof_dump.py, so two runs can be compared, blended and diagnosed. --against and --blend
read such dumps (v3 = LightGBM v3, made by oof_dump.py). Only aggregate scores are printed.

No competition label is read; as in cv.py, only the shape of its ball-gap runs.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from . import atd, augment, cv, dfl, features, seq, train
from .common import CACHE_DIR, DFL_DIR, X

DIAG_DIR = os.path.join(CACHE_DIR, "diag")  # data/ is gitignored
EXAM_CONTROLLED_S = 2.0
WEIGHTS = (0.25, 0.5, 0.75)
BLEND_GRID = {"team_cost": (1.0, 2.0, 4.0), "dead_cost": (32.0, 64.0), "min_len": (50, 75, 100)}
KEYS = ("proba", "labels", "seen", "halves", "prior")


def save_dump(tag, exams, probas, priors, folder=DIAG_DIR):
    """Out-of-fold probabilities and what the exams were made of, as <folder>/<tag>.npz."""
    os.makedirs(folder, exist_ok=True)
    out = {}
    for i in probas:
        out[i + "__proba"] = np.asarray(probas[i], dtype=np.float32)
        out[i + "__labels"] = exams[i]["labels"]
        out[i + "__seen"] = ~np.isnan(exams[i]["ball"][:, 0])
        out[i + "__halves"] = np.array(exams[i]["halves"])
        out[i + "__prior"] = np.asarray(priors[i])
    path = os.path.join(folder, tag + ".npz")
    np.savez_compressed(path, ids=np.array(list(probas)), **out)
    return path


def load_dump(tag, folder=DIAG_DIR):
    """{match id: {"proba", "labels", "seen", "halves", "prior"}} from save_dump / oof_dump.py."""
    z = np.load(os.path.join(folder, tag + ".npz"))
    return {str(i): {k: z[str(i) + "__" + k] for k in KEYS} for i in z["ids"]}


def check_same_exams(a, b):
    """ValueError unless the two dumps sat the same exams: the same matches, answers, halves and
    ball mask (the answers alone do not show how the exam copy was corrupted)."""
    if set(a) != set(b):
        raise ValueError("different matches: {} vs {}".format(sorted(a), sorted(b)))
    for i in a:
        for k in ("labels", "halves", "seen"):
            if not np.array_equal(a[i][k], b[i][k]):
                raise ValueError("{}: the two runs sat different exam copies ({})".format(i, k))


def blend(a, b, w):
    """A dump whose probabilities are w * a + (1 - w) * b, after checking both sat the same exams."""
    check_same_exams(a, b)
    return {i: dict(a[i], proba=w * a[i]["proba"].astype(np.float64)
                    + (1.0 - w) * b[i]["proba"].astype(np.float64)) for i in a}


def score_dump(dump, decoder):
    """{match id: cv._scores} for one decoder setting."""
    return {i: cv._scores(m["labels"], train.decode_match(
        m["proba"], {"halves": [tuple(int(v) for v in h) for h in m["halves"]]}, decoder,
        m["prior"])) for i, m in dump.items()}


def out_of_fold_gru(extra, epochs, hidden, device, n_folds=None, log=print, dfl_dir=DFL_DIR):
    """Train the GRU once per fold. Returns (config, exams, probas, priors) like cv.out_of_fold."""
    ids = dfl.match_ids(dfl_dir)
    ball_runs = augment.ball_run_lengths(atd.load())
    views, exams, exam_x, n_features = {}, {}, {}, None
    for k, i in enumerate(ids):
        raw = dfl.load(i, dfl_dir)
        views[i] = []
        for v in cv.training_views(raw, ball_runs, k, train.COPIES):
            f, names, mask = features.build(v, extra=extra)
            views[i].append((f, v["labels"], v["halves"], mask & (v["labels"] != X)))
        exams[i] = cv.exam_copy(raw, ball_runs, k, cv._frames(EXAM_CONTROLLED_S))
        exam_x[i] = features.build(exams[i], extra=extra)[0]
        n_features = len(names)
    log("built training views and exam copies for {} matches, {} features".format(
        len(ids), n_features))

    turns = cv.folds(ids)[:n_folds] if n_folds else cv.folds(ids)
    probas, priors = {}, {}
    for exam_id, train_ids in turns:
        assert exam_id not in train_ids, "exam match leaked into its own training set"
        t0 = time.time()
        fold = [view for j in train_ids for view in views[j]]
        model, mean, std = seq.fit(fold, seed=train.SEED + ids.index(exam_id), device=device,
                                   epochs=epochs, hidden=hidden, log=log)
        counts = sum(np.bincount(y[g], minlength=3)[:3] for _, y, _, g in fold)
        probas[exam_id] = seq.predict_match(model, seq.scale(exam_x[exam_id], mean, std),
                                            exams[exam_id]["halves"], device)
        priors[exam_id] = counts / counts.sum()
        log("{}: GRU trained on {} views, exam predicted, {:.0f} s".format(
            exam_id, len(fold), time.time() - t0))

    config = {"model": "gru", "ids": ids, "exams": [e for e, _ in turns], "epochs": epochs,
              "hidden": hidden, "step": seq.STEP, "window": seq.WINDOW, "layers": seq.LAYERS,
              "dropout": seq.DROPOUT, "batch": seq.BATCH, "lr": seq.LR, "seed": train.SEED,
              "copies": train.COPIES, "extra_features": extra, "n_features": n_features,
              "exam_seed_offset": cv.EXAM_SEED_OFFSET, "exam_controlled_s": EXAM_CONTROLLED_S,
              "train_controlled_s": None, "device": str(device), "torch": torch.__version__,
              "exam_fingerprints": {i: cv.exam_fingerprint(exams[i]) for i in ids}}
    return config, exams, probas, priors


def _mean(scores, key="spell_f1"):
    return float(np.mean([s[key] for s in scores.values()]))


def print_against(new, old, label_new, label_old):
    """Match-by-match spellF1 of two score dicts, and the paired summary."""
    print("\n{} against {} (spellF1, match by match)".format(label_new, label_old))
    for k in new:
        print("  {:<18}{:.4f} -> {:.4f}   {:+.4f}".format(
            k, old[k]["spell_f1"], new[k]["spell_f1"], new[k]["spell_f1"] - old[k]["spell_f1"]))
    s = cv.paired_summary({k: v["spell_f1"] for k, v in new.items()},
                          {k: v["spell_f1"] for k, v in old.items()})
    print("  mean {:.4f} -> {:.4f}, change {:+.4f} | better on {} of {}, worse on {}".format(
        _mean(old), _mean(new), s["mean_delta"], s["wins"], s["n"], s["losses"]))


def run_blend(ref_tag, new_tag, weights, grid):
    ref, new = load_dump(ref_tag), load_dump(new_tag)
    check_same_exams(ref, new)
    base = score_dump(ref, train.DECODER)
    print("decoder {} | {} alone: mean spellF1 {:.4f} | {} alone: {:.4f}".format(
        train.DECODER, ref_tag, _mean(base), new_tag, _mean(score_dump(new, train.DECODER))))
    best = None
    for w in weights:
        scores = score_dump(blend(ref, new, w), train.DECODER)
        s = cv.paired_summary({k: v["spell_f1"] for k, v in scores.items()},
                              {k: v["spell_f1"] for k, v in base.items()})
        print("  w = {:.2f} ({} share): mean spellF1 {:.4f}, macroF1 {:.4f}, change {:+.4f} | "
              "better on {} of {}, worse on {}".format(
                  w, ref_tag, _mean(scores), _mean(scores, "macro_f1"), s["mean_delta"],
                  s["wins"], s["n"], s["losses"]))
        if best is None or _mean(scores) > best[0]:
            best = (_mean(scores), w, scores)
    print_against(best[2], base, "blend w = {:.2f}".format(best[1]), ref_tag)
    if grid:
        mixed = blend(ref, new, best[1])
        rows = []
        for setting in cv.grid_settings(BLEND_GRID):
            scores = score_dump(mixed, setting)
            rows.append((_mean(scores), setting, scores))
        rows.sort(key=lambda r: -r[0])
        print("\ndecoder grid on the blend w = {:.2f}: {} settings".format(best[1], len(rows)))
        for mean, setting, _ in rows[:6]:
            print("  {:.4f}  team_cost {:g} dead_cost {:g} min_len {}".format(
                mean, setting["team_cost"], setting["dead_cost"], setting["min_len"]))
        print_against(rows[0][2], base, "best grid setting", ref_tag)


def main(argv=None):
    p = argparse.ArgumentParser(description="The GRU frame scorer on the DFL exam, and blends.")
    p.add_argument("--tag", help="save the dump as data/cache/diag/<tag>.npz and the scores as "
                                 "data/cache/cv/<tag>.json")
    p.add_argument("--against", help="a saved dump to compare with, match by match (e.g. v3)")
    p.add_argument("--extra", default=str(train.EXTRA_FEATURES),
                   help="features.build's extra groups (default: train.EXTRA_FEATURES)")
    p.add_argument("--epochs", type=int, default=seq.EPOCHS)
    p.add_argument("--hidden", type=int, default=seq.HIDDEN)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--folds", type=int, help="sit only the first N exams (smoke test)")
    p.add_argument("--blend", nargs=2, metavar=("REF", "NEW"),
                   help="blend two saved dumps: w * REF + (1 - w) * NEW")
    p.add_argument("--weights", type=cv.parse_floats, default=WEIGHTS, metavar="W1,W2,...")
    p.add_argument("--grid", action="store_true", help="with --blend: a small decoder grid")
    a = p.parse_args(argv)
    t0 = time.time()

    if a.blend:
        run_blend(a.blend[0], a.blend[1], a.weights, a.grid)
    else:
        if not a.tag:
            p.error("--tag is needed to save the exam run")
        extra = int(a.extra) if a.extra.isdigit() else a.extra
        config, exams, probas, priors = out_of_fold_gru(extra, a.epochs, a.hidden, a.device,
                                                        a.folds)
        save_dump(a.tag, exams, probas, priors)
        mine = load_dump(a.tag)
        scores = score_dump(mine, train.DECODER)
        for k, s in scores.items():
            print("  {:<18} spellF1 {:.4f}  macroF1 {:.4f}  accuracy {:.4f}  poss {}/{}".format(
                k, s["spell_f1"], s["macro_f1"], s["accuracy"], s["n_pred"], s["n_gt"]))
        print("mean spellF1 {:.4f} | decoder {}".format(_mean(scores), train.DECODER))
        config["decoder"] = train.DECODER
        result = {"config": config, "matches": {
            i: {"model": scores[i],
                "baseline": cv._scores(exams[i]["labels"], cv.baseline_codes(exams[i]))}
            for i in scores}}
        os.makedirs(cv.CV_DIR, exist_ok=True)
        with open(os.path.join(cv.CV_DIR, a.tag + ".json"), "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        if a.against:
            ref = load_dump(a.against)
            ref = {i: ref[i] for i in mine if i in ref}
            check_same_exams(mine, ref)
            print_against(scores, score_dump(ref, train.DECODER), a.tag, a.against)
    print("total {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
