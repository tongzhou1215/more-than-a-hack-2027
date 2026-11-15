"""Score the competition match with the trained models and write a submission.

    python -m possession.predict                       # -> evaluation/results/lgbm_viterbi.csv
    python -m possession.predict --out my_team.csv

Every frame's H / A / D probabilities are lgbm_weight x LightGBM + (1 - lgbm_weight) x GRU, with
the weight from decoder.json; a model directory without lgbm_weight is LightGBM alone and needs no
PyTorch. The GRU runs on the CPU in float64 (see gru_proba). Frames outside the two halves are
written as X (never graded). Within a half, frames before its kickoff are dead ball by definition
(see pre_kickoff). Viterbi then turns the probabilities into spells.
"""

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np

from . import atd, features
from .common import ROOT
from .train import MODEL_DIR, decode_match

sys.path.insert(0, ROOT)
from games import GAMES  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

LABELS = np.array(["H", "A", "D", "X"])
KICKOFF_CODE = "set pieces"
KICKOFF_TYPE = "kick off (start)"
HALF_NAMES = ("1st half", "2nd half")


def pre_kickoff(tags, halves, n_frames):
    """Frames of each half before its tagged kickoff, where the ball cannot be in play yet.

    A half starts with its kickoff, so nothing before it is live. The kickoff comes from Metrica's
    automatic Smart Tagging ("Kick Off (Start)", one per half). Its window opens a few seconds
    before the real kickoff, so every frame up to the window's start is certainly before it. No
    frame number of this match appears here, and a match without such a tag is left unchanged.
    """
    mask = np.zeros(n_frames, dtype=bool)
    for h, (start, stop) in enumerate(halves[:len(HALF_NAMES)]):
        kicks = [t["start_frame"] for t in tags
                 if (t["code"] or "").strip().lower() == KICKOFF_CODE
                 and (t["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE
                 and (t["tags"].get("Half") or "").strip().lower() == HALF_NAMES[h]]
        if kicks:
            mask[start:max(start, min(min(kicks), stop))] = True
    return mask


def gru_proba(f, names, halves, model_dir=MODEL_DIR):
    """The GRU's probabilities (T, 3) for the match's features f. They are computed on the CPU in
    float64, the saved float32 weights upcast exactly, so that last-bit differences between
    machines cannot change a decision."""
    from . import seq  # PyTorch is needed only when the model directory holds a GRU
    model, mean, std, info = seq.load(model_dir)
    assert info["features"] == names, "the GRU was trained on other features; retrain"
    return seq.predict_match(model.double(), seq.scale(f, mean, std), halves, "cpu",
                             info["step"], info["window"])


def predict(match=None, model_dir=MODEL_DIR, kickoff_rule=True):
    match = match or atd.load()
    with open(os.path.join(model_dir, "decoder.json")) as handle:
        cfg = json.load(handle)
    booster = lgb.Booster(model_file=os.path.join(model_dir, "lgbm.txt"))
    f, names, _ = features.build(match, extra=cfg.get("extra_features", 0))
    assert names == cfg["features"], "feature list changed since training; retrain"
    proba = booster.predict(f, num_threads=1)
    weight = cfg.get("lgbm_weight", 1.0)
    if weight < 1.0:
        proba = weight * proba + (1.0 - weight) * gru_proba(f, names, match["halves"], model_dir)
    if kickoff_rule and match["name"] in GAMES:
        dead = pre_kickoff(load_smart_tagging(match["name"]), match["halves"], len(proba))
        proba = proba.copy()
        proba[dead] = (0.0, 0.0, 1.0)
    return decode_match(proba, match, cfg["decoder"], cfg["prior"]), proba


def write(path, codes):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LABELS[c]) for i, c in enumerate(codes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "lgbm_viterbi.csv"))
    args = parser.parse_args()
    codes, _ = predict()
    write(args.out, codes)
    counts = {LABELS[k]: int(np.sum(codes == k)) for k in range(4)}
    print("wrote", args.out, counts)


if __name__ == "__main__":
    main()
