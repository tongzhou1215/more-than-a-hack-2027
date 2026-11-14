"""Score the competition match with the trained model and write a submission.

    python -m possession.predict                       # -> evaluation/results/lgbm_viterbi.csv
    python -m possession.predict --out my_team.csv

Frames outside the two halves are written as X (never graded).
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from . import atd, features, viterbi
from .common import ROOT, X
from .train import MODEL_DIR

LABELS = np.array(["H", "A", "D", "X"])


def predict(match=None, model_dir=MODEL_DIR):
    match = match or atd.load()
    cfg = json.load(open(os.path.join(model_dir, "decoder.json")))
    booster = lgb.Booster(model_file=os.path.join(model_dir, "lgbm.txt"))
    f, names, _ = features.build(match)
    assert names == cfg["features"], "feature list changed since training; retrain"
    proba = booster.predict(f, num_threads=1)
    d = cfg["decoder"]
    out = np.full(len(proba), X, dtype=np.int8)
    for start, stop in match["halves"]:
        out[start:stop] = viterbi.decode(proba[start:stop], d["team_cost"], d["dead_cost"],
                                         (d["min_len"],) * 3, cfg["prior"], d["prior_power"])
    return out, proba


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
