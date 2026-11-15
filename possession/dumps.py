"""Exam dumps: out-of-fold probabilities of one exam run, saved so runs can be compared and blended.

cv.py (LightGBM, --dump) and seq_cv.py (the GRU) both write them as data/cache/diag/<tag>.npz:
per exam match its frame probabilities, the exam answers, whether the ball was seen in the exam
copy, the halves and the training prior. Two dumps can only be blended or compared when they sat
the same exam copies, which check_same_exams verifies. numpy only, so no PyTorch is needed here.
"""

import os

import numpy as np

from .common import CACHE_DIR

DIAG_DIR = os.path.join(CACHE_DIR, "diag")  # data/ is gitignored
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
    """{match id: {"proba", "labels", "seen", "halves", "prior"}} from save_dump."""
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
