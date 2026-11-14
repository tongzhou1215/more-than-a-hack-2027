"""The one match representation every loader produces.

A match is a dict:
    home   float32 (T, Ph, 2)  player positions in metres, origin at the centre spot, NaN = unseen
    away   float32 (T, Pa, 2)
    ball   float32 (T, 2)      NaN = unseen
    halves list of (start, stop) half-open frame spans that are actually played
    labels int8 (T,) or None   0=H 1=A 2=D 3=X (ungraded), same codes as evaluation/possession_metrics
    pitch  (length, width) in metres
    name   str

Slot order inside home/away carries no meaning: features must be permutation invariant, because
the target match has tracklets, not players.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "cache")  # data/ is gitignored, so derived arrays stay out of git
DFL_DIR = os.environ.get("DFL_DATA_DIR") or os.path.join(os.path.dirname(ROOT), "dfl_data")

H, A, D, X = 0, 1, 2, 3
FPS = 25


def ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR
