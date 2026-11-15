"""The challenge's "controlled possession" reading, applied to labels from another source.

CHALLENGE.md: possession changes only when the other team gains *controlled* possession. A touch
without control or a deflection does not count, and while the ball is contested or loose,
possession stays with the last team that had it. After a stoppage, the restarting team is a clear
possessor.

controlled() turns that into a rule on run lengths, so that DFL's frame-level possession, which
flips on brief touches, reads the way the challenge counts possessions. Within live play, the first
spell after a stoppage keeps its team. After that, a spell of the other team takes over only if it
lasts at least min_frames; otherwise its frames go to the team that already had the ball. Dead-ball
(D) and ungraded (X) frames are never changed, and both count as a stoppage.
"""

import numpy as np

from .common import A, H


def controlled(labels, min_frames):
    """A new int8 array: labels read as controlled possession, with min_frames as the threshold."""
    labels = np.asarray(labels)
    out = labels.astype(np.int8, copy=True)
    if len(labels) == 0:
        return out
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))

    current = None  # the team in possession; None right after a stoppage
    for start, end in zip(starts, ends):
        value = labels[start]
        if value != H and value != A:
            current = None
        elif current is None:
            current = value
        elif value != current:
            if end - start >= min_frames:
                current = value
            else:
                out[start:end] = current
    return out
