"""Possession pipeline: per-frame scores from LightGBM and a bidirectional GRU, blended, then
Viterbi segmentation.

Every match, whatever its source, is turned into the same dict (see common.py) before features
are computed, so the model trained on other games sees exactly the representation it is applied to.
"""
