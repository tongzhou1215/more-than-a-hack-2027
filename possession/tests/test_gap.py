"""Tests for possession/gap.py, the domain-gap check. Synthetic arrays only."""

import numpy as np
import pytest

from possession import gap


def test_auc_counts_how_often_a_positive_outranks_a_negative():
    assert gap.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert gap.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert gap.auc([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5
    assert gap.auc([1, 2, 2, 3], [0, 0, 1, 1]) == pytest.approx(7 / 8)


def test_separability_finds_the_shifted_and_the_missing_feature():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2000, 3)), rng.normal(size=(2000, 3))
    b[:, 0] += 3.0
    a[rng.random(2000) < 0.3, 1] = np.nan
    s = gap.separability(a, b)
    assert s[0] > 0.95 and 0.6 < s[1] < 0.7 and abs(s[2] - 0.5) < 0.05


def test_block_folds_are_contiguous_and_balanced():
    assert gap.block_folds(10, 3).tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_domain_auc_is_chance_for_one_distribution_and_high_for_two():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(3000, 4)), rng.normal(size=(1000, 4))
    fa, fb = gap.block_folds(3000, 5), gap.block_folds(1000, 5)
    same, _ = gap.domain_auc(a, b, fa, fb, rounds=30)
    b[:, 1] += 1.5
    far, share = gap.domain_auc(a, b, fa, fb, rounds=30)
    assert abs(same - 0.5) < 0.05
    assert far > 0.8 and int(np.argmax(share)) == 1


def test_rate_by_bins_gives_the_share_in_each_bin():
    v = np.array([0.1, 0.2, np.nan, 0.6, 0.7, 0.9])
    flags = np.array([1, 1, 1, 0, 1, 0])
    assert gap.rate_by_bins(v, flags, [0.5]).tolist() == pytest.approx([1.0, 1 / 3])
