"""Tests for possession/seq.py, the GRU frame scorer. Synthetic arrays only; needs PyTorch."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from possession import seq  # noqa: E402
from possession.common import X  # noqa: E402


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Echo(torch.nn.Module):
    """Logits = the first three input features, so each output says which frame it belongs to."""

    def forward(self, x):
        return x[..., :3]


class Where(torch.nn.Module):
    """Logits = [0.1 * the step's position inside the window it was run in, 0, 0]."""

    def forward(self, x):
        pos = torch.arange(x.shape[1], dtype=x.dtype).expand(x.shape[0], -1) * 0.1
        zero = torch.zeros_like(pos)
        return torch.stack([pos, zero, zero], dim=-1)


def _delayed_rule(rng, n=400, delay=3):
    """Two noise features; the label is H when feature 0 was positive `delay` steps earlier."""
    f = rng.normal(size=(n, 2)).astype(np.float32)
    y = np.where(np.roll(f[:, 0], delay) > 0, 0, 1).astype(np.int64)
    y[:delay] = X
    return f, y


def test_scaler_uses_masked_frames_and_handles_nan_and_inf():
    x = np.array([[1.0, 10.0, np.nan, 0.0],
                  [3.0, np.nan, np.nan, np.inf],
                  [5.0, 10.0, np.nan, 0.0],
                  [100.0, 100.0, np.nan, 0.0]], dtype=np.float32)
    mask = np.array([True, True, True, False])
    mean, std = seq.fit_scaler([x], [mask])
    assert np.allclose(mean, [3.0, 10.0, 0.0, 0.0])
    assert np.allclose(std, [np.sqrt(8.0 / 3.0), 1.0, 1.0, 1.0])
    z = seq.scale(x, mean, std)
    assert z.dtype == np.float32
    assert z[1, 1] == 0.0          # NaN -> 0
    assert z[3, 0] == seq.CLIP     # far outside the training range -> clipped
    assert np.all(z[:, 2] == 0.0)  # a column without a single value
    assert z[1, 3] == seq.CLIP     # +inf -> +CLIP


def test_scaler_pools_every_view():
    a = np.array([[0.0], [2.0]], dtype=np.float32)
    b = np.array([[4.0]], dtype=np.float32)
    mean, std = seq.fit_scaler([a, b], [np.ones(2, bool), np.ones(1, bool)])
    assert np.allclose(mean, [2.0])  # not the mean of the two views' means (2.5)
    assert np.allclose(std, [np.sqrt(8.0 / 3.0)])


def test_phases_cover_each_frame_of_a_half_once():
    idx = seq.phase_index(10, 33, 5)
    assert len(idx) == 5
    assert np.array_equal(np.sort(np.concatenate(idx)), np.arange(10, 33))
    assert all(np.all(np.diff(i) == 5) for i in idx)
    assert [list(i) for i in seq.phase_index(0, 3, 5)] == [[0], [1], [2]]


def test_sequences_split_halves_into_phases():
    x = np.arange(20, dtype=np.float32)[:, None]
    y = (np.arange(20) % 3).astype(np.int8)
    out = seq.sequences(x, y, [(0, 10), (12, 20)], step=5)
    assert len(out) == 10
    assert np.array_equal(out[0][0][:, 0], [0, 5])
    assert np.array_equal(out[5][0][:, 0], [12, 17])
    assert np.array_equal(out[5][1], [0, 2])  # the labels of frames 12 and 17
    assert out[0][0].dtype == np.float32 and out[0][1].dtype == np.int64


def test_windows_have_the_asked_shape_and_pad_short_sequences():
    long_seq = (np.ones((50, 4), np.float32), np.zeros(50, np.int64))
    short_seq = (np.ones((5, 4), np.float32), np.ones(5, np.int64))
    xb, yb = seq.sample_windows([long_seq, short_seq], 200, 20, np.random.default_rng(0))
    assert xb.shape == (200, 20, 4) and xb.dtype == np.float32
    assert yb.shape == (200, 20) and yb.dtype == np.int64
    short = yb[:, 0] == 1
    assert 0 < short.sum() < 60  # picked in proportion to length: about 5 / 55 of the windows
    assert np.all(yb[short, 5:] == X) and np.all(xb[short, 5:] == 0)
    assert np.all(yb[~short] == 0) and np.all(xb[~short] == 1)


@pytest.mark.parametrize("L", [3, 7, 103, 1003])
def test_prediction_puts_every_frame_back_in_its_place(L):
    x = np.random.default_rng(1).normal(size=(L, 4)).astype(np.float32)
    p = seq.predict_half(Echo(), x, "cpu", step=5, window=20)
    assert p.shape == (L, 3) and p.dtype == np.float32
    assert np.allclose(p, _softmax(x[:, :3]), atol=1e-6)


def test_prediction_keeps_the_centre_of_overlapping_windows():
    p = seq.predict_half(Where(), np.zeros((100, 1), np.float32), "cpu", step=1, window=20)
    pos = np.round(10 * np.log(p[:, 0] / p[:, 1])).astype(int)
    assert pos[0] == 0 and pos[99] == 19  # the two ends come from the first and the last window
    assert np.all((pos[15:85] >= 5) & (pos[15:85] <= 14))


def test_predict_match_scores_halves_and_leaves_the_rest_uniform():
    x = np.random.default_rng(2).normal(size=(40, 4)).astype(np.float32)
    p = seq.predict_match(Echo(), x, [(5, 20), (25, 40)], "cpu", step=5, window=20)
    assert np.allclose(p[:5], 1 / 3) and np.allclose(p[20:25], 1 / 3)
    assert np.allclose(p[5:20], _softmax(x[5:20, :3]), atol=1e-6)
    assert np.allclose(p[25:40], _softmax(x[25:40, :3]), atol=1e-6)


def test_seqnet_reads_both_directions():
    torch.manual_seed(0)
    net = seq.SeqNet(4, hidden=8).eval()
    x = torch.zeros(1, 10, 4)
    late, early = x.clone(), x.clone()
    late[0, -1] = 5.0
    early[0, 0] = 5.0
    with torch.no_grad():
        base = net(x)
        assert base.shape == (1, 10, 3)
        assert not torch.equal(net(late)[0, 0], base[0, 0])     # the first step sees the future
        assert not torch.equal(net(early)[0, -1], base[0, -1])  # the last step sees the past


def test_training_learns_a_rule_that_needs_the_past():
    # The label is the sign of the previous step's feature, which the current step says nothing
    # about: a frame-by-frame model is right half the time, a model with memory nearly always.
    rng = np.random.default_rng(0)
    data = [_delayed_rule(rng, delay=1) for _ in range(8)]
    net = seq.train_model(data, 2, seed=0, epochs=60, hidden=16, window=50, batch=16)
    f, y = _delayed_rule(rng, delay=1)
    p = seq.predict_half(net, f, "cpu", step=1, window=50)
    assert np.mean(p[1:].argmax(axis=1) == y[1:]) > 0.9


def test_training_is_repeatable_on_cpu():
    rng = np.random.default_rng(1)
    data = [_delayed_rule(rng, n=120) for _ in range(2)]
    a = seq.train_model(data, 2, seed=1, epochs=2, hidden=8, window=50, batch=4)
    b = seq.train_model(data, 2, seed=1, epochs=2, hidden=8, window=50, batch=4)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_training_survives_windows_with_no_graded_frame():
    rng = np.random.default_rng(2)
    blank = (rng.normal(size=(200, 2)).astype(np.float32), np.full(200, X, np.int64))
    net = seq.train_model([blank, _delayed_rule(rng, n=200)], 2, seed=2, epochs=5, hidden=8,
                          window=50, batch=1)
    assert all(torch.isfinite(p).all() for p in net.parameters())


def _views(rng, n_views=2, T=120):
    """Synthetic training views for seq.fit: (features, labels, halves, graded mask)."""
    out = []
    for _ in range(n_views):
        f, y = _delayed_rule(rng, n=T, delay=1)
        f[rng.random(T) < 0.1, 1] = np.nan
        out.append((f, y.astype(np.int8), [(0, T // 2), (T // 2, T)], y != X))
    return out


def _small_model(seed=4, n_features=5):
    torch.manual_seed(seed)
    return seq.SeqNet(n_features, hidden=4).eval()


def test_fit_is_scaler_then_sequences_then_training():
    views = _views(np.random.default_rng(3))
    model, mean, std = seq.fit(views, seed=3, epochs=1, hidden=4, window=20, batch=4)
    m0, s0 = seq.fit_scaler([v[0] for v in views], [v[3] for v in views])
    seqs = [s for f, y, h, _ in views for s in seq.sequences(seq.scale(f, m0, s0), y, h)]
    ref = seq.train_model(seqs, 2, seed=3, epochs=1, hidden=4, window=20, batch=4)
    assert np.array_equal(mean, m0) and np.array_equal(std, s0)
    sa, sb = model.state_dict(), ref.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_save_and_load_give_back_the_same_model(tmp_path):
    model = _small_model()
    mean = np.array([0.1, -2.5, 3e-7, 1e6, 0.0], np.float32)
    std = np.array([1.0, 0.3, 2.0, 1e-3, 7.0], np.float32)
    seq.save(str(tmp_path), model, mean, std, {"epochs": 6})
    back, m2, s2, info = seq.load(str(tmp_path))
    assert np.array_equal(m2, mean) and m2.dtype == np.float32
    assert np.array_equal(s2, std) and s2.dtype == np.float32
    assert info["epochs"] == 6 and info["n_features"] == 5 and info["hidden"] == 4
    assert not back.training
    sa, sb = model.state_dict(), back.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    x = np.random.default_rng(4).normal(size=(30, 5)).astype(np.float32)
    assert np.array_equal(seq.predict_half(model, x, step=1, window=10),
                          seq.predict_half(back, x, step=1, window=10))


def test_saving_twice_writes_the_same_bytes(tmp_path):
    model = _small_model()
    mean, std = np.zeros(5, np.float32), np.ones(5, np.float32)
    for name in ("a", "b"):
        seq.save(str(tmp_path / name), model, mean, std, {"epochs": 6})
    for f in seq.MODEL_FILES:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_load_refuses_weights_that_do_not_fit(tmp_path):
    seq.save(str(tmp_path), _small_model(), np.zeros(5, np.float32), np.ones(5, np.float32))
    weights = str(tmp_path / seq.MODEL_FILES[0])
    np.save(weights, np.load(weights)[:-1])
    with pytest.raises(ValueError):
        seq.load(str(tmp_path))


def test_prediction_runs_in_the_model_dtype():
    model = _small_model()
    x = np.random.default_rng(5).normal(size=(64, 5)).astype(np.float32)
    p32 = seq.predict_match(model, x, [(0, 64)], step=2, window=16)
    p64 = seq.predict_match(copy.deepcopy(model).double(), x, [(0, 64)], step=2, window=16)
    assert p32.dtype == np.float32 and p64.dtype == np.float64
    assert np.allclose(p32, p64, atol=1e-5) and not np.array_equal(p32, p64)
