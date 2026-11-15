"""A bidirectional GRU frame scorer: the same per-frame features LightGBM sees, read in order.

LightGBM scores every frame on its own and knows the surrounding seconds only through the rolling
means in features.py. The GRU reads each half forwards and backwards, so it can learn what
happens around a change of possession: who the ball left, who received it.

Each half is read at 5 frames per second in STEP phase-shifted sequences (frames o, o+5, o+10,
...), so every frame is still scored and a WINDOW-step window covers 60 s. Training draws random
windows. Prediction runs overlapping windows and takes each step from the window whose centre is
nearest, where the GRU sees the most context on both sides.

fit() is the whole training recipe (feature scaler, then the network), shared by the exam
(seq_cv.py) and the final model (train.py). save() / load() keep a model as MODEL_FILES: the
parameters as one flat float32 array and a JSON file with everything else, so the same model always
gives the same bytes. Prediction runs in the model's dtype: predict.py upcasts the float32 weights
to float64, where last-bit differences between machines cannot change a decision.

Needs PyTorch (possession/requirements.txt pins 2.7.1). Seeded and deterministic.
"""

import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; before CUDA

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .common import X  # noqa: E402

STEP = 5           # frames between two steps of a sequence: 5 per second at 25 fps
WINDOW = 300       # steps per training and prediction window: 60 s
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 6         # an epoch is about one pass over every training step; chosen on the exam,
                   # where 30 over-trains (g1 0.6241 against g2's 0.6573 at 6)
BATCH = 64
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
CLIP = 10.0        # standardised features are clipped to +/- CLIP
MIN_STD = 1e-6     # a feature that varies less than this is treated as constant
MODEL_FILES = ("gru.npy", "gru.json")  # the parameters (flat float32) and everything else
SEED_STRIDE = 1000  # member s of an ensemble seeded k is seeded k + SEED_STRIDE * s


def fit_scaler(xs, masks):
    """Per-feature mean and standard deviation over the masked frames of all views together,
    ignoring non-finite values. A feature without a value gets mean 0, a constant one std 1."""
    n_feat = xs[0].shape[1]
    count, total = np.zeros(n_feat), np.zeros(n_feat)
    for x, m in zip(xs, masks):
        v = x[m].astype(np.float64)
        ok = np.isfinite(v)
        count += ok.sum(axis=0)
        total += np.where(ok, v, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros(n_feat), where=count > 0)
    dev = np.zeros(n_feat)
    for x, m in zip(xs, masks):
        v = x[m].astype(np.float64)
        dev += np.where(np.isfinite(v), (v - mean) ** 2, 0.0).sum(axis=0)
    std = np.sqrt(np.divide(dev, count, out=np.zeros(n_feat), where=count > 0))
    std = np.where(std > MIN_STD, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def scale(x, mean, std):
    """(x - mean) / std as float32: NaN -> 0, +/-inf -> +/-CLIP, then clipped to +/-CLIP."""
    with np.errstate(invalid="ignore"):
        z = (x.astype(np.float32) - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=CLIP, neginf=-CLIP)
    return np.clip(z, -CLIP, CLIP).astype(np.float32)


def phase_index(start, stop, step=STEP):
    """Frame indices of each phase of the half [start, stop): o, o+step, ... for o < step."""
    return [i for i in (np.arange(start + o, stop, step) for o in range(step)) if len(i)]


def sequences(x, labels, halves, step=STEP):
    """Every half's phase sequences, in half order then phase order: [(x (L, F), y (L,))]."""
    return [(x[i].astype(np.float32), labels[i].astype(np.int64))
            for start, stop in halves for i in phase_index(start, stop, step)]


def sample_windows(seqs, n, length, rng):
    """n random windows of `length` steps. A sequence is picked in proportion to its length and
    a start uniformly inside it; a shorter sequence is padded with zeros labelled X."""
    lengths = np.array([len(y) for _, y in seqs], dtype=float)
    pick = rng.choice(len(seqs), size=n, p=lengths / lengths.sum())
    xb = np.zeros((n, length, seqs[0][0].shape[1]), dtype=np.float32)
    yb = np.full((n, length), X, dtype=np.int64)
    for row, i in enumerate(pick):
        x, y = seqs[i]
        start = int(rng.integers(0, max(1, len(y) - length + 1)))
        k = min(length, len(y) - start)
        xb[row, :k] = x[start:start + k]
        yb[row, :k] = y[start:start + k]
    return xb, yb


class SeqNet(torch.nn.Module):
    """Linear -> GELU -> bidirectional GRU -> Linear: logits for H, A, D at every step."""

    def __init__(self, n_features, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.inp = torch.nn.Linear(n_features, hidden)
        self.gru = torch.nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = torch.nn.Linear(2 * hidden, 3)

    def forward(self, x):
        h, _ = self.gru(F.gelu(self.inp(x)))
        return self.out(h)


def train_model(seqs, n_features, seed, device="cpu", epochs=EPOCHS, hidden=HIDDEN,
                window=WINDOW, batch=BATCH, log=None):
    """A SeqNet trained on phase sequences [(x, y)], X labels ignored; returned in eval mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SeqNet(n_features, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    per_epoch = max(1, math.ceil(sum(len(y) for _, y in seqs) / (batch * window)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * per_epoch)
    model.train()
    for epoch in range(epochs):
        losses = []
        for _ in range(per_epoch):
            xb, yb = sample_windows(seqs, batch, window, rng)
            if np.all(yb == X):
                continue  # nothing graded: the loss would be 0 / 0
            logits = model(torch.from_numpy(xb).to(device))
            loss = F.cross_entropy(logits.reshape(-1, 3), torch.from_numpy(yb).to(device).reshape(-1),
                                   ignore_index=X)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if log:
            log("epoch {}/{}: training loss {:.4f}".format(
                epoch + 1, epochs, float(np.mean(losses)) if losses else float("nan")))
    model.eval()
    return model


def member_seeds(seed, n):
    """The seeds of an n-member ensemble seeded `seed`; member 0 keeps `seed` itself."""
    return [seed + SEED_STRIDE * s for s in range(n)]


def fit_many(views, seeds, device="cpu", epochs=EPOCHS, hidden=HIDDEN, window=WINDOW,
             batch=BATCH, log=None):
    """The scaler fitted on every graded frame of the views, then one SeqNet per seed trained on
    their phase sequences. The members see the same data and differ only in their seed.
    views: [(features (T, F), labels (T,), halves, graded mask (T,))].
    Returns ([models in eval mode], mean, std)."""
    mean, std = fit_scaler([f for f, _, _, _ in views], [g for _, _, _, g in views])
    seqs = []
    for f, y, halves, _ in views:
        seqs += sequences(scale(f, mean, std), y, halves)
    models = [train_model(seqs, views[0][0].shape[1], s, device, epochs, hidden, window, batch,
                          log) for s in seeds]
    return models, mean, std


def fit(views, seed, device="cpu", epochs=EPOCHS, hidden=HIDDEN, window=WINDOW, batch=BATCH,
        log=None):
    """fit_many with one seed: (model in eval mode, mean, std)."""
    models, mean, std = fit_many(views, [seed], device, epochs, hidden, window, batch, log)
    return models[0], mean, std


def save(folder, model, mean, std, meta=None):
    """Write MODEL_FILES into folder: the parameters as one flat float32 array (np.save stores no
    timestamp, so the same model always gives the same bytes) and a JSON file with `meta`, the
    network's size, the step and window prediction uses (this module's STEP and WINDOW), the
    parameters' names and shapes, and the feature scaler."""
    os.makedirs(folder, exist_ok=True)
    state = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    info = dict(meta or {})
    info.update(n_features=model.inp.in_features, hidden=model.gru.hidden_size,
                layers=model.gru.num_layers, dropout=model.gru.dropout, step=STEP, window=WINDOW,
                params=[[k, list(v.shape)] for k, v in state.items()],
                mean=[float(v) for v in mean], std=[float(v) for v in std])
    flat = np.concatenate([v.ravel() for v in state.values()]).astype(np.float32)
    np.save(os.path.join(folder, MODEL_FILES[0]), flat, allow_pickle=False)
    with open(os.path.join(folder, MODEL_FILES[1]), "w", newline="\n") as handle:
        json.dump(info, handle, indent=2)


def load(folder):
    """(model on the CPU in eval mode, mean, std, info) from the MODEL_FILES save() wrote."""
    with open(os.path.join(folder, MODEL_FILES[1])) as handle:
        info = json.load(handle)
    flat = np.load(os.path.join(folder, MODEL_FILES[0]), allow_pickle=False)
    sizes = [int(np.prod(shape)) for _, shape in info["params"]]
    if flat.size != sum(sizes):
        raise ValueError("{} holds {} numbers but {} describes {}".format(
            MODEL_FILES[0], flat.size, MODEL_FILES[1], sum(sizes)))
    model = SeqNet(info["n_features"], info["hidden"], info["layers"], info["dropout"])
    parts = np.split(flat, np.cumsum(sizes)[:-1])
    model.load_state_dict({name: torch.from_numpy(p.reshape(shape).copy())
                           for (name, shape), p in zip(info["params"], parts)})
    model.eval()
    return (model, np.array(info["mean"], dtype=np.float32),
            np.array(info["std"], dtype=np.float32), info)


def _dtypes(model):
    """(torch dtype, numpy dtype) a model computes in: its parameters', float32 without any."""
    p = next(model.parameters(), None)
    dtype = torch.float32 if p is None else p.dtype
    return dtype, torch.empty(0, dtype=dtype).numpy().dtype


def _window_starts(n, window):
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, window // 2))
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


def _predict_sequence(model, x, device, window):
    """Probabilities (n, 3) for one phase sequence from overlapping windows; every step is taken
    from the window whose centre is nearest."""
    n = len(x)
    w = min(window, n)
    starts = _window_starts(n, w)
    dtype, np_dtype = _dtypes(model)
    batch = torch.from_numpy(np.stack([x[s:s + w] for s in starts])).to(device=device, dtype=dtype)
    with torch.no_grad():
        prob = torch.softmax(model(batch), dim=-1).cpu().numpy()
    out = np.empty((n, 3), dtype=np_dtype)
    for j, s in enumerate(starts):
        lo = 0 if j == 0 else (starts[j - 1] + s) // 2 + w // 2
        hi = n if j == len(starts) - 1 else (s + starts[j + 1]) // 2 + w // 2
        out[lo:hi] = prob[j, lo - s:hi - s]
    return out


def predict_half(model, x_half, device="cpu", step=STEP, window=WINDOW):
    """Probabilities (L, 3) for every frame of one half: each phase sequence is predicted on its
    own and put back into its frames."""
    model.eval()
    out = np.empty((len(x_half), 3), dtype=_dtypes(model)[1])
    for i in phase_index(0, len(x_half), step):
        out[i] = _predict_sequence(model, np.ascontiguousarray(x_half[i], dtype=np.float32),
                                   device, window)
    return out


def predict_match(model, x, halves, device="cpu", step=STEP, window=WINDOW):
    """Probabilities (T, 3) for a whole match: the halves scored, every other frame 1/3 each."""
    out = np.full((len(x), 3), 1.0 / 3.0, dtype=_dtypes(model)[1])
    for start, stop in halves:
        out[start:stop] = predict_half(model, x[start:stop], device, step, window)
    return out


def predict_members(models, x, halves, device="cpu", step=STEP, window=WINDOW):
    """The mean of the members' predict_match probabilities, summed in member order."""
    total = None
    for model in models:
        p = predict_match(model, x, halves, device, step, window)
        total = p if total is None else total + p
    return total / len(models)
