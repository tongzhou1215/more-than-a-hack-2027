# Possession pipeline (team Mendota Metrics)

This folder is our entry for the More than a Hack 2027 Initial Challenge: one label (`H`, `A`, `D` or `X`) for each of the 199,319 frames of the BAR-ATH tactical video.

This page explains how to install it and run it. The method, the training data and the limitations are described in [`docs/technical_document.md`](../docs/technical_document.md).

Run every command from the repository root.

## 1. Install

Use Python 3.12, the version we tested.

```bash
pip install -r possession/requirements.txt
```

The file pins numpy 2.4.4, pandas 3.0.2, lightgbm 4.7.0 and torch 2.7.1. The CPU build of PyTorch is enough to regenerate the submission. A CUDA build only matters for retraining; the comments in `requirements.txt` give the index URL for each build.

## 2. Put the competition data in place

Unpack the data bundle into `data/` as [`data/README.md`](../data/README.md) describes, or point `MTAH_DATA_DIR` at it. The pipeline reads three files:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

The match data is not in this repository and must not be committed to it. `.gitignore` already excludes `data/`.

## 3. Regenerate the submission

```bash
python -m possession.predict
```

- It writes `evaluation/results/lgbm_viterbi.csv`. The name is historical: the file comes from the LightGBM + GRU blend.
- It takes about 20 seconds on a laptop and needs no GPU.
- The first run parses the tracking file and caches it in `data/cache/`, which is git-ignored.
- `--out some/path.csv` writes the file somewhere else.

The file should have this sha256:

```
e82e6b31bb93d6905f90cfa4e3adee77deaaeb2fc700ce14015b7d8eee6678ee
```

Check the hash with `sha256sum evaluation/results/lgbm_viterbi.csv` on Linux, `shasum -a 256` on macOS, or `Get-FileHash evaluation\results\lgbm_viterbi.csv` in Windows PowerShell.

Then check the format with the organisers' validator:

```bash
python evaluation/validate_submission.py evaluation/results/lgbm_viterbi.csv
```

It should report 199,319 rows and 242 possessions with a mean of 14.1 s.

## 4. Retrain the models (optional)

The submission is produced from the four committed files in `possession/model/`. You only need to retrain if you want to check where they come from.

First download the training data: the Sportec Open DFL dataset, 7 matches, 2.6 GB, CC BY 4.0.

```bash
python -m possession.download_dfl
```

The files go to `../dfl_data`, a folder next to the repository. Set `DFL_DATA_DIR` to use another folder. You can also download the 21 XML files by hand from <https://doi.org/10.6084/m9.figshare.28196177> into one folder.

Then train:

```bash
python -m possession.train
```

- It takes about 2.5 minutes on an RTX 5060 laptop GPU and overwrites the files in `possession/model/`.
- It needs the competition tracking file as well as the DFL data. The training data is corrupted to resemble the competition tracking, and the lengths of its ball gaps are measured from that file. No label of the competition match is read.
- The GRU trains on a GPU when one is available. `--device cpu` forces the CPU.

On the setup we used (Windows 11, RTX 5060 laptop GPU, PyTorch 2.7.1 with CUDA 12.8), a retrain writes the same four files byte for byte. On a CPU or a different GPU the GRU's weights come out slightly different, and the regenerated CSV may then differ in a few frames. The committed files are the reference for the submission; `git checkout -- possession/model` restores them.

| committed file | sha256 |
|---|---|
| `possession/model/lgbm.txt` | `40bd17f3983f4a1b8a51b7d8845f154543517cd9f6b2b994a840f445c462a7b4` |
| `possession/model/gru.npy` | `12f635fae171e26c3c560e04b9db837c5f12bf48ca957609ea9e64c8d2928364` |
| `possession/model/gru.json` | `507eddea56fa44dffaa3abd93c824ece2d3ded2f927b859c73b0f9768ffd5c0f` |
| `possession/model/decoder.json` | `18ecad7af9f189fadf85e21d2a575c00dcf57f5803d404c784ffe1f0ecc9fc41` |

## 5. Exams, diagnostics and tests

These reproduce the numbers in the technical document. All of them need the DFL data and the competition tracking file. None reads a label of the competition match.

| command | what it does | time |
|---|---|---|
| `python -m possession.cv --exam-controlled 2 --decoder 2,64,75 --extra-features 1 --copies 2 --dump lgbm_exam` | LightGBM on the leave-one-match-out exam: mean spellF1 0.6592, official baseline 0.1949 | about 5 min, CPU |
| `python -m possession.seq_cv --tag gru_exam --epochs 6` | The GRU on the same exam: 0.7012 on our GPU | about 8 min, GPU |
| `python -m possession.seq_cv --blend lgbm_exam gru_exam --weights 0.35` | The submitted blend on the exam: 0.7122 | under 1 min |
| `python -m possession.gap` | How far the corrupted training data is from the competition match, from inputs only | about 2 min |
| `python -m possession.diagnose` | Feature percentiles of the two, side by side | under 1 min |

The exam commands save their results under `data/cache/`.

To run the tests:

```bash
pip install pytest
```

```bash
python -m pytest possession/tests
```

There are 99 tests. Those that need PyTorch are skipped when it is not installed.

## 6. What is where

| file | role |
|---|---|
| `atd.py` | Loads the competition tracking into a common representation, in metres. |
| `dfl.py`, `download_dfl.py` | Load and download the DFL matches. |
| `augment.py` | Corrupts clean DFL tracking so that it resembles the competition tracking. |
| `features.py` | The 109 per-frame features. |
| `seq.py` | The bidirectional GRU: training, saving, loading, prediction. |
| `viterbi.py` | The minimum-duration Viterbi decoder. |
| `train.py` | Trains both models on all 7 DFL matches and writes `possession/model/`. |
| `predict.py` | Scores the competition match, applies the kickoff rule, decodes and writes the CSV. |
| `labels.py` | The 2-second "controlled possession" reading used for the exam's answers. |
| `cv.py`, `seq_cv.py`, `dumps.py` | The leave-one-match-out exam for LightGBM and the GRU, and blends of exam runs. |
| `gap.py`, `diagnose.py` | Checks of how the training data differs from the competition match, from inputs only. |
| `model/` | The trained models: `lgbm.txt`, `gru.npy`, `gru.json`, and `decoder.json` with the blend weight and decoder settings. |
| `tests/` | Unit tests. |

## Data credit

The models are trained on official match data of the Deutsche Fußball Liga (DFL), published under CC BY 4.0:

Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12(1), 195. <https://doi.org/10.1038/s41597-025-04505-y>
