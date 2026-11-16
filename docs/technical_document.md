# Mendota Metrics: technical document

More than a Hack 2027, Initial Challenge: frame-level ball possession on BAR-ATH.

- Repository: <https://github.com/tongzhou1215/more-than-a-hack-2027>. Our code is in `possession/`, with installation and commands in [`possession/README.md`](../possession/README.md); everything else is the organisers' starter repository.
- Submission file: written by `python -m possession.predict`, sha256 `e82e6b31bb93d690…` (section 6).

## Summary

Two models trained on other matches give every frame a probability for home, away and dead ball: a gradient-boosted tree model (LightGBM) and a bidirectional GRU. Both read the same 109 features, computed from the provided tactical tracking. Their probabilities are blended 0.35 / 0.65, and a Viterbi decoder with a 3-second minimum duration turns them into possessions and stoppages.

```
tactical ATD ─► 109 features ─┬─► LightGBM ─┐
(players, ball)   per frame   └─► GRU ──────┴─► 0.35 / 0.65 blend ─► kickoff rule ─► Viterbi, 3 s minimum ─► H / A / D
                                                                          ▲
                              Smart Tagging "Kick Off (Start)" ───────────┘            outside the two halves: X
```

- **Training data:** the 7 public matches of the Sportec Open DFL dataset, corrupted to resemble the provided tracking. No label of the provided game was used to train, tune or select anything.
- **Estimated quality:** mean spellF1 0.712 on a leave-one-match-out exam over the 7 DFL matches, where the official baseline scores 0.195.
- **Public 5-minute window,** checked once, on the final file: spellF1 0.690, macroF1 0.908.
- **Reproduction:** the repository regenerates the submitted file byte for byte.

## 1. Approach

### 1.1 What we built

| step | module | what it does |
|---|---|---|
| load | `possession/atd.py` | Reads the tactical ATD into metres (players and ball), with the graded halves from `games.half_video_blocks`. |
| features | `possession/features.py` | 109 numbers per frame, independent of the order of the player slots, computed within each half and oriented so that the home team always defends the same goal. |
| frame scorer 1 | `possession/train.py` | LightGBM, 3 classes, 126 rounds of 63-leaf trees, trained on every 5th frame. |
| frame scorer 2 | `possession/seq.py` | Bidirectional GRU (2 layers, 64 hidden units, 131,843 parameters) reading each half at 5 steps per second, in 60-second windows. |
| blend | `possession/predict.py` | 0.35 × LightGBM + 0.65 × GRU. |
| kickoff rule | `possession/predict.py` | Frames of a half before its tagged kickoff become dead ball (section 4). |
| decoder | `possession/viterbi.py` | Viterbi over H, A and D, on each half. Every state lasts at least 75 frames (3 s); changing team costs 2 and changing between alive and dead costs 64, in log-probability units. |

The features describe the **ball** (seen or not, time since and until it is seen, interpolated position and speed, distance to the lines, off the pitch or not), the **players around it** (each team's nearest and second-nearest player, its players within 5 m and 15 m, how fast its three nearest close in), each team's **shape and motion** (centroid, spread, deepest and most advanced player, visible players, speeds, how fast the block moves forward) and the **time context**: the main signals averaged over ±1, 3, 10 and 20 s, plus the 5 s before and the 5 s after given separately, because a change of possession flips who is near the ball and which way each block moves. Speeds are divided by their median over the half: the provided tracking measures players about 30% slower than DFL (median 1.5 against 2.1 m/s), and raw speeds would read as dead ball.

### 1.2 Why this and not something else

1. **The metric counts whole possessions.** Frame-by-frame argmax flickers. In our first version (trained on five DFL matches, scored on the other two) it scored spellF1 0.11 with about 2,800 fragments per match; the same probabilities through the decoder scored 0.50. The decoder is half of the method.
2. **The ball is missing on 41% of graded frames,** where a nearest-player rule like the baseline has nothing to say. Most of our features therefore describe the players.
3. **No label of the provided game may be used,** so we learn from other matches whose labels mean the same thing. DFL's tracking carries a ball status and a possessing team on every frame, which is the construction CHALLENGE.md gives for the ground truth. The price is a domain gap, which we reduce (section 3) and measure (section 5).

LightGBM sees each frame on its own and the surrounding seconds only through rolling means; the GRU reads the sequence in both directions and can learn what happens around a hand-over, so the two complement each other. We considered but did not build ball detection in the video, a CRF over player trajectories in the style of PathCRF, and open tracking data produced from broadcast video, such as SkillCorner's.

### 1.3 How every choice was made

We could not score ourselves on the provided match, so we built an exam from DFL (`possession/cv.py`, `possession/seq_cv.py`). Each of the 7 matches is held out once and the models are trained on the other 6. The held-out match is corrupted like the training data, with a fixed seed, so every run sits the same exam. The answers are read as controlled possession with a 2-second threshold (section 4) and scored with the organisers' `possession_metrics.evaluate`. A change was adopted only if it was better on at least 5 of the 7 matches and raised the mean spellF1 by at least 0.005, a rule fixed before the experiments of section 2. The public 5 minutes took no part in any choice.

| spellF1 on the 7 held-out matches | official baseline | LightGBM | GRU | blend (submitted) |
|---|---|---|---|---|
| mean | 0.1949 | 0.6592 | 0.7012 | **0.7122** |
| worst to best match | 0.1671–0.2078 | 0.6367–0.6750 | 0.6725–0.7339 | 0.6900–0.7391 |

The blend beats LightGBM on all 7 matches and the GRU on 6 of 7. Its mean macroF1 is 0.904, against 0.471 for the baseline, the nearest-player rule with the ±50-frame vote from `starter/baseline.py`. The blend weight and the GRU's 6 epochs were picked on these exams, so 0.712 is slightly optimistic, though not fragile: weights of 0.20 and 0.50 give 0.710 and 0.708. Nor can the exam see the differences between DFL's tracking and the provided tracking that our corruption fails to imitate (section 5).

## 2. Assets used and dropped

| provided asset | used | how |
|---|---|---|
| Tactical ATD (`BAR-ATH_tactical_atd.txt`, `_atd.xml`) | yes | The only model input: positions of every player slot and of the ball, and each slot's team. Speed is recomputed from positions, so that the same code runs on DFL. |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`) | one rule | The "Kick Off (Start)" tag of each half drives the kickoff rule (section 4); also the sanity check below. Not used for training or tuning. |
| `games.py` constants | yes | Graded half boundaries, pitch size, frame alignment. |
| Public 5-minute ground truth | pipeline check | The final file was scored on it once. Never used to train, tune or choose between versions. |
| Tactical video and homography | no | Only needed for ball detection, which we did not attempt. |
| Broadcast video, ATD and homography | no | A different timeline, with no mapping to the tactical video. |

Some statistics of the provided match's **inputs**, none of them involving a label, shape the training data: the lengths of its tracked and missing ball stretches, how often its ball is seen off the pitch, and how many of its players are visible (section 3). As a check without ground truth, our file agrees with Smart Tagging better than `baseline_smooth50` does. In phases tagged as a team defending, 68% of frames go to the other team and 13% to that team (baseline: 63% and 26%), and 56 of the 59 tagged set pieces have dead ball in the second before or during them (baseline: 48). Phases are not possessions, so this is evidence, not a score.

What we tried and dropped, on the exam (mean spellF1), each against the best version at the time:

| tried | result | verdict |
|---|---|---|
| Touch features: which team last had a player within 1–2 m of the ball with no opponent as close | +0.012, better on only 4 of 7 | dropped (code kept, switched off) |
| Average of 3 GRUs with different seeds | GRU alone 0.710 against 0.701 (7 of 7); in the blend 0.711 against 0.712 (3 of 7) | dropped |
| GRU trained for 30 epochs instead of 6 | 0.624 against 0.657 (earlier features) | over-trained |
| Ball gaps placed in dead play, in all / half of the training copies | standard exam: 0.642 / 0.698 against 0.712; exam with the same gaps: 0.761 / 0.758 against 0.741 | dropped (section 5) |
| LightGBM: more rounds, more leaves, a slower learning rate, 3 seeds, more regularisation | from −0.006 to +0.002 | dropped |
| 36 more before-and-after features / ball direction of travel | +0.004 / −0.007 | dropped |
| 3 corrupted copies of each match instead of 2 | −0.003 | dropped |
| Training answers read as controlled possession, 1 s / 2 s | −0.003 / −0.009 | dropped |
| Decoder: team-change cost 4 / a separate minimum for dead ball | +0.001 / no gain | dropped |

## 3. Training data

**One labelled dataset: Sportec Open DFL.** Bassek, Rein, Weber and Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12, 195 (2025), <https://doi.org/10.1038/s41597-025-04505-y>. The data are at <https://doi.org/10.6084/m9.figshare.28196177>, published with the authorisation of the Deutsche Fußball Liga (DFL) under CC BY 4.0.

- **What and how much:** official DFL match data for 7 matches of the 2022/23 season, 2 from the Bundesliga and 5 from the 2. Bundesliga, with positions recorded at 25 Hz by the TRACAB Gen5 multi-camera system. That is 1,002,644 labelled frames (11.1 hours): 44.8% dead ball, 29.3% home and 25.8% away, with 437 possessions per match and a mean of 7.2 s.
- **The labels:** every ball frame carries `BallStatus` (0 inactive, 1 active) and `BallPossession` (1 home, 2 away). We map them to D where the status is 0, otherwise to H or A (`possession/dfl.py`). The data descriptor defines possession by "the player in control of the ball" and says the ball is inactive when the referee has interrupted the match.
- **Who labelled it:** the labels are part of the official match data collected for the DFL. The descriptor says the position data are checked by live operators and again after the match; it does not say how these two fields are set, so we claim no more than that. Nobody on our team labelled anything, on any match.
- **What we read:** each match's position file and match-information file. The event files are never opened.

**Nothing comes from the provided game:** no ground truth, public or private, no hand labels, no event data, no manual correction of its tracking, and no frame number of it in our code. The graded half boundaries come from the organisers' `games.py`.

**Making DFL look like the provided tracking** (`possession/augment.py`). DFL's tracking is nearly complete and the provided tracking is not, so each DFL match is corrupted before training:

- The ball is removed wherever it is more than 1 m off the pitch: DFL tracks it there on 20% of its ball frames, the provided tracking on 4%.
- The ball is also removed in whole stretches, with visible and missing lengths drawn from the provided match's own ball track (medians of 345 and 268 frames).
- Every player slot disappears now and then, for about 7% of its frames, since the provided tracking shows about 20.5 of 22 players.
- Gaussian noise is added: 0.15 m on players, 0.3 m on the ball.

Each match gives 2 corrupted copies, each with a twin in which home and away are exchanged, so that the models treat the two teams alike: 28 training views in all. LightGBM trains on every 5th frame of them (802,128 rows), the GRU on every frame.

## 4. Possession rules we implemented

The challenge asks for *controlled* possession and three states. Few rules are written by hand: most of the behaviour is learnt from the DFL labels and then constrained by the decoder.

| situation | what our labels do | where it comes from |
|---|---|---|
| Play stopped by the referee | D, never a team: we do not guess who owns a dead ball. | Learnt from DFL's `BallStatus`. A stoppage lasts at least 3 s, and switching between alive and dead costs 64. |
| Before a half's kickoff | D | Kickoff rule, below. |
| Restart | The first team the models see with the ball after the stoppage, for at least 3 s. | Decoder. |
| Loose ball, 50/50 challenge, aerial duel | Possession stays with the team that had it, unless the other team then keeps the ball for about 3 s. | Minimum duration, plus a cost of 2 for changing team. |
| A touch without control, a deflection | No change of possession. | The same. |
| Change of possession | Declared when the evidence for the other team lasts, at the boundary that best fits the frame probabilities. | Viterbi. |
| Ball not tracked | Still labelled, from the players' shape and movement and from how long the ball has been unseen. We never abstain inside a half. | Features. |
| Outside the two graded halves | X | `games.half_video_blocks`. |

**Minimum duration.** No possession or stoppage in the file is shorter than 3 s (75 frames), except possibly the last segment of a half. The decoder settings (75 frames, team cost 2, dead cost 64) were chosen on the exam.

**Kickoff rule.** The ball cannot be in play before a half's kickoff, so every frame from the start of a half to the start of its "Kick Off (Start)" tag in Smart Tagging becomes dead ball. This matters in the second half, whose graded window opens before play resumes: dead ball in that stretch rose from 52% to 97%. The first half is graded from the kickoff itself and is unchanged. The rule holds no frame number, and a match without the tag is left alone.

**"Controlled" in our exam.** DFL's labels change hands on brief touches: 30% of its possessions are shorter than 2 s. The exam's answers are therefore rewritten by one rule (`possession/labels.py`). In live play, the other team takes over only if its spell lasts at least 2 s; otherwise those frames stay with the team that had the ball. The first spell after a stoppage keeps its team, and dead-ball frames never change. This leaves 288 possessions per match, with a mean of 11.0 s. The 2 s are the low end of CHALLENGE.md's "keeps the ball for a few seconds" and were not tuned on the public window.

The submitted file holds 73,662 D frames (46% of graded frames), 52,111 H and 33,084 A (Barcelona 61% of live play) and 40,462 X: 242 possessions with a mean of 14.1 s.

## 5. Limitations

**The team is the weak part, not dead ball.** Replacing one part of our exam output with the truth and decoding again raises mean spellF1 from 0.712 to 0.750 for perfect alive-or-dead, to 0.931 for a perfect team, and to 0.973 for perfect frames, which is what the decoder alone costs. Of the live frames, 87.7% go to the right team, 9.3% to the wrong team and 3.0% to dead ball. With the ball missing, the right team falls to 83.3% (90.7% when it is seen).

**Hand-overs are timed loosely.** Boundaries next to a stoppage are sharp: the median error is 0.12 s where a possession starts after a dead ball and 0.32 s where it ends in one. Boundaries between the two teams are off by a median of 0.76 to 0.80 s, and only about 35% fall within half a second. The metric is strict here: a perfect file shifted by 1 s scores 0.65.

**Short possessions are lost.** The file cannot hold a possession shorter than 3 s, yet 17.6% of the exam's true possessions are that short. Of the 638 of 2,017 true possessions we miss at IoU 0.5, 36% are shorter than 3 s, 28% are swallowed by a longer predicted possession and 23% go mostly to the wrong team. We predict 256 possessions per match where there are 288.

**The training data still does not look like the provided match.** A classifier that tells corrupted DFL frames from frames of the provided match, using inputs only (`python -m possession.gap`), reaches an AUC of 0.991, where one DFL match against the other six gives 0.59. The largest difference is *when* the ball is missing. With frames grouped into tenths by players' speed, the provided match's ball is missing on 90% of the frames of the slowest tenth and on 10% of the fastest, much like DFL's own share of dead ball (94% down to 2%). Our corruption hides it at random, on 40–45% of the frames of every tenth.

A corruption that hides the ball in dead play instead (`augment.dead_gaps`, calibrated on the provided match's inputs only) matches that profile, at 82% down to 12%. Yet the classifier's AUC only falls to 0.987; the rest comes from each team's deepest players, the number of visible players and how speeds vary. Models trained with it scored 0.761 against 0.741 on an exam corrupted the same way, but 0.642 against 0.712 on the standard exam, more than the 0.01 drop our rule allowed, so we kept random gaps. That is the cautious choice, not a proven one: the provided match does lose its ball the new way.

**What this means for the real score.** The 0.712 is an estimate on other matches and another tracking system. The submitted method also scores 0.741 on the exam with the realistic ball gaps, so the standard exam may be the harder one, but neither exam sees the remaining differences. The only direct evidence is the public window (spellF1 0.690). It holds 17 possessions, and EVALUATION.md notes that a single missed possession moves spellF1 by about 0.06 there.

**With more time**, in this order:

1. Close the input gap: imitate the missing deepest players, then re-test dead-play ball gaps, checking both on open tracking data produced from broadcast video.
2. Sharpen hand-overs with a second stage that refines each boundary between the two teams, where most of the remaining loss is.
3. Replace the hard 3-second minimum with a soft one, so that clear short possessions survive.
4. Find the ball in the video where the tracking lost it during live play.

## 6. Reproduction

Full instructions, including retraining and the exam commands, are in [`possession/README.md`](../possession/README.md). In short, from the repository root with Python 3.12 and the competition bundle unpacked into `data/`:

```bash
pip install -r possession/requirements.txt
python -m possession.predict
```

This writes `evaluation/results/lgbm_viterbi.csv` from the four committed model files in `possession/model/`, in about 20 seconds and without a GPU. Its sha256 is `e82e6b31bb93d6905f90cfa4e3adee77deaaeb2fc700ce14015b7d8eee6678ee`, and the submitted file is a copy of it under the team's name. `python -m possession.download_dfl` fetches the DFL dataset (2.6 GB), and `python -m possession.train` then retrains both models in about 2.5 minutes on an RTX 5060 laptop GPU.

Every random step is seeded, LightGBM runs in deterministic mode, and the GRU predicts on the CPU in float64, so that last-bit differences between machines cannot change a decision. On 2026-11-15, on the machine the models were trained on (Windows 11, x86-64, Python 3.12), two training runs and two prediction runs reproduced the committed model files and the CSV byte for byte, and `evaluation/validate_submission.py` accepts the CSV. A retrain on a CPU or a different GPU gives slightly different GRU weights, so the committed model files are the reference for the submission; other operating systems are untested.

## Tools and credits

- Training data: Deutsche Fußball Liga (DFL), via Bassek et al. (2025), CC BY 4.0, cited in section 3.
- Software: Python 3.12, NumPy 2.4.4, pandas 3.0.2, LightGBM 4.7.0, PyTorch 2.7.1, and the organisers' starter code and scorer. The pipeline contains no pre-trained model, no commercial API and no LLM.
- We wrote the code with the help of an AI coding assistant (Claude Code). In line with the Terms and Conditions, no competition video, dataset or extract of them was given to it: every command that reads the match data ran on our own machine and printed only aggregate numbers such as scores, counts and medians.
