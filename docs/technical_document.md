# Mendota Metrics: technical document

More than a Hack 2027, Initial Challenge: frame-level ball possession on BAR-ATH.

- Repository: <https://github.com/tongzhou1215/more-than-a-hack-2027>. Our code is in `possession/`; everything else is the organisers' starter repository.
- Submission file: written by `python -m possession.predict`, sha256 `e82e6b31bb93d690…` (section 6).

## Summary

Two models trained on other matches give every frame a probability for home, away and dead ball. One is a gradient-boosted tree model (LightGBM), the other a bidirectional GRU. Both read the same 109 features, computed from the provided tactical tracking. Their probabilities are averaged with weights 0.35 and 0.65, and a Viterbi decoder with a 3-second minimum duration turns them into possessions and stoppages.

```
tactical ATD ─► 109 features ─┬─► LightGBM ─┐
(players, ball)   per frame   └─► GRU ──────┴─► 0.35 / 0.65 blend ─► kickoff rule ─► Viterbi, 3 s minimum ─► H / A / D
                                                                          ▲
                              Smart Tagging "Kick Off (Start)" ───────────┘            outside the two halves: X
```

- **Training data:** the 7 public matches of the Sportec Open DFL dataset, corrupted to resemble the provided automatic tracking. No label of the provided game was used to train, tune or select anything.
- **Our estimate of quality:** mean spellF1 0.712 on a leave-one-match-out test over the 7 DFL matches, which this document calls the exam. The official baseline scores 0.195 on the same exam.
- **Public 5-minute window:** checked once, on the final file: spellF1 0.690, macroF1 0.908.
- **Reproduction:** the repository regenerates the submitted file byte for byte.

## 1. Approach

### 1.1 What we built

| step | module | what it does |
|---|---|---|
| load | `possession/atd.py` | Reads the tactical ATD into metres: home players, away players and the ball, with the two graded halves from `games.half_video_blocks`. |
| features | `possession/features.py` | 109 numbers per frame. They do not depend on the order of the player slots, are computed inside each half, and are oriented so that the home team always defends the same goal. |
| frame scorer 1 | `possession/train.py` | LightGBM, 3 classes, 126 rounds of 63-leaf trees, trained on every 5th frame. |
| frame scorer 2 | `possession/seq.py` | Bidirectional GRU (2 layers, 64 hidden units, 131,843 parameters). It reads each half at 5 steps per second in 60-second windows, forwards and backwards. |
| blend | `possession/predict.py` | 0.35 × LightGBM + 0.65 × GRU. |
| kickoff rule | `possession/predict.py` | Frames of a half before its tagged kickoff are set to dead ball (section 4). |
| decoder | `possession/viterbi.py` | Viterbi over the states H, A, D, run on each half separately. Every state must last at least 75 frames (3 s). Changing team costs 2 and changing between alive and dead costs 64, in log-probability units. |

The 109 features fall into four groups:

- **Ball:** whether it is seen, how long since it was last seen and until it is seen again, its interpolated position, its distance to the touchlines and goal lines, whether it is off the pitch, its speed.
- **Players and ball:** for each team, the distance of its nearest and second-nearest player to the ball, how many players are within 5 m and 15 m, and how fast its three nearest players close in on the ball; which team is nearer.
- **Team shape and motion:** each team's centroid, spread, deepest and most advanced player, number of visible players, mean and maximum speed, and how fast its block moves forward.
- **Time context:** the main signals averaged over ±1 s, ±3 s, ±10 s and ±20 s, and the 5 seconds before and the 5 seconds after given separately. A change of possession flips who is near the ball and which way each block moves, so "before against after" is what shows it.

Speeds are divided by their median over the half. The provided tracking measures players about 30% slower than DFL (median 1.5 against 2.1 m/s), and "everyone slowed down" is the strongest dead-ball cue, so raw speeds would read as dead ball on the provided match.

### 1.2 Why this and not something else

Three facts about the task decided the design.

1. **The ranking metric counts whole possessions.** A model that picks the likeliest label frame by frame flickers. In our first version (trained on five DFL matches, scored on the other two), per-frame argmax scored spellF1 0.11 with about 2,800 fragments per match. The same probabilities through the decoder scored 0.50. The decoder is therefore half of the method, not post-processing.
2. **The ball is missing on 41% of graded frames.** A rule that needs the ball position, such as the baseline's nearest player, has nothing to say there. Most of our features describe the players instead, and the ball features include how long it has been unseen.
3. **No label of the provided game may be used.** Supervised learning then needs other matches whose labels mean the same thing. The DFL tracking data carries a ball status and a possessing team on every frame, which is the construction CHALLENGE.md gives for the ground truth. The price is a domain gap between clean stadium tracking and the provided automatic tracking. We reduce it by corrupting the DFL data (section 3) and we measured what is left (section 5).

We use two frame scorers because they see time differently. LightGBM scores each frame on its own and knows the surrounding seconds only through rolling means. The GRU reads the sequence in both directions and can learn what happens around a hand-over. On the exam the GRU alone scores 0.701 and LightGBM alone 0.659. The blend scores 0.712: better than LightGBM on 7 of 7 matches and better than the GRU on 6 of 7.

We did not build three things we considered: detecting the ball in the video where the tracking lost it, a CRF over player trajectories in the style of PathCRF, and bringing in open tracking data produced from broadcast video, such as SkillCorner's. Section 5 says which we would try first.

### 1.3 How every choice was made

We could not score ourselves on the provided match, so we built an exam from DFL (`possession/cv.py` for LightGBM, `possession/seq_cv.py` for the GRU):

- Each of the 7 DFL matches is held out once, and the models are trained on the other 6.
- The held-out match is corrupted like the training data, with a seed of its own, so every run sits the same exam and two versions can be compared match by match.
- The answers are read as controlled possession with a 2-second threshold (section 4).
- The score is the organisers' own `possession_metrics.evaluate`.

A change was adopted only if it was better on at least 5 of the 7 matches and raised the mean spellF1 by at least 0.005. This rule was fixed before the experiments of section 2 were run.

The public 5 minutes took no part in any choice. We scored the final file on them once.

| match held out | official baseline | LightGBM | GRU | blend (submitted) |
|---|---|---|---|---|
| DFL-MAT-J03WMX | 0.2017 | 0.6516 | 0.6896 | 0.7017 |
| DFL-MAT-J03WN1 | 0.2018 | 0.6367 | 0.6725 | 0.6982 |
| DFL-MAT-J03WOH | 0.2078 | 0.6733 | 0.6876 | 0.6900 |
| DFL-MAT-J03WOY | 0.1671 | 0.6532 | 0.7024 | 0.7190 |
| DFL-MAT-J03WPY | 0.2013 | 0.6750 | 0.7031 | 0.7161 |
| DFL-MAT-J03WQQ | 0.1929 | 0.6549 | 0.7339 | 0.7217 |
| DFL-MAT-J03WR9 | 0.1916 | 0.6700 | 0.7195 | 0.7391 |
| **mean spellF1** | **0.1949** | **0.6592** | **0.7012** | **0.7122** |

The official baseline here is the nearest-player rule with the ±50-frame vote from `starter/baseline.py`, sitting the same exams. Mean macroF1 on the exam is 0.904 for the blend and 0.471 for the baseline.

Two caveats on the 0.712:

- The blend weight (0.35) and the GRU's 6 epochs were picked on these same exams, so the number is slightly optimistic. It does not hinge on the exact weight: 0.20 gives 0.710 and 0.50 gives 0.708.
- The exam measures how well the method carries over to a new match of the same tracking system. It cannot see differences between DFL's tracking and the provided tracking that our corruption fails to imitate (section 5).

## 2. Assets used and dropped

| provided asset | used | how |
|---|---|---|
| Tactical ATD (`BAR-ATH_tactical_atd.txt`, `_atd.xml`) | yes | The only model input: x and y of every player slot and of the ball, and each slot's team. The ATD's own speed value is ignored; speed is recomputed from positions so that the same code runs on DFL. |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`) | yes, for one rule | The "Kick Off (Start)" tag of each half drives the kickoff rule (section 4). We also compared our output with the tagged phases as a sanity check without ground truth (below). It was not used for training or tuning. |
| `games.py` constants | yes | Half boundaries (which frames are graded), pitch size, frame alignment. |
| Public 5-minute ground truth | pipeline check only | The final file was scored on it once. Never used to train, tune or choose between versions. |
| Tactical video, tactical homography | no | Only needed for ball detection, which we did not attempt. |
| Broadcast video, ATD, homography | no | They run on a different timeline and no mapping to the tactical video is provided. |

A few statistics of the provided match's **inputs** shape the training data, and none involves a label: the lengths of the stretches where its ball is tracked and where it is missing, how often its ball is seen off the pitch, and how many of its players are visible (section 3).

The sanity check against Smart Tagging, on the whole match. Phases are not possessions, so this is evidence and not a score:

| | submitted file | `baseline_smooth50` |
|---|---|---|
| phases tagged as a team attacking: frames given to that team / the other team / dead ball | 50% / 20% / 30% | 50% / 30% / 20% |
| phases tagged as a team defending: frames given to the other team / that team / dead ball | 68% / 13% / 19% | 63% / 26% / 11% |
| tagged set pieces with dead ball in the second before or during them | 56 of 59 | 48 of 59 |

What we tried and dropped, each measured on the 7-match exam (mean spellF1):

| tried | result | verdict |
|---|---|---|
| Touch features: which team last had a player within 1–2 m of the ball with no opponent as close | +0.012, better on only 4 of 7 | dropped (the code stays in `features.py`, switched off) |
| GRU trained for 30 epochs instead of 6 | 0.624 against 0.657 (on an earlier version of the features) | over-trained |
| Average of 3 GRUs with different seeds | GRU alone 0.710 against 0.701 (7 of 7); in the blend 0.711 against 0.712 (3 of 7) | dropped |
| Decoder with a team-change cost of 4 | +0.001 | below the bar |
| Ball gaps placed in dead play (see section 5) | 0.761 against 0.741 on an exam built the same way; 0.642 against 0.712 on the standard exam | dropped |
| The same for half of the training copies | 0.758 against 0.741; 0.698 against 0.712 | dropped |
| LightGBM: 300 rounds / 127 leaves / learning rate 0.025 with 400 rounds | +0.002 / −0.001 / +0.001 | dropped |
| LightGBM: average of 3 seeds | +0.001 | dropped |
| 3 corrupted copies of each match instead of 2 | −0.003 | dropped |
| 36 more before-and-after features | +0.004 | below the bar |
| Ball direction-of-travel features | −0.007 | dropped |
| Stronger regularisation | −0.006 | dropped |
| Training answers read as controlled possession, 1 s / 2 s | −0.003 / −0.009 | dropped: training uses the raw DFL labels |
| A separate minimum duration for dead ball | no gain | dropped |

Each row is compared with the best version at the time it was tried.

## 3. Training data

**One labelled dataset: Sportec Open DFL.** Bassek, Rein, Weber and Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12, 195 (2025), <https://doi.org/10.1038/s41597-025-04505-y>. The data are at <https://doi.org/10.6084/m9.figshare.28196177>, published with the authorisation of the Deutsche Fußball Liga (DFL) under CC BY 4.0.

- **What it is:** official DFL match data for 7 matches of the 2022/23 season, 2 from the Bundesliga and 5 from the 2. Bundesliga. The position data were recorded at 25 Hz with the TRACAB Gen5 multi-camera system.
- **How much:** 1,002,644 labelled frames, or 11.1 hours of match time (91.5 to 97.9 minutes per match). 44.8% of the frames are dead ball, 29.3% home and 25.8% away. The raw labels hold 437 possessions per match with a mean of 7.2 s.
- **What the labels are:** every ball frame of the position data carries `BallStatus` (0 inactive, 1 active) and `BallPossession` (1 home, 2 away). The data descriptor defines possession by "the player in control of the ball" and says the ball is inactive when the referee has interrupted the match. We map them directly: D where the status is 0, otherwise H or A (`possession/dfl.py`).
- **Who labelled it:** the labels are part of the official match data collected for the DFL. The descriptor says the position data are checked by live operators and again after the match. It does not say how the two fields are set frame by frame, so we claim no more than that. Nobody on our team labelled anything, on any match.
- **What we read:** the position file and the match-information file (which team is home, pitch size) of each match. The event files are downloaded with the rest and never opened.

**Nothing comes from the provided game:** no ground truth, public or private, no hand labels, no event data, no manual correction of its tracking. Our code holds no frame number of the match; the graded half boundaries come from the organisers' `games.py`.

**Making DFL look like the provided tracking** (`possession/augment.py`). DFL's tracking is nearly complete; the provided tracking is not. Each DFL match is therefore corrupted before training:

- The ball is removed wherever it is more than 1 m off the pitch. DFL keeps tracking it there (20% of its ball frames) and the provided tracking almost never does (4%).
- The ball is removed in whole stretches, not single frames. Visible and missing stretches alternate, with lengths drawn from the provided match's own ball track (median 345 frames visible, 268 frames missing).
- Every player slot disappears now and then, for about 7% of its frames, because the provided tracking shows about 20.5 of 22 players on average.
- Gaussian noise is added: 0.15 m on players, 0.3 m on the ball.

Each match gives 2 corrupted copies, and each copy a twin with home and away exchanged, so the models treat the two teams alike. That makes 28 training views of the 7 matches. LightGBM trains on every 5th frame of them (802,128 rows) and the GRU on all frames.

## 4. Possession rules we implemented

The challenge asks for *controlled* possession and three states. We wrote very few rules by hand. Most of the behaviour is learnt from the DFL labels and then constrained by the decoder.

| situation | what our labels do | where it comes from |
|---|---|---|
| Play stopped by the referee | D, never a team. We do not guess who owns a dead ball. | Learnt from DFL's `BallStatus`. A stoppage must last at least 3 s, and switching between alive and dead costs 64. |
| Before a half's kickoff | D | Kickoff rule, below. |
| Restart | The first team the models see with the ball after the stoppage. That possession must last at least 3 s. | Decoder. |
| Loose ball, 50/50 challenge, aerial duel | Possession stays with the team that had it, unless the other team then keeps the ball for about 3 s. | Minimum duration, plus a cost of 2 for changing team. |
| A touch without control, a deflection | No change of possession. | The same. |
| Change of possession | Declared when the evidence for the other team lasts. The decoder puts the boundary where it fits the frame probabilities best. | Viterbi. |
| Ball not tracked | Still labelled, from the players' shape and movement and from how long the ball has been unseen. We never abstain inside a half. | Features. |
| Outside the two graded halves | X | `games.half_video_blocks`. |

**Minimum duration.** Every state in the output lasts at least 75 frames, so the file contains no possession and no stoppage shorter than 3 s, except possibly the last segment of a half. The three decoder settings (75 frames, team cost 2, dead cost 64) were chosen on the exam. A later grid of 18 settings on the blended probabilities put 75 frames in all of its top six.

**Kickoff rule.** A half starts with its kickoff, so the ball cannot be in play before it. For each half we take the "Kick Off (Start)" set piece from Smart Tagging and force every frame from the start of the half up to the start of that tag to dead ball. This matters in the second half, whose graded window opens before play resumes: the share of dead ball in that stretch rose from 52% to 97% when we added the rule. The first half is graded from the kickoff itself and is unchanged. The rule holds no frame number, and a match without such a tag is left alone.

**What "controlled" means in our exam.** DFL's labels change hands on brief touches: 437 possessions per match, 30% of them shorter than 2 s. To score ourselves the way the challenge counts possessions, the exam's answers are rewritten by one rule (`possession/labels.py`): within live play, a spell of the other team takes over only if it lasts at least 2 s, and otherwise its frames go to the team that already had the ball. The first spell after a stoppage keeps its team, and dead-ball frames are never changed. This leaves 288 possessions per match with a mean of 11.0 s. The 2 seconds are the low end of CHALLENGE.md's "keeps the ball for a few seconds". They were not tuned on the public window.

The submitted file as a whole: 73,662 frames D (46% of graded frames), 52,111 H and 33,084 A (Barcelona 61% of live play), 40,462 X, and 242 possessions with a mean of 14.1 s.

## 5. Limitations

**The team is the weak part, not dead ball.** On the exam we replaced one part of the model's output with the truth and decoded again:

| what is made perfect | mean spellF1 |
|---|---|
| nothing (the submitted method) | 0.712 |
| alive or dead | 0.750 |
| which team | 0.931 |
| every frame (what the decoder alone costs) | 0.973 |

Of the live frames, 87.7% go to the right team, 9.3% to the wrong team and 3.0% are called dead. With the ball missing, the right team falls to 83.3% (90.7% when it is seen).

**Hand-overs are timed loosely.** Boundaries next to a stoppage are sharp: the median error is 0.12 s where a possession starts after a dead ball and 0.32 s where it ends in one. Boundaries between the two teams have a median error of 0.76 to 0.80 s, and only about 35% fall within half a second. The metric is strict here: a perfect file shifted by 1 s scores 0.65.

**Short possessions are lost.** The file cannot hold a possession shorter than 3 s. In the exam's answers 17.6% of possessions are that short. Of the 638 true possessions we miss at IoU 0.5 (out of 2,017), 36% are shorter than 3 s, 28% are swallowed by a longer predicted possession and 23% go mostly to the wrong team. We predict 256 possessions per match where there are 288.

**The training data still does not look like the provided match.** `python -m possession.gap` trains a classifier to tell corrupted DFL frames from frames of the provided match, using inputs only. Its AUC is 0.991, where 0.5 would mean it cannot tell them apart. One DFL match against the other six gives 0.59, so playing style does not explain it. The largest difference is *when* the ball is missing. With frames sorted by players' speed into ten groups:

| players' speed, slowest to fastest | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| provided match: ball missing | 0.90 | 0.77 | 0.65 | 0.54 | 0.45 | 0.27 | 0.16 | 0.13 | 0.10 | 0.10 |
| our corrupted DFL: ball missing | 0.45 | 0.45 | 0.44 | 0.44 | 0.44 | 0.44 | 0.42 | 0.40 | 0.40 | 0.41 |
| DFL's own labels: dead ball | 0.94 | 0.87 | 0.79 | 0.68 | 0.52 | 0.33 | 0.19 | 0.09 | 0.05 | 0.02 |

The provided tracking loses the ball mostly when play is dead. Our corruption hides it at random, so in fast live play our training data lacks the ball about four times as often as the provided match does.

We built a corruption that hides the ball in dead play, calibrated on the provided match's inputs only (`augment.dead_gaps`). It brings the middle row of the table close to the first (0.82 falling to 0.12), yet the classifier's AUC only drops to 0.987. What remains is the position of each team's deepest players, the number of visible players and how speeds vary. Models trained this way scored higher on an exam corrupted the same way (0.761 against 0.741) and much lower on the standard exam (0.642 against 0.712). Our rule for a change to the corruption was that the standard exam may not fall by more than 0.01, so we kept random gaps. That is the cautious choice, not a proven one: the table says the provided match loses its ball the new way. With more time we would settle this on tracking data closer to the provided match.

**What this means for the real score.** The 0.712 is an estimate on other matches and another tracking system. The submitted method also scores 0.741 on the exam with the more realistic ball gaps, so the standard exam may be the harder one in that respect, but neither exam sees the remaining differences. The only direct evidence is the public window (spellF1 0.690). It holds 17 possessions, and EVALUATION.md notes that a single missed possession moves spellF1 by about 0.06 there.

**Reproduction was verified on one machine.** See section 6.

**With more time**, in this order:

1. Close the input gap: imitate the missing deepest players, then re-test dead-play ball gaps, checking both on open tracking data produced from broadcast video, such as SkillCorner's.
2. Sharpen hand-overs: a second stage that refines each boundary between the two teams, where most of the remaining loss is.
3. Replace the hard 3-second minimum with a soft one, so that clear short possessions survive.
4. Find the ball in the video on frames where the tracking lost it during live play.

## 6. Reproduction

Full instructions are in [`possession/README.md`](../possession/README.md). In short, from the repository root with Python 3.12:

```bash
pip install -r possession/requirements.txt
```

Unpack the competition bundle into `data/`. Three files are read: `data/tactical/BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_smart_tagging.xml`. Then:

```bash
python -m possession.predict
```

This writes `evaluation/results/lgbm_viterbi.csv` from the four committed model files in `possession/model/`. It takes about 20 seconds and needs no GPU. Its sha256 is `e82e6b31bb93d6905f90cfa4e3adee77deaaeb2fc700ce14015b7d8eee6678ee`, and the submitted file is a copy of it under the team's name.

```bash
python evaluation/validate_submission.py evaluation/results/lgbm_viterbi.csv
```

To retrain from scratch (optional):

```bash
python -m possession.download_dfl
```

```bash
python -m possession.train
```

The first command fetches the DFL dataset (2.6 GB) into `../dfl_data`. The second takes about 2.5 minutes on an RTX 5060 laptop GPU.

**What is deterministic.** Every random step is seeded. LightGBM runs in deterministic mode. The GRU's weights are stored as a plain float32 array, and prediction runs on the CPU in float64 so that last-bit differences between machines cannot change a decision.

**What we verified.** On 2026-11-15, on the machine the models were trained on, we trained twice and predicted twice. Both training runs wrote four model files identical, byte for byte, to the committed ones. Both prediction runs wrote the CSV with the sha256 above, and `validate_submission.py` accepts it.

**What we did not verify.** The GRU's weights depend on the device it is trained on, so a retrain on a CPU or on a different GPU gives slightly different weights. The committed model files are the reference for the submission. We also had only one machine for the final version (Windows 11, x86-64, Python 3.12), so regeneration on another operating system is expected but untested.

The exam numbers of section 1.3 come from:

```bash
python -m possession.cv --exam-controlled 2 --decoder 2,64,75 --extra-features 1 --copies 2 --dump lgbm_exam
```

```bash
python -m possession.seq_cv --tag gru_exam --epochs 6
```

```bash
python -m possession.seq_cv --blend lgbm_exam gru_exam --weights 0.35
```

## Tools and credits

- Training data: Deutsche Fußball Liga (DFL), via Bassek et al. (2025), CC BY 4.0, cited in section 3.
- Software: Python 3.12, NumPy 2.4.4, pandas 3.0.2, LightGBM 4.7.0, PyTorch 2.7.1, and the organisers' starter code and scorer.
- The pipeline contains no pre-trained model, no commercial API and no LLM.
- We wrote the code with the help of an AI coding assistant (Claude Code). In line with the Terms and Conditions, no competition video, dataset or extract of them was given to it: every command that reads the match data ran on our own machine and printed only aggregate numbers such as scores, counts and medians.
