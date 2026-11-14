"""Load a Sportec Open DFL match (Bassek et al. 2025, CC BY 4.0) into the common representation.

The ball FrameSet carries BallPossession (1 home, 2 away) and BallStatus (1 alive, 0 dead) on
every frame. That is the same construction as the competition ground truth -- dead where the
tracking marks the ball dead, otherwise the team the tracking names -- so it becomes the training
label directly: D if dead, else H or A.

Data: Deutsche Fussball Liga (DFL), via https://doi.org/10.6084/m9.figshare.28196177
"""

import glob
import os
import re

import numpy as np

from .common import A, CACHE_DIR, D, DFL_DIR, H, X, ensure_cache

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
_BALL_STATE = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
_TEAM = re.compile(r'<Team TeamId="([^"]+)"[^>]*Role="(\w+)"')
_PITCH = re.compile(r'PitchX="([\d.]+)" PitchY="([\d.]+)"')


def match_ids(data_dir=DFL_DIR):
    paths = glob.glob(os.path.join(data_dir, "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(os.path.basename(p).rsplit("_", 1)[-1][:-4] for p in paths)


def _info(match_id, data_dir):
    path = glob.glob(os.path.join(data_dir, "DFL_02_01_matchinformation_*{}.xml".format(match_id)))[0]
    text = open(path, encoding="utf-8").read()
    roles = {role: team for team, role in _TEAM.findall(text)}
    length, width = (float(v) for v in _PITCH.search(text).groups())
    return roles["home"], roles["guest"], (length, width)


def _parse(match_id, data_dir):
    home_id, away_id, pitch = _info(match_id, data_dir)
    path = glob.glob(os.path.join(data_dir, "DFL_04_03_positions_raw_observed_*{}.xml".format(match_id)))[0]
    sets = []  # (half, team, frames, xs, ys, poss, status)
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = _FRAME.match(line)
                current[2].append(int(m.group(1)))
                current[3].append(float(m.group(2)))
                current[4].append(float(m.group(3)))
                if current[1] == "BALL":
                    s = _BALL_STATE.search(line)
                    current[5].append(int(s.group(1)))
                    current[6].append(int(s.group(2)))
            elif line.startswith("<FrameSet "):
                half, team = _FRAMESET.match(line).groups()
                current = (half, team, [], [], [], [], [])
                sets.append(current)

    halves = []
    for half in ("firstHalf", "secondHalf"):
        ball = [s for s in sets if s[0] == half and s[1] == "BALL"][0]
        n0, n1 = min(ball[2]), max(ball[2]) + 1
        halves.append((half, n0, n1))

    # Lay the two halves end to end with a 1-frame X gap so no temporal feature leaks across.
    offsets, total = {}, 0
    for half, n0, n1 in halves:
        offsets[half] = total - n0
        total += (n1 - n0) + 1

    def team_array(team_id):
        slots = [s for s in sets if s[1] == team_id]
        out = np.full((total, len(slots), 2), np.nan, dtype=np.float32)
        for k, s in enumerate(slots):
            idx = np.asarray(s[2]) + offsets[s[0]]
            out[idx, k, 0] = s[3]
            out[idx, k, 1] = s[4]
        return out

    ball = np.full((total, 2), np.nan, dtype=np.float32)
    labels = np.full(total, X, dtype=np.int8)
    for s in sets:
        if s[1] != "BALL":
            continue
        idx = np.asarray(s[2]) + offsets[s[0]]
        ball[idx, 0] = s[3]
        ball[idx, 1] = s[4]
        poss, status = np.asarray(s[5]), np.asarray(s[6])
        labels[idx] = np.where(status == 0, D, np.where(poss == 1, H, A))

    spans = [(n0 + offsets[h], n1 + offsets[h]) for h, n0, n1 in halves]
    return team_array(home_id), team_array(away_id), ball, labels, spans, pitch


def load(match_id, data_dir=DFL_DIR, use_cache=True):
    cache = os.path.join(CACHE_DIR, "dfl_{}.npz".format(match_id))
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        home, away, ball, labels = z["home"], z["away"], z["ball"], z["labels"]
        spans = [tuple(s) for s in z["spans"].tolist()]
        pitch = tuple(z["pitch"].tolist())
    else:
        home, away, ball, labels, spans, pitch = _parse(match_id, data_dir)
        ensure_cache()
        np.savez_compressed(cache, home=home, away=away, ball=ball, labels=labels,
                            spans=np.array(spans), pitch=np.array(pitch))
    return {"name": match_id, "home": home, "away": away, "ball": ball,
            "halves": spans, "labels": labels, "pitch": pitch}
