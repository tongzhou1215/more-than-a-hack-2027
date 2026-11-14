"""Load the competition match (Metrica FIFA/EPTS ATD) into the common representation."""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .common import CACHE_DIR, ROOT, X, ensure_cache

sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402


def _slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]  # FIFA/EPTS: first team is home (Barcelona here, per DATA.md)
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")])


def _parse_raw(raw_path, n_frames, n_slots):
    """(n_frames, n_slots, 2) player xy and (n_frames, 2) ball xy, normalised 0-1, NaN unseen."""
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            players[frame] = values.reshape(n_slots, 3)[:, :2]
            b = parts[2].split(",")
            if b[0][0] != "N":
                ball[frame] = (float(b[0]), float(b[1]))
    return players, ball


def load(game=GAME, data_dir=None, use_cache=True):
    cache = os.path.join(CACHE_DIR, "{}_tactical.npz".format(game))
    length, width = GAMES[game]["pitch"]
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        players, ball, teams = z["players"], z["ball"], z["teams"]
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
        teams = _slot_teams(meta_path)
        players, ball = _parse_raw(raw_path, source_frames("tactical", game), len(teams))
        ensure_cache()
        np.savez_compressed(cache, players=players, ball=ball, teams=teams)

    scale = np.array([length, width], dtype=np.float32)
    players = (players - 0.5) * scale
    ball = (ball - 0.5) * scale
    halves = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    return {
        "name": game,
        "home": players[:, teams == 0],
        "away": players[:, teams == 1],
        "ball": ball,
        "halves": halves,
        "labels": None,
        "pitch": (length, width),
    }


def public_labels(game=GAME):
    """The public 5-minute ground truth as codes (X outside). For pipeline checks only, never tuning."""
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    path = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(game))
    return pm.label_codes(pm.read_label_csv(path))
