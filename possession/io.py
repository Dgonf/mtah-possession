"""Readers for the provided ATD (FIFA/EPTS) and Smart Tagging (SportsCode XML)."""
import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

FPS = 25
L, W = 105.3, 68.0          # pitch in metres, as carried by the organisers' games.py
N_SLOTS = 46
# the two graded halves in video frames (first, last), published by the organisers in DATA.md
GRADED = ((11185, 90651), (110808, 190197))


def asset(data_dir, suffix):
    return os.path.join(data_dir, "tactical", "BAR-ATH_tactical" + suffix)


def load_atd(data_dir):
    """players (N, 46, 3) x, y, speed -- x, y normalised 0-1; ball (N, 3); team (46,) 0=home 1=away."""
    raw, meta = asset(data_dir, "_atd.txt"), asset(data_dir, "_atd.xml")
    txt = open(raw).read().replace(":", ",").replace(";", ",")
    arr = pd.read_csv(io.StringIO(txt), header=None, na_values=["NaN"], dtype=np.float32).values
    if not (arr[:, 0].astype(int) == np.arange(1, len(arr) + 1)).all():
        raise ValueError("ATD frame ids are not 1..N -- video_frame = frame_id - 1 would not hold")
    players = arr[:, 1:1 + N_SLOTS * 3].reshape(len(arr), N_SLOTS, 3)
    ball = arr[:, 1 + N_SLOTS * 3:1 + N_SLOTS * 3 + 3]
    root = ET.parse(meta).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]     # first team = home (FIFA/EPTS)
    team = np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")], np.int8)
    return players, ball, team


TEAM = {"barcelona": 0, "atletic club": 1}


def load_tags(path_or_dir):
    path = path_or_dir if path_or_dir.endswith(".xml") else asset(path_or_dir, "_smart_tagging.xml")
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        out.append(dict(code=inst.findtext("code").strip().upper(),
                        s=int(round(float(inst.findtext("start")) * FPS)),
                        e=int(round(float(inst.findtext("end")) * FPS)),
                        team=TEAM.get((tags.get("Team") or "").strip().lower(), -1),
                        type=(tags.get("Type") or "").strip().lower()))
    return out
