"""Convert the public DFL / IDSSE matches (other games, CC-BY 4.0) to the arrays v2 uses.

    python tools/dfl_convert.py --dfl external/dfl --out external/dfl_npz

Reads only the DFL_04_03_positions_raw_observed_*.xml files, streaming (they are ~400 MB each).
Writes one <match>.npz per match with the fields tools/dfl_grid.py expects:
    players (N,46,3) ball (N,3) team (46,) labels (N,) graded (N,) kickoffs (2,)

Layout of each output, in frames:  [10 s empty] [first half] [5 min empty] [second half]
The empty frames let v2's rule layer find kick-offs and half-time the way it does on the provided
match; they are not graded.

Labels come from the ball's BallStatus (0 dead, 1 alive) and BallPossession in the DFL feed.
Which BallPossession value means which team is not assumed: it is detected from the data (the team
of the player nearest the ball should usually be the team in possession) and printed.
Speeds are left NaN on purpose (DFL's unit differs from the ATD); v2 then derives them from positions.
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
N_SLOTS, PER_TEAM = 46, 23
PAD_START, HALFTIME_GAP = 10 * FPS, 5 * 60 * FPS
SECTIONS = ("firstHalf", "secondHalf")


def parse(path):
    """-> pitch (Lx, Ly), {(section, team, person): dict of numpy arrays}, ball attribute names"""
    sets, pitch, ball_keys = {}, None, None
    cur, cols, is_ball, root = None, None, False, None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        if ev == "start":
            if root is None:
                root = el
            if el.tag == "PitchSize":
                pitch = (float(el.get("X")), float(el.get("Y")))
            elif el.tag == "FrameSet":
                cur = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
                cols = {"N": [], "X": [], "Y": [], "Z": [], "BS": [], "BP": []}
                is_ball = None
            continue
        if el.tag == "Frame":
            a = el.attrib
            if is_ball is None:
                is_ball = ("BallStatus" in a) or (cur[1] or "").lower() == "ball"
                if is_ball and ball_keys is None:
                    ball_keys = sorted(a.keys())
            cols["N"].append(int(a["N"]))
            cols["X"].append(float(a["X"]))
            cols["Y"].append(float(a["Y"]))
            if is_ball:
                cols["Z"].append(float(a.get("Z", "nan")))
                cols["BS"].append(int(a.get("BallStatus", -1)))
                cols["BP"].append(int(a.get("BallPossession", -1)))
            el.clear()
        elif el.tag == "FrameSet":
            d = {k: np.array(v) for k, v in cols.items() if len(v)}
            d["ball"] = bool(is_ball)
            sets[cur] = d
            root.clear()
    return pitch, sets, ball_keys


def convert(path):
    pitch, sets, ball_keys = parse(path)
    Lx, Ly = pitch
    print(f"  pitch {Lx} x {Ly} m, ball frame attributes: {ball_keys}")
    ball_sets = {k: v for k, v in sets.items() if v["ball"]}
    teams = sorted({k[1] for k, v in sets.items() if not v["ball"] and (k[1] or "").lower() != "referee"})
    if len(teams) != 2:
        raise SystemExit(f"expected 2 teams, found {teams}")
    persons = [sorted({k[2] for k in sets if k[1] == t}) for t in teams]
    for t, p in zip(teams, persons):
        print(f"  team slot {teams.index(t)}: {len(p)} players")
        if len(p) > PER_TEAM:
            raise SystemExit(f"team {t} has {len(p)} players, more than {PER_TEAM} slots")
    slot = {(t, p): i * PER_TEAM + j for i, t in enumerate(teams) for j, p in enumerate(persons[i])}
    team = np.array([0] * PER_TEAM + [1] * PER_TEAM, np.int8)

    halves = []
    for sec in SECTIONS:
        b = [v for k, v in ball_sets.items() if k[0] == sec]
        if len(b) != 1:
            raise SystemExit(f"{sec}: expected one ball FrameSet, found {len(b)}")
        b = b[0]
        n0, n = b["N"].min(), b["N"].max() - b["N"].min() + 1
        P = np.full((n, N_SLOTS, 3), np.nan, np.float32)
        for (s, t, p), v in sets.items():
            if s != sec or (t, p) not in slot:
                continue
            i = v["N"] - n0
            ok = (i >= 0) & (i < n)
            P[i[ok], slot[(t, p)], 0] = (v["X"][ok] + Lx / 2) / Lx
            P[i[ok], slot[(t, p)], 1] = (v["Y"][ok] + Ly / 2) / Ly
        B = np.full((n, 3), np.nan, np.float32)
        BS = np.full(n, -1); BP = np.full(n, -1)
        i = b["N"] - n0
        B[i, 0] = (b["X"] + Lx / 2) / Lx
        B[i, 1] = (b["Y"] + Ly / 2) / Ly
        B[i, 2] = b["Z"]
        BS[i] = b["BS"]; BP[i] = b["BP"]
        halves.append((P, B, BS, BP))
        print(f"  {sec}: {n} frames, BallStatus {dict(zip(*np.unique(BS, return_counts=True)))}, "
              f"BallPossession {dict(zip(*np.unique(BP, return_counts=True)))}")

    def pad(k):
        return (np.full((k, N_SLOTS, 3), np.nan, np.float32), np.full((k, 3), np.nan, np.float32),
                np.full(k, -1), np.full(k, -1))

    parts = [pad(PAD_START), halves[0], pad(HALFTIME_GAP), halves[1]]
    players, ball, BS, BP = (np.concatenate([p[j] for p in parts]) for j in range(4))
    graded = np.concatenate([np.zeros(PAD_START, bool), np.ones(len(halves[0][1]), bool),
                             np.zeros(HALFTIME_GAP, bool), np.ones(len(halves[1][1]), bool)])
    kickoffs = np.array([PAD_START, PAD_START + len(halves[0][1]) + HALFTIME_GAP])

    # which BallPossession value is team slot 0: the nearest player's team should agree with it
    vals = [v for v in np.unique(BP[graded & (BS == 1)]) if v >= 0]
    if len(vals) != 2:
        raise SystemExit(f"expected 2 BallPossession values in live play, found {vals}")
    d = np.hypot((players[:, :, 0] - ball[:, None, 0]) * Lx, (players[:, :, 1] - ball[:, None, 1]) * Ly)
    near = np.where(np.isnan(d), np.inf, d)
    k1 = team[np.argmin(near, 1)]
    close = graded & (BS == 1) & (np.min(near, 1) < 1.5)
    agree = np.mean((BP[close] == vals[0]) == (k1[close] == 0))
    home_val = vals[0] if agree >= 0.5 else vals[1]
    print(f"  BallPossession={home_val} -> slot 0 (nearest-player agreement {max(agree, 1 - agree):.3f})")

    labels = np.full(len(BS), "D", "<U1")
    live = graded & (BS == 1)
    labels[live & (BP == home_val)] = "H"
    labels[live & (BP >= 0) & (BP != home_val)] = "A"
    unknown = live & (BP < 0)
    for t in np.flatnonzero(unknown):           # live but no possession value: keep the last team
        labels[t] = labels[t - 1] if labels[t - 1] != "D" else "D"
    print(f"  live share {live[graded].mean():.3f}, live frames without possession {unknown.sum()}")
    return dict(players=players, ball=ball, team=team, labels=labels, graded=graded, kickoffs=kickoffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default="external/dfl")
    ap.add_argument("--out", default="external/dfl_npz")
    ap.add_argument("--only", default=None, help="convert one match id, e.g. J03WMX")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.dfl, "**", "DFL_04_03_positions_raw_observed_*.xml"), recursive=True))
    for f in files:
        m = f.rsplit("DFL-MAT-", 1)[-1].split(".")[0]
        if a.only and m != a.only:
            continue
        print(m)
        np.savez_compressed(os.path.join(a.out, m + ".npz"), **convert(f))


if __name__ == "__main__":
    main()