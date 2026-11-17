"""Team correction from jersey colour in the tactical video.

The automatic tracking sometimes carries a person in the wrong team's slot (on ~10 % of graded
frames one team has 12+ tracked players and the other 10 or fewer). Slots also swap identity, so the
fix has to be per tracklet, not per slot. This module produces a per-frame team matrix (N, 46):
0 = home, 1 = away, -1 = not a player (match official), used by features.basic() and geometry().

    python -m possession.teamfix --data data                  # full run, prints the checks
    python -m possession.teamfix --data data --max-frames 15000   # quick test on the first 10 min

Method, all algorithmic, no hand labels:
  1. Tracklets: each slot is cut into continuous runs, split wherever tracking drops out or the
     position jumps more than JUMP_M in one frame (an identity switch).
  2. Colour samples: every STRIDE frames, each player who is not within MIN_SEP_M of anyone else is
     projected into the video with the provided homography (normalised pitch -> pixels). A torso
     patch is cut above the feet, scaled by the local pixels-per-metre, grass pixels are dropped, and
     the median CIELAB colour is kept.
  3. Two team colours = median colour of each team as the tracking labels it. Most labels are right,
     so the median is robust to the few wrong ones; it is recomputed once after correction.
  4. Each tracklet votes with its samples. A tracklet is reassigned only with >= MIN_VOTES usable
     samples and >= AGREE agreement. Samples far from both team colours (goalkeepers, officials)
     do not vote. A tracklet whose colour is far from both teams and which stays away from both
     goals is marked as an official (-1).
Samples are cached in data/derived/ (gitignored with data/), so re-runs skip the video decode.
Deterministic: fixed stride, medians, no random state.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .io import L, W, asset, load_atd, load_tags

STRIDE = 5             # frames between colour samples
JUMP_M = 1.0           # m in one frame (25 m/s) = identity switch, start a new tracklet
MIN_SEP_M = 1.2        # m to the nearest other person, below which a patch may mix two people
TORSO_M = (0.9, 1.5)   # m above the feet: the torso band of the patch
HALF_W_M = 0.25        # m, half width of the patch
MIN_PX_PER_M = 6.0     # smaller players are too few pixels to read
MIN_NONGRASS = 0.3     # share of non-grass pixels a patch needs
L_WEIGHT = 0.5         # lightness matters less than hue (shadows, floodlights)
R_MULT = 3.0           # usable sample: within R_MULT x the median own-team distance of a team colour
MIN_TRACKED = 18       # sample only frames with a full match on the pitch (no warm-up, half-time)
MIN_VOTES = 3
AGREE = 0.7
OFFICIAL_FAR = 0.8     # share of a tracklet's samples far from both teams to call it an official
GOAL_M = 30.0          # an official stays further than this from both goals (goalkeepers do not)
KEEPER_M = 20.0        # a tracklet whose median distance to the nearest goal line is under this is
                       # treated as a goalkeeper: its kit is neither team colour, so it is never moved
EXCESS = 1.0           # a move from team A to B needs A to have, on average over the tracklet's frames,
                       # at least this many more tracked players than B (a move must fix a count)


def derived_dir(data_dir):
    d = os.path.join(data_dir, "derived")
    os.makedirs(d, exist_ok=True)
    return d


def tracklets(players):
    """(N, 46) tracklet id per frame and slot, -1 where the slot is empty."""
    n, S = players.shape[:2]
    x, y = players[:, :, 0].astype(float) * L, players[:, :, 1].astype(float) * W
    ok = ~np.isnan(x)
    step = np.zeros((n, S))
    step[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    tid = np.full((n, S), -1, np.int32)
    base = 0
    for s in range(S):
        with np.errstate(invalid="ignore"):
            start = ok[:, s] & np.r_[True, ~ok[:-1, s] | (step[1:, s] > JUMP_M)]
        ids = np.cumsum(start) - 1 + base
        tid[ok[:, s], s] = ids[ok[:, s]]
        base += int(start.sum())
    return tid


def first_kickoff(data_dir):
    """First frame of play, from the Smart Tagging kick-off tag (automatic, in the provided data)."""
    ks = [t["s"] for t in load_tags(data_dir) if t["code"] == "SET PIECES" and "kick off" in t["type"]]
    return min(ks) if ks else 0


def load_homography(data_dir, n):
    h = pd.read_csv(asset(data_dir, "_homography.csv"))
    cols = [f"h{i}{j}" for i in range(3) for j in range(3)]
    Hm = np.full((n, 3, 3), np.nan)
    ok = h["valid"].values.astype(bool) & (h["frame"].values < n)
    Hm[h["frame"].values[ok]] = h[cols].values[ok].reshape(-1, 3, 3)
    return Hm


def project(Hm, x, y):
    p = Hm @ np.vstack([x, y, np.ones_like(x)])
    return p[0] / p[2], p[1] / p[2]


def sample_colours(data_dir, players, start, stop, log=print):
    """Colour samples on frames [start, stop). The video is read sequentially from frame 0 (no
    seeking), so video frame t is guaranteed to be tracking row t."""
    import cv2
    n = stop
    Hm = load_homography(data_dir, len(players))
    tracked = (~np.isnan(players[:, :, 0])).sum(1)
    cap = cv2.VideoCapture(asset(data_dir, ".mp4"))
    if not cap.isOpened():
        raise SystemExit("cannot open the tactical video")
    out_t, out_s, out_c = [], [], []
    t0 = time.time()
    for t in range(n):
        if t < start or t % STRIDE or tracked[t] < MIN_TRACKED:
            cap.grab()
            continue
        ok, img = cap.read()
        if not ok:
            break
        if np.isnan(Hm[t, 0, 0]):
            continue
        x01, y01 = players[t, :, 0].astype(float), players[t, :, 1].astype(float)
        live = np.flatnonzero(~np.isnan(x01))
        if len(live) < 2:
            continue
        xm, ym = x01[live] * L, y01[live] * W
        dd = np.hypot(xm[:, None] - xm[None, :], ym[:, None] - ym[None, :])
        np.fill_diagonal(dd, np.inf)
        alone = live[dd.min(1) > MIN_SEP_M]
        if not len(alone):
            continue
        u, v = project(Hm[t], x01[alone], y01[alone])
        u1, v1 = project(Hm[t], x01[alone] + 1.0 / L, y01[alone])
        ppm = np.hypot(u1 - u, v1 - v)
        hgt, wid = img.shape[:2]
        for s, uu, vv, k in zip(alone, u, v, ppm):
            if not np.isfinite(k) or k < MIN_PX_PER_M:
                continue
            c0, c1 = int(uu - HALF_W_M * k), int(uu + HALF_W_M * k) + 1
            r0, r1 = int(vv - TORSO_M[1] * k), int(vv - TORSO_M[0] * k) + 1
            c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, wid), min(r1, hgt)
            if c1 - c0 < 2 or r1 - r0 < 2:
                continue
            patch = img[r0:r1, c0:c1]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 40)
            keep = ~grass
            if keep.mean() < MIN_NONGRASS:
                continue
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[keep].astype(float)
            out_t.append(t); out_s.append(int(s)); out_c.append(np.median(lab, 0))
        if t % (STRIDE * 2500) == 0 and t:
            log(f"  video frame {t}/{n}  {time.time() - t0:.0f}s  {len(out_t)} samples")
    cap.release()
    return np.array(out_t, np.int32), np.array(out_s, np.int16), np.array(out_c, np.float32).reshape(-1, 3)


def assign(t, s, colour, tid, team, players):
    """Per-tracklet team from colour votes. Returns (T (N,46), report dict)."""
    n = len(tid)
    tr = tid[t, s]
    # per-tracklet context from the tracking alone: team-count excess and distance to goal
    f_all, s_all = np.nonzero(tid >= 0)
    tr_all = tid[f_all, s_all]
    team_all = team[s_all]
    xm_all = players[f_all, s_all, 0].astype(float) * L
    goal_dist = pd.Series(np.minimum(xm_all, L - xm_all)).groupby(tr_all).median()
    feat = colour * np.array([L_WEIGHT, 1.0, 1.0])
    lab = team[s].astype(int)                      # start from the tracking's own labels
    n_tr = int(tid.max()) + 1
    decided = np.full(n_tr, -2, np.int8)           # -2 = keep the tracking's label
    for _ in range(3):
        # team-count excess on each tracklet's frames, officials found so far left out
        off = decided[tr_all] == -1
        h_cnt = np.bincount(f_all[(team_all == 0) & ~off], minlength=n)
        a_cnt = np.bincount(f_all[(team_all == 1) & ~off], minlength=n)
        excess = pd.Series((h_cnt - a_cnt)[f_all]).groupby(tr_all).mean()   # > 0: home over-counted
        cen = np.stack([np.median(feat[lab == j], 0) for j in (0, 1)])
        d = np.linalg.norm(feat[:, None, :] - cen[None, :, :], axis=2)
        # usable radius: R_MULT x the typical own-team spread, never less than half the gap
        # between the two team colours (keeps it meaningful when kits are very uniform)
        R = max(R_MULT * np.median(d[np.arange(len(d)), lab]), 0.5 * np.linalg.norm(cen[0] - cen[1]))
        usable = d.min(1) < R
        vote1 = (d[:, 1] < d[:, 0]).astype(float)
        df = pd.DataFrame(dict(tr=tr, uv=vote1 * usable, u=usable.astype(float)))
        g = df.groupby("tr")
        cnt = g.u.sum()
        frac1 = g.uv.sum() / cnt.replace(0, np.nan)
        far = 1 - g.u.mean()
        total = g.size()
        prev_off = decided == -1
        decided[:] = -2
        decided[prev_off] = -1
        for k in cnt.index:
            if decided[k] == -1:
                continue
            if goal_dist[k] < KEEPER_M:
                continue                           # goalkeeper: never moved
            if cnt[k] >= MIN_VOTES and frac1[k] >= AGREE and excess[k] >= EXCESS:
                decided[k] = 1                     # looks away, and home has one too many here
            elif cnt[k] >= MIN_VOTES and frac1[k] <= 1 - AGREE and excess[k] <= -EXCESS:
                decided[k] = 0                     # looks home, and away has one too many here
            elif total[k] >= MIN_VOTES and far[k] >= OFFICIAL_FAR and goal_dist[k] > GOAL_M:
                decided[k] = -1
        new = decided[tr]
        lab = np.where(new >= 0, new, lab)
        lab = np.where(new == -1, team[s], lab)    # officials do not shape the team colours
    T = np.broadcast_to(team, tid.shape).astype(np.int8).copy()
    m = (tid >= 0) & (decided[np.maximum(tid, 0)] != -2)
    T[m] = decided[tid[m]]
    slot_of = np.full(n_tr, -1)
    f_, s_ = np.nonzero(tid >= 0)
    slot_of[tid[f_, s_]] = s_
    orig = team[np.maximum(slot_of, 0)]
    rep = dict(tracklets=n_tr, sampled=int(len(np.unique(tr))),
               kept=int(((decided == orig) | (decided == -2)).sum()),
               to_home=int(((decided == 0) & (orig == 1)).sum()),
               to_away=int(((decided == 1) & (orig == 0)).sum()),
               officials=int((decided == -1).sum()))
    return T, rep


def imbalance(players, T):
    """Share of frames with 20+ tracked players where one team has 12+ and the other 10 or fewer."""
    tracked = ~np.isnan(players[:, :, 0])
    h = (tracked & (T == 0)).sum(1)
    a = (tracked & (T == 1)).sum(1)
    busy = tracked.sum(1) >= 20
    bad = ((h >= 12) & (a <= 10)) | ((a >= 12) & (h <= 10))
    return float(bad[busy].mean())


def team_matrix(data_dir, players, team, start=None, stop=None, log=print):
    k1 = first_kickoff(data_dir)
    start = k1 if start is None else max(start, k1)
    stop = len(players) if stop is None else min(stop, len(players))
    cache = os.path.join(derived_dir(data_dir), f"teamfix_samples_s{STRIDE}_t{MIN_TRACKED}_{start}_{stop}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        t, s, c = z["t"], z["s"], z["c"]
    else:
        log(f"teamfix: sampling jersey colours from video frames {start}-{stop} (one-off, cached)")
        t, s, c = sample_colours(data_dir, players, start, stop, log)
        np.savez_compressed(cache, t=t, s=s, c=c)
    tid = tracklets(players)
    T, rep = assign(t, s, c, tid, team, players)
    rep["samples"] = int(len(t))
    rep["window"] = (int(start), int(stop))
    return T, tid, rep


def export_checks(data_dir, players, team, T, tid, k=20, seed=0):
    """Write k random corrected tracklets as images for a visual check (local only, never upload)."""
    import cv2
    out = os.path.join(derived_dir(data_dir), "teamfix_check")
    os.makedirs(out, exist_ok=True)
    Hm = load_homography(data_dir, len(players))
    changed = (tid >= 0) & (T != team[None, :])
    ids = np.unique(tid[changed])
    rng = np.random.default_rng(seed)
    pick = rng.choice(ids, size=min(k, len(ids)), replace=False) if len(ids) else []
    names = {0: "home", 1: "away", -1: "official"}
    want = []
    for trk in pick:
        f_idx, s_idx = np.nonzero(tid == trk)
        mid = len(f_idx) // 2
        want.append((int(f_idx[mid]), int(s_idx[mid])))
    want.sort()
    open(os.path.join(out, "index.txt"), "w").close()
    cap = cv2.VideoCapture(asset(data_dir, ".mp4"))
    t = -1
    for i, (f, s) in enumerate(want):
        while t < f - 1:                       # sequential read: no seeking, so frames stay aligned
            cap.grab(); t += 1
        ok, img = cap.read(); t += 1
        if not ok:
            continue
        u, v = project(Hm[f], np.array([players[f, s, 0]], float), np.array([players[f, s, 1]], float))
        u, v = int(u[0]), int(v[0])
        crop = img[max(v - 160, 0):v + 40, max(u - 100, 0):u + 100].copy()
        cv2.circle(crop, (min(u, 100), min(v, 160)), 6, (0, 255, 255), 2)
        label = f"frame {f} slot {s}: {names[int(team[s])]} -> {names[int(T[f, s])]}"
        path = os.path.join(out, f"{i:02d}_f{f}_s{s}.png")
        cv2.imwrite(path, crop)
        with open(os.path.join(out, "index.txt"), "a") as fh:
            fh.write(label + "\n")
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", type=int, default=None, help="first frame to sample (default: first kick-off)")
    ap.add_argument("--max-frames", type=int, default=None, help="frames to sample from --start")
    ap.add_argument("--checks", type=int, default=20)
    a = ap.parse_args()
    players, ball, team = load_atd(a.data)
    t0 = time.time()
    start = a.start if a.start is not None else first_kickoff(a.data)
    stop = None if a.max_frames is None else start + a.max_frames
    T, tid, rep = team_matrix(a.data, players, team, start, stop)
    a0, b0 = rep["window"]
    w = slice(a0, b0)
    base = np.broadcast_to(team, T.shape)
    print(f"teamfix done in {time.time() - t0:.0f}s: {rep}")
    print(f"frames 12-vs-10 (20+ tracked) in frames {a0}-{b0}: before {imbalance(players[w], base[w]):.3f}  "
          f"after {imbalance(players[w], T[w]):.3f}   (target < 0.02)")
    live = ~np.isnan(players[w, :, 0])
    print(f"player-frames whose team changed: {np.mean((T[w] != base[w])[live]):.3f}")
    if a.checks:
        sub = np.full_like(tid, -1); sub[w] = tid[w]
        print("check images:", export_checks(a.data, players, team, T, sub, a.checks))


if __name__ == "__main__":
    main()