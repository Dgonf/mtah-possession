"""Dead-ball model trained on other games (public DFL / IDSSE data), applied to this one.

Why: on seven public Bundesliga matches with human in-play labels, v2's self-supervised stoppage
model was the bottleneck (alive/dead F1 0.74; with a perfect dead-ball signal spellF1 rose from
0.45 to 0.67). A classifier trained on the DFL in-play status closes most of that gap, using
player positions only.

Design choices that make it transfer to the provided match's noisier automatic tracking:
  - Player-only features. Nothing about the ball goes in: in the DFL data the ball is always
    tracked, here it is missing on 41 % of graded frames, so ball features would not transfer.
  - No team labels either (collective speed and formation only), so tracking team errors do not matter.
  - Every feature is replaced by its quantile within the match (0..1000, among frames of the two
    halves). Scale differences between tracking providers (speed estimation, jitter) drop out.

The training table (possession/dfl_dead_train.npz) holds every 25th in-half frame of the seven
DFL matches: quantised features, the DFL BallStatus label (1 = dead), and the match index. It is
built by tools/dfl_dead_table.py from the public download and is the only labelled data used.
No label of any kind from the provided match is involved.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features
from .io import FPS, L, W

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dfl_dead_train.npz")
MOTION = ("med", "mean", "p90")          # dead_features() columns that depend on player speed only
Q = 1000                                 # quantile resolution


def formation(F):
    """Team-agnostic formation features: how many are slow or fast, how spread, who is in the boxes."""
    px, py, sp = F["px"], F["py"], F["sp"]
    with np.errstate(invalid="ignore"):
        cnt = np.maximum((~np.isnan(px)).sum(1), 1)
        f = {
            "slow": np.nansum(sp < 1.0, 1) / cnt,
            "fast": np.nansum(sp > 4.0, 1) / cnt,
            "walk": np.nansum(sp < 2.0, 1) / cnt,
            "spstd": np.nanstd(sp, 1),
            "xstd": np.nanstd(px, 1),
            "ystd": np.nanstd(py, 1),
            "cx": np.abs(np.nanmean(px, 1) - L / 2),
            "cy": np.abs(np.nanmean(py, 1) - W / 2),
            "box": ((np.abs(py - W / 2) < 20.16) & ((px < 16.5) | (px > L - 16.5))).sum(1) / cnt,
            "wide": ((py < 5) | (py > W - 5)).sum(1) / cnt,
            "deep": ((px < 25) | (px > L - 25)).sum(1) / cnt,
        }
    cols = {}
    for k, v in f.items():
        v = np.nan_to_num(v.astype(float))
        for w in (1, 3, 8):
            cols[f"{k}_m{w}"] = features.roll(v, w * FPS)
        cols[f"{k}_past3"] = features.roll(v, 3 * FPS, center=False)
        cols[f"{k}_next3"] = features.roll(v[::-1], 3 * FPS, center=False)[::-1]
    for k in ("slow", "walk", "spstd"):
        v = np.nan_to_num(f[k].astype(float))
        cols[f"{k}_m20"] = features.roll(v, 20 * FPS)
        cols[f"{k}_past8"] = features.roll(v, 8 * FPS, center=False)
        cols[f"{k}_next8"] = features.roll(v[::-1], 8 * FPS, center=False)[::-1]
    return pd.DataFrame(cols)


def player_features(F):
    """(X float32 (N, 80), column names): speed statistics from dead_features() plus formation()."""
    Xd = features.dead_features(F)
    motion = [c for c in Xd.columns if c.startswith(MOTION)]
    Fm = formation(F)
    X = np.hstack([Xd[motion].values, Fm.values]).astype(np.float32)
    return X, motion + list(Fm.columns)


def quantise(X, ref):
    """Each column -> its quantile (0..Q) among the rows where ref is True."""
    out = np.empty(X.shape, np.uint16)
    for j in range(X.shape[1]):
        s = np.sort(X[ref, j])
        out[:, j] = np.round(Q * np.searchsorted(s, X[:, j], side="left") / len(s))
    return out


def classifier():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def predict(F, ref, exclude_match=None):
    """P(ball is dead) for every frame. ref: bool mask of the frames inside the two halves.
    exclude_match leaves one DFL match out of training (used only to validate on that match)."""
    X, cols = player_features(F)
    tab = np.load(TABLE, allow_pickle=False)
    if list(tab["cols"]) != cols:
        raise ValueError("feature columns differ from the training table; rebuild it with tools/dfl_dead_table.py")
    keep = np.ones(len(tab["y"]), bool) if exclude_match is None else tab["match"] != exclude_match
    clf = classifier().fit(tab["X"][keep].astype(np.float32), tab["y"][keep])
    return clf.predict_proba(quantise(X, ref).astype(np.float32))[:, 1]
