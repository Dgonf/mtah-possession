"""A/B test of decoder variants on the public DFL matches (other games, CC-BY 4.0), with the
organisers' scorer, leave-one-match-out for the dead-ball model.

    python tools/dfl_ab.py --dfl external/dfl_npz --metrics <challenge>/evaluation/possession_metrics.py

Two conditions per match:
  clean   the ball hidden at random in long blocks until 41 % of frames have no ball (dfl_grid.py)
  hard    the ball hidden preferentially where two opponents contest it (the provided match loses
          the ball "disproportionately on the interesting frames", FAQ.md), and synthetic team-slot
          errors: a player moved to an empty slot of the other team, until ~14 % of graded frames
          show a 12-vs-10 count (TECHNICAL.md section 9.2)
No data from the provided game is read here. Deterministic (fixed seeds).
"""
import argparse
import glob
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dfl_grid as G                           # noqa: E402
from possession import run                     # noqa: E402
from possession.io import FPS, L, W            # noqa: E402

V3 = dict(W_LIVE=0.0, W_CTRL_BAD=1.0, W_RESTART=0.0)
VARIANTS = {
    "v3": {},
    "live0.5": dict(W_LIVE=0.5), "live1": dict(W_LIVE=1.0), "live2": dict(W_LIVE=2.0),
    "ctrlbad0.3": dict(W_CTRL_BAD=0.3), "ctrlbad0": dict(W_CTRL_BAD=0.0),
    "restart0.5": dict(W_RESTART=0.5), "restart1": dict(W_RESTART=1.0),
    "live1+restart0.5": dict(W_LIVE=1.0, W_RESTART=0.5),
    "live1+restart0.5+ctrlbad0.3": dict(W_LIVE=1.0, W_RESTART=0.5, W_CTRL_BAD=0.3),
    "live1+ctrlbad0.3": dict(W_LIVE=1.0, W_CTRL_BAD=0.3),
    "live1.5+ctrlbad0.3": dict(W_LIVE=1.5, W_CTRL_BAD=0.3),
    "live1+ctrlbad0.5": dict(W_LIVE=1.0, W_CTRL_BAD=0.5),
}
SHIFTS = (0, 25, 50)
NOISE_SHARE = 0.14


def duel_frames(players, ball, team):
    px, py = players[:, :, 0] * L, players[:, :, 1] * W
    d = np.hypot(px - ball[:, None, 0] * L, py - ball[:, None, 1] * W)
    d = np.where(np.isnan(d), np.inf, d)
    return (d[:, team == 0].min(1) < 2.0) & (d[:, team == 1].min(1) < 2.0)


def mask_ball_duels(ball, graded, duel, seed=0):
    """As dfl_grid.mask_ball, but block starts are drawn 1 s before a duel 3 times out of 4."""
    rng = np.random.default_rng(seed)
    ball = ball.copy()
    n = len(ball)
    sigma = np.log(G.GAP_P90_S / G.GAP_MEDIAN_S) / 1.2816
    target = G.MISS_SHARE * graded.sum()
    miss = np.isnan(ball[:, 0]) & graded
    idx, didx = np.flatnonzero(graded), np.flatnonzero(graded & duel)
    while miss.sum() < target:
        length = int(np.exp(rng.normal(np.log(G.GAP_MEDIAN_S), sigma)) * FPS)
        if len(didx) and rng.random() < 0.75:
            start = max(0, int(rng.choice(didx)) - int(rng.uniform(0, 1) * length))
        else:
            start = int(rng.choice(idx))
        miss[start:min(n, start + length)] = True
        miss &= graded
    ball[miss] = np.nan
    return ball


def team_noise(players, team, graded, seed=0):
    """Move players into empty slots of the other team for 5-60 s until NOISE_SHARE of graded
    frames carry one wrong-slot player."""
    rng = np.random.default_rng(seed + 1)
    P = players.copy()
    n = len(P)
    bad = np.zeros(n, bool)
    idx = np.flatnonzero(graded)
    tries = 0
    while bad[graded].mean() < NOISE_SHARE and tries < 100000:
        tries += 1
        length = int(rng.uniform(5, 60) * FPS)
        s = int(rng.choice(idx)); e = min(n, s + length)
        if bad[s:e].any():
            continue
        k = int(rng.integers(2))
        own = [j for j in np.flatnonzero(team == k) if not np.isnan(P[s:e, j, 0]).any()]
        free = [j for j in np.flatnonzero(team == 1 - k) if np.isnan(P[s:e, j, 0]).all()]
        if not own or not free:
            continue
        a, b = int(rng.choice(own)), free[0]
        P[s:e, b] = P[s:e, a]
        P[s:e, a] = np.nan
        bad[s:e] = True
    return P


def set_params(p):
    for k, v in {**V3, **p}.items():
        setattr(run, k, v)


def one_match(job):
    f, i, metrics, cond, shift = job
    pm = G.load_pm(metrics)
    d = np.load(f)
    g, players, team = d["graded"], d["players"], d["team"]
    if cond == "clean":
        ball = G.mask_ball(d["ball"], g, seed=0)
    else:
        ball = mask_ball_duels(d["ball"], g, duel_frames(players, d["ball"], team), seed=0)
        players = team_noise(players, team, g, seed=0)
    run.SHAPE_SHIFT = shift
    S = run.build_arrays(players, ball, team, G.kickoff_tags(d["kickoffs"]), log=lambda *_: None,
                         ref=g, exclude_match=i)
    gc = pm.label_codes(np.where(g, G.merge_short(d["labels"], 3.0), "X"))
    rows = []
    for name, p in VARIANTS.items():
        set_params(p)
        pred = run.decode(S, use_tag_prior=False)
        r = pm.evaluate(gc, pm.label_codes(np.where(g, pred, "X")))
        rows.append(dict(match=os.path.basename(f)[:-4], cond=cond, shift=shift, variant=name,
                         spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"],
                         aliveF1=r["alive_dead_f1"], n_pred=r["n_pred_spells"], n_gt=r["n_gt_spells"]))
    print(f"{rows[0]['match']} {cond} shift {shift}: v3 {rows[0]['spellF1']:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default="external/dfl_npz")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="dfl_ab.csv")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--conds", default="clean,hard")
    ap.add_argument("--shifts", default=",".join(map(str, SHIFTS)))
    ap.add_argument("--variants", default=None, help="comma-separated subset of VARIANTS")
    a = ap.parse_args()
    if a.variants:
        keep = a.variants.split(",")
        for k in list(VARIANTS):
            if k not in keep:
                del VARIANTS[k]
    files = sorted(glob.glob(os.path.join(a.dfl, "*.npz")))
    jobs = [(f, i, a.metrics, c, s) for (i, f), c, s in
            itertools.product(enumerate(files), a.conds.split(","), map(int, a.shifts.split(",")))]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one_match, jobs):
            rows += r
            pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    t = df.groupby(["cond", "shift", "variant"])[["spellF1", "macroF1", "aliveF1", "n_pred", "n_gt"]].mean()
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
