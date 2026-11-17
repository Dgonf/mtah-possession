"""Grid search of the v2 decoder weights on the public DFL / IDSSE matches (other games, CC-BY 4.0),
scored with the organisers' own scorer (evaluation/possession_metrics.py, spellF1 = ranking metric).

    python tools/dfl_grid.py --dfl external/dfl_npz --metrics ../../more-than-a-hack-2027/evaluation/possession_metrics.py

Input: one .npz per DFL match from tools/dfl_convert.py:
    players (N,46,3)  ball (N,3)  team (46,)  labels (N,) H/A/D  graded (N,) bool  kickoffs (2,)

Per match:
  1. hide the ball in long blocks until ~41 % of graded frames have no ball (the provided match's
     share; block lengths log-normal, median 10.7 s, p90 34 s, from TECHNICAL.md section 1),
  2. run the v2 build once on the masked data,
  3. decode with every weight combination in GRID and score it against the DFL labels with the
     organisers' scorer. Ungraded frames are X, so the two halves are scored separately, exactly
     as the official scorer treats the provided match.
DFL counts very short touches as possessions (median 3.7 s); the challenge's ground truth has a mean
possession of 14.2 s. So the DFL labels are also scored with short spells merged (GT_MIN_S).
Results are reported leave-one-match-out: weights are picked on six matches and scored on the
seventh. No data from the provided game is read here. Deterministic.
"""
import argparse
import glob
import importlib.util
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from possession import run                     # noqa: E402
from possession.decode import viterbi3         # noqa: E402
from possession.io import FPS                  # noqa: E402

GRID = {
    "SWITCH_COST": [6.0, 8.0, 12.0],
    "DEAD_COST":   [10.0, 15.0, 25.0],     # used for both ENTER_DEAD and LEAVE_DEAD
    "W_GEO_MISS":  [0.05, 0.10, 0.20],
    "W_DEAD":      [0.25, 0.5, 1.0],
}                                          # 81 combinations
V2 = dict(SWITCH_COST=8.0, DEAD_COST=15.0, W_GEO_MISS=0.10, W_DEAD=0.5)

MISS_SHARE, GAP_MEDIAN_S, GAP_P90_S = 0.41, 10.7, 34.0
GT_MIN_S = (1.0, 2.0, 3.0)                 # DFL possessions shorter than this are merged


def load_pm(path):
    spec = importlib.util.spec_from_file_location("possession_metrics", path)
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    return pm


def merge_short(lab, min_s):
    """Give live spells shorter than min_s to the team before them (after them if a dead ball
    comes before). Single touches and 50/50s do not change possession in the challenge's labels."""
    lab = lab.copy()
    if min_s <= 0:
        return lab
    st = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    en = np.r_[st[1:], len(lab)]
    for s, e in zip(st, en):
        if lab[s] == "D" or e - s >= min_s * FPS:
            continue
        prev = lab[s - 1] if s > 0 else "D"
        nxt = lab[e] if e < len(lab) else "D"
        fill = prev if prev != "D" else nxt
        if fill != "D":
            lab[s:e] = fill
    return lab


def mask_ball(ball, graded, seed=0):
    """Hide the ball in long blocks until MISS_SHARE of graded frames have no ball."""
    rng = np.random.default_rng(seed)
    ball = ball.copy()
    n = len(ball)
    sigma = np.log(GAP_P90_S / GAP_MEDIAN_S) / 1.2816
    target = MISS_SHARE * graded.sum()
    miss = np.isnan(ball[:, 0]) & graded
    idx = np.flatnonzero(graded)
    while miss.sum() < target:
        length = int(np.exp(rng.normal(np.log(GAP_MEDIAN_S), sigma)) * FPS)
        start = int(rng.choice(idx))
        miss[start:min(n, start + length)] = True
        miss &= graded
    ball[miss] = np.nan
    return ball


def kickoff_tags(kickoffs):
    """Only the two half starts, as the rule layer expects them. No DFL human events are used."""
    return [dict(code="SET PIECES", s=int(k), e=int(k) + FPS, team=-1, type="kick off (start)")
            for k in kickoffs]


def decode_with(S, p):
    run.SWITCH_COST = p["SWITCH_COST"]
    run.ENTER_DEAD = run.LEAVE_DEAD = p["DEAD_COST"]
    run.W_GEO_MISS = p["W_GEO_MISS"]
    run.W_DEAD = p["W_DEAD"]
    em = run.emissions(S, use_tag_prior=False)
    path = viterbi3(em, run.SWITCH_COST, run.ENTER_DEAD, run.LEAVE_DEAD)
    return np.array(["H", "A", "D"])[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default="external/dfl_npz")
    ap.add_argument("--metrics", required=True, help="path to the challenge's evaluation/possession_metrics.py")
    ap.add_argument("--out", default="dfl_grid.csv")
    a = ap.parse_args()
    pm = load_pm(a.metrics)
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    v2_combo = combos.index(V2)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dfl, "*.npz"))):
        m = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        g = d["graded"]
        ball = mask_ball(d["ball"], g, seed=0)
        t0 = time.time()
        S = run.build_arrays(d["players"], ball, d["team"], kickoff_tags(d["kickoffs"]), log=lambda *_: None)
        gts = {}
        for ms in GT_MIN_S:
            gc = pm.label_codes(np.where(g, merge_short(d["labels"], ms), "X"))
            runs = pm.valid_runs(gc)
            gts[ms] = (gc, runs, pm.possession_stats(gc, runs))
        print(f"{m}: built in {time.time() - t0:.0f}s, ball missing {np.isnan(ball[g, 0]).mean():.2f}, "
              + ", ".join(f"GT<{ms:g}s {s['n_spells']} poss/{s['mean_duration_s']:.1f}s" for ms, (_, _, s) in gts.items()))
        for i, p in enumerate(combos):
            pc = pm.label_codes(np.where(g, decode_with(S, p), "X"))
            for ms, (gc, runs, gst) in gts.items():
                sp = pm.spell_scores(gc, pc, runs)
                rows.append(dict(match=m, combo=i, gt_min_s=ms, **p,
                                 spellF1=sp["mean_f1"],
                                 macroF1=pm.macro_f1(pm.scored_confusion(pm.confusion(gc, pc))),
                                 n_poss=sp["n_pred"], n_true=sp["n_gt"],
                                 dur=pm.possession_stats(pc, runs)["mean_duration_s"],
                                 dur_true=gst["mean_duration_s"]))
        pd.DataFrame(rows).to_csv(a.out, index=False)        # saved after every match
    df = pd.DataFrame(rows)

    print("\nspellF1 is the ranking metric; gaps under ~0.04 are noise on one match (scorer docs).")
    for ms in GT_MIN_S:
        dm = df[df.gt_min_s == ms]
        held = []
        for m in sorted(dm.match.unique()):
            best = dm[dm.match != m].groupby("combo")["spellF1"].mean().idxmax()
            held.append(dm[(dm.match == m) & (dm.combo == best)]["spellF1"].item())
        best_all = dm.groupby("combo")["spellF1"].mean().idxmax()
        b, v = dm[dm.combo == best_all], dm[dm.combo == v2_combo]
        print(f"\nGT merged < {ms:g}s: true {v.n_true.mean():.0f} poss/match, mean {v.dur_true.mean():.1f}s")
        print(f"  v2 weights       spellF1 {v.spellF1.mean():.3f}  macroF1 {v.macroF1.mean():.3f}  "
              f"{v.n_poss.mean():.0f} poss, {v.dur.mean():.1f}s")
        print(f"  held-out picked  spellF1 {np.mean(held):.3f}")
        print(f"  best on all      spellF1 {b.spellF1.mean():.3f}  macroF1 {b.macroF1.mean():.3f}  "
              f"{b.n_poss.mean():.0f} poss, {b.dur.mean():.1f}s  {dict((k, float(b[k].iloc[0])) for k in GRID)}")


if __name__ == "__main__":
    main()