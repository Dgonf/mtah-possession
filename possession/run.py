"""End to end: provided data in, frame,label CSV out.

    python -m possession.run --data data --out submission.csv
"""
import argparse
import time
import warnings

import numpy as np

from . import deadball, deadsup, features, models
from .decode import A, D, H, viterbi3
from .io import FPS, GRADED, load_atd, load_tags

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- decoder weights (per frame, in "frames of clear control" units) ---
W_CTRL = 1.0         # one frame of clear control by a team
W_GEO_VIS = 0.02     # team-shape model while the ball is tracked (control evidence dominates)
W_GEO_MISS = 0.10    # team-shape model while the ball is not tracked
W_DEAD = 0.5         # self-supervised dead-ball model (v2), times its clipped log-odds
W_DEAD_SUP = 1.0     # dead-ball model trained on the public DFL matches (possession/deadsup.py)
W_RULE = 3.0         # a rule-detected stoppage (ball out, set piece, goal): high precision
W_TAG = 0.01         # Smart Tagging phase team, a weak prior
W_TAKER = 0.2        # restart taker's team, for the 2 s after a restart
SWITCH_COST = 8.0    # H <-> A
ENTER_DEAD = 15.0    # live -> D
LEAVE_DEAD = 15.0    # D -> live
GEO_CLIP, DEAD_CLIP = 3.0, 4.0

ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
          "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL"}
DEFEND = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
          "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def logit(p, clip):
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-4, 1 - 1e-4)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def build_arrays(players, ball, team, tags, log=print, ref=None, exclude_match=None):
    """ref: bool mask of the frames inside the two halves. Given, the dead-ball model is the one
    trained on the public DFL matches; None reproduces v2 (self-supervised on this match only)."""
    t0 = time.time()
    F = features.basic(players, ball, team)
    R = deadball.rules(F, players, tags)
    log(f"rules        {time.time() - t0:5.1f}s  rule-dead {R['rule_dead'].mean():.3f}")

    sign = features.attack_sign(F, R["halves"])
    hv, av = features.geometry(F, sign)
    p_home, _ = models.possession_from_shape(F, hv, av, R["rule_dead"], R["halves"])
    log(f"shape model  {time.time() - t0:5.1f}s")
    if ref is None:
        Xd = features.dead_features(F)
        p_dead, _ = models.dead_from_motion(F, Xd, R["rule_dead"], R["hard"], R["halves"])
    else:
        p_dead = deadsup.predict(F, ref, exclude_match)
    R = deadball.fix_goals(R, p_dead, tags)
    log(f"dead model   {time.time() - t0:5.1f}s")
    return dict(F=F, R=R, tags=tags, p_home=p_home, p_dead=p_dead, n=F["n"], sup=ref is not None)


def graded_mask(n):
    m = np.zeros(n, bool)
    for a, b in GRADED:
        m[a:b + 1] = True
    return m


def build(data_dir, use_tag_prior=True, log=print, teamfix=False, v2=False):
    players, ball, team = load_atd(data_dir)
    if teamfix:                       # per-frame teams from jersey colour (possession/teamfix.py)
        from . import teamfix as tfx
        team, _, rep = tfx.team_matrix(data_dir, players, team, log=log)
        log(f"teamfix      {rep}")
    ref = None if v2 else graded_mask(len(players))
    return build_arrays(players, ball, team, load_tags(data_dir), log=log, ref=ref)


def emissions(S, use_tag_prior=True):
    F, R, n = S["F"], S["R"], S["n"]
    em = np.zeros((n, 3))
    ch, ca = models.control(F)
    em[:, H] += W_CTRL * ch
    em[:, A] += W_CTRL * ca
    g = logit(S["p_home"], GEO_CLIP)
    w = np.where(F["has"], W_GEO_VIS, W_GEO_MISS)
    em[:, H] += w * g / 2
    em[:, A] -= w * g / 2
    em[:, D] += (W_DEAD_SUP if S.get("sup") else W_DEAD) * logit(S["p_dead"], DEAD_CLIP)
    em[R["rule_dead"], D] += W_RULE
    em[R["hard"], D] += 1e3
    for rs, tm in R["restarts"].items():
        if tm >= 0:
            em[rs:rs + 2 * FPS, tm] += W_TAKER
    if use_tag_prior:
        for t in S["tags"]:
            if t["team"] < 0:
                continue
            if t["code"] in ATTACK:
                em[t["s"]:t["e"], t["team"]] += W_TAG
            elif t["code"] in DEFEND:
                em[t["s"]:t["e"], 1 - t["team"]] += W_TAG
    return em


def decode(S, use_tag_prior=True):
    path = viterbi3(emissions(S, use_tag_prior), SWITCH_COST, ENTER_DEAD, LEAVE_DEAD)
    return np.array(["H", "A", "D"])[path]


def write(labels, path):
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--teamfix", action="store_true", help="correct team labels from jersey colour")
    ap.add_argument("--v2", action="store_true", help="v2: self-supervised dead-ball model, no external data")
    a = ap.parse_args()
    S = build(a.data, teamfix=a.teamfix, v2=a.v2)
    lab = decode(S)
    write(lab, a.out)
    print(f"wrote {a.out}: {len(lab)} rows")


if __name__ == "__main__":
    main()