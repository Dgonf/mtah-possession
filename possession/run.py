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

# --- v4 terms (0 = off = v3 behaviour). Values are set from the DFL experiments (tools/dfl_ab.py) ---
W_LIVE = 1.0         # ball tracked inside the pitch and moving or controlled: evidence against D
LIVE_IN_M = 1.0      # m inside the lines for that
LIVE_V = 2.0         # m/s, ball speed that counts as moving
W_CTRL_BAD = 0.3     # control weight for the team with 12+ tracked players (a wrong-slot player)
W_RESTART = 0.0      # 2nd pass: per frame, for 2 s, to the team that controls the ball at a restart
RESTART_PRE_S = 2.0  # s before a D -> live boundary in which the taker's control is read
SHAPE_SHIFT = 0      # frames: the shape model reads features from t + shift (look-ahead)

ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
          "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL"}
DEFEND = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
          "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def logit(p, clip):
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-4, 1 - 1e-4)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def shift_rows(df, k):
    """Row t gets the values of row t + k (the last k rows repeat the final row)."""
    idx = np.minimum(np.arange(len(df)) + k, len(df) - 1)
    out = df.iloc[idx].copy()
    out.index = df.index
    return out


def team_counts(F):
    """Tracked players per team in each frame (static slot teams)."""
    ok = ~np.isnan(F["px"])
    if F["team"].ndim == 1:
        return ok[:, F["team"] == 0].sum(1), ok[:, F["team"] == 1].sum(1)
    return (ok & (F["team"] == 0)).sum(1), (ok & (F["team"] == 1)).sum(1)


def live_ball(F):
    """Frames where the tracked ball is clearly in play: inside the pitch and moving or controlled."""
    ch, ca = models.control(F)
    with np.errstate(invalid="ignore"):
        moving = np.nan_to_num(F["bv"], nan=0.0) > LIVE_V
        inn = F["has"] & features.inside(F["bx"], F["by"], LIVE_IN_M)
    return inn & (moving | ch | ca)


def build_arrays(players, ball, team, tags, log=print, ref=None, exclude_match=None):
    """ref: bool mask of the frames inside the two halves. Given, the dead-ball model is the one
    trained on the public DFL matches; None reproduces v2 (self-supervised on this match only)."""
    t0 = time.time()
    F = features.basic(players, ball, team)
    R = deadball.rules(F, players, tags)
    log(f"rules        {time.time() - t0:5.1f}s  rule-dead {R['rule_dead'].mean():.3f}")

    sign = features.attack_sign(F, R["halves"])
    hv, av = features.geometry(F, sign)
    if SHAPE_SHIFT:                   # team shape reacts after a turnover: read it a little later
        hv, av = shift_rows(hv, SHAPE_SHIFT), shift_rows(av, SHAPE_SHIFT)
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
    wh = np.full(n, W_CTRL)
    wa = np.full(n, W_CTRL)
    if W_CTRL_BAD != 1.0:             # a team with 12+ players carries a player of the other team
        nh, na = team_counts(F)
        wh[(nh >= 12) & (na <= 10)] *= W_CTRL_BAD
        wa[(na >= 12) & (nh <= 10)] *= W_CTRL_BAD
    em[:, H] += wh * ch
    em[:, A] += wa * ca
    if W_LIVE:
        em[live_ball(F) & ~R["rule_dead"], D] -= W_LIVE
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


def restart_teams(S, path):
    """For each D -> live boundary of a first decode: the team that controls the ball in the
    RESTART_PRE_S before it (the taker standing at the ball), else the tagged set-piece team."""
    F = S["F"]
    ch, ca = models.control(F)
    b = np.flatnonzero((path[:-1] == D) & (path[1:] != D)) + 1
    pre = int(RESTART_PRE_S * FPS)
    tagged = [(t["s"], t["team"]) for t in S["tags"] if t["code"] == "SET PIECES" and t["team"] >= 0]
    out = {}
    for t in b:
        h, a = ch[max(0, t - pre):t + 5].sum(), ca[max(0, t - pre):t + 5].sum()
        if h != a:
            out[int(t)] = H if h > a else A
            continue
        near = [tm for s, tm in tagged if abs(s - t) < 3 * FPS]
        if near:
            out[int(t)] = near[0]
    return out


def decode(S, use_tag_prior=True):
    em = emissions(S, use_tag_prior)
    path = viterbi3(em, SWITCH_COST, ENTER_DEAD, LEAVE_DEAD)
    if W_RESTART:                     # 2nd pass: the restart taker's team starts the possession
        for t, tm in restart_teams(S, path).items():
            em[t:t + 2 * FPS, tm] += W_RESTART
        path = viterbi3(em, SWITCH_COST, ENTER_DEAD, LEAVE_DEAD)
    return np.array(["H", "A", "D"])[path]


N_FRAMES = 199319
KICKOFFS = (11185, 113742)            # published in the organisers' DATA.md


def check_build(S):
    """Fail loudly where the rule layer could go wrong silently on the provided match."""
    if S["n"] != N_FRAMES:
        raise ValueError(f"ATD has {S['n']} frames, the video has {N_FRAMES}")
    ks = [h[0] for h in S["R"]["halves"]]
    for k, ref in zip(ks, KICKOFFS):
        if abs(k - ref) > 2 * FPS:
            raise ValueError(f"kick-off detected at frame {k}, published {ref}: check the Smart Tagging")


def write(labels, path):
    if not set(np.unique(labels)) <= {"H", "A", "D", "X"}:
        raise ValueError("invalid label")
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
    check_build(S)
    lab = decode(S)
    write(lab, a.out)
    print(f"wrote {a.out}: {len(lab)} rows")


if __name__ == "__main__":
    main()