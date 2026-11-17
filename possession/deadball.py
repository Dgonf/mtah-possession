"""Rule layer for the ball being out of play.

Produces three things the decoder uses:
  hard_dead   frames that are dead whatever else says (before kick-off, half-time, ball tracked out)
  rule_dead   stoppages found by clear events (ball out, set-piece tag, goal), onset -> restart
  restarts    {restart frame: team taking it, -1 if unknown}
and the half boundaries / kick-offs the rest of the pipeline needs.
"""
import numpy as np
import pandas as pd

from .features import inside
from .io import FPS, L, W

OUT_M = 0.5         # m beyond a line = ball out of play
LINE_M = 2.0        # m from a line = "at the line" when the ball vanishes
VANISH_S = 1.5      # s missing after vanishing at a line
SLOW = 1.6          # m/s median player speed = players have stopped running
STOP_RATIO = 0.7    # ... or slowed to 70 % of the previous 2 s
MIN_DEAD_S = 2.0    # s, the shortest stoppage
KICK_V = 4.0        # m/s, ball speed of a restart kick / throw
INSIDE_M = 0.3      # m inside the lines for that restart to count
MAX_DEAD_S = 300.0  # s, longest stoppage searched (VAR)
MIN_LIVE_S = 1.0    # s, shorter live stretches between two stoppages are the ball being moved
CENTRE_M = 6.0      # m from the centre spot for a kick-off (ball tracking is noisy)


def find_restart(F, start, earliest=None, centre=False):
    bx, by, bv = F["bx"], F["by"], F["bv"]
    lo = start if earliest is None else max(start, earliest)
    hi = min(len(bx), start + int(MAX_DEAD_S * FPS))
    with np.errstate(invalid="ignore"):
        ok = inside(bx[lo:hi], by[lo:hi], INSIDE_M) & (np.nan_to_num(bv[lo:hi]) > KICK_V)
        if centre:     # a kick-off is taken from the centre spot
            ok_c = ok & (np.hypot(bx[lo:hi] - L / 2, by[lo:hi] - W / 2) < CENTRE_M)
            idx = np.flatnonzero(ok_c)
            if len(idx) and idx[0] < 30 * FPS:
                return lo + idx[0]
    idx = np.flatnonzero(ok)
    return lo + idx[0] if len(idx) else hi


def ball_out(F):
    with np.errstate(invalid="ignore"):
        return F["has"] & ~inside(F["bx"], F["by"], -OUT_M)


def onsets_from_ball(F):
    has, bx, by, m = F["has"], F["bx"], F["by"], F["motion"]
    with np.errstate(invalid="ignore"):
        at_line = has & ~inside(bx, by, LINE_M)
    out = ball_out(F)
    pre = pd.Series(m).rolling(2 * FPS, min_periods=1).mean().values
    post = pd.Series(m[::-1]).rolling(4 * FPS, min_periods=1).mean().values[::-1]
    post = np.r_[post[FPS:], np.full(FPS, np.nan)]
    on = set((np.flatnonzero(out[1:] & ~out[:-1]) + 1).tolist())
    for t in np.flatnonzero(has[:-1] & ~has[1:]):
        if at_line[t] and post[t] < max(SLOW, STOP_RATIO * pre[t]):
            if not has[t + 1:t + 1 + int(VANISH_S * FPS)].any():
                on.add(int(t + 1))
    return np.array(sorted(on), dtype=int)


def kickoffs(F, tags):
    return sorted(find_restart(F, t["s"] - 2 * FPS, centre=True) for t in tags
                  if t["code"] == "SET PIECES" and t["type"] == "kick off (start)")


def halftime(F, players, k1, k2):
    tracked = (~np.isnan(players[:, :, 0])).sum(1)
    few = pd.Series(tracked < 12).rolling(5 * FPS, center=True, min_periods=1).mean().values > 0.9
    few[:k1] = False; few[k2:] = False
    s = np.flatnonzero(few & ~np.r_[False, few[:-1]])
    e = np.flatnonzero(few & ~np.r_[few[1:], False]) + 1
    if not len(s):
        return (k1 + k2) // 2
    i = int(np.argmax(e - s))
    return int(s[i])


def goal_onset(F, onsets, s):
    """Kick-off after a goal at s: the ball has been dead since the goal."""
    win = onsets[(onsets >= s - 150 * FPS) & (onsets <= s - 20 * FPS)]
    if len(win):
        return int(win[-1])
    lo = s - 150 * FPS
    miss = ~F["has"][lo:s]
    gs = np.flatnonzero(miss & ~np.r_[False, miss[:-1]])
    ge = np.flatnonzero(miss & ~np.r_[miss[1:], False]) + 1
    long_gaps = [a for a, b in zip(gs, ge) if b - a >= 5 * FPS and a <= 130 * FPS]
    return int(lo + long_gaps[-1]) if long_gaps else s - 60 * FPS


def rules(F, players, tags, use_tags=True):
    n = F["n"]
    onsets = onsets_from_ball(F)
    events = []                                   # (onset, earliest restart, team)
    if use_tags:
        for t in tags:
            if t["code"] != "SET PIECES":
                continue
            s = t["s"]
            if "after goal" in t["type"]:
                onset = goal_onset(F, onsets, s)
                events.append((onset, s - 2 * FPS, -2))       # -2: kick-off, team unknown
                continue
            elif "kick off" in t["type"]:
                onset = s - FPS
            else:
                prev = onsets[(onsets <= s) & (onsets >= s - 45 * FPS)]
                onset = int(prev[-1]) if len(prev) else s - FPS
            events.append((onset, s, t["team"]))
    used = {e[0] for e in events}
    events += [(int(o), None, -1) for o in onsets if o not in used]
    events.sort(key=lambda e: e[0])

    rule_dead = np.zeros(n, bool)
    restarts = {}
    goals = []                                    # (onset guess, kick-off) per goal
    i = 0
    while i < len(events):
        onset, earliest, tm = events[i]
        rs = find_restart(F, onset + int(MIN_DEAD_S * FPS), earliest, centre=(tm == -2))
        j = i + 1
        while j < len(events) and events[j][0] < rs:
            if events[j][1] is not None:
                earliest = events[j][1] if earliest is None else max(earliest, events[j][1])
                tm = events[j][2] if events[j][2] >= 0 else tm
                rs = find_restart(F, onset + int(MIN_DEAD_S * FPS), earliest, centre=(tm == -2))
            j += 1
        rule_dead[onset:rs] = True
        restarts[int(rs)] = max(tm, -1)
        if tm == -2:
            goals.append((onset, int(rs)))
        i = j

    ks = kickoffs(F, tags) if use_tags else []
    if len(ks) >= 2:
        k1, k2 = ks[0], ks[1]
    else:                                        # no tags: kick-offs from the tracking alone
        k1, k2 = None, None
    hard = ball_out(F).copy()
    if k1 is not None:
        ht = halftime(F, players, k1, k2)
        hard[:k1] = True
        hard[ht:k2] = True
        restarts[k1] = -1; restarts[k2] = -1
        halves = [(k1, ht), (k2, n)]
    else:
        halves = [(0, n // 2), (n // 2, n)]
    rule_dead |= hard

    live = ~rule_dead
    ls = np.flatnonzero(live & ~np.r_[False, live[:-1]])
    le = np.flatnonzero(live & ~np.r_[live[1:], False]) + 1
    for a, b in zip(ls, le):
        if 0 < a and b < n and b - a < MIN_LIVE_S * FPS:
            rule_dead[a:b] = True
            restarts[int(b)] = restarts.pop(int(a), -1)
    return dict(rule_dead=rule_dead, hard=hard, restarts=restarts, halves=halves, goals=goals)


def fix_goals(R, p_dead, tags):
    """After a goal the ball is dead until the kick-off. The goal itself is the end of the last
    sustained stretch of play (>= 10 s judged live by the dead-ball model) after the GOALS tag
    starts. Everything from there to the kick-off becomes hard dead."""
    goal_tags = sorted(t["s"] for t in tags if t["code"] == "GOALS")
    live = np.nan_to_num(p_dead, nan=1.0) < 0.5
    for onset, ko in R["goals"]:
        starts = [g for g in goal_tags if ko - 300 * FPS <= g < ko]
        lo = starts[-1] if starts else ko - 150 * FPS
        seg = live[lo:ko - 15 * FPS]
        s_ = np.flatnonzero(seg & ~np.r_[False, seg[:-1]])
        e_ = np.flatnonzero(seg & ~np.r_[seg[1:], False]) + 1
        runs = [(a, b) for a, b in zip(s_, e_) if b - a >= 10 * FPS]
        g = lo + runs[-1][1] if runs else onset
        R["hard"][g:ko] = True
        R["rule_dead"][g:ko] = True
    return R
