"""3-state (H, A, D) Viterbi over per-frame scores, with the possession rules as transition costs.

  H <-> A   SWITCH_COST: a team must out-control the other for a moment to take the ball over,
            so single touches, deflections and 50/50s do not change possession, and a loose ball
            stays with the team that last had it.
  live <-> D  ENTER/LEAVE_DEAD: stoppages are seconds long, not frames.
A dead ball always separates two possessions, so the same team either side of it is two spells.
"""
import numpy as np

H, A, D = 0, 1, 2


def viterbi3(em, switch, enter_dead, leave_dead):
    n = len(em)
    T = np.array([[0.0, -switch, -enter_dead],
                  [-switch, 0.0, -enter_dead],
                  [-leave_dead, -leave_dead, 0.0]])
    back = np.zeros((n, 3), np.int8)
    s = em[0].copy()
    for t in range(1, n):
        cand = s[:, None] + T                  # from i to j
        b = np.argmax(cand, 0)
        s = cand[b, np.arange(3)] + em[t]
        back[t] = b
    path = np.zeros(n, np.int8)
    path[-1] = int(np.argmax(s))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
