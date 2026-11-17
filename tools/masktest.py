"""Label-free test of how possession is filled in while the ball is not tracked.

Take 11 s windows of live play where the ball IS tracked, hide the ball there (drop the control
evidence, switch to the 'ball missing' weights), decode again, and compare with the unmasked
decode. The unmasked decode is the reference; no ground truth is involved.

    python tools/masktest.py --data data
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from possession import models, run                       # noqa: E402
from possession.decode import viterbi3                    # noqa: E402

WIN, HIDE_FROM, HIDE_TO = 375, 50, 325                    # 15 s windows, hide 2..13 s


def masked_decode(S, mask, w_miss):
    F = S["F"]
    em = run.emissions(S)
    ch, ca = models.control(F)
    em[mask, 0] -= run.W_CTRL * ch[mask]
    em[mask, 1] -= run.W_CTRL * ca[mask]
    g = run.logit(S["p_home"], run.GEO_CLIP)
    dw = w_miss - run.W_GEO_VIS
    em[mask, 0] += dw * g[mask] / 2
    em[mask, 1] -= dw * g[mask] / 2
    return np.array(["H", "A", "D"])[viterbi3(em, run.SWITCH_COST, run.ENTER_DEAD, run.LEAVE_DEAD)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    S = run.build(a.data)
    F, R, n = S["F"], S["R"], S["n"]
    ref = run.decode(S)
    inh = models.in_halves(n, R["halves"])
    starts = [t for t in range(0, n - WIN, WIN)
              if inh[t:t + WIN].all() and F["has"][t:t + WIN].mean() > 0.95 and (ref[t:t + WIN] != "D").all()]
    mask = np.zeros(n, bool)
    for t in starts:
        mask[t + HIDE_FROM:t + HIDE_TO] = True
    print(f"{len(starts)} windows, {mask.sum()} hidden frames")
    for w in (0.0, 0.05, run.W_GEO_MISS, 0.2, 0.4):
        p = masked_decode(S, mask, w)
        print(f"W_GEO_MISS {w:4.2f}: agreement with unmasked decode {(p[mask] == ref[mask]).mean():.3f}")


if __name__ == "__main__":
    main()
