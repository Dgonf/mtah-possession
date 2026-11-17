"""Label-free checks for a submission -- what we compare versions on instead of the public window.

    python tools/selfcheck.py --data data sub_a.csv sub_b.csv ...

1. Smart Tagging agreement (an independent automatic source, not ground truth):
   - phase team: in attacking phases the tagged team should be the one in possession,
     in defending phases the other one
   - set pieces: a dead ball just before the tag, and the first live label after it = taker's team
2. Plausibility of the possession structure (counts, share of dead ball, micro-possessions).
3. The public 5-minute window, printed for completeness. 17 possessions: a sanity check, not a ranking.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from possession.io import load_tags

FPS = 25
GRADED = [(11185, 90651), (110808, 190197)]
ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
          "ATTACKING TRANSITION"}
DEFEND = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD"}


def graded_mask(n):
    m = np.zeros(n, bool)
    for a, b in GRADED:
        m[a:b + 1] = True
    return m


def segments(lab, mask):
    out = []
    for a, b in GRADED:
        q = lab[a:b + 1]
        ch = np.flatnonzero(q[1:] != q[:-1]) + 1
        st = np.r_[0, ch]; en = np.r_[ch, len(q)]
        out += [(a + s, a + e, q[s]) for s, e in zip(st, en)]
    return out


def check(lab, tags):
    n = len(lab); g = graded_mask(n); r = {}
    team = np.where(lab == "H", 0, np.where(lab == "A", 1, -1))
    # phase agreement
    agree = tot = 0
    for t in tags:
        if t["team"] < 0 or t["code"] not in ATTACK | DEFEND:
            continue
        want = t["team"] if t["code"] in ATTACK else 1 - t["team"]
        tt = team[t["s"]:t["e"]]; tt = tt[tt >= 0]
        agree += (tt == want).sum(); tot += len(tt)
    r["phase_agree"] = agree / max(tot, 1)
    # set pieces
    sp = [t for t in tags if t["code"] == "SET PIECES" and g[min(t["s"], n - 1)]]
    dead_near = np.mean([(lab[max(0, t["s"] - 10 * FPS):t["s"] + 5 * FPS] == "D").any() for t in sp])
    first_ok = []
    for t in sp:
        if t["team"] < 0:
            continue
        w = team[t["s"]:t["s"] + 40 * FPS]; w = w[w >= 0]
        if len(w):
            first_ok.append(w[0] == t["team"])
    r["setpiece_dead"] = dead_near
    r["setpiece_team"] = np.mean(first_ok)
    # structure
    seg = segments(lab, g)
    poss = [(e - s) / FPS for s, e, l in seg if l in "HA"]
    r["possessions"] = len(poss)
    r["mean_s"] = np.mean(poss)
    r["under_1s"] = int(np.sum(np.array(poss) < 1))
    r["dead_share"] = (lab[g] == "D").mean()
    r["dead_stops"] = sum(1 for s, e, l in seg if l == "D")
    r["home_share"] = (lab[g] == "H").sum() / max((lab[g] != "D").sum(), 1)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--challenge", default=None, help="path to the challenge repo, for the public scorer")
    ap.add_argument("subs", nargs="+")
    a = ap.parse_args()
    tags = load_tags(os.path.join(a.data, "tactical", "BAR-ATH_tactical_smart_tagging.xml"))
    rows = []
    for s in a.subs:
        lab = pd.read_csv(s).label.values.astype("<U1")
        r = check(lab, tags); r["sub"] = os.path.basename(s); rows.append(r)
    df = pd.DataFrame(rows).set_index("sub")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())
