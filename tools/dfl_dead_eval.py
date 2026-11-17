"""Validate the DFL-trained dead-ball model on the DFL matches themselves, leave-one-match-out:
the model that scores a match never saw it. Same protocol as tools/dfl_grid.py (ball hidden on
~41 % of frames in long blocks, organisers' scorer, DFL possessions under 3 s merged).

    python tools/dfl_dead_eval.py --dfl external/dfl_npz --metrics <challenge>/evaluation/possession_metrics.py
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dfl_grid as G                           # noqa: E402
from possession import run                     # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default="external/dfl_npz")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-v2", action="store_true", help="skip the v2 run (halves the time)")
    a = ap.parse_args()
    pm = G.load_pm(a.metrics)
    rows = []
    for i, f in enumerate(sorted(glob.glob(os.path.join(a.dfl, "*.npz")))):
        m = os.path.splitext(os.path.basename(f))[0]
        if a.only and m != a.only:
            continue
        d = np.load(f)
        g = d["graded"]
        ball = G.mask_ball(d["ball"], g, seed=0)
        gc = pm.label_codes(np.where(g, G.merge_short(d["labels"], 3.0), "X"))
        out = []
        for ref, ex in ((None, None), (g, i)):                  # v2, then DFL model without match i
            if ref is None and a.no_v2:
                out += [np.nan] * 4
                continue
            S = run.build_arrays(d["players"], ball, d["team"], G.kickoff_tags(d["kickoffs"]),
                                 log=lambda *_: None, ref=ref, exclude_match=ex)
            pred = run.decode(S, use_tag_prior=False)
            r = pm.evaluate(gc, pm.label_codes(np.where(g, pred, "X")))
            out += [r["spell_f1_mean"], r["alive_dead_f1"], r["macro_f1"], r["n_pred_spells"]]
        rows.append(out)
        print(f"{m}: v2 spellF1 {out[0]:.3f} aliveF1 {out[1]:.3f} | DFL-trained spellF1 {out[4]:.3f} "
              f"aliveF1 {out[5]:.3f} macroF1 {out[6]:.3f}  possessions {out[7]}/{r['n_gt_spells']}", flush=True)
    r = np.array(rows).mean(0)
    print(f"mean: v2 spellF1 {r[0]:.3f} aliveF1 {r[1]:.3f} macroF1 {r[2]:.3f} | "
          f"DFL-trained spellF1 {r[4]:.3f} aliveF1 {r[5]:.3f} macroF1 {r[6]:.3f}")
