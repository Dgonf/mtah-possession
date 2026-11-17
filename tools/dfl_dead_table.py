"""Build possession/dfl_dead_train.npz, the training table of the supervised dead-ball model, from
the public DFL / IDSSE matches (Bassek et al. 2025, Scientific Data 12:195, CC-BY 4.0, data owner DFL).

    python tools/dfl_convert.py --dfl external/dfl --out external/dfl_npz     # once
    python tools/dfl_dead_table.py --dfl external/dfl_npz

Labels: the BallStatus field of the DFL position feed (0 = dead, 1 = alive), recorded by the DFL's
official data collection. Seven matches, every 25th frame inside the two halves. Deterministic.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from possession import deadsup, features          # noqa: E402

STRIDE = 25

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default="external/dfl_npz")
    a = ap.parse_args()
    Xs, ys, ms, names = [], [], [], []
    for i, f in enumerate(sorted(glob.glob(os.path.join(a.dfl, "*.npz")))):
        d = np.load(f)
        g = d["graded"]
        F = features.basic(d["players"], d["ball"], d["team"])
        X, cols = deadsup.player_features(F)
        q = deadsup.quantise(X, g)[g][::STRIDE]
        y = (d["labels"][g][::STRIDE] == "D").astype(np.uint8)
        Xs.append(q); ys.append(y); ms.append(np.full(len(y), i, np.uint8))
        names.append(os.path.splitext(os.path.basename(f))[0])
        print(names[-1], q.shape, "dead share %.3f" % y.mean())
    np.savez_compressed(deadsup.TABLE, X=np.vstack(Xs), y=np.concatenate(ys), match=np.concatenate(ms),
                        cols=np.array(cols), names=np.array(names))
    print("wrote", deadsup.TABLE, np.vstack(Xs).shape)
