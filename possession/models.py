"""Self-supervised models. No human labels: every training label below is produced by the rules.

possession_from_shape
    Where the ball is visible and one team clearly controls it, the rules know who has the ball.
    A classifier learns to read that from the shape of the two teams alone (no ball), and is then
    used where the ball is not tracked -- 22 % of live play, often exactly when possession changes.

dead_from_motion
    Where the rules are sure the ball is out of play (ball out, set piece) or sure it is in play
    (controlled, moving), a classifier learns what a stoppage looks like in the players' movement,
    and finds the stoppages the rules miss (fouls, injuries, celebrations).

Both are trained and predicted out-of-fold over contiguous time blocks, so no frame is scored by
a model that saw it.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .io import FPS

R_CTRL = 1.5        # m, ball within this of a player = possible control
MARGIN = 0.75       # m, nearest opponent at least this much farther
V_CTRL = 8.0        # m/s, faster than this the ball is travelling, not controlled
N_FOLDS = 5
STRIDE = 5          # train on every 5th frame: neighbouring frames are near duplicates


def control(F):
    with np.errstate(invalid="ignore"):
        c = (F["d1"] < R_CTRL) & (F["d_opp"] - F["d1"] > MARGIN) & (np.nan_to_num(F["bv"], nan=99) < V_CTRL)
    return c & (F["k1"] == 0), c & (F["k1"] == 1)


def _near(mask, w):
    c = np.cumsum(np.r_[0, mask.astype(int)])
    n = len(mask)
    i = np.arange(n)
    return (c[np.minimum(n, i + w + 1)] - c[np.maximum(0, i - w)]) > 0


def _clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def _oof(X, y, w_rows, n, predict_rows=None):
    """Out-of-fold probabilities for every frame. Folds are contiguous blocks of the match."""
    folds = np.minimum((np.arange(n) * N_FOLDS) // n, N_FOLDS - 1)
    p = np.full(n, np.nan)
    for f in range(N_FOLDS):
        tr = w_rows & (folds != f)
        m = _clf().fit(X[tr], y[tr])
        te = folds == f if predict_rows is None else (folds == f) & predict_rows
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def in_halves(n, halves):
    m = np.zeros(n, bool)
    for a, b in halves:
        m[a:b] = True
    return m


def possession_from_shape(F, home_view, away_view, rule_dead, halves):
    n = F["n"]
    ch, ca = control(F)
    near_h, near_a = _near(ch, 12), _near(ca, 12)
    far_h, far_a = _near(ch, 25), _near(ca, 25)
    lab = np.full(n, -1)
    lab[near_h & ~far_a] = 1           # home view label: home has the ball
    lab[near_a & ~far_h] = 0
    ok = (lab >= 0) & F["has"] & ~rule_dead & in_halves(n, halves) & (np.arange(n) % STRIDE == 0)

    Xh, Xa = home_view.values.astype(np.float32), away_view.values.astype(np.float32)
    # one symmetric model: home view labelled "home has it", away view labelled "away has it"
    X = np.vstack([Xh, Xa])
    y = np.r_[lab, np.where(lab >= 0, 1 - lab, -1)]
    rows = np.r_[ok, ok]
    folds = np.minimum((np.arange(n) * N_FOLDS) // n, N_FOLDS - 1)
    folds2 = np.r_[folds, folds]
    p_home = np.full(n, np.nan); p_away = np.full(n, np.nan)
    for f in range(N_FOLDS):
        tr = rows & (folds2 != f)
        m = _clf().fit(X[tr], y[tr])
        te = folds == f
        p_home[te] = m.predict_proba(Xh[te])[:, 1]
        p_away[te] = m.predict_proba(Xa[te])[:, 1]
    p = 0.5 * (p_home + (1 - p_away))      # P(home in possession)
    return p, lab


def dead_from_motion(F, Xd, rule_dead, hard, halves):
    n = F["n"]
    ch, ca = control(F)
    ctrl_near = _near(ch | ca, FPS)
    dead_core = rule_dead & ~_near(~rule_dead, FPS)          # 1 s inside a rule stoppage
    live_core = ~_near(rule_dead, 3 * FPS)                   # 3 s away from any rule stoppage
    pos = dead_core
    neg = live_core & ctrl_near & F["has"]
    # live play with the ball briefly untracked: a gap bracketed by control on both sides
    gap_live = live_core & ~F["has"] & _near(ch | ca, 5 * FPS)
    y = np.full(n, -1)
    y[neg | gap_live] = 0
    y[pos] = 1
    rows = (y >= 0) & ~hard & in_halves(n, halves) & (np.arange(n) % STRIDE == 0)
    X = Xd.values.astype(np.float32)
    return _oof(X, y, rows, n), y
