"""Per-frame features. Everything here is a deterministic function of the tracking."""
import numpy as np
import pandas as pd

from .io import FPS, L, W


def roll(x, w, center=True):
    return pd.Series(x).rolling(int(w), center=center, min_periods=1).mean().values


def inside(bx, by, m):
    return (bx > m) & (bx < L - m) & (by > m) & (by < W - m)


def basic(players, ball, team):
    """Ball, nearest players, collective motion."""
    n = len(ball)
    bx, by = ball[:, 0].astype(float) * L, ball[:, 1].astype(float) * W
    px, py = players[:, :, 0].astype(float) * L, players[:, :, 1].astype(float) * W
    has = ~np.isnan(bx)

    bv = np.full(n, np.nan)
    bv[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * FPS / 2

    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    d1 = d[np.arange(n), j]
    if team.ndim == 1:                                      # one team per slot (v2)
        k1 = team[j]
        d_opp = np.where(team[None, :] != k1[:, None], d, np.inf).min(1)
        d_home = d[:, team == 0].min(1)
        d_away = d[:, team == 1].min(1)
    else:                                                   # per-frame teams from teamfix
        d = np.where(team >= 0, d, np.inf)                  # officials (-1) never control the ball
        j = np.argmin(d, 1)
        d1 = d[np.arange(n), j]
        k1 = team[np.arange(n), j]
        d_opp = np.where((team != k1[:, None]) & (team >= 0), d, np.inf).min(1)
        d_home = np.where(team == 0, d, np.inf).min(1)
        d_away = np.where(team == 1, d, np.inf).min(1)

    sp = players[:, :, 2].astype(float)
    vx = np.full_like(px, np.nan); vy = np.full_like(py, np.nan)
    vx[1:-1] = (px[2:] - px[:-2]) * FPS / 2
    vy[1:-1] = (py[2:] - py[:-2]) * FPS / 2
    sp = np.where(np.isnan(sp), np.hypot(vx, vy), sp)
    sp = np.where(sp > 12, np.nan, sp)                      # identity switches teleport players
    med = np.nanmedian(sp, 1)
    motion = roll(med, FPS)

    return dict(n=n, bx=bx, by=by, has=has, bv=bv, d1=d1, k1=k1, d_opp=d_opp,
                d_home=d_home, d_away=d_away, motion=motion, sp=sp,
                px=px, py=py, vx=vx, vy=vy, team=team)


def attack_sign(F, halves):
    """+1 where the home team attacks towards x = L, -1 where it attacks towards x = 0."""
    sign = np.zeros(F["n"])
    for a, b in halves:
        if F["team"].ndim == 1:
            hx = np.nanmean(F["px"][a:b][:, F["team"] == 0])
            ax = np.nanmean(F["px"][a:b][:, F["team"] == 1])
        else:
            hx = np.nanmean(np.where(F["team"][a:b] == 0, F["px"][a:b], np.nan))
            ax = np.nanmean(np.where(F["team"][a:b] == 1, F["px"][a:b], np.nan))
        sign[a:b] = 1.0 if hx < ax else -1.0                  # home defends the side it stands on
    return sign


def _team_stats(x, y, vx, sp, mask_team):
    if mask_team.ndim == 1:                                 # one team per slot (v2)
        xs, ys, vs, ss = x[:, mask_team], y[:, mask_team], vx[:, mask_team], sp[:, mask_team]
    else:                                                   # (N, 46) mask from teamfix
        xs, ys = np.where(mask_team, x, np.nan), np.where(mask_team, y, np.nan)
        vs, ss = np.where(mask_team, vx, np.nan), np.where(mask_team, sp, np.nan)
    out = {
        "cx": np.nanmean(xs, 1), "cy": np.nanmean(ys, 1),
        "sx": np.nanstd(xs, 1), "sy": np.nanstd(ys, 1),
        "xmax": np.nanmax(xs, 1), "xmin": np.nanmin(xs, 1),
        "vx": np.nanmean(vs, 1),
        "sp": np.nanmean(ss, 1),
        "sp90": np.nanpercentile(ss, 90, axis=1),
        "nfin": np.sum(xs > 2 * L / 3, 1).astype(float),
        "nown": np.sum(xs < L / 3, 1).astype(float),
        "cnt": np.sum(~np.isnan(xs), 1).astype(float),
    }
    return out


def geometry(F, sign):
    """Team-shape features in an 'attack to the right' frame for each team, home and away.

    Returned twice -- as (own, other) for home and for away -- so one model scores both teams.
    """
    px, py, vx, sp, team = F["px"], F["py"], F["vx"], F["sp"], F["team"]
    s = sign[:, None]
    # home-oriented coordinates: home attacks +x
    hx = np.where(s > 0, px, L - px)
    hvx = vx * s
    # away-oriented: away attacks +x
    ax = L - hx
    avx = -hvx

    H_h = _team_stats(hx, py, hvx, sp, team == 0)       # home seen from home
    A_h = _team_stats(hx, py, hvx, sp, team == 1)       # away seen from home
    H_a = _team_stats(ax, py, avx, sp, team == 0)       # home seen from away
    A_a = _team_stats(ax, py, avx, sp, team == 1)       # away seen from away

    # pressure: mean distance from each player to the nearest opponent (chunked: N x 46 x 46)
    n = len(px)
    nn = np.full((n, px.shape[1]), np.nan, np.float32)
    if team.ndim == 1:
        opp = team[:, None] != team[None, :]
    for a in range(0, n, 10000):
        x, y = px[a:a + 10000].astype(np.float32), py[a:a + 10000].astype(np.float32)
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        if team.ndim == 1:
            d[:, ~opp] = np.nan
        else:                                               # same team or an official: not an opponent
            tc = team[a:a + 10000]
            d[(tc[:, :, None] == tc[:, None, :]) | (tc[:, None, :] < 0)] = np.nan
        nn[a:a + 10000] = np.nanmin(d, 2)
    if team.ndim == 1:
        pres_h = np.nanmean(nn[:, team == 0], 1)
        pres_a = np.nanmean(nn[:, team == 1], 1)
    else:
        pres_h = np.nanmean(np.where(team == 0, nn, np.nan), 1)
        pres_a = np.nanmean(np.where(team == 1, nn, np.nan), 1)

    def pack(own, oth, p_own, p_oth):
        cols = {}
        for k in own:
            cols["own_" + k] = own[k]
            cols["oth_" + k] = oth[k]
        cols["d_cx"] = own["cx"] - oth["cx"]
        cols["d_vx"] = own["vx"] - oth["vx"]
        cols["d_sp"] = own["sp"] - oth["sp"]
        cols["own_len"] = own["xmax"] - own["xmin"]
        cols["oth_len"] = oth["xmax"] - oth["xmin"]
        cols["own_area"] = own["sx"] * own["sy"]
        cols["oth_area"] = oth["sx"] * oth["sy"]
        cols["own_free"] = p_own                 # how unmarked my players are
        cols["oth_free"] = p_oth
        df = pd.DataFrame(cols)
        # temporal context: 1 s and 4 s means, 2 s trend of the key quantities
        for k in ["d_cx", "d_vx", "own_vx", "oth_vx", "own_free", "oth_free", "own_area",
                  "oth_area", "own_cx", "own_len", "oth_len"]:
            v = df[k].values
            df[k + "_m1"] = roll(v, FPS)
            df[k + "_m4"] = roll(v, 4 * FPS)
            df[k + "_tr"] = roll(np.r_[np.full(FPS, np.nan), v[2 * FPS:] - v[:-2 * FPS], np.full(FPS, np.nan)], FPS)
        return df

    home_view = pack(H_h, A_h, pres_h, pres_a)
    away_view = pack(A_a, H_a, pres_a, pres_h)
    return home_view, away_view


def dead_features(F):
    """Features for 'is the ball in play': collective motion, its change, ball visibility."""
    sp, n = F["sp"], F["n"]
    med = np.nanmedian(sp, 1)
    mean = np.nanmean(sp, 1)
    p90 = np.nanpercentile(sp, 90, axis=1)
    has = F["has"].astype(float)
    bx, by = F["bx"], F["by"]
    with np.errstate(invalid="ignore"):
        line = np.minimum(np.minimum(bx, L - bx), np.minimum(by, W - by))   # <0 = outside
    bv = np.nan_to_num(F["bv"], nan=0.0)
    # time since the ball was last seen / until it is next seen (s), capped
    last = np.where(F["has"], np.arange(n), -1)
    last = np.maximum.accumulate(last)
    since = np.where(last >= 0, np.arange(n) - last, 10 ** 6) / FPS
    nxt = np.where(F["has"], np.arange(n), 10 ** 9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    until = np.where(nxt < 10 ** 9, nxt - np.arange(n), 10 ** 6) / FPS
    tracked = np.sum(~np.isnan(F["px"]), 1).astype(float)

    cols = {"since": np.minimum(since, 60), "until": np.minimum(until, 60),
            "tracked": tracked, "d1": np.minimum(np.nan_to_num(F["d1"], posinf=60), 60)}
    for name, v in [("med", med), ("mean", mean), ("p90", p90), ("has", has),
                    ("bv", bv), ("line", np.nan_to_num(line, nan=0.0))]:
        for w in (1, 3, 8):
            cols[f"{name}_m{w}"] = roll(v, w * FPS)
        cols[f"{name}_past3"] = roll(v, 3 * FPS, center=False)
        cols[f"{name}_next3"] = roll(v[::-1], 3 * FPS, center=False)[::-1]
    cols["med_drop"] = cols["med_past3"] - cols["med_next3"]
    return pd.DataFrame(cols)