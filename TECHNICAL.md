# Frame-level ball possession from uncorrected automatic tracking

*More than a Hack 2027 · Phase 1 · technical document (v4)*

## 1. Approach in one paragraph

We read the match as a sequence of three states (`H`, `A`, `D`) and decode it with a single 3-state Viterbi pass.
The football rules live in the transition costs. A team has to out-control the other for a moment before possession changes, and a stoppage lasts seconds, not frames.
Per-frame evidence comes from four sources:

1. **Ball control**, wherever the ball is tracked.
2. **A rule layer** for stoppages with an unambiguous cause: ball out, set piece, goal, kick-off, half-time.
3. **A dead-ball classifier trained on other games.** It reads "is the ball in play" from how the players move and stand, with no ball input. It is trained on the in-play status of seven public Bundesliga matches (§4).
4. **A self-supervised possession classifier** that reads possession from the *shape of the two teams*, trained only on labels our own rules produce on this match.

No human label from the provided match is used anywhere, for training or for validation.

Why this design: the ball is missing on 41 % of graded frames, and in long blocks (median gap 10.7 s, 90th percentile 34 s). A method that only looks at the ball has to guess for half a minute at a time. The players are always tracked, and they carry both signals we need: whether play has stopped, and who is attacking.

**What changed from v2, and why.** We validated v2 against real labels on other games (§6.1). Its weak point was the dead-ball decision, not the decoder weights: with a perfect dead-ball signal its spellF1 on those games rose from 0.45 to 0.70, while a grid search over 81 weight combinations moved it by 0.001. v3 therefore replaces v2's self-supervised stoppage model with the classifier of point 3. Everything else is v2.

**What changed from v3 (v4), and why.** Two decoder terms, each tested on the seven DFL matches before adoption (§6.5):
1. *Live-ball evidence against `D`.* The v3 dead-ball model reads only the players, so a slow build-up looks like a stoppage, and nothing in the decoder argued the other way while the ball was tracked in play. v4 subtracts 1 from the `D` score on every frame where the tracked ball is more than 1 m inside the pitch and moving above 2 m/s or controlled.
2. *Down-weighted control from a team with too many players.* When one team has 12+ tracked players and the other 10 or fewer, one slot carries a player of the other team, and a control frame from that team is suspect. Its control weight drops from 1 to 0.3.

## 2. Assets

| asset | used | how |
|---|---|---|
| tactical ATD (players + ball) | yes | everything below |
| Smart Tagging | yes | set-piece timing and taker team; goals and kick-offs; a very weak phase-team prior |
| graded half boundaries (DATA.md) | yes | the reference frames for the per-match feature quantiles (§3.4) |
| tactical video + homography | tried, dropped | jersey-colour team correction (`possession/teamfix.py`): it lowered the share of frames with a 12-vs-10 player count only from 13.6 % to 13.5 %, so it is off by default and not part of the submission |
| broadcast video, broadcast ATD | no | |
| external data | yes | seven public DFL matches, §4 |
| any human labels / event data about this match | **no** | not used anywhere |

The public 5-minute ground truth was used only to confirm alignment and file format, and is reported in §7 as a sanity check. No parameter was chosen on it.

## 3. Pipeline

### 3.1 Features (deterministic)
- Ball position in metres (105.3 × 68.0) and ball speed from a central difference.
- For every frame: the nearest player to the ball, their team, and the distance to the nearest opponent.
- Player speeds. Speeds above 12 m/s are discarded as identity switches.
- Team shape, computed for each team in its own attacking direction (detected per half from where each team stands): centroid, spread, length and area; mean forward velocity and speed; players in the final and own third; how unmarked each team's players are; 1 s and 4 s means of each, plus 2 s trends.
- Dead-ball features, **players only, no ball and no team labels** (80 columns): median, mean and 90th-percentile player speed; the share of players who are slow (< 1 m/s), walking (< 2 m/s) or fast (> 4 m/s); the spread of speeds; the spread and centroid of all players; the share of players in the penalty areas, near the touchlines and in the end zones. Each over 1, 3 and 8 s windows and the past and next 3 s (some also over 20 s and the past and next 8 s).

### 3.2 Ball control
A team has *control* in a frame when one of its players is within 1.5 m of the ball, at least 0.75 m closer than any opponent, and the ball is moving below 8 m/s.
Contested balls and travelling balls give no evidence at all.

### 3.3 Rule layer for stoppages

**Onset**, when one of these happens:
- the ball is tracked more than 0.5 m beyond a line,
- the ball vanishes within 2 m of a line for at least 1.5 s while the players slow down,
- a Smart Tagging set piece starts. The onset is the ball-out event in the 45 s before the tag, else 1 s before it.

**Restart**: the first frame after onset + 2 s where the ball is inside the pitch and moving above 4 m/s. For a tagged set piece the search starts at the tag. Kick-offs are kicks within 6 m of the centre spot.

**Goals**: the ball is dead from the goal to the kick-off. The goal moment is the end of the last ≥ 10 s stretch that the dead-ball model judges live, after the GOALS tag starts.

**Half-time**: the longest stretch between the two kick-offs where fewer than 12 players are tracked. Everything before the first kick-off and during half-time is hard `D`.

### 3.4 Models
Both are scikit-learn `HistGradientBoostingClassifier` with `random_state=0`.

- **Dead ball, trained on other games** (`possession/deadsup.py`). *Input*: the 80 player-only features of §3.1, each replaced by its quantile within the match (0–1000, among the frames of the two halves). *Label*: the in-play status of the DFL data (§4). *Training set*: every 25th in-half frame of the seven matches, 40,109 rows, shipped in the repository as `possession/dfl_dead_train.npz`; the model is fitted from it at run time (about 20 s).
  Three choices are there for transfer to this match's noisier tracking: no ball features (in the DFL data the ball is always tracked, here it is not, so they would not carry over); no team labels (team errors in the tracking do not matter); per-match quantiles (differences in how speed is estimated between tracking providers drop out).
- **Possession from shape, self-supervised on this match** (unchanged from v2). *Pseudo-label*: a frame where the ball is tracked, one team has control within ±0.5 s, and the other has none within ±1 s. *Input*: team shape only, no ball. One symmetric model serves both teams. Trained on every 5th frame and predicted out-of-fold over 5 contiguous time blocks.

### 3.5 Decoding
Per-frame scores, in units of "one frame of clear control":

| evidence | weight |
|---|---|
| control by team *k* | +1 to *k* (0.3 if *k* has 12+ tracked players and the other 10 or fewer; v4) |
| ball tracked > 1 m inside the pitch, moving > 2 m/s or controlled (v4) | −1 to `D` |
| shape model, ball not tracked | ±0.10 · clip(logit p, 3) / 2 |
| shape model, ball tracked | ±0.02 · … |
| dead-ball model | +1.0 · clip(logit p, 4) to `D` |
| rule stoppage | +3 to `D` |
| hard dead (ball out, pre-kick-off, half-time, post-goal) | forced `D` |
| restart taker's team, 2 s after a tagged restart | +0.2 |
| Smart Tagging phase team | +0.01 |

| transition | cost |
|---|---|
| `H` ↔ `A` | 8 |
| live → `D` | 15 |
| `D` → live | 15 |

**Possession rules this implements.** A team gains possession when it out-controls the other for about a third of a second.
The following do **not** change possession: single touches, deflections, won aerial duels without control, and 50/50s (which give no evidence).
A loose ball stays with the last team, because nothing pays for a switch. A dead ball always separates possessions. Dead-ball ownership is never predicted.
No `X` is written: on the ranking metric it would mean "repeat the previous label", which the decoder already does where it has no evidence.

## 4. Training data from other games

| | |
|---|---|
| dataset | "An integrated dataset of spatiotemporal and event data in elite soccer", Bassek, Rein, Weber, Memmert, *Scientific Data* 12:195 (2025). Data owner: Deutsche Fußball Liga (DFL). Licence CC-BY 4.0. Download: https://doi.org/10.6084/m9.figshare.28196177 |
| content | 7 full matches of the German Bundesliga and 2. Bundesliga, season 2022/23: positions of all players and the ball at 25 Hz |
| labels we use | two fields of the position feed, recorded by the DFL's official match-data collection: `BallStatus` (ball in play or not) and, for validation only, `BallPossession` (team) |
| who labelled it | the DFL's data provider (Sportec Solutions), not us. We labelled nothing by hand, on any game |
| how much | training: 40,109 frames (every 25th in-half frame of the 7 matches). Validation: all 1,002,644 frames |
| what is in the repository | the derived training table only (`possession/dfl_dead_train.npz`: quantised features, the label, the match index). `tools/dfl_convert.py` and `tools/dfl_dead_table.py` rebuild it from the public download |

None of it is about the provided match. The DFL event data (human-annotated passes, fouls, set pieces) is not used at all.

## 5. Handling missing and noisy tracking
- **Ball missing**: the shape model carries possession through gaps, and the dead-ball model needs no ball at all.
- **Identity switches**: positions are never linked over time per player; every feature is per frame or a team aggregate, and implausible speeds are discarded.
- **Team-label errors**: the dead-ball model ignores team labels. The shape model uses team aggregates, which dilute single wrong labels.
- **Ball-tracking noise at restarts**: restarts need a kick inside the pitch, and kick-offs a kick near the centre spot with a fallback.

## 6. Validation

The public window has 17 possessions and cannot choose between versions (the organisers' own baselines show it picks the wrong smoothing window). We validate on other games with real labels, and check transfer to this match without labels.

### 6.1 On other games, with real labels (`tools/dfl_grid.py`, `tools/dfl_dead_eval.py`)
Protocol, on the seven DFL matches:
- The ball is hidden in long blocks until 41 % of in-half frames have no ball (block lengths as in this match), so the conditions resemble the task.
- The pipeline gets no Smart Tagging there (only the two half starts), so the set-piece rules are not exercised.
- Scoring uses the organisers' `possession_metrics.py`, the two halves scored separately.
- DFL possession labels are finer than the challenge's (median spell 3.7 s). Spells under 3 s are merged into their neighbour, which gives a mean spell of 13.8 s (the public ground truth: 14.2 s).
- The dead-ball model is scored **leave-one-match-out**: the model that labels a match was trained on the other six.

| | spellF1 | alive/dead F1 | macroF1 |
|---|---|---|---|
| v2 (self-supervised stoppage model) | 0.446 | 0.741 | 0.672 |
| **v3 (dead-ball model from the other six matches)** | **0.620** | **0.939** | **0.861** |
| v3 with a perfect dead-ball signal (ceiling) | 0.700 | 1 | 0.915 |
| v3 with a perfect shape model (ceiling) | 0.770 | | 0.909 |

v3 per match: 0.60, 0.56, 0.67, 0.58, 0.66, 0.65, 0.63. Every match improves by far more than the scorer's noise band (0.04). The dead-ball classifier alone has a frame-level AUC of 0.981 on unseen matches.

These numbers are not a forecast for this match. DFL tracking is much cleaner, and the absolute level will be lower here. What they show is which part to fix and that the fix holds on unseen games.

**Decoder weights.** A grid of 81 combinations (switch cost 6/8/12, dead-ball costs 10/15/25, shape weight 0.05/0.10/0.20, stoppage weight 0.25/0.5/1), picked leave-one-match-out, improved v2 by 0.001 spellF1. The v2 weights are kept. For the new dead-ball model, weight 0.5 / 1 / 2 gives 0.589 / 0.620 / 0.619; we use 1.

### 6.2 Transfer to this match, without labels
The dead-ball model was never fitted to this match, so we check that it behaves sensibly here:

| check | result |
|---|---|
| share of graded frames it calls dead (p > 0.5) | 46.1 % (DFL matches: 41–52 % dead) |
| AUC against the rule layer's own certain cases (rule stoppages vs. frames with a controlled ball) | 0.930 |
| tagged set pieces where it says "dead" in the 3 s before the tag | 41 / 53 (corners 12/12, free kicks 5/5, throw-ins 18/26, goal kicks 6/10) |

### 6.3 Smart Tagging agreement and structure (`tools/selfcheck.py`)

| | baseline_smooth50 | v2 | **v3** |
|---|---|---|---|
| phase-team agreement | 0.675 | 0.840 | 0.834 |
| set piece preceded by `D` | 0.828 | 0.983 | 0.983 |
| first team after a set piece = taker | 0.630 | 0.870 | 0.731 |
| possessions (full match) | 594 | 251 | 240 |
| mean possession | 7.2 s | 14.5 s | 13.7 s |
| possessions under 1 s | 124 | 4 | 4 |
| dead-ball share of graded frames | 32 % | 42.5 % | 48.3 % |

Two caveats on this table. Smart Tagging is also an input (taker bonus, phase prior), so the agreement rows partly measure the method against itself. And "first team after a set piece" drops in v3 because the dead-ball model can end a stoppage before the rule layer sees the restart kick (the rule needs a tracked ball), and the taker bonus is tied to the rule's restart. We did not tune for this row.

### 6.4 Masked-ball test (`tools/masktest.py`)
We hide the ball for 11 s inside windows of tracked live play, decode again, and compare with the unmasked decode. The shape-model weight 0.10 was chosen on this test in v2 (agreement 0.928, against 0.840 for "hold the last team"). The test compares the method with itself, so it supports the choice of weight, not the accuracy.

### 6.5 v4 decoder terms on other games (`tools/dfl_ab.py`)
Same protocol as §6.1 (leave-one-match-out dead-ball model, DFL spells under 3 s merged, organisers' scorer), in two conditions:
- *clean*: the ball hidden at random in long blocks, as in §6.1;
- *hard*: the ball hidden preferentially around duels (75 % of blocks start near a frame where both teams are within 2 m of the ball), plus synthetic team-slot errors: a player moved to an empty slot of the other team for 5–60 s, until 14 % of frames show a wrong-slot player. This is closer to the provided match (FAQ: the ball is lost "disproportionately on the interesting frames"; §9.2).

| variant | spellF1 clean | Δ, wins/losses | spellF1 hard | Δ, wins/losses |
|---|---|---|---|---|
| v3 | 0.621 | | 0.589 | |
| live-ball term (1.0) | 0.627 | +0.006, 7/0 | 0.595 | +0.006, 5/2 |
| control down-weight (0.3) | 0.621 | 0, 0/0 | 0.595 | +0.006, 5/0 |
| **v4 = both** | **0.627** | **+0.006, 7/0** | **0.601** | **+0.012, 6/1** |

macroF1 rises by 0.004 (clean) and 0.005 (hard). The gains are small, under the scorer's per-match noise band, but they are consistent across matches and come with no loss elsewhere.

Tested and rejected:
- *Second decoding pass that gives the first frames after a restart to the team that controlled the ball just before it*: −0.005 clean (0/7 wins), −0.001 hard.
- *Shape model reading the team shape 1 or 2 s later (look-ahead)*: v3 −0.005 to −0.009 clean, −0.000 to +0.003 hard, inside the noise.
- The hard condition lowers v3 by 0.03, so §6.1's numbers are optimistic for this match.

## 7. Public window (sanity check only)

| | macroF1 | spellF1 |
|---|---|---|
| baseline_smooth50 | 0.621 | 0.326 |
| v2 | 0.766 | 0.529 |

Reported for transparency for the version it was first run on. With 17 possessions, one missed possession moves spellF1 by about six points; no version was chosen on these numbers.

## 8. Tried and dropped
- **Decoder weight tuning** on other games: no gain beyond noise (§6.1).
- **Ball features in the dead-ball model**: frame AUC 0.994 on DFL matches, but it relies on the ball always being tracked there. Dropped for transfer.
- **Possession-from-shape trained on other games**: +0.02 spellF1 on DFL matches (inside the noise band), and on this match it agrees with the control pseudo-labels slightly less than the in-match model (0.846 vs 0.856). Dropped.
- **Jersey-colour team correction** from the tactical video: see §2.

## 9. Limitations and next steps
1. **Possession while the ball is untracked is now the bottleneck.** On other games a perfect shape model would lift spellF1 from 0.62 to 0.77. Detecting the ball in the tactical video on the frames where the ATD lost it, or a sequence model over player trajectories, is the next step.
2. **Team-label errors in the automatic tracking.** On about 14 % of frames with 20+ tracked players one team has 12+ and the other 10 or fewer. Our colour-based correction did not fix it.
3. **Restart timing and ownership.** The taker bonus is tied to the rule layer's restart, which needs a tracked ball (§6.3).
4. **The dead-ball model was validated on cleaner tracking.** Its transfer here is checked only indirectly (§6.2).

## 10. Reproduction
```bash
pip install -r requirements.txt          # Python 3.11+
python -m possession.run --data data --out submission.csv     # about 2 min on 2 CPU cores
python tools/selfcheck.py --data data submission.csv
python -m possession.run --data data --out v2.csv --v2        # the previous version, no external data
```
`data/` is the organisers' bundle, unchanged. The output is byte-identical across runs.
The run needs nothing but the provided data and this repository; the DFL-derived training table is included.
To rebuild that table and repeat §6.1 from the public DFL download:
```bash
python tools/dfl_convert.py --dfl external/dfl --out external/dfl_npz
python tools/dfl_dead_table.py --dfl external/dfl_npz
python tools/dfl_dead_eval.py --dfl external/dfl_npz --metrics <challenge-repo>/evaluation/possession_metrics.py
OMP_NUM_THREADS=4 python tools/dfl_ab.py --dfl external/dfl_npz --metrics <challenge-repo>/evaluation/possession_metrics.py --shifts 0   # §6.5
```

**Tools.** Parts of the code were written with an LLM assistant. The match data stayed on our machine; the assistant worked from code, documentation and aggregate statistics printed by our scripts.
