# Frame-level ball possession from uncorrected automatic tracking

**More than a Hack 2027 · Phase 1 (Ball Possession) · Short technical document**

| | |
|---|---|
| Repository | https://github.com/Dgonf/mtah-possession (public) |
| Command that regenerates the submission | `python -m possession.run --data data --out submission.csv` |
| Runtime | about 2–3 minutes on a laptop CPU, no GPU |
| Full documentation | `TECHNICAL.md` in the repository |

## 1. Summary

We label every frame of the tactical video as `H`, `A` or `D` from the provided automatic tracking (ATD) and Smart Tagging. The ball is missing on 41 % of graded frames, often for long stretches (median gap 10.7 s, 90th percentile 34 s), so a method that relies on the ball has to guess for long periods. Our system therefore draws most of its evidence from the players, who are always tracked, and combines it into whole possessions with a three-state Viterbi decoder.

No human label of the provided match is used anywhere, for training or for validation. The public 5-minute window was used only to check alignment and file format; no version or parameter was chosen on it.

## 2. Approach

Per frame, four sources of evidence:

1. **Ball control.** A team controls the ball when one of its players is within 1.5 m of it, at least 0.75 m closer than any opponent, and the ball moves below 8 m/s. Contested and travelling balls give no evidence. When one team has 12+ tracked players and the other 10 or fewer (a player carried in the wrong team's slot), that team's control counts 0.3 instead of 1.
2. **Rule layer for stoppages.** Ball tracked beyond a line; ball vanishing at a line while the players slow down; Smart Tagging set pieces, goals and kick-offs; half-time. A restart is the first kick of the ball inside the pitch.
3. **Dead-ball classifier trained on other games.** It reads whether the ball is in play from how the players move and stand (speed statistics, share of slow and fast players, spread, players in the boxes and near the lines), with no ball input and no team labels, so it transfers to noisy tracking. Each feature is converted to its quantile within the match. It is trained on seven public Bundesliga matches (§4). Where the ball is tracked inside the pitch and moving or controlled, this is counted as evidence against `D`.
4. **Possession from team shape,** a self-supervised classifier trained only on frames where our own control rule is certain, used mainly while the ball is not tracked.

**Decoding.** A three-state Viterbi pass (`H`, `A`, `D`) turns this evidence into one consistent sequence. Changing possession costs 8 (about a third of a second of clear control); entering or leaving a stoppage costs 15.

**Possession rules implemented.** A team gains possession when it out-controls the other for about a third of a second. Single touches, deflections, aerial duels without control and 50/50s do not change possession. A loose ball stays with the last team. A dead ball always separates possessions, and ownership of a dead ball is never predicted. No `X` is written.

## 3. Assets

| asset | use |
|---|---|
| Tactical ATD (players and ball) | all features |
| Smart Tagging | set-piece timing and taker, goals, kick-offs, a very weak phase prior |
| Graded half boundaries (DATA.md) | reference frames for the per-match quantiles; a safety check on detected kick-offs |
| Tactical video and homography | tried (jersey-colour team correction), dropped: it changed the share of 12-vs-10 frames only from 13.6 % to 13.5 % |
| Broadcast video and ATD | not used |
| Public DFL data (other games) | training and validation, §4 |

## 4. Training data (other games only)

| | |
|---|---|
| Dataset | Bassek, Rein, Weber, Memmert (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12:195. Data owner: DFL. Licence CC-BY 4.0. https://doi.org/10.6084/m9.figshare.28196177 |
| Content | 7 full matches, Bundesliga and 2. Bundesliga 2022/23, positions of all players and the ball at 25 Hz |
| Labels used | `BallStatus` (in play or not) from the official position feed, for training; `BallPossession` for validation only |
| Labelled by | the DFL's official data provider. We labelled nothing by hand, on any game |
| Amount | training: 40,109 frames (every 25th in-half frame); validation: all 1,002,644 frames |
| In the repository | the derived training table only (`possession/dfl_dead_train.npz`); scripts in `tools/` rebuild it from the public download |

## 5. Validation

The public window holds 17 possessions and cannot rank versions, so we validated on the seven DFL matches with real labels:
- the ball hidden in long blocks on 41 % of frames, as in the provided match;
- the dead-ball model scored leave-one-match-out (the model that labels a match never saw it);
- the organisers' `possession_metrics.py`;
- DFL possession spells under 3 s merged, which gives a mean spell of 13.8 s (public ground truth: 14.2 s).

A second, harder condition hides the ball preferentially around duels and adds synthetic wrong-slot players on 14 % of frames.

| system | spellF1, standard | spellF1, hard |
|---|---|---|
| player-shape and rules only, no external data | 0.446 | |
| + dead-ball classifier trained on other games | 0.621 | 0.589 |
| **+ live-ball evidence and control down-weighting (submitted)** | **0.627** | **0.601** |

The dead-ball classifier raised spellF1 on every one of the seven matches; the two decoder terms added a small but consistent gain. A grid search over 81 decoder-weight combinations changed spellF1 by 0.001, so the weights were not tuned further. These numbers are not a forecast for the provided match: DFL tracking is much cleaner.

On the provided match, without labels, the dead-ball classifier behaves as expected: it calls 46 % of graded frames dead (41–52 % in the DFL matches), separates the rule layer's certain stoppages from frames with a controlled ball with AUC 0.93, and calls the ball dead before 41 of 53 tagged set pieces (all 12 corners, all 5 free kicks). The submitted file reports 235 possessions of mean 14.2 s (organisers' `validate_submission.py`).

## 6. Limitations

1. **Possession while the ball is untracked** is the main remaining error. On the DFL matches, a perfect shape model would lift spellF1 from 0.62 to 0.77. Detecting the ball in the tactical video where the ATD lost it is the next step.
2. **Team-slot errors** in the tracking (about 14 % of frames with 20+ tracked players show 12 vs 10) are only down-weighted, not corrected.
3. **Restart ownership** relies on the rule layer's restart, which needs a tracked ball.
4. **Transfer** of the dead-ball classifier to this match is checked only indirectly (§5).

Also tried and dropped: jersey-colour team correction, long-context (8–20 s) shape features, a shape model trained on the DFL matches, a second decoding pass for restarts, and a look-ahead shape model. None improved spellF1 on the DFL matches.

## 7. Reproduction and access

```bash
git clone https://github.com/Dgonf/mtah-possession.git
cd mtah-possession
pip install -r requirements.txt        # Python 3.11+; numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1
# place the organisers' bundle in data/ (data/tactical/BAR-ATH_tactical_atd.txt, ..._atd.xml, ..._smart_tagging.xml)
python -m possession.run --data data --out submission.csv
```

The output is deterministic and byte-identical across runs. The run needs only the provided data and the repository; the DFL-derived training table is included. The match data is not in the repository. Scripts to rebuild the training table and repeat the validation from the public DFL download are in `tools/` (see `TECHNICAL.md`, §10).

**Compliance.** No hand labels on the provided match; no human-produced data about it; the public window was used for format checks only. Parts of the code were written with an LLM assistant; the match data never left our machine, and the assistant worked from code, documentation and aggregate statistics.
