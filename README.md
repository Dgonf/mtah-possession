# More than a Hack 2027 · Phase 1: Ball Possession

Labels every frame of the tactical video `H` / `A` / `D` from the provided tracking (ATD) and Smart Tagging.
It uses a 3-state Viterbi over ball control (with live-ball evidence against stoppages), a rule layer for stoppages, a self-supervised model for possession from team shape, and a dead-ball model trained on seven public DFL matches (other games).
No human label from the provided match is used anywhere. Read `TECHNICAL.md` for the method and its validation.

## Run

```bash
pip install -r requirements.txt                               # Python 3.11+
python -m possession.run --data data --out submission.csv     # ~2 min, byte-identical every run
python <challenge-repo>/evaluation/validate_submission.py submission.csv
```

`data/` is the organisers' bundle as shipped (`data/tactical/BAR-ATH_tactical_atd.txt`, `..._atd.xml`, `..._smart_tagging.xml`).
It is gitignored. Never commit it.

## Layout

```
possession/
  io.py         ATD + Smart Tagging readers
  features.py   ball / control / motion / team-shape / stoppage features
  deadball.py   rule layer: ball out, set pieces, goals, kick-offs, half-time
  models.py     self-supervised classifiers (out-of-fold, pseudo-labels from the rules)
  deadsup.py    dead-ball model trained on the public DFL matches (dfl_dead_train.npz)
  teamfix.py    jersey-colour team correction from the video (tried, off by default, needs opencv-python)
  decode.py     3-state Viterbi
  run.py        weights, emissions, end-to-end entry point
tools/
  selfcheck.py  label-free comparison of submissions (Smart Tagging agreement, structure)
  masktest.py   hides the ball where it is tracked, checks how gaps are filled
  dfl_ab.py     A/B test of decoder variants on the DFL matches (clean and hard conditions)
  dfl_*.py      convert the public DFL matches, build the training table, validate on them
```

## Ground rules for the team

- Do not tune on the public 5 minutes. Compare versions with `tools/selfcheck.py`, `tools/masktest.py`, and by looking at the video.
- Hand labels on this match are never allowed, in any amount.
- Keep the data out of the repo, and never upload it or extracts of it to a cloud LLM (Terms and Conditions).

## External data

`possession/dfl_dead_train.npz` is derived from: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12:195. Data owner: Deutsche Fußball Liga (DFL). Licence CC-BY 4.0. https://doi.org/10.6084/m9.figshare.28196177
