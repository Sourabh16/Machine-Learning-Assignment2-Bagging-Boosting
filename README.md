# Bagging, Boosting and Bias–Variance on Olivetti Faces

Machine Learning, Assignment 2 — Sourabh Singh (M25CPS009)

Everything in the report is produced by the code in this repository. Bagging and
multiclass SAMME AdaBoost are implemented from scratch in `src/ensembles.py`;
only `sklearn.tree.DecisionTreeClassifier` is used as the base learner.

## Reproduce

```bash
conda env create -f environment.yml && conda activate ml-a2   # or: pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` runs, in order:

| command | what it produces | time (2 cores) |
|---|---|---|
| `python scripts/q1.py` | tree grid, bagging for B ∈ {1,…,200}, OOB / disagreement / margin, clean vs masked | ~2 min |
| `python scripts/q2.py` | SAMME toy round, depth × rounds grid, diagnostics, bagging vs boosting, masked analysis | ~4 min |
| `python scripts/q3.py` | 30 resampled training sets, bias² and variance, depth study | ~2 h |
| `python scripts/figures.py` | every figure in `figures/` from `results/*.json` | seconds |

Each script can also be run on its own (Q2 needs Q1's outputs, Q3 needs Q1 and Q2).
`q3.py` caches every refit in `results/q3_cache/`, so an interrupted run resumes
where it stopped; delete that folder to recompute from scratch.

The report is built with `cd report && pdflatex main.tex && pdflatex main.tex`.

## Layout

```
configs/default.json      seed, grids and sizes used by every script
splits/split_M25CPS009.json  the saved 6/2/2 split (image indices into fetch_olivetti_faces order)
src/common.py             data loading, split creation/loading, mask, logging
src/ensembles.py          Bagging and SAMME (from scratch), vote/margin/disagreement helpers
scripts/q1.py q2.py q3.py experiments
scripts/figures.py        figure script
results/*.json            all reported numbers (final logs of every experiment)
logs/*.log                console logs of the final run
figures/                  figures used in the report
report/                   CVPR-format LaTeX source and PDF
```

## Protocol details

* **Seed.** All numeric characters of the roll number in order: `M25CPS009 → 25009`.
  The split uses `numpy.random.default_rng(25009)` directly: for each identity the
  ten images are permuted and assigned 6 train / 2 validation / 2 test. Other
  random streams (tree tie-breaking, bootstraps, Q3 resampling) use
  `default_rng([25009, tag])` with a fixed tag per experiment.
* **Features.** Raw 4096 pixels in [0, 1]; no PCA or other reduction.
* **Masked test set.** Rows and columns 24–39 (inclusive) of each 64×64 test image set to 0.
  Models are trained on clean images only.
* **Selection.** Clean validation accuracy only. Ties: fewer leaves (tree), fewer
  trees (bagging), fewer rounds then shallower learners (boosting).
* **Bagging.** Bootstrap of size n = 240 per tree; hard majority vote; confidence =
  winning vote fraction. The (bootstrap, seed) pair is drawn per tree in turn, so
  the first B trees of a 200-tree fit are exactly the B-tree model.
* **SAMME.** α_t = log((1−ε_t)/ε_t) + log(K−1); misclassified weights multiplied by
  exp(α_t) then renormalised. A learner with ε_t ≥ 1 − 1/K is rejected and refitted
  with a new tie-breaking seed (stop after 5 consecutive rejections); ε_t = 0 is
  clipped to 1e−10 and boosting stops. Confidence q_k = S_k / Σα_t.
* **Q3.** Each of 30 training sets draws, within each identity, 6 of its 6 training
  faces with replacement. Class vectors: leaf class distribution (tree), vote
  fractions (bagging), normalised weighted vote q (boosting).

## Data note

`fetch_olivetti_faces()` downloads the data from figshare on first use. If that
host is unreachable, place a copy in scikit-learn's cache
(`~/scikit_learn_data/olivetti_py3.pkz`); the loader and the split are unchanged.
