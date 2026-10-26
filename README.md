# Machine Learning Assignment 2: Bagging, Boosting and Bias–Variance

**Sourabh Singh · Roll no. M25CPS009**

Face recognition on the Olivetti faces dataset using a single decision tree,
bagging and multiclass SAMME AdaBoost. Both ensembles are implemented from
scratch in [`src/ensembles.py`](src/ensembles.py); only scikit-learn's
`DecisionTreeClassifier` is used as the base learner. The report is in
[`report/main.pdf`](report/main.pdf).

## Results

Seed `25009` (digits of the roll number), 6/2/2 split per identity, raw 4096
pixels, no PCA. The masked test set zeroes rows and columns 24–39.

| Model | Clean test acc. | Masked test acc. | Bias² | Variance |
|---|---|---|---|---|
| Single tree (`max_depth=None`, `min_samples_leaf=4`) | 55.0% | 23.8% | 0.475 | 0.670 |
| Bagging, 200 trees | **81.3%** | **67.5%** | 0.558 | 0.020 |
| SAMME, depth-4 trees, 100 rounds | 46.3% | 35.0% | 0.885 | **0.018** |

Bias² and variance are measured over 30 training sets resampled within each
identity (Q3).

Main findings:

- **Q1:** Bagging fixes 23 of the tree's 36 clean-test errors. The remaining
  shared errors are systematic confusions between look-alike identities, which
  averaging cannot fix.
- **Q1:** The mean number of distinct images per bootstrap is 151.6, against a
  theoretical 151.9.
- **Q2:** SAMME's update always leaves the correctly classified images with
  only 1/K of the total weight. This is why boosting keeps returning to the
  same hard faces.
- **Q2:** Boosting's margins are small (mean 0.026). Masking pushes many faces
  into a single "sink" identity.
- **Q3:** Bagging removes 96–97% of the tree's variance. The largest accuracy
  gain from bagging is at depth 16, not for the deepest tree.
- **Q3:** The model with the lowest variance (boosting) is not the most
  accurate.

## Reproduce

```bash
git clone https://github.com/Sourabh16/Machine-Learning-Assignment2-Bagging-Boosting.git
cd Machine-Learning-Assignment2-Bagging-Boosting
conda env create -f environment.yml && conda activate ml-a2   # or: pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` runs, in order:

| Command | What it produces | Time (2 cores) |
|---|---|---|
| `python scripts/q1.py` | tree grid, bagging for B ∈ {1, 5, 10, 25, 50, 100, 200}, OOB / disagreement / margin, clean vs masked | ~2 min |
| `python scripts/q2.py` | SAMME toy round, depth × rounds grid, diagnostics, bagging vs boosting, masked analysis | ~4 min |
| `python scripts/q3.py` | 30 resampled training sets, bias² and variance, depth study | ~80 min |
| `python scripts/figures.py` | every figure in `figures/` from `results/*.json` | seconds |

The scripts can also be run one at a time: Q2 needs Q1's outputs, and Q3
needs both. `q3.py` caches every refit in `results/q3_cache/`, so an
interrupted run resumes where it stopped. Delete that folder to recompute from
scratch.

To rebuild the report:

```bash
cd report && pdflatex main.tex && pdflatex main.tex
```

## Repository layout

```
configs/default.json          seed, grids and sizes used by every script
splits/split_M25CPS009.json   saved train/val/test split (indices in fetch_olivetti_faces order)
src/common.py                 data loading, split, mask, logging
src/ensembles.py              Bagging and SAMME (from scratch), vote/margin/disagreement helpers
scripts/q1.py q2.py q3.py     experiments for each question
scripts/figures.py            figure script
results/*.json, *.npz         all reported numbers and test-set predictions
logs/*.log                    console logs of the final run
figures/                      figures used in the report
report/                       CVPR-format LaTeX source and PDF
```

## Protocol details

**Seed.**
- All numeric characters of the roll number, in order: `M25CPS009 → 25009`.
- The split uses `numpy.random.default_rng(25009)` directly. Within each
  identity the ten images are permuted and assigned 6 train / 2 validation /
  2 test.
- Other random streams use `default_rng([25009, tag])` with a fixed tag per
  experiment.

**Selection.**
- Clean validation accuracy only.
- Ties go to the simpler model: fewer leaves (tree), fewer trees (bagging),
  then fewer rounds and shallower learners (boosting).

**Bagging.**
- Bootstrap of size n = 240 per tree.
- Prediction is a hard majority vote; confidence is the winning vote fraction.
- A (bootstrap, seed) pair is drawn per tree in turn, so the first B trees of
  a 200-tree fit are exactly the B-tree model.

**SAMME.**
- α_t = log((1 − ε_t)/ε_t) + log(K − 1).
- Misclassified weights are multiplied by exp(α_t), then all weights are
  renormalised.
- A learner with ε_t ≥ 1 − 1/K is rejected and refitted with a new seed.
  Boosting stops after 5 consecutive rejections.
- ε_t = 0 is clipped to 1e−10, and boosting stops after that learner.
- Confidence is q_k = S_k / Σ α_t.

**Q3.**
- Each of the 30 training sets draws, within each identity, 6 of its 6
  training faces with replacement.
- Class vectors are the leaf class distribution (tree), vote fractions
  (bagging) and normalised weighted vote q (boosting).

## Data note

`fetch_olivetti_faces()` downloads the dataset from figshare on first use. If
that host is unreachable, place a copy in scikit-learn's cache
(`~/scikit_learn_data/olivetti_py3.pkz`). The loader and the split are
unchanged.
