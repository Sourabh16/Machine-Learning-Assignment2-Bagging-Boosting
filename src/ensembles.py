"""Bagging and multiclass SAMME AdaBoost, written from scratch.
Only DecisionTreeClassifier is taken from scikit-learn."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from common import K


def tree_proba(tree, X, k=K):
    """predict_proba mapped onto all k classes (a bootstrap sample can miss a class)."""
    p = np.zeros((len(X), k))
    p[:, tree.classes_] = tree.predict_proba(X)
    return p


def vote_counts(preds, k=K):
    """preds: (T, N) integer labels -> (N, k) vote counts."""
    T, N = preds.shape
    counts = np.zeros((N, k))
    np.add.at(counts, (np.tile(np.arange(N), T), preds.ravel()), 1)
    return counts


def margin(frac):
    """winning fraction minus runner-up fraction, per row."""
    s = np.sort(frac, axis=1)
    return s[:, -1] - s[:, -2]


def pairwise_disagreement(preds, k=K):
    """Mean over all tree pairs of the fraction of points where they differ.
    Uses: agreeing pairs at a point = sum_k n_k (n_k - 1) / 2."""
    T = preds.shape[0]
    if T < 2:
        return float("nan")
    n = vote_counts(preds, k)
    agree = (n * (n - 1)).sum(1) / (T * (T - 1))
    return float(np.mean(1 - agree))


def _fit_tree(X, y, idx, rs, params):
    t = DecisionTreeClassifier(random_state=rs, **params)
    return t.fit(X[idx], y[idx])


class Bagging:
    """Bootstrap aggregation with hard majority voting.
    The first B trees of a fitted model form exactly the model that would be
    obtained by fitting with n_estimators=B and the same seed."""

    def __init__(self, tree_params, n_estimators, seed, n_jobs=1):
        self.params, self.T, self.seed, self.n_jobs = tree_params, n_estimators, seed, n_jobs

    def fit(self, X, y):
        n = len(X)
        rng = np.random.default_rng(self.seed)
        # draw (bootstrap, tree seed) per tree in turn, so a fit with B trees is
        # an exact prefix of a fit with more trees
        self.boot, states = [], []
        for _ in range(self.T):
            self.boot.append(rng.integers(0, n, n))
            states.append(rng.integers(0, 2**31 - 1))
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(X, y, b, int(s), self.params) for b, s in zip(self.boot, states))
        self.n_train = n
        return self

    def tree_preds(self, X):
        return np.stack([t.predict(X) for t in self.trees])

    def votes(self, X, B=None, preds=None):
        preds = self.tree_preds(X) if preds is None else preds
        return vote_counts(preds[:B or self.T])

    def predict(self, X, B=None, preds=None):
        # ties are broken towards the smaller label (np.argmax)
        return self.votes(X, B, preds).argmax(1)

    def vote_fraction(self, X, B=None, preds=None):
        v = self.votes(X, B, preds)
        return v / v.sum(1, keepdims=True)

    def distinct_counts(self):
        return np.array([len(np.unique(b)) for b in self.boot])

    def oob_accuracy(self, train_preds, y, B):
        """OOB vote using only trees whose bootstrap sample did not contain i."""
        n = len(y)
        inbag = np.zeros((B, n), bool)
        for b in range(B):
            inbag[b, self.boot[b]] = True
        counts = np.zeros((n, K))
        for b in range(B):
            oob = ~inbag[b]
            np.add.at(counts, (np.flatnonzero(oob), train_preds[b, oob]), 1)
        covered = counts.sum(1) > 0
        acc = float(np.mean(counts[covered].argmax(1) == y[covered])) if covered.any() else float("nan")
        return acc, float(covered.mean())


class SAMME:
    """Multiclass AdaBoost (SAMME, Zhu et al. 2009).

    alpha_t = log((1 - eps_t) / eps_t) + log(K - 1)
    w_i <- w_i * exp(alpha_t * 1[h_t(x_i) != y_i]), then renormalise.
    A weak learner with eps_t >= 1 - 1/K (no better than random guessing among
    K classes) gets alpha_t <= 0. Such a learner is rejected: it is not added,
    weights are left unchanged, and we refit with a different tie-breaking seed.
    After `max_retries` rejected learners in a row boosting stops early.
    A perfect learner (eps_t = 0) would get infinite alpha, so eps is clipped
    to 1e-10 and boosting stops after adding it (weights cannot change)."""

    def __init__(self, depth, n_rounds, seed, k=K, max_retries=5, eps_floor=1e-10):
        self.depth, self.T, self.seed, self.k = depth, n_rounds, seed, k
        self.max_retries, self.eps_floor = max_retries, eps_floor

    def fit(self, X, y, log=None):
        n, k = len(X), self.k
        rng = np.random.default_rng(self.seed)
        w = np.full(n, 1.0 / n)
        self.learners, self.alphas, self.errors = [], [], []
        self.weight_history = [w.copy()]   # weights before round 1, after each round
        self.train_pred_history = []        # h_t(x_i) for every accepted learner
        self.rejected = []
        retries = 0
        while len(self.learners) < self.T:
            h = DecisionTreeClassifier(max_depth=self.depth,
                                       random_state=int(rng.integers(2**31 - 1)))
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)
            miss = pred != y
            eps = float(np.sum(w[miss]) / np.sum(w))
            if eps >= 1 - 1 / k:
                self.rejected.append({"round": len(self.learners) + 1, "eps": eps})
                if log: log(f"  rejected weak learner at round {len(self.learners)+1}: eps={eps:.4f} >= 1-1/K")
                retries += 1
                if retries > self.max_retries:
                    if log: log("  stopping early: no weak learner better than random")
                    break
                continue
            retries = 0
            eps_c = max(eps, self.eps_floor)
            alpha = np.log((1 - eps_c) / eps_c) + np.log(k - 1)
            w = w * np.exp(alpha * miss)
            w = w / w.sum()
            self.learners.append(h); self.alphas.append(alpha); self.errors.append(eps)
            self.weight_history.append(w.copy()); self.train_pred_history.append(pred)
            if eps == 0:
                if log: log(f"  perfect weak learner at round {len(self.learners)}; stopping")
                break
        self.alphas = np.array(self.alphas); self.errors = np.array(self.errors)
        self.weight_history = np.array(self.weight_history)
        self.train_pred_history = np.array(self.train_pred_history)
        return self

    def learner_preds(self, X):
        return np.stack([h.predict(X) for h in self.learners])

    def scores(self, X, T=None, preds=None):
        """S_k(x) = sum_t alpha_t 1[h_t(x) = k] for the first T learners."""
        preds = self.learner_preds(X) if preds is None else preds
        T = T or len(self.learners)
        S = np.zeros((preds.shape[1], self.k))
        for t in range(T):
            S[np.arange(preds.shape[1]), preds[t]] += self.alphas[t]
        return S

    def q(self, X, T=None, preds=None):
        """normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t"""
        T = T or len(self.learners)
        return self.scores(X, T, preds) / self.alphas[:T].sum()

    def predict(self, X, T=None, preds=None):
        return self.scores(X, T, preds).argmax(1)
