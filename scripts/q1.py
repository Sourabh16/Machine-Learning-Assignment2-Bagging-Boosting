"""Q1: single-tree baseline and bagging from scratch."""
import os
import sys
import time

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import (RESULTS, Logger, load_all, load_config, rng_for, save_json, tree_params)
from ensembles import Bagging, margin, pairwise_disagreement, tree_proba, vote_counts

cfg = load_config()
log = Logger("q1")
D = load_all(cfg)
Xtr, ytr, Xva, yva, Xte, yte, Xm = D["Xtr"], D["ytr"], D["Xva"], D["yva"], D["Xte"], D["yte"], D["Xte_mask"]
n = len(ytr)
log(f"seed={cfg['seed']} train={len(ytr)} val={len(yva)} test={len(yte)} features={Xtr.shape[1]}")
os.makedirs(os.path.join(RESULTS, "models"), exist_ok=True)

acc = lambda p, y: float(np.mean(p == y))


def leaf_stats(tree):
    leaves = tree.apply(Xtr)
    _, counts = np.unique(leaves, return_counts=True)
    return dict(n_leaves=int(tree.get_n_leaves()), depth=int(tree.get_depth()),
                mean_per_leaf=float(counts.mean()), frac_single=float(np.mean(counts == 1)))


# ---------------- (a) single tree grid ----------------
tree_seed = int(rng_for(cfg, 1).integers(2**31 - 1))
grid = []
for d in cfg["q1"]["max_depth_grid"]:
    for msl in cfg["q1"]["min_samples_leaf_grid"]:
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=msl, random_state=tree_seed).fit(Xtr, ytr)
        row = dict(max_depth=d, min_samples_leaf=msl, train=acc(t.predict(Xtr), ytr),
                   val=acc(t.predict(Xva), yva), test=acc(t.predict(Xte), yte), **leaf_stats(t))
        grid.append(row)
        log("tree", row)
# selection on validation accuracy only; ties -> fewer leaves (simpler tree)
best = max(grid, key=lambda r: (r["val"], -r["n_leaves"]))
sel = dict(max_depth=best["max_depth"], min_samples_leaf=best["min_samples_leaf"], random_state=tree_seed)
log("SELECTED tree:", best)
tree = DecisionTreeClassifier(**tree_params(sel), random_state=tree_seed).fit(Xtr, ytr)
joblib.dump(tree, os.path.join(RESULTS, "models", "tree.joblib"))

# ---------------- (b, c) bagging ----------------
Bs = cfg["q1"]["bagging_B"]
Bmax = max(Bs)
t0 = time.time()
bag = Bagging(tree_params(sel), Bmax, seed=int(rng_for(cfg, 2).integers(2**31 - 1)), n_jobs=cfg["n_jobs"]).fit(Xtr, ytr)
log(f"fitted {Bmax} bagged trees in {time.time()-t0:.1f}s")
P = {s: bag.tree_preds(X) for s, X in [("tr", Xtr), ("va", Xva), ("te", Xte), ("tm", Xm)]}
distinct = bag.distinct_counts()
theory = n * (1 - (1 - 1 / n) ** n)

rows = []
for B in Bs:
    oob, cover = bag.oob_accuracy(P["tr"], ytr, B)
    fr_va = bag.vote_fraction(None, B, P["va"])
    row = dict(B=B, train=acc(bag.predict(None, B, P["tr"]), ytr), val=acc(bag.predict(None, B, P["va"]), yva),
               test=acc(bag.predict(None, B, P["te"]), yte),
               distinct_mean=float(distinct[:B].mean()), distinct_std=float(distinct[:B].std()),
               distinct_theory=theory, oob=oob, oob_coverage=cover,
               disagreement_val=pairwise_disagreement(P["va"][:B]),
               disagreement_te=pairwise_disagreement(P["te"][:B]),
               margin_val=float(margin(fr_va).mean()),
               margin_te=float(margin(bag.vote_fraction(None, B, P["te"])).mean()))
    rows.append(row)
    log("bag", row)

# dense curves (B = 1..Bmax) for the plots
curve = dict(B=list(range(1, Bmax + 1)), val=[], oob=[], dis=[], margin=[])
for B in curve["B"]:
    curve["val"].append(acc(bag.predict(None, B, P["va"]), yva))
    curve["oob"].append(bag.oob_accuracy(P["tr"], ytr, B)[0])
    curve["dis"].append(pairwise_disagreement(P["va"][:B]))
    curve["margin"].append(float(margin(bag.vote_fraction(None, B, P["va"])).mean()))

# select B on validation accuracy only; ties -> smaller B
bestB = max(rows, key=lambda r: (r["val"], -r["B"]))["B"]
log("SELECTED bagging B:", bestB)
bag_final = Bagging(tree_params(sel), bestB, seed=bag.seed, n_jobs=1)
bag_final.trees, bag_final.boot, bag_final.n_train = bag.trees[:bestB], bag.boot[:bestB], n
joblib.dump(bag_final, os.path.join(RESULTS, "models", "bag.joblib"))

# ---------------- (d) clean vs masked ----------------
def tree_eval(X, y):
    p = tree_proba(tree, X)
    return p.argmax(1), p.max(1)


def bag_eval(preds, y):
    v = vote_counts(preds[:bestB])
    return v.argmax(1), v.max(1) / bestB, v


d_rows = {}
cases = {}
for name, X, key in [("clean", Xte, "te"), ("masked", Xm, "tm")]:
    tp, tc = tree_eval(X, yte)
    bp, bc, bv = bag_eval(P[key], yte)
    d_rows[name] = dict(tree_acc=acc(tp, yte), tree_conf=float(tc.mean()),
                        tree_conf_correct=float(tc[tp == yte].mean()) if (tp == yte).any() else None,
                        tree_conf_wrong=float(tc[tp != yte].mean()) if (tp != yte).any() else None,
                        bag_acc=acc(bp, yte), bag_conf=float(bc.mean()),
                        bag_conf_correct=float(bc[bp == yte].mean()), bag_conf_wrong=float(bc[bp != yte].mean()),
                        bag_margin=float(margin(bv / bestB).mean()),
                        disagreement=pairwise_disagreement(P[key][:bestB]))
    log("Q1d", name, d_rows[name])
    if name == "clean":
        fixed = np.flatnonzero((tp != yte) & (bp == yte))
        both = np.flatnonzero((tp != yte) & (bp != yte))
        # most convincing corrections (largest bagging margin) and the most
        # confident joint failures (largest wrong-vote share)
        fixed = fixed[np.argsort(-margin(bv[fixed] / bestB))]
        both = both[np.argsort(-bc[both])]
        log(f"bagging corrects tree on {len(fixed)} faces; both wrong on {len(both)}; "
            f"tree right & bagging wrong on {int(np.sum((tp == yte) & (bp != yte)))}")
        def describe(i):
            top = np.argsort(-bv[i])[:3]
            same_as_tree = int(bv[i][tp[i]])
            return dict(test_pos=int(i), image_id=int(D["idx_te"][i]), true=int(yte[i]),
                        tree_pred=int(tp[i]), tree_conf=float(tc[i]),
                        bag_top=[(int(c), int(bv[i][c])) for c in top],
                        bag_votes_true=int(bv[i][yte[i]]), bag_votes_tree_pred=same_as_tree,
                        bag_pred=int(bp[i]))
        cases = dict(n_fixed=len(fixed), n_both=len(both),
                     n_broken=int(np.sum((tp == yte) & (bp != yte))),
                     fixed=[describe(i) for i in fixed[:2]], both=[describe(i) for i in both[:2]],
                     all_both=[describe(i) for i in both], all_fixed=[describe(i) for i in fixed])
        for c in cases["fixed"] + cases["both"]:
            log("case", c)

np.savez_compressed(os.path.join(RESULTS, "q1_preds.npz"), **P)
save_json(dict(grid=grid, selected=best, selected_params=sel, bagging=rows, curve=curve,
               selected_B=bestB, theory_distinct=theory, distinct_all=distinct.tolist(),
               clean_vs_masked=d_rows, cases=cases, bag_seed=bag.seed), "q1.json")
log("done")
