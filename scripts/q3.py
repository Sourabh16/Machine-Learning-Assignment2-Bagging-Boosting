"""Q3: bias and variance measured over 30 stratified bootstrap training sets."""
import os
import sys
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import K, RESULTS, Logger, load_all, load_config, load_json, rng_for, save_json
from ensembles import SAMME, Bagging, pairwise_disagreement, tree_proba, vote_counts

cfg = load_config()
log = Logger("q3")
D = load_all(cfg)
Xtr, ytr, Xte, yte, Xm = D["Xtr"], D["ytr"], D["Xte"], D["yte"], D["Xte_mask"]
q1, q2 = load_json("q1.json"), load_json("q2.json")
sel = q1["selected_params"]
B_final = q1["selected_B"]
bd, bT = q2["selected"]["depth"], q2["selected"]["rounds"]
R = cfg["q3"]["n_repeats"]
depths = [tuple(d) for d in cfg["q3"]["depth_study"]]   # (max_depth, min_samples_leaf)
B_study = cfg["q3"]["depth_study_bag_size"]
SEL = (sel["max_depth"], sel["min_samples_leaf"])
log(f"tree={sel} bag B={B_final} boost depth={bd} rounds={bT} repeats={R} depths={depths}")

# ---- resampled training sets: within each identity draw 6 of its 6 faces with replacement
rng = rng_for(cfg, 4)
resamples = []
for r in range(R):
    idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), 6, replace=True) for c in range(K)])
    resamples.append(idx)
seeds = rng_for(cfg, 5).integers(0, 2**31 - 1, size=(R, 2))   # bagging seed, boosting seed


def onehot(y):
    e = np.zeros((len(y), K)); e[np.arange(len(y)), y] = 1; return e


E = onehot(yte)


def bias_var(P):
    """P: (R, N, K) class vectors -> (bias^2, variance) as defined in the brief."""
    pbar = P.mean(0)
    bias2 = float(np.mean(np.sum((pbar - E) ** 2, 1)))
    var = float(np.mean(np.sum((P - pbar[None]) ** 2, 2)))
    return bias2, var


CACHE = os.path.join(RESULTS, "q3_cache")   # one file per refit, so an interrupted run resumes
os.makedirs(CACHE, exist_ok=True)


def cached(fn):
    def wrapper(*args):
        path = os.path.join(CACHE, fn.__name__ + "_" + "_".join(str(a) for a in args).replace(" ", "") + ".joblib")
        if os.path.exists(path):
            return joblib.load(path)
        out = fn(*args)
        joblib.dump(out, path + ".tmp"); os.replace(path + ".tmp", path)
        return out
    wrapper.__name__ = fn.__name__
    return wrapper


@cached
def tree_job(r, depth):
    idx = resamples[r]
    X, y = Xtr[idx], ytr[idx]
    params = dict(max_depth=depth[0], min_samples_leaf=depth[1])
    t = DecisionTreeClassifier(random_state=sel["random_state"], **params).fit(X, y)
    leaves = t.apply(X)
    _, cnt = np.unique(leaves, return_counts=True)
    nb = max(B_final, B_study) if depth == SEL else B_study
    bag = Bagging(params, nb, seed=int(seeds[r, 0])).fit(X, y)
    Pte, Ptm = bag.tree_preds(Xte), bag.tree_preds(Xm)
    bag_leaf_single = float(np.mean([np.mean(np.unique(tt.apply(X[b]), return_counts=True)[1] == 1)
                                     for tt, b in zip(bag.trees, bag.boot)]))
    return dict(r=r, depth=depth,
                tree_te=tree_proba(t, Xte), tree_tm=tree_proba(t, Xm),
                n_leaves=int(t.get_n_leaves()), frac_single=float(np.mean(cnt == 1)),
                mean_per_leaf=float(cnt.mean()), tree_depth=int(t.get_depth()),
                bag_leaf_single=bag_leaf_single,
                bag_leaves=float(np.mean([tt.get_n_leaves() for tt in bag.trees])),
                preds_te=Pte.astype(np.int16), preds_tm=Ptm.astype(np.int16))


@cached
def boost_job(r):
    idx = resamples[r]
    m = SAMME(depth=bd, n_rounds=bT, seed=int(seeds[r, 1])).fit(Xtr[idx], ytr[idx])
    return dict(r=r, te=m.q(Xte), tm=m.q(Xm), n_learners=len(m.alphas), rejected=len(m.rejected))


t0 = time.time()
jobs = [(r, d) for d in sorted(depths, key=lambda d: (-(d[0] or 999), d[1])) for r in range(R)]  # slow jobs first
tree_out = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(delayed(tree_job)(r, d) for r, d in jobs)
log(f"tree/bagging refits done in {time.time()-t0:.0f}s")
t0 = time.time()
boost_out = Parallel(n_jobs=cfg["n_jobs"])(delayed(boost_job)(r) for r in range(R))
log(f"boosting refits done in {time.time()-t0:.0f}s")


def collect(depth):
    rows = sorted([o for o in tree_out if o["depth"] == depth], key=lambda o: o["r"])
    return rows


def summarise(P_te, P_tm):
    b, v = bias_var(P_te); bm, vm = bias_var(P_tm)
    a = np.array([np.mean(p.argmax(1) == yte) for p in P_te])
    am = np.array([np.mean(p.argmax(1) == yte) for p in P_tm])
    return dict(bias2=b, var=v, acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
                bias2_masked=bm, var_masked=vm, acc_masked_mean=float(am.mean()),
                acc_masked_std=float(am.std(ddof=1)), acc_all=a.tolist(),
                acc_of_mean_vector=float(np.mean(P_te.mean(0).argmax(1) == yte)))


def bag_vectors(rows, B, key):
    return np.stack([vote_counts(o[key][:B].astype(int)) / B for o in rows])


# ---- (a) selected tree, final bagging, selected boosting
rows_sel = collect(SEL)
main = {
    "tree": summarise(np.stack([o["tree_te"] for o in rows_sel]), np.stack([o["tree_tm"] for o in rows_sel])),
    "bagging": summarise(bag_vectors(rows_sel, B_final, "preds_te"), bag_vectors(rows_sel, B_final, "preds_tm")),
    "boosting": summarise(np.stack([o["te"] for o in sorted(boost_out, key=lambda o: o["r"])]),
                          np.stack([o["tm"] for o in sorted(boost_out, key=lambda o: o["r"])])),
}
for k, v in main.items():
    log("Q3a", k, {kk: vv for kk, vv in v.items() if kk != "acc_all"})

# ---- (b, c) depth study
study = []
for d in depths:
    rows = collect(d)
    st = summarise(np.stack([o["tree_te"] for o in rows]), np.stack([o["tree_tm"] for o in rows]))
    sb = summarise(bag_vectors(rows, B_study, "preds_te"), bag_vectors(rows, B_study, "preds_tm"))
    dis = [pairwise_disagreement(o["preds_te"][:B_study].astype(int)) for o in rows]
    row = dict(depth=d[0], min_samples_leaf=d[1], tree=st, bag=sb,
               n_leaves=float(np.mean([o["n_leaves"] for o in rows])),
               frac_single=float(np.mean([o["frac_single"] for o in rows])),
               mean_per_leaf=float(np.mean([o["mean_per_leaf"] for o in rows])),
               bag_leaf_single=float(np.mean([o["bag_leaf_single"] for o in rows])),
               bag_leaves=float(np.mean([o["bag_leaves"] for o in rows])),
               tree_depth=float(np.mean([o["tree_depth"] for o in rows])),
               disagreement=float(np.mean(dis)),
               var_removed=st["var"] - sb["var"], var_removed_frac=(st["var"] - sb["var"]) / st["var"],
               acc_gain=sb["acc_mean"] - st["acc_mean"],
               paired_gain_std=float(np.std(np.array(sb["acc_all"]) - np.array(st["acc_all"]), ddof=1)))
    study.append(row)
    log("depth", d, {k: v for k, v in row.items() if k not in ("tree", "bag")},
        "tree:", {k: st[k] for k in ("bias2", "var", "acc_mean")}, "bag:", {k: sb[k] for k in ("bias2", "var", "acc_mean")})

save_json(dict(main=main, study=study, R=R, B_final=B_final, B_study=B_study,
               boost_learners=[o["n_learners"] for o in boost_out],
               boost_rejected=[o["rejected"] for o in boost_out],
               resample_distinct=[float(np.mean([len(np.unique(idx[ytr[idx] == c])) for c in range(K)]))
                                  for idx in resamples]), "q3.json")
log("done")
