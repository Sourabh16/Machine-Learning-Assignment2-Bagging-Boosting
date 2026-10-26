"""Q2: multiclass SAMME AdaBoost from scratch."""
import os
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import (RESULTS, Logger, load_all, load_config, load_json, masked_feature_ids,
                    rng_for, save_json, tree_params)
from ensembles import SAMME, Bagging, margin, pairwise_disagreement, tree_proba, vote_counts

cfg = load_config()
log = Logger("q2")
acc = lambda p, y: float(np.mean(p == y))

# ---------------- (a) toy three-class example ----------------
Xtoy = np.array([[1.0, 1.0], [1.5, 2.0], [2.0, 1.2],      # class 0
                 [4.0, 4.0], [4.5, 3.5], [2.2, 1.6],      # class 1 (last one sits among class 0)
                 [7.0, 1.0], [6.5, 2.0], [4.2, 3.8]])     # class 2 (last one sits among class 1)
ytoy = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
toy = SAMME(depth=1, n_rounds=1, seed=0, k=3).fit(Xtoy, ytoy, log=log)
h = toy.learners[0]
w0, w1 = toy.weight_history[0], toy.weight_history[1]
pred = toy.train_pred_history[0]
eps, alpha = float(toy.errors[0]), float(toy.alphas[0])
# independent hand calculation of the same round, to check the implementation
miss = pred != ytoy
eps_hand = w0[miss].sum() / w0.sum()
alpha_hand = np.log((1 - eps_hand) / eps_hand) + np.log(3 - 1)
w_hand = w0 * np.exp(alpha_hand * miss); w_hand /= w_hand.sum()
assert np.isclose(eps, eps_hand) and np.isclose(alpha, alpha_hand) and np.allclose(w1, w_hand)
toy_log = dict(X=Xtoy, y=ytoy, w0=w0, pred=pred, miss=miss.astype(int), eps=eps, alpha=alpha,
               w1=w1, split_feature=int(h.tree_.feature[0]), split_threshold=float(h.tree_.threshold[0]),
               unnormalised=w0 * np.exp(alpha * miss))
log("TOY round 1")
log(f"  stump: x[{toy_log['split_feature']}] <= {toy_log['split_threshold']:.3f}")
for i in range(len(ytoy)):
    log(f"  i={i} x={Xtoy[i]} y={ytoy[i]} w0={w0[i]:.4f} h={pred[i]} miss={int(miss[i])} w1={w1[i]:.4f}")
log(f"  eps={eps:.4f} alpha=log((1-eps)/eps)+log(2)={alpha:.4f}  (hand check passed)")

# ---------------- face data ----------------
D = load_all(cfg)
Xtr, ytr, Xva, yva, Xte, yte, Xm = D["Xtr"], D["ytr"], D["Xva"], D["yva"], D["Xte"], D["yte"], D["Xte_mask"]
q1 = load_json("q1.json")
sel_tree = q1["selected_params"]

# ---------------- (b) grid over depth x rounds ----------------
depths, rounds = cfg["q2"]["depth_grid"], cfg["q2"]["rounds_grid"]
Tmax = max(rounds)
boost_seed = int(rng_for(cfg, 3).integers(2**31 - 1))
models, grid, curves = {}, [], {}
for d in depths:
    t0 = time.time()
    m = SAMME(depth=d, n_rounds=Tmax, seed=boost_seed).fit(Xtr, ytr, log=log)
    fit_time = time.time() - t0
    models[d] = m
    Ptr, Pva, Pte = m.learner_preds(Xtr), m.learner_preds(Xva), m.learner_preds(Xte)
    curves[d] = dict(train=[acc(m.predict(None, T, Ptr), ytr) for T in range(1, len(m.alphas) + 1)],
                     val=[acc(m.predict(None, T, Pva), yva) for T in range(1, len(m.alphas) + 1)])
    for T in rounds:
        row = dict(depth=d, rounds=T, train=acc(m.predict(None, T, Ptr), ytr),
                   val=acc(m.predict(None, T, Pva), yva), test=acc(m.predict(None, T, Pte), yte))
        grid.append(row); log("boost", row)
    log(f"depth {d}: {len(m.alphas)} learners, {len(m.rejected)} rejected, {fit_time:.1f}s")
best = max(grid, key=lambda r: (r["val"], -r["rounds"], -r["depth"]))
bd, bT = best["depth"], best["rounds"]
log("SELECTED boosting:", best)
full = models[bd]
boost = SAMME(depth=bd, n_rounds=bT, seed=boost_seed)   # = first bT rounds of the fitted run
boost.learners, boost.alphas, boost.errors = full.learners[:bT], full.alphas[:bT], full.errors[:bT]
boost.weight_history, boost.train_pred_history = full.weight_history[:bT + 1], full.train_pred_history[:bT]
boost.rejected = [r for r in full.rejected if r["round"] <= bT]
joblib.dump(boost, os.path.join(RESULTS, "models", "boost.joblib"))
W = full.weight_history[1:]                                # weights after each round
diag = dict(eps=full.errors, alpha=full.alphas, max_w=W.max(1),
            top10=np.sort(W, 1)[:, -10:].sum(1), uniform=1 / len(ytr), random_eps=1 - 1 / 40)

# ---------------- (c) boosting vs bagging at equal size ----------------
cmp_rows = []
for M in cfg["q2"]["compare_sizes"]:
    t0 = time.time(); b = SAMME(depth=bd, n_rounds=M, seed=boost_seed).fit(Xtr, ytr); tb = time.time() - t0
    t0 = time.time(); g = Bagging(tree_params(sel_tree), M, seed=q1["bag_seed"], n_jobs=1).fit(Xtr, ytr); tg = time.time() - t0
    for name, model, tt in [("boosting", b, tb), ("bagging", g, tg)]:
        row = dict(model=name, learners=M, train=acc(model.predict(Xtr), ytr), val=acc(model.predict(Xva), yva),
                   test=acc(model.predict(Xte), yte), time_s=tt)
        cmp_rows.append(row); log("compare", row)

# heavily weighted training face in the selected model
Wsel = boost.weight_history
i_heavy = int(np.argmax(Wsel[-1]))
hist = boost.train_pred_history[:, i_heavy]
cumS = np.cumsum(boost.alphas[:, None] * (np.arange(40)[None, :] == hist[:, None]), axis=0)
true_c = ytr[i_heavy]
others = cumS.copy(); others[:, true_c] = -np.inf
heavy = dict(train_pos=i_heavy, image_id=int(D["idx_tr"][i_heavy]), label=int(true_c),
             final_weight=float(Wsel[-1, i_heavy]), weight_traj=Wsel[:, i_heavy],
             rank_final=int((Wsel[-1] > Wsel[-1, i_heavy]).sum()) + 1,
             correct_hist=(hist == true_c).astype(int), pred_hist=hist,
             n_correct=int((hist == true_c).sum()), n_rounds=len(hist),
             cum_true=cumS[:, true_c], cum_best_other=others.max(1),
             ensemble_pred_final=int(cumS[-1].argmax()),
             wrong_labels=np.unique(hist[hist != true_c], return_counts=True))
log("heavy face:", {k: v for k, v in heavy.items() if k not in ("weight_traj", "correct_hist", "pred_hist", "cum_true", "cum_best_other")})

# ---------------- (d) frozen models on clean and masked test ----------------
tree = joblib.load(os.path.join(RESULTS, "models", "tree.joblib"))
bag = joblib.load(os.path.join(RESULTS, "models", "bag.joblib"))
q1p = np.load(os.path.join(RESULTS, "q1_preds.npz"))
B = len(bag.trees)
mask_ids = set(masked_feature_ids(cfg).tolist())


def path_uses_mask(est, X):
    """For every sample: does its root-to-leaf path test a masked pixel?"""
    dp = est.decision_path(X).toarray().astype(bool)
    feat = est.tree_.feature
    node_masked = np.array([f in mask_ids for f in feat])
    return (dp & node_masked[None, :]).any(1)


def split_fraction(est):
    f = est.tree_.feature; f = f[f >= 0]
    return float(np.mean([x in mask_ids for x in f])) if len(f) else 0.0


res_d, per_face = {}, {}
Pb_te = boost.learner_preds(Xte); Pb_tm = boost.learner_preds(Xm)
for name, X, key, Pb in [("clean", Xte, "te", Pb_te), ("masked", Xm, "tm", Pb_tm)]:
    pt = tree_proba(tree, X)
    vb = vote_counts(q1p[key][:B]) / B
    qb = boost.q(None, preds=Pb)
    res_d[name] = dict(tree_acc=acc(pt.argmax(1), yte), tree_conf=float(pt.max(1).mean()),
                       bag_acc=acc(vb.argmax(1), yte), bag_conf=float(vb.max(1).mean()),
                       boost_acc=acc(qb.argmax(1), yte), boost_conf=float(qb.max(1).mean()),
                       bag_disagreement=pairwise_disagreement(q1p[key][:B]),
                       boost_margin=float(margin(qb).mean()), bag_margin=float(margin(vb).mean()))
    per_face[name] = dict(tree=pt, bag=vb, boost=qb)
    log("Q2d", name, res_d[name])

split_frac = dict(tree=split_fraction(tree), bag=float(np.mean([split_fraction(t) for t in bag.trees])),
                  boost_alpha_weighted=float(np.average([split_fraction(h) for h in boost.learners], weights=boost.alphas)),
                  boost_unweighted=float(np.mean([split_fraction(h) for h in boost.learners])),
                  pixel_share=len(mask_ids) / 4096)
uses_bag = np.stack([path_uses_mask(t, Xte) for t in bag.trees])        # (B, N)
uses_boost = np.stack([path_uses_mask(h, Xte) for h in boost.learners])  # (T, N)
log("split fractions in masked block:", split_frac)
log(f"mean share of bagged trees whose path crosses the block: {uses_bag.mean():.3f}; "
    f"alpha-weighted share for boosting: {np.average(uses_boost.mean(1), weights=boost.alphas):.3f}")

# faces where bagging and boosting react differently to the mask
ok = {m: {s: per_face[s][m].argmax(1) == yte for s in ("clean", "masked")} for m in ("bag", "boost")}
d_bag = per_face["clean"]["bag"][np.arange(len(yte)), yte] - per_face["masked"]["bag"][np.arange(len(yte)), yte]
d_boost = per_face["clean"]["boost"][np.arange(len(yte)), yte] - per_face["masked"]["boost"][np.arange(len(yte)), yte]
diff_outcome = np.flatnonzero((ok["bag"]["clean"] == ok["boost"]["clean"]) &
                              (ok["bag"]["masked"] != ok["boost"]["masked"]))
cand = diff_outcome[np.argsort(-np.abs(d_boost[diff_outcome] - d_bag[diff_outcome]))]
log(f"faces where only one ensemble flips under masking: {len(diff_outcome)}")


def face_story(i):
    changed_bag = q1p["te"][:B, i] != q1p["tm"][:B, i]
    changed_boost = Pb_te[:, i] != Pb_tm[:, i]
    c = int(yte[i])
    # total boosting weight on this identity's training faces when each learner was fitted
    id_weight = boost.weight_history[:-1][:, ytr == c].sum(1)
    return dict(test_pos=int(i), image_id=int(D["idx_te"][i]), true=c,
                bag_pred=[int(per_face[s]["bag"][i].argmax()) for s in ("clean", "masked")],
                boost_pred=[int(per_face[s]["boost"][i].argmax()) for s in ("clean", "masked")],
                bag_true=[float(per_face[s]["bag"][i, c]) for s in ("clean", "masked")],
                bag_win=[float(per_face[s]["bag"][i].max()) for s in ("clean", "masked")],
                boost_true=[float(per_face[s]["boost"][i, c]) for s in ("clean", "masked")],
                boost_win=[float(per_face[s]["boost"][i].max()) for s in ("clean", "masked")],
                bag_frac_changed=float(changed_bag.mean()), bag_frac_path_in_mask=float(uses_bag[:, i].mean()),
                bag_dis=[pairwise_disagreement(q1p[k][:B, [i]]) for k in ("te", "tm")],
                boost_alpha_changed=float(boost.alphas[changed_boost].sum() / boost.alphas.sum()),
                boost_alpha_path_in_mask=float(np.average(uses_boost[:, i], weights=boost.alphas)),
                boost_changed_rounds=(np.flatnonzero(changed_boost) + 1).tolist(),
                boost_changed_alpha=boost.alphas[changed_boost].tolist(),
                boost_changed_lost_true=int(np.sum(changed_boost & (Pb_te[:, i] == c))),
                boost_changed_gained_true=int(np.sum(changed_boost & (Pb_tm[:, i] == c))),
                id_weight_at_changed=id_weight[changed_boost].tolist(), id_weight_uniform=6 / len(ytr),
                masked_runner=int(np.argsort(-per_face["masked"]["boost"][i])[1]))


stories = [face_story(i) for i in cand[:4]]
for s in stories:
    log("face", s)

np.savez_compressed(os.path.join(RESULTS, "q2_preds.npz"), boost_te=Pb_te, boost_tm=Pb_tm)
save_json(dict(toy=toy_log, grid=grid, curves=curves, selected=best, boost_seed=boost_seed,
               rejected={d: models[d].rejected for d in depths}, diag=diag, compare=cmp_rows,
               heavy=heavy, clean_vs_masked=res_d, split_frac=split_frac,
               bag_path_share=float(uses_bag.mean()),
               boost_path_share=float(np.average(uses_boost.mean(1), weights=boost.alphas)),
               n_diff_outcome=len(diff_outcome), stories=stories), "q2.json")
log("done")
