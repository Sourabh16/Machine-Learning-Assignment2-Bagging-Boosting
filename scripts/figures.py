"""Builds every figure in the report from results/*.json (no retraining)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import ROOT, load_all, load_config, load_json

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "lines.linewidth": 1.2})
C = {"tree": "#7f7f7f", "bag": "#1f77b4", "boost": "#d62728", "val": "#2ca02c", "oob": "#9467bd", "train": "#ff7f0e"}
W1 = 3.3   # one CVPR column in inches
cfg = load_config()
D = load_all(cfg)
q1 = load_json("q1.json")
have = lambda f: os.path.exists(os.path.join(ROOT, "results", f))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


def face(ax, img, title, color="k"):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6.5, color=color, pad=2)
    for s in ax.spines.values(): s.set_visible(False)


def first_train_image(c):
    return D["images"][D["idx_tr"][D["ytr"] == c][0]]


# ---------- Q1a: depth sweep ----------
g = [r for r in q1["grid"] if r["min_samples_leaf"] == q1["selected"]["min_samples_leaf"]]
labels = [str(r["max_depth"]) for r in g]
fig, ax = plt.subplots(1, 2, figsize=(W1, 1.35))
x = np.arange(len(g))
for k, c in [("train", C["train"]), ("val", C["val"]), ("test", C["tree"])]:
    ax[0].plot(x, [r[k] for r in g], "o-", ms=3, color=c, label=k)
ax[0].set_xticks(x, labels); ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("accuracy"); ax[0].legend(frameon=False)
ax[1].bar(x - 0.2, [r["n_leaves"] for r in g], 0.4, color=C["bag"], label="leaves")
ax[1].bar(x + 0.2, [r["depth"] for r in g], 0.4, color=C["oob"], label="realised depth")
ax[1].set_xticks(x, labels); ax[1].set_xlabel("max_depth"); ax[1].legend(frameon=False)
save(fig, "q1a_depth.pdf")

# ---------- Q1c: OOB, disagreement, margin ----------
cv = q1["curve"]; rows = q1["bagging"]; Bs = [r["B"] for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(W1 * 2.05, 1.45))
fig.subplots_adjust(wspace=0.4)
ax[0].plot(cv["B"], cv["oob"], color=C["oob"], label="OOB")
ax[0].plot(cv["B"], cv["val"], color=C["val"], alpha=0.8, label="validation")
ax[0].plot(Bs, [r["oob"] for r in rows], "o", ms=3, color=C["oob"])
ax[0].plot(Bs, [r["val"] for r in rows], "s", ms=3, color=C["val"])
ax[0].set_ylabel("accuracy"); ax[0].legend(frameon=False)
ax[1].plot(cv["B"], cv["dis"], color=C["bag"]); ax[1].plot(Bs[1:], [r["disagreement_val"] for r in rows[1:]], "o", ms=3, color=C["bag"])
ax[1].set_ylabel("pairwise disagreement")
ax[2].plot(cv["B"], cv["margin"], color=C["boost"]); ax[2].plot(Bs, [r["margin_val"] for r in rows], "o", ms=3, color=C["boost"])
ax[2].set_ylabel("mean vote margin")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("B (trees)"); a.axvline(q1["selected_B"], ls=":", c="k", lw=0.8)
save(fig, "q1c_bagging.pdf")

# ---------- Q1d: case faces ----------
cases = q1["cases"]["fixed"] + q1["cases"]["both"]
fig, ax = plt.subplots(2, 4, figsize=(W1, 2.35))
for j, cs in enumerate(cases):
    img = D["images"][cs["image_id"]]
    top = ", ".join(f"{c}:{v}" for c, v in cs["bag_top"][:2])
    col = "#1a7f37" if j < 2 else "#b42318"
    face(ax[0, j], img, f"true {cs['true']}, tree {cs['tree_pred']}\nvotes {top}", col)
    ax[0, j].title.set_fontsize(5.5)
    wrong = cs["tree_pred"] if j < 2 else cs["bag_pred"]
    face(ax[1, j], first_train_image(wrong), f"train img, id {wrong}")
    ax[1, j].title.set_fontsize(5.5)
fig.text(0.31, 0.02, "bagging corrects the tree", ha="center", fontsize=7, color="#1a7f37")
fig.text(0.72, 0.02, "both wrong", ha="center", fontsize=7, color="#b42318")
fig.subplots_adjust(wspace=0.25, hspace=0.3, bottom=0.08)
save(fig, "q1d_cases.pdf")

# ---------- masked example ----------
fig, ax = plt.subplots(1, 2, figsize=(1.3, 0.75))
face(ax[0], D["Xte"][0].reshape(64, 64), "clean"); face(ax[1], D["Xte_mask"][0].reshape(64, 64), "masked")
save(fig, "mask_example.pdf")

if have("q2.json"):
    q2 = load_json("q2.json")
    # ---------- Q2b: grid heat map + diagnostics ----------
    dg = q2["diag"]; T = np.arange(1, len(dg["eps"]) + 1); Tsel = q2["selected"]["rounds"]
    fig, ax = plt.subplots(1, 4, figsize=(W1 * 2.05, 1.35))
    ax[0].plot(T, dg["eps"], color=C["boost"]); ax[0].axhline(dg["random_eps"], ls="--", c="k", lw=0.7)
    ax[0].text(T[-1], dg["random_eps"], "1-1/K", ha="right", va="bottom", fontsize=6); ax[0].set_ylabel(r"weighted error $\epsilon_t$")
    ax[1].plot(T, dg["alpha"], color=C["bag"]); ax[1].axhline(np.log(39), ls="--", c="k", lw=0.7)
    ax[1].text(T[-1], np.log(39), "log(K-1)", ha="right", va="bottom", fontsize=6); ax[1].set_ylabel(r"learner weight $\alpha_t$")
    ax[2].plot(T, dg["max_w"], color=C["oob"]); ax[2].axhline(dg["uniform"], ls="--", c="k", lw=0.7)
    ax[2].set_yscale("log"); ax[2].set_ylabel("max sample weight")
    ax[3].plot(T, dg["top10"], color=C["train"]); ax[3].axhline(10 * dg["uniform"], ls="--", c="k", lw=0.7)
    ax[3].set_ylabel("weight on 10 heaviest")
    for a in ax:
        a.set_xlabel("round t"); a.axvline(Tsel, ls=":", c="k", lw=0.8)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "q2b_diagnostics.pdf")

    fig, ax = plt.subplots(figsize=(W1 * 0.5, 1.3))
    for d, cvv in q2["curves"].items():
        ax.plot(np.arange(1, len(cvv["val"]) + 1), cvv["val"], label=f"depth {d}")
    ax.set_xlabel("rounds"); ax.set_ylabel("val accuracy"); ax.legend(frameon=False, fontsize=6)
    save(fig, "q2b_valcurves.pdf")

    # ---------- Q2c: heavy face ----------
    h = q2["heavy"]
    fig = plt.figure(figsize=(W1, 1.45))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 0.55, 2.0], hspace=0.6, wspace=0.6)
    lab = np.array(h["wrong_labels"][0]); cnt = np.array(h["wrong_labels"][1])
    face(fig.add_subplot(gs[0, 0]), D["images"][h["image_id"]], f"id {h['label']}")
    if len(lab):
        mc = int(lab[np.argmax(cnt)])
        face(fig.add_subplot(gs[0, 1]), first_train_image(mc), f"id {mc}")
    a = fig.add_subplot(gs[:, 2])
    t = np.arange(len(h["weight_traj"]))
    a.plot(t, h["weight_traj"], color=C["boost"], label="weight of this face")
    corr = np.array(h["correct_hist"]).astype(bool); tt = np.arange(1, len(corr) + 1)
    w = np.array(h["weight_traj"])[1:]
    a.plot(tt[~corr], w[~corr], "x", ms=3, color="k", label=r"$h_t$ wrong")
    a.plot(tt[corr], w[corr], "o", ms=3, mfc="none", color=C["val"], label=r"$h_t$ right")
    a.axhline(1 / 240, ls="--", c="k", lw=0.6); a.set_yscale("log"); a.set_xlabel("round", fontsize=7)
    a.legend(frameon=False, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
    a2 = fig.add_subplot(gs[1, :2])
    a2.plot(np.arange(1, len(corr) + 1), np.array(h["cum_true"]) - np.array(h["cum_best_other"]), color=C["bag"])
    a2.axhline(0, c="k", lw=0.6); a2.set_xlabel("round", fontsize=6); a2.set_ylabel(r"$S_y-\max S_k$", fontsize=6)
    save(fig, "q2c_heavy.pdf")

    # ---------- Q2d: story face ----------
    if q2["stories"]:
        s = q2["stories"][0]; i = s["test_pos"]
        fig, ax = plt.subplots(1, 3, figsize=(W1, 1.15), gridspec_kw={"width_ratios": [1, 1, 2.2]})
        face(ax[0], D["Xte"][i].reshape(64, 64), f"clean (id {s['true']})")
        face(ax[1], D["Xte_mask"][i].reshape(64, 64), "masked")
        xb = np.arange(2)
        ax[2].bar(xb - 0.2, s["bag_true"], 0.4, color=C["bag"], label="bagging: vote for true id")
        ax[2].bar(xb + 0.2, s["boost_true"], 0.4, color=C["boost"], label=r"boosting: $q_y$")
        ax[2].plot(xb - 0.2, s["bag_win"], "_", ms=12, color="k"); ax[2].plot(xb + 0.2, s["boost_win"], "_", ms=12, color="k")
        ax[2].set_xticks(xb, ["clean", "masked"]); ax[2].legend(frameon=False, fontsize=5.5, loc="upper right")
        for k in range(2):
            ax[2].text(xb[k] - 0.2, s["bag_win"][k] + 0.02, f"pred {s['bag_pred'][k]}", ha="center", fontsize=5.5)
            ax[2].text(xb[k] + 0.2, s["boost_win"][k] + 0.02, f"pred {s['boost_pred'][k]}", ha="center", fontsize=5.5)
        ax[2].set_ylim(0, max(max(s["bag_win"]), max(s["boost_win"])) * 1.6)
        save(fig, "q2d_face.pdf")

if have("q3.json"):
    q3 = load_json("q3.json")
    st = q3["study"]
    lab = [f"{r['depth']}" if r["min_samples_leaf"] != 1 else "None\nleaf=1" for r in st]
    x = np.arange(len(st))
    fig, ax = plt.subplots(1, 3, figsize=(W1 * 2.05, 1.5))
    fig.subplots_adjust(wspace=0.35)
    ax[0].plot(x, [r["tree"]["bias2"] for r in st], "o-", ms=3, color=C["tree"], label="tree bias$^2$")
    ax[0].plot(x, [r["bag"]["bias2"] for r in st], "o-", ms=3, color=C["bag"], label="bag bias$^2$")
    ax[0].plot(x, [r["tree"]["var"] for r in st], "s--", ms=3, color=C["tree"], label="tree var")
    ax[0].plot(x, [r["bag"]["var"] for r in st], "s--", ms=3, color=C["bag"], label="bag var")
    ax[0].legend(frameon=False, fontsize=5.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    ax[1].plot(x, [r["tree"]["acc_mean"] for r in st], "o-", ms=3, color=C["tree"], label="tree")
    ax[1].plot(x, [r["bag"]["acc_mean"] for r in st], "o-", ms=3, color=C["bag"], label="bag (100)")
    ax[1].bar(x, [r["acc_gain"] for r in st], 0.5, color=C["bag"], alpha=0.25, label="gain")
    ax[1].set_ylabel("test accuracy (mean of 30)"); ax[1].legend(frameon=False, fontsize=6)
    ax[2].plot(x, [r["disagreement"] for r in st], "o-", ms=3, color=C["oob"], label="disagreement")
    ax[2].plot(x, [r["bag_leaf_single"] for r in st], "s-", ms=3, color=C["train"], label="singleton-leaf frac.")
    ax[2].plot(x, [r["var_removed_frac"] for r in st], "^-", ms=3, color=C["boost"], label="var. removed (frac.)")
    ax[2].legend(frameon=False, fontsize=6)
    for a in ax:
        a.set_xticks(x, lab, fontsize=6); a.set_xlabel("max_depth")
    save(fig, "q3_depth.pdf")
print("figures written to", FIG)
