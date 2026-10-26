"""Data loading, the saved split, the masked test set and small helpers."""
import json
import os
import time

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
SPLITS = os.path.join(ROOT, "splits")
K = 40  # identities


def load_config(path=None):
    path = path or os.path.join(ROOT, "configs", "default.json")
    with open(path) as f:
        return json.load(f)


def seed_from_roll(roll):
    """All numeric characters of the roll number, in order."""
    return int("".join(ch for ch in roll if ch.isdigit()))


def rng_for(cfg, tag):
    """Independent stream for one experiment. The split itself uses
    default_rng(seed) directly, as the assignment asks."""
    return np.random.default_rng([cfg["seed"], tag])


def load_faces():
    faces = fetch_olivetti_faces(shuffle=False)
    return faces.images.astype(np.float32), faces.target.astype(int)


def make_split(cfg, y):
    rng = np.random.default_rng(cfg["seed"])
    a, b = cfg["n_train_per_id"], cfg["n_train_per_id"] + cfg["n_val_per_id"]
    tr, va, te = [], [], []
    for c in range(K):
        idx = rng.permutation(np.flatnonzero(y == c))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


def get_split(cfg, y):
    """Create the split once, save it, and reload it everywhere afterwards."""
    os.makedirs(SPLITS, exist_ok=True)
    path = os.path.join(SPLITS, f"split_{cfg['roll_number']}.json")
    if not os.path.exists(path):
        tr, va, te = make_split(cfg, y)
        with open(path, "w") as f:
            json.dump({"seed": cfg["seed"], "roll_number": cfg["roll_number"],
                       "train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}, f)
    with open(path) as f:
        s = json.load(f)
    assert s["seed"] == cfg["seed"]
    return np.array(s["train"]), np.array(s["val"]), np.array(s["test"])


def mask_block(images, cfg):
    r0, r1 = cfg["mask_rows"]; c0, c1 = cfg["mask_cols"]
    out = images.copy()
    out[:, r0:r1 + 1, c0:c1 + 1] = 0.0
    return out


def masked_feature_ids(cfg):
    r0, r1 = cfg["mask_rows"]; c0, c1 = cfg["mask_cols"]
    m = np.zeros((64, 64), bool); m[r0:r1 + 1, c0:c1 + 1] = True
    return np.flatnonzero(m.ravel())


def load_all(cfg):
    """Returns a dict with flattened clean/masked arrays for each part."""
    images, y = load_faces()
    tr, va, te = get_split(cfg, y)
    flat = images.reshape(len(images), -1)
    masked = mask_block(images[te], cfg).reshape(len(te), -1)
    return dict(images=images, y=y, idx_tr=tr, idx_va=va, idx_te=te,
                Xtr=flat[tr], ytr=y[tr], Xva=flat[va], yva=y[va],
                Xte=flat[te], yte=y[te], Xte_mask=masked)


class Logger:
    def __init__(self, name):
        os.makedirs(LOGS, exist_ok=True)
        self.f = open(os.path.join(LOGS, f"{name}.log"), "w")
        self.t0 = time.time()

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        stamp = f"[{time.time() - self.t0:7.1f}s] "
        print(stamp + line, flush=True)
        self.f.write(stamp + line + "\n"); self.f.flush()


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=_np_default)


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _np_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def tree_params(d):
    """Selected tree setting as kwargs for DecisionTreeClassifier."""
    return {"max_depth": d["max_depth"], "min_samples_leaf": d["min_samples_leaf"]}
