"""Datasets: HAM10000 loader and a synthetic imbalanced proxy for quick pipeline checks."""
from pathlib import Path
import numpy as np

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_COUNTS = {"akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142}

# (lesion RGB, radius fraction, boundary irregularity, texture frequency) - deliberately overlapping
_PROTO = {
    0: ((200, 120, 110), .30, .10, 3), 1: ((190, 150, 140), .28, .15, 5),
    2: ((150, 110, 80), .32, .08, 2), 3: ((170, 120, 95), .18, .05, 2),
    4: ((70, 45, 40), .30, .35, 6), 5: ((120, 85, 65), .27, .05, 1),
    6: ((180, 60, 80), .15, .05, 4),
}


def make_synthetic(n=4000, img=64, seed=0):
    """Synthetic dermoscopy-like images with HAM10000-like class imbalance.
    NOT real medical data: it only exists so the whole pipeline can be verified quickly."""
    rng = np.random.default_rng(seed)
    p = np.array([HAM_COUNTS[c] for c in CLASSES], float)
    p /= p.sum()
    counts = np.maximum(24, np.round(n * p).astype(int))
    y = rng.permutation(np.repeat(np.arange(len(CLASSES)), counts))
    yy, xx = np.mgrid[0:img, 0:img] / (img - 1) * 2 - 1
    X = np.zeros((len(y), img, img, 3), np.uint8)
    for i, k in enumerate(y):
        col, rad, irr, freq = _PROTO[int(k)]
        bg = np.array([225, 185, 160]) + rng.uniform(-18, 18, 3)
        light = 1 + 0.08 * (xx * rng.uniform(-1, 1) + yy * rng.uniform(-1, 1))
        im = bg[None, None, :] * light[..., None]
        cx, cy = rng.uniform(-.12, .12, 2)
        dx, dy = xx - cx, yy - cy
        dist, th = np.hypot(dx, dy), np.arctan2(dy, dx)
        r = rad * 2 * rng.uniform(.75, 1.25)
        bnd = r * (1 + irr * np.sin(rng.integers(2, 6) * th + rng.uniform(0, 6.28)))
        m = 1 / (1 + np.exp(-(bnd - dist) / 0.04))
        c = np.array(col) + rng.uniform(-16, 16, 3)
        tex = 12 * np.sin(freq * 6 * xx + rng.uniform(0, 6.28)) * np.sin(freq * 6 * yy + rng.uniform(0, 6.28))
        im = im * (1 - m[..., None]) + (c[None, None, :] + tex[..., None]) * m[..., None]
        im += rng.normal(0, 9, im.shape)
        X[i] = np.clip(im, 0, 255).astype(np.uint8)
    return X, y.astype(np.int64), np.arange(len(y))


def load_ham10000(root, img=64):
    """Load HAM10000 (HAM10000_metadata.csv + the .jpg images anywhere under `root`)."""
    import pandas as pd
    from PIL import Image
    root = Path(root)
    cache = root / f"cache_{img}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["groups"]
    meta = pd.read_csv(root / "HAM10000_metadata.csv")
    paths = {p.stem: p for p in root.rglob("*.jpg")}
    missing = [i for i in meta.image_id if i not in paths]
    if missing:
        raise SystemExit(f"{len(missing)} images from the metadata were not found under {root}")
    X = np.zeros((len(meta), img, img, 3), np.uint8)
    try:
        from tqdm import tqdm
        meta_iter = enumerate(tqdm(meta.image_id, desc="Loading & resizing HAM10000", unit="img", dynamic_ncols=True))
    except ImportError:
        meta_iter = enumerate(meta.image_id)
    for i, iid in meta_iter:
        X[i] = np.asarray(Image.open(paths[iid]).convert("RGB").resize((img, img), Image.BILINEAR))
    y = meta.dx.map({c: i for i, c in enumerate(CLASSES)}).to_numpy().astype(np.int64)
    groups = pd.factorize(meta.lesion_id)[0]
    np.savez_compressed(cache, X=X, y=y, groups=groups)
    return X, y, groups


def split_data(y, groups, seed=0):
    """~71/14/14 train/val/test split, stratified by class and grouped by lesion (no leakage)."""
    from sklearn.model_selection import StratifiedGroupKFold
    idx = np.arange(len(y))
    rest, te = next(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(idx, y, groups))
    tr, va = next(StratifiedGroupKFold(6, shuffle=True, random_state=seed).split(rest, y[rest], groups[rest]))
    return rest[tr], rest[va], te


def subsample(idx, y, frac, seed=0, min_per_class=2):
    """Stratified label-budget subsample: keeps `frac` of each class (at least a couple of samples)."""
    if frac >= 1:
        return np.asarray(idx)
    rng = np.random.default_rng(seed)
    out = []
    for k in np.unique(y[idx]):
        members = idx[y[idx] == k]
        take = min(len(members), max(min_per_class, int(round(frac * len(members)))))
        out.append(rng.choice(members, take, replace=False))
    return np.sort(np.concatenate(out))
