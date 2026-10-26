import copy
import numpy as np
import torch
import torch.nn as nn
from .metrics import compute_metrics
from .models import MaskedPatchPretrainer

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, *args, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def set_postfix(self, *args, **kwargs):
            pass
        def close(self):
            pass


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def to_tensor(xb, device):
    t = torch.from_numpy(xb).to(device).permute(0, 3, 1, 2).float() / 255.0
    return (t - 0.5) / 0.5


def augment(x):
    """Per-sample dihedral flips/rotations + mild brightness jitter (lesions have no canonical orientation)."""
    B = x.size(0)
    for op in (lambda z: z.flip(3), lambda z: z.flip(2), lambda z: z.transpose(2, 3)):
        m = (torch.rand(B, device=x.device) < .5).view(-1, 1, 1, 1)
        x = torch.where(m, op(x), x)
    return x * (1 + 0.08 * torch.randn(B, 1, 1, 1, device=x.device))


@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval()
    out = [model(to_tensor(X[i:i + bs], device)).argmax(1).cpu() for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def train_classifier(model, Xtr, ytr, Xva, yva, device, epochs, lr, bs=64, weighted=False,
                     wd=0.05, seed=0, k=7, desc=None):
    model.to(device)
    w = None
    if weighted:
        cnt = np.bincount(ytr, minlength=k).astype(float)
        w = torch.tensor(cnt.sum() / (k * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = max(1, int(np.ceil(len(Xtr) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    best, best_state = -1.0, None
    pbar = tqdm(total=epochs, desc=desc, unit="ep", dynamic_ncols=True) if desc else None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot_loss, nb = 0.0, 0
        n_batches = int(np.ceil(len(perm) / bs))
        for bi, i in enumerate(range(0, len(perm), bs)):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = augment(to_tensor(Xtr[b], device))
            yb = torch.from_numpy(ytr[b]).to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            nb += 1
            if pbar and (bi % 5 == 0 or bi == n_batches - 1):
                pbar.set_postfix(loss=f"{tot_loss / max(nb, 1):.3f}")
        f1 = compute_metrics(yva, predict(model, Xva, device), k)["macro_f1"]
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{tot_loss / max(nb, 1):.3f}", val_f1=f"{f1:.3f}", best=f"{best:.3f}")
    if pbar:
        pbar.close()
    model.load_state_dict(best_state)
    return model, best


def pretrain_ssl(vit, X, device, epochs, lr=1e-3, bs=128, seed=0, mask_ratio=0.6, desc=None):
    """Self-supervised pretraining on UNLABELLED training images (labels never used)."""
    mp = MaskedPatchPretrainer(vit, mask_ratio).to(device)
    opt = torch.optim.AdamW(mp.parameters(), lr=lr, weight_decay=0.05)
    rng = np.random.default_rng(seed)
    losses = []
    pbar = tqdm(total=epochs, desc=desc, unit="ep", dynamic_ncols=True) if desc else None
    for ep in range(epochs):
        mp.train()
        perm = rng.permutation(len(X))
        tot, nb = 0.0, 0
        n_batches = int(np.ceil(len(perm) / bs))
        for bi, i in enumerate(range(0, len(perm), bs)):
            b = perm[i:i + bs]
            loss = mp(augment(to_tensor(X[b], device)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(mp.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
            if pbar and (bi % 5 == 0 or bi == n_batches - 1):
                pbar.set_postfix(loss=f"{tot / max(nb, 1):.4f}")
        avg_loss = tot / max(nb, 1)
        losses.append(avg_loss)
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
    if pbar:
        pbar.close()
    return vit, losses
