"""Run the full experiment: CNN vs ViT(scratch) vs ViT+masked pretraining across label budgets."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from src.data import CLASSES, make_synthetic, load_ham10000, split_data, subsample
from src.models import PatchViT, MobileNetV2, SmallCNN, n_params
from src.train import set_seed, to_tensor, predict, train_classifier, pretrain_ssl
from src.metrics import compute_metrics
from src.figures import fig_samples, make_result_figures

METHODS = {"mobilenet_v2": (2e-3, False), "vit_scratch": (1e-3, False),
           "vit_ssl": (5e-4, False), "vit_ssl_weighted": (5e-4, True)}
PRESETS = {  # epochs, pretrain epochs, seeds, fractions, synthetic size
    "quick": dict(epochs=6, pre=6, seeds=1, fractions=[0.25, 1.0], n=1500),
    "cpu": dict(epochs=15, pre=15, seeds=2, fractions=[0.1, 0.5, 1.0], n=2500),
    "standard": dict(epochs=40, pre=60, seeds=3, fractions=[0.1, 0.25, 0.5, 1.0], n=4000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "ham10000"], default="synthetic")
    ap.add_argument("--data-root", default="data/HAM10000")
    ap.add_argument("--preset", choices=["auto"] + list(PRESETS), default="auto",
                    help="auto = standard on GPU, cpu preset on CPU")
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.preset == "auto":
        a.preset = "standard" if device.type == "cuda" else "cpu"
    P = PRESETS[a.preset]
    print(f"Device: {device} | preset: {a.preset} | dataset: {a.dataset}", flush=True)
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "models").mkdir(exist_ok=True)
    t0 = time.time()

    if a.dataset == "ham10000":
        X, y, groups = load_ham10000(a.data_root, a.img)
        label = "HAM10000 (Tschandl et al., 2018), resized to %dx%d" % (a.img, a.img)
    else:
        X, y, groups = make_synthetic(P["n"], a.img, 0)
        label = "Synthetic imbalanced proxy dataset (NOT real skin lesions), %dx%d" % (a.img, a.img)
    k = len(CLASSES)
    fig_samples(X, y, CLASSES, out / "figures" / "fig_samples.png")
    vit_cfg = dict(img=a.img, patch=a.patch, dim=128, depth=4, heads=4)
    meta = dict(dataset=a.dataset, dataset_label=label, preset=a.preset, img=a.img, patch=a.patch,
                epochs=P["epochs"], pretrain_epochs=P["pre"], seeds=list(range(P["seeds"])),
                fractions=P["fractions"], methods=list(METHODS), classes=CLASSES, vit_cfg=vit_cfg,
                device=str(device), torch=torch.__version__, n_total=int(len(y)),
                params={"vit": n_params(PatchViT(**vit_cfg)), "mobilenet_v2": n_params(MobileNetV2(k))})
    rows = []

    def save(final=False):
        meta["runtime_min"] = round((time.time() - t0) / 60, 1)
        (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows, "complete": final}))

    total_runs = len(meta["seeds"]) * (1 + len(P["fractions"]) * len(METHODS))
    run_idx = 0
    print(f"\nStarting {total_runs} training tasks ({len(meta['seeds'])} seed(s), {len(P['fractions'])} data split(s), {len(METHODS)} method(s))\n", flush=True)

    for seed in meta["seeds"]:
        tr, va, te = split_data(y, groups, seed)
        if seed == 0:
            meta.update(n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
                        class_counts_train=np.bincount(y[tr], minlength=k).tolist(),
                        class_counts_test=np.bincount(y[te], minlength=k).tolist())
        set_seed(seed)
        run_idx += 1
        ssl_desc = f"[{run_idx}/{total_runs}] SSL Pretrain (s={seed})"
        print(f"--- Task {run_idx}/{total_runs}: Masked pre-training on {len(tr)} unlabelled images (seed {seed}, {P['pre']} epochs) ---", flush=True)
        vit, losses = pretrain_ssl(PatchViT(**vit_cfg, n_classes=k), X[tr], device, P["pre"], seed=seed, desc=ssl_desc)
        ssl_state = {kk: v.detach().cpu().clone() for kk, v in vit.state_dict().items()}
        if seed == 0:
            meta["pretrain_loss"] = losses
        print(f"    Completed SSL pretraining | final loss: {losses[-1]:.4f}\n", flush=True)

        Xte, yte = X[te], y[te]
        for frac in P["fractions"]:
            tri, vai = subsample(tr, y, frac, seed), subsample(va, y, frac, seed)
            ep = int(np.ceil(P["epochs"] * min(4.0, 1.0 / frac)))   # keep optimiser steps comparable
            pct_label = f"{int(frac * 100)}%" if frac < 1.0 else "100%"
            for name, (lr, weighted) in METHODS.items():
                run_idx += 1
                clf_desc = f"[{run_idx}/{total_runs}] {name} (s={seed}, {pct_label})"
                set_seed(seed)
                if name in ("mobilenet_v2", "cnn"):
                    model = MobileNetV2(k)
                else:
                    model = PatchViT(**vit_cfg, n_classes=k)
                    if name.startswith("vit_ssl"):
                        model.load_state_dict(ssl_state); model.head.reset_parameters()
                model, best = train_classifier(model, X[tri], y[tri], X[vai], y[vai], device, ep, lr,
                                               weighted=weighted, seed=seed, k=k, desc=clf_desc)
                m = compute_metrics(yte, predict(model, Xte, device), k)
                rows.append(dict(method=name, fraction=frac, seed=seed, n_train_labeled=int(len(tri)),
                                 val_macro_f1=float(best), **m))
                print(f"  [{run_idx}/{total_runs}] seed={seed} frac={frac:<4} {name:<17} "
                      f"test macro-F1={m['macro_f1']:.3f} acc={m['acc']:.3f} (val_f1={best:.3f})", flush=True)
                if name == "vit_ssl" and frac == max(P["fractions"]) and seed == 0:
                    torch.save({"state": model.state_dict(), "cfg": vit_cfg, "classes": CLASSES,
                                "dataset": a.dataset}, out / "models" / "vit_ssl_best.pt")
                save()
    save(final=True)
    make_result_figures(json.loads((out / "results.json").read_text()), out / "figures")
    print(f"\nDone in {meta['runtime_min']} min. Results: {out/'results.json'}")


if __name__ == "__main__":
    main()
