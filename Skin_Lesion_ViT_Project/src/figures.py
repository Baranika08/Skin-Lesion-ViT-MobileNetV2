"""Black/white publication-style figures. Needs only numpy + matplotlib."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": "black",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
NAMES = {"cnn": "MobileNetV2 baseline", "mobilenet_v2": "MobileNetV2 baseline",
         "vit_scratch": "ViT (scratch)", "vit_ssl": "ViT + masked pretraining",
         "vit_ssl_weighted": "ViT + masked pretr. + weighted loss"}
STYLE = {"cnn": ("s", ":", "#777777", ""), "mobilenet_v2": ("s", ":", "#777777", ""),
         "vit_scratch": ("o", "--", "#000000", ""),
         "vit_ssl": ("^", "-", "#000000", "//"), "vit_ssl_weighted": ("D", "-.", "#444444", "xx")}


def agg(rows, method, frac, key="macro_f1"):
    v = [r[key] for r in rows if r["method"] == method and abs(r["fraction"] - frac) < 1e-9]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def fig_samples(X, y, classes, path, per=6):
    fig, ax = plt.subplots(len(classes), per, figsize=(per * .9, len(classes) * .9))
    rng = np.random.default_rng(0)
    for k, c in enumerate(classes):
        idx = rng.choice(np.where(y == k)[0], per, replace=False)
        for j in range(per):
            ax[k, j].imshow(X[idx[j]]); ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
            for s in ax[k, j].spines.values():
                s.set_visible(True)
        ax[k, 0].set_ylabel(c, rotation=0, ha="right", va="center")
    fig.savefig(path); plt.close(fig)


def fig_class_dist(meta, path):
    k = meta["classes"]; x = np.arange(len(k)); w = .38
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x - w / 2, meta["class_counts_train"], w, color="black", label="Train")
    ax.bar(x + w / 2, meta["class_counts_test"], w, color="white", edgecolor="black", hatch="//", label="Test")
    ax.set_xticks(x); ax.set_xticklabels(k); ax.set_ylabel("Images"); ax.set_yscale("log")
    ax.legend(frameon=False); fig.savefig(path); plt.close(fig)


def fig_macro_f1(res, path):
    rows, fr = res["rows"], res["meta"]["fractions"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m in res["meta"]["methods"]:
        mk, ls, col, _ = STYLE[m]
        mu = [agg(rows, m, f)[0] for f in fr]; sd = [agg(rows, m, f)[1] for f in fr]
        ax.errorbar([f * 100 for f in fr], mu, yerr=sd, marker=mk, ls=ls, color=col, capsize=3,
                    lw=1.4, ms=5, mfc="white" if m in ("cnn", "mobilenet_v2") else col, label=NAMES[m])
    ax.set_xscale("log"); ax.set_xticks([f * 100 for f in fr]); ax.set_xticklabels([f"{f*100:g}%" for f in fr])
    ax.set_xlabel("Labelled training data used"); ax.set_ylabel("Test macro-F1")
    ax.grid(axis="y", color="#cccccc", lw=.5); ax.legend(frameon=False, fontsize=8)
    fig.savefig(path); plt.close(fig)


def fig_recall(res, path):
    rows, meta = res["rows"], res["meta"]
    f0 = min(meta["fractions"]); cls = meta["classes"]; ms = meta["methods"]
    fig, ax = plt.subplots(figsize=(7, 3.6)); w = .8 / len(ms); x = np.arange(len(cls))
    shades = ["#bbbbbb", "#888888", "#000000", "#ffffff"]
    for i, m in enumerate(ms):
        rec = np.mean([r["recall"] for r in rows if r["method"] == m and abs(r["fraction"] - f0) < 1e-9], 0)
        ax.bar(x + (i - (len(ms) - 1) / 2) * w, rec, w, color=shades[i % 4], edgecolor="black",
               hatch=STYLE[m][3], label=NAMES[m], lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(cls); ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Test recall ({f0*100:g}% labels)"); ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(path); plt.close(fig)


def fig_confusion(res, path):
    rows, meta = res["rows"], res["meta"]; f0 = min(meta["fractions"]); cls = meta["classes"]
    pair = [m for m in ("vit_scratch", "vit_ssl") if m in meta["methods"]]
    fig, axs = plt.subplots(1, len(pair), figsize=(4.4 * len(pair), 4))
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, pair):
        C = np.sum([r["confusion"] for r in rows if r["method"] == m and abs(r["fraction"] - f0) < 1e-9], 0).astype(float)
        Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
        ax.imshow(Cn, cmap="Greys", vmin=0, vmax=1)
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, f"{Cn[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Cn[i, j] > .5 else "black")
        ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=45); ax.set_yticks(range(len(cls)))
        ax.set_yticklabels(cls); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(NAMES[m], fontsize=9)
    fig.savefig(path); plt.close(fig)


def fig_pretrain(meta, path):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(meta["pretrain_loss"]) + 1), meta["pretrain_loss"], color="black", lw=1.4)
    ax.set_xlabel("Pre-training epoch"); ax.set_ylabel("Masked-patch MSE"); ax.grid(color="#cccccc", lw=.5)
    fig.savefig(path); plt.close(fig)


def make_result_figures(res, outdir):
    from pathlib import Path
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    fig_class_dist(res["meta"], out / "fig_class_distribution.png")
    fig_macro_f1(res, out / "fig_macro_f1.png")
    fig_recall(res, out / "fig_recall.png")
    fig_confusion(res, out / "fig_confusion.png")
    if res["meta"].get("pretrain_loss"):
        fig_pretrain(res["meta"], out / "fig_pretrain_loss.png")
