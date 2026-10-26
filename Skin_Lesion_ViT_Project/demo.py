"""Classify one image with the trained ViT (research demo - NOT a medical device)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.models import PatchViT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input dermoscopy image")
    ap.add_argument("--weights", default="results/models/vit_ssl_best.pt")
    a = ap.parse_args()
    wp = Path(a.weights)
    if not wp.exists():
        raise SystemExit(
            f"Model weights not found at: {wp}\n"
            f"Please run RUN_ME.bat to train or use the web app to generate a model."
        )
    ck = torch.load(wp, map_location="cpu")
    cfg = ck["cfg"]
    m = PatchViT(**cfg, n_classes=len(ck["classes"]))
    m.load_state_dict(ck["state"])
    m.eval()
    im = Image.open(a.image).convert("RGB").resize((cfg["img"], cfg["img"]), Image.BILINEAR)
    x = (torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()[None] / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        p = m(x).softmax(1)[0]
    print(f"Model trained on: {ck.get('dataset', 'unknown')}")
    for i in p.argsort(descending=True):
        print(f"  {ck['classes'][i]:<6} {p[i]:.3f}")
    print("\nResearch prototype only - not a clinical diagnosis. Consult a certified dermatologist.")


if __name__ == "__main__":
    main()
