"""30-second install/code check: builds every model and runs one training step of each."""
import numpy as np, torch
from src.models import PatchViT, SmallCNN, MaskedPatchPretrainer
from src.data import make_synthetic, split_data, subsample
from src.train import to_tensor, augment, train_classifier, predict, pretrain_ssl

print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
X, y, g = make_synthetic(300, 64, 0)
tr, va, te = split_data(y, g, 0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = PatchViT(img=64, patch=8, dim=128, depth=4, heads=4)
x = augment(to_tensor(X[:8], dev)); vit.to(dev)
assert vit(x).shape == (8, 7), "ViT output shape wrong"
vit, losses = pretrain_ssl(vit, X[tr], dev, epochs=1)
print("masked pre-training step OK, loss", round(losses[0], 3))
for name, m in (("ViT", vit), ("MobileNetV2", SmallCNN(7))):
    m, best = train_classifier(m, X[tr], y[tr], X[va], y[va], dev, epochs=1, lr=1e-3)
    p = predict(m, X[te], dev)
    print(f"{name}: train+predict OK (val macro-F1 {best:.3f})")
print("SELF-TEST PASSED - you can run the full project.")
