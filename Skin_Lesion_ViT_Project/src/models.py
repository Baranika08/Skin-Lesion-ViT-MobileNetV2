"""Compact Vision Transformer, masked-patch pretraining head, and MobileNetV2 CNN baseline."""
import torch
import torch.nn as nn

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class PatchViT(nn.Module):
    def __init__(self, img=64, patch=8, dim=128, depth=4, heads=4, n_classes=7, drop=0.1):
        super().__init__()
        assert img % patch == 0
        self.patch, self.n = patch, (img // patch) ** 2
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=.02)
        nn.init.trunc_normal_(self.cls, std=.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def tokens(self, x):                      # (B,3,H,W) -> (B,N,D)
        return self.proj(x).flatten(2).transpose(1, 2)

    def encode(self, t):                      # (B,N,D) patch tokens -> (B,N+1,D)
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], 1) + self.pos
        return self.norm(self.enc(t))

    def forward(self, x):
        return self.head(self.encode(self.tokens(x))[:, 0])


class MaskedPatchPretrainer(nn.Module):
    """SimMIM-style self-supervised pretraining: hide patches, reconstruct their (normalised) pixels."""
    def __init__(self, vit, mask_ratio=0.6):
        super().__init__()
        dim = vit.head.in_features
        self.vit, self.ratio = vit, mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dec = nn.Linear(dim, vit.patch * vit.patch * 3)

    def forward(self, x):
        B, p = x.size(0), self.vit.patch
        t = self.vit.tokens(x)
        N = t.size(1)
        mask = torch.rand(B, N, device=x.device) < self.ratio
        t = torch.where(mask.unsqueeze(-1), self.mask_token.expand(B, N, -1), t)
        pred = self.dec(self.vit.encode(t)[:, 1:])
        tgt = x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 4, 5, 1).reshape(B, N, p * p * 3)
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.var(-1, keepdim=True) + 1e-6).sqrt()
        loss = ((pred - tgt) ** 2).mean(-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1)


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2 (Sandler et al., 2018)."""
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 CNN architecture with inverted residuals and linear bottlenecks."""
    def __init__(self, n_classes=7, pretrained=False, width_mult=1.0):
        super().__init__()
        if HAS_TORCHVISION:
            weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.net = tv_models.mobilenet_v2(weights=weights)
            self.net.classifier[1] = nn.Linear(self.net.last_channel, n_classes)
        else:
            input_channel = int(32 * width_mult)
            last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            features = [
                nn.Sequential(
                    nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True),
                )
            ]
            for t, c, n, s in inverted_residual_setting:
                output_channel = int(c * width_mult)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            features.extend([
                nn.Sequential(
                    nn.Conv2d(input_channel, last_channel, 1, bias=False),
                    nn.BatchNorm2d(last_channel),
                    nn.ReLU6(inplace=True),
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            ])
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(last_channel, n_classes),
            )
            self.net = nn.Sequential(self.features, self.classifier)

    def forward(self, x):
        return self.net(x)


# Backward-compatible alias
SmallCNN = MobileNetV2


def n_params(m):
    return sum(p.numel() for p in m.parameters())
