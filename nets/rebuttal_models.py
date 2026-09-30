"""Model variants used by the IEEE Sensors Journal rebuttal experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mobilenetv3 import mobilenet_v3_small
from .mobilenetv3_fractal import mobilenetv3_fractal


class FractalOnlyHead(nn.Module):
    def __init__(self, source_head: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.frac = source_head.frac
        hidden_dim = source_head.classifier[0].out_features
        in_channels = source_head.classifier[0].in_features // 2
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.frac(x))


class GatedFractalHead(nn.Module):
    def __init__(self, source_head: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.gap = source_head.gap
        self.frac = source_head.frac
        in_channels = source_head.classifier[0].in_features // 2
        hidden_dim = source_head.classifier[0].out_features
        self.gate = nn.Sequential(nn.Linear(in_channels * 2, in_channels), nn.Hardsigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = self.gap(x).flatten(1)
        frac = self.frac(x)
        alpha = self.gate(torch.cat([gap, frac], dim=1))
        return self.classifier(alpha * gap + (1.0 - alpha) * frac)


class ParameterMatchedGAPHead(nn.Module):
    """GAP head whose classifier parameter count matches a hybrid head.

    The fractal descriptor itself has no trainable parameters, but concatenating
    it with GAP doubles the first classifier input. This head feeds two copies
    of the same GAP vector into an identically sized classifier, preventing a
    parameter-count increase from being mistaken for a fractal-branch gain.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = self.gap(x).flatten(1)
        return self.classifier(torch.cat([gap, gap], dim=1))


class WideGAPHead(nn.Module):
    """GAP-only control with a genuinely wider trainable hidden layer.

    The hidden width is chosen so that the total model parameter count is
    within a few parameters of the GAP+Fractal head.  Unlike the historical
    duplicated-GAP control, every input to this classifier is an independent
    learned projection of the GAP descriptor.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 2029,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = self.gap(x).flatten(1)
        return self.classifier(gap)


class GAPGMPHead(nn.Module):
    """GAP+GMP control using the same two-descriptor classifier as Fractal."""

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 1024,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = self.gap(x).flatten(1)
        gmp = self.gmp(x).flatten(1)
        return self.classifier(torch.cat([gap, gmp], dim=1))


class PenultimateFractalHead(nn.Module):
    """Fuse GAP from the final feature map with fractal statistics from the
    penultimate feature map, keeping branch-location comparisons controlled."""

    def __init__(self, final_channels: int, penultimate_channels: int,
                 num_classes: int, hidden_dim: int = 1280, dropout: float = 0.2,
                 box_sizes=(1, 2, 4), threshold: float = 0.5) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        from .mobilenetv3_fractal import FractalPooling2D
        self.frac = FractalPooling2D(box_sizes=box_sizes, threshold=threshold)
        self.classifier = nn.Sequential(
            nn.Linear(final_channels + penultimate_channels, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, final_features: torch.Tensor, penultimate_features: torch.Tensor) -> torch.Tensor:
        gap = self.gap(final_features).flatten(1)
        frac = self.frac(penultimate_features)
        return self.classifier(torch.cat([gap, frac], dim=1))


class PenultimateFractalModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.features = base_model.features
        final_channels = base_model.head.classifier[0].in_features // 2
        penultimate_channels = self.features[-2].out_channels
        hidden_dim = base_model.head.classifier[0].out_features
        self.head = PenultimateFractalHead(final_channels, penultimate_channels, num_classes, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, fractal_features=None) -> torch.Tensor:
        penultimate = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == len(self.features) - 2:
                penultimate = x
        if penultimate is None:
            raise RuntimeError("Feature extractor has no penultimate activation")
        return self.head(x, penultimate)

    def freeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def Unfreeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = True


def mobilenetv3_gap(pretrained=False, progress=True, num_classes=1000, **kwargs):
    """Vanilla MobileNetV3-Small with the original GAP classifier."""
    return mobilenet_v3_small(pretrained=pretrained, progress=progress, num_classes=num_classes)


def mobilenetv3_gap_matched(pretrained=False, progress=True, num_classes=1000, **kwargs):
    """Historical duplicated-GAP control retained for audit only."""
    model = mobilenetv3_fractal(pretrained=pretrained, num_classes=num_classes, **kwargs)
    in_channels = model.head.classifier[0].in_features // 2
    hidden_dim = model.head.classifier[0].out_features
    model.head = ParameterMatchedGAPHead(in_channels, num_classes, hidden_dim)
    return model


def mobilenetv3_gap_wide(pretrained=False, progress=True, num_classes=1000, **kwargs):
    """GAP-only baseline with parameter-matched trainable classifier width."""
    model = mobilenetv3_fractal(pretrained=pretrained, num_classes=num_classes, **kwargs)
    in_channels = model.head.classifier[0].in_features // 2
    model.head = WideGAPHead(in_channels, num_classes, hidden_dim=2029)
    return model


def mobilenetv3_gap_gmp(pretrained=False, progress=True, num_classes=1000, **kwargs):
    """GAP+GMP control with the same classifier capacity as GAP+Fractal."""
    model = mobilenetv3_fractal(pretrained=pretrained, num_classes=num_classes, **kwargs)
    in_channels = model.head.classifier[0].in_features // 2
    hidden_dim = model.head.classifier[0].out_features
    model.head = GAPGMPHead(in_channels, num_classes, hidden_dim=hidden_dim)
    return model


def mobilenetv3_fractal_only(pretrained=False, progress=True, num_classes=1000, **kwargs):
    model = mobilenetv3_fractal(pretrained=pretrained, num_classes=num_classes, **kwargs)
    model.head = FractalOnlyHead(model.head, num_classes)
    return model


def mobilenetv3_gated_fractal(pretrained=False, progress=True, num_classes=1000, **kwargs):
    model = mobilenetv3_fractal(pretrained=pretrained, num_classes=num_classes, **kwargs)
    model.head = GatedFractalHead(model.head, num_classes)
    return model


def mobilenetv3_fractal_penultimate(pretrained=False, progress=True, num_classes=1000, **kwargs):
    base = mobilenetv3_fractal(pretrained=pretrained, num_classes=num_classes, **kwargs)
    return PenultimateFractalModel(base, num_classes)
