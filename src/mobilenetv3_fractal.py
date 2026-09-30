"""MobileNetV3-Fractal inference model, derived from the MIT-licensed base project.

Copyright (c) 2020 Bubbliiiing. Changes for the channel-wise fractal head are
distributed under the repository MIT License; see LICENSE and NOTICE.
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Callable, Any, Optional, List, Sequence
import os
import numpy as np
import math

class FractalPooling2D(nn.Module):
    # 对每个通道做 box-counting fractal pooling
    # 输入:  [B, C, H, W]
    # 输出:  [B, C]
    def __init__(self, box_sizes=(1, 2, 4), threshold=0.5, eps=1e-6):
        super().__init__()
        self.box_sizes = tuple(box_sizes)
        self.threshold = threshold
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # per-channel min-max normalize
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + self.eps)

        # binarize
        bmap = (x_norm >= self.threshold).float()

        xs = []
        ys = []

        for k in self.box_sizes:
            if k <= 0:
                raise ValueError("box sizes must be positive integers")
            Hk = (H // k) * k
            Wk = (W // k) * k
            if Hk == 0 or Wk == 0:
                continue

            xk = bmap[:, :, :Hk, :Wk]

            # reshape into non-overlapping kxk boxes
            xk = xk.view(B, C, Hk // k, k, Wk // k, k)
            xk = xk.amax(dim=3).amax(dim=4)   # [B, C, H//k, W//k]
            nk = xk.sum(dim=(2, 3))           # [B, C]

            xs.append(torch.full((1,), math.log(1.0 / k), device=x.device, dtype=x.dtype))
            ys.append(torch.log(nk + self.eps))

        if len(xs) < 2:
            return torch.zeros(B, C, device=x.device, dtype=x.dtype)

        x_log = torch.cat(xs)                 # [M]
        y_log = torch.stack(ys, dim=-1)       # [B, C, M]

        x_mean = x_log.mean()
        y_mean = y_log.mean(dim=-1, keepdim=True)

        num = ((x_log - x_mean) * (y_log - y_mean)).sum(dim=-1)
        den = ((x_log - x_mean) ** 2).sum() + self.eps

        D = num / den                         # [B, C]
        return D

class HybridFractalHead(nn.Module):
    # GAP + Fractal Pooling
    # 输入: [B, C, H, W]
    # 输出: [B, num_classes]
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 1280, dropout: float = 0.2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.frac = FractalPooling2D(box_sizes=(1, 2, 4), threshold=0.5)

        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap_feat = self.gap(x).flatten(1)   # [B, C]
        frac_feat = self.frac(x)            # [B, C]
        feat = torch.cat([gap_feat, frac_feat], dim=1)
        return self.classifier(feat)

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=norm_layer is None,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if activation_layer is nn.Sigmoid else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input

class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)

class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result

class MobileNetV3Fractal(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        fractal_dim: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)

        # Replace the original avgpool and classifier with HybridFractalHead
        self.head = HybridFractalHead(in_channels=lastconv_output_channels, num_classes=num_classes, hidden_dim=last_channel, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, fractal_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

def mobilenetv3_fractal(pretrained=False, num_classes=1000, fractal_dim=4, **kwargs):
    width_mult = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    reduce_divider = 1
    dilation = 1

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    model = MobileNetV3Fractal(inverted_residual_setting, last_channel, num_classes=num_classes, fractal_dim=fractal_dim, **kwargs)

    if pretrained:
        model_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_data")
        weight_path = os.path.join(model_data_path, "mobilenet_v3_small-047dcff4.pth")

        if os.path.exists(weight_path):
            print(f"Loading pretrained weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location="cpu")
        else:
            try:
                from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
                weights = MobileNet_V3_Small_Weights.DEFAULT
                state_dict = mobilenet_v3_small(weights=weights).state_dict()
                print("Loading pretrained weights from torchvision")

                os.makedirs(model_data_path, exist_ok=True)
                torch.save(state_dict, weight_path)
                print(f"Saved downloaded weights to {weight_path}")
            except ImportError:
                from torch.hub import load_state_dict_from_url
                state_dict = load_state_dict_from_url("https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth", progress=True)
                print("Loading pretrained weights from url")

                os.makedirs(model_data_path, exist_ok=True)
                torch.save(state_dict, weight_path)
                print(f"Saved downloaded weights to {weight_path}")

        # Remove mismatched classifier weights (if num_classes != 1000)
        # 既然我们替换了分类器，需要完全丢弃原版 classifier 的权重
        state_dict.pop("classifier.0.weight", None)
        state_dict.pop("classifier.0.bias", None)
        state_dict.pop("classifier.3.weight", None)
        state_dict.pop("classifier.3.bias", None)

        model_dict = model.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in state_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print(f"\nSuccessful Load Key Num in MobileNetV3Fractal: {len(load_key)}")
        print(f"Fail To Load Key num in MobileNetV3Fractal: {len(no_load_key)}")
        if len(no_load_key) > 0:
            print(f"Failed keys: {no_load_key[:10]}...")

    return model
