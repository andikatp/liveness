"""Model configuration and creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.mobilenetv4.ft_net import FTNet


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    num_classes: int = 3
    pretrained: bool = False
    image_size: int = 224


def create_model(
    cfg: ModelConfig, *, device: Optional[torch.device] = None
) -> nn.Module:
    model = FTNet(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        input_size=cfg.image_size,
    )

    if device is not None:
        model = model.to(device)

    return model


def freeze_backbone(model: nn.Module) -> None:
    if not isinstance(model, FTNet):
        raise ValueError("Model must be FTNet")

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.ft_gen.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
