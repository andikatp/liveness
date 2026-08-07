from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from src.mobilenetv4.labels import LabelSpec


class ConditionalRandomResizedCrop:
    def __init__(self, size, scale=(0.9, 1.1), ratio=(0.75, 1.3333333333333333)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio):
        width, height = img.size
        area = height * width

        log_ratio = (
            torch.log(torch.tensor(ratio[0])),
            torch.log(torch.tensor(ratio[1])),
        )
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.crop(img, i, j, h, w)

        crop_size = min(h, w)
        target_size = min(self.size[0], self.size[1])

        img_np = np.array(img)
        if len(img_np.shape) == 3:
            if crop_size < target_size:
                interpolation = cv2.INTER_LANCZOS4
            else:
                interpolation = cv2.INTER_AREA
            img_resized = cv2.resize(
                img_np, (self.size[1], self.size[0]), interpolation=interpolation
            )
            img = Image.fromarray(img_resized)
        else:
            if crop_size < target_size:
                interpolation = transforms.InterpolationMode.LANCZOS
            else:
                interpolation = transforms.InterpolationMode.BOX
            img = F.resize(img, self.size, interpolation=interpolation)

        return img


@dataclass(frozen=True)
class DataPaths:
    data_root: str
    train_json: str
    val_json: str


class JsonImageDataset(Dataset):
    def __init__(
        self,
        *,
        root_dir: str,
        json_path: str,
        label_spec: LabelSpec,
        transform: Optional[Any] = None,
        strip_prefix: Optional[str] = None,
        mode: str = "train",
    ) -> None:
        self.root_dir = root_dir
        self.json_path = json_path
        self.label_spec = label_spec
        self.transform = transform
        self.strip_prefix = strip_prefix
        self.mode = mode

        try:
            meta = pd.read_json(json_path, orient="index")
        except ValueError:
            meta = pd.read_json(json_path)

        self.meta = meta
        self.img_paths = meta.index.tolist()

        if label_spec.label_column not in meta.columns:
            raise ValueError(
                f"Label column {label_spec.label_column!r} not found in metadata columns: {list(meta.columns)[:20]}"
            )
        self.raw_labels = meta[label_spec.label_column].tolist()

        valid_labels = {0, 1, 2, 3, 7, 8, 9}
        valid_indices = [
            i for i, label in enumerate(self.raw_labels) if label in valid_labels
        ]
        if len(valid_indices) < len(self.raw_labels):
            filtered_count = len(self.raw_labels) - len(valid_indices)
            print(
                f"Filtered out {filtered_count} samples with invalid labels (5, 6, 10)"
            )
            self.meta = self.meta.iloc[valid_indices]
            self.img_paths = [self.img_paths[i] for i in valid_indices]
            self.raw_labels = [self.raw_labels[i] for i in valid_indices]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        max_retries = min(100, len(self))
        original_idx = idx

        for attempt in range(max_retries):
            rel_path = self.img_paths[idx]
            if not isinstance(rel_path, str):
                rel_path = str(rel_path)

            if self.strip_prefix and rel_path.startswith(self.strip_prefix):
                rel_path = rel_path[len(self.strip_prefix) :]

            if rel_path.startswith("Data/"):
                rel_path = rel_path[5:]

            img_full_path = os.path.join(self.root_dir, rel_path)

            try:
                image = Image.open(img_full_path).convert("RGB")
                class_id = self.label_spec.to_class_id(self.raw_labels[idx])

                if self.transform is not None:
                    image = self.transform(image)

                return image, class_id
            except (FileNotFoundError, Image.UnidentifiedImageError, OSError, IOError):
                idx = (idx + 1) % len(self)
                if idx == original_idx:
                    raise RuntimeError(
                        f"Failed to load image after {max_retries} attempts starting from index {original_idx}"
                    )

        raise RuntimeError(
            f"Failed to load image after {max_retries} attempts starting from index {original_idx}"
        )


def build_transforms(image_size: int) -> Tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            ConditionalRandomResizedCrop(
                image_size,
                scale=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.1)),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (
                    x + (torch.randn_like(x) * 0.02)
                    if torch.rand(1).item() < 0.3
                    else x
                )
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform
