"""Explicit TorchVision weight versions, local-only loading after download."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
DINOV2_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"

CATALOG = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        960,
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weights(directory, manifest_path):
    """Only official PyTorch URLs; verify publisher's hash prefix and save full hash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "Verified upstream pretraining only; no plant-stress training",
        "models": {},
    }
    for name, (_, weights, dim) in CATALOG.items():
        filename = weights.url.rsplit("/", 1)[1]
        path = directory / filename
        prefix = filename.rsplit("-", 1)[1].split(".")[0]
        if not path.exists():
            torch.hub.download_url_to_file(
                weights.url, str(path), hash_prefix=prefix, progress=False
            )
        digest = sha256(path)
        if not digest.startswith(prefix):
            raise ValueError(f"Checksum mismatch: {path}")
        manifest["models"][name] = {
            "weights": str(weights),
            "url": weights.url,
            "filename": filename,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "embedding_dim": dim,
            "pretraining": "ImageNet-1K",
        }
        print(f"Verified {name}: {path.stat().st_size:,} bytes", flush=True)
    dino_path = directory / DINOV2_URL.rsplit("/", 1)[1]
    if not dino_path.exists():
        torch.hub.download_url_to_file(DINOV2_URL, str(dino_path), progress=False)
    if sha256(dino_path) != DINOV2_SHA256:
        raise ValueError(f"DINOv2 checkpoint checksum mismatch: {dino_path}")
    manifest["models"]["dinov2_vits14"] = {
        "weights": "Meta DINOv2 ViT-S/14 backbone only",
        "url": DINOV2_URL,
        "filename": dino_path.name,
        "sha256": DINOV2_SHA256,
        "bytes": dino_path.stat().st_size,
        "embedding_dim": 384,
        "pretraining": "LVD-142M; Apache-2.0 model card",
    }
    print(f"Verified dinov2_vits14: {dino_path.stat().st_size:,} bytes", flush=True)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_backbone(name, directory, num_classes=None):
    """No implicit network access. Five-class heads are newly initialized."""
    constructor, weights, _ = CATALOG[name]
    path = Path(directory) / weights.url.rsplit("/", 1)[1]
    prefix = path.stem.rsplit("-", 1)[1]
    if not path.is_file():
        raise FileNotFoundError(f"Run scripts/download_models.py first: {path}")
    if not sha256(path).startswith(prefix):
        raise ValueError(f"Corrupted checkpoint: {path}")
    model = constructor(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    if num_classes is not None:
        index = {"efficientnet_b0": 1, "convnext_tiny": 2, "mobilenet_v3_large": 3}[name]
        model.classifier[index] = nn.Linear(model.classifier[index].in_features, num_classes)
    return model


def load_dinov2(directory):
    """Frozen official DINOv2 ViT-S/14 from a full-SHA verified local checkpoint."""
    import timm

    path = Path(directory) / DINOV2_URL.rsplit("/", 1)[1]
    if not path.is_file():
        raise FileNotFoundError(f"Run scripts/download_models.py first: {path}")
    if sha256(path) != DINOV2_SHA256:
        raise ValueError(f"DINOv2 checkpoint checksum mismatch: {path}")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Official hub model has a mask token absent from timm's inference graph.
    state.pop("mask_token")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def dinov2_transform():
    """Deterministic 518px preprocessing, divisible by the ViT's 14px patch."""
    return image_transform(training=False, size=518)


def embeddings(model, images):
    """Caller controls eval/train mode and gradient tracking."""
    features = model.avgpool(model.features(images)).flatten(1)
    if isinstance(model, models.ConvNeXt):
        features = model.classifier[0](features[:, :, None, None]).flatten(1)
    return features


def image_transform(training=False, size=384):
    ops = [transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC)]
    if training:
        ops = [
            transforms.RandomResizedCrop(
                size, scale=(0.85, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        ]
    return transforms.Compose(
        ops
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class GatedCrossAttention(nn.Module):
    """F3 comparison head; random initialization, not a reproduced paper checkpoint.

    Input: image tokens (batch, patches, image_dim), spectral vector (batch, bands).
    """

    def __init__(self, image_dim=1280, spectral_dim=21, hidden=64, classes=5):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, hidden)
        self.spectral_projection = nn.Linear(1, hidden)
        self.band_embedding = nn.Parameter(torch.randn(1, spectral_dim, hidden) * 0.02)
        self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.output = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, classes)
        )

    def forward(self, image_tokens, spectra):
        image = self.image_projection(image_tokens)
        spectral = self.spectral_projection(spectra.unsqueeze(-1)) + self.band_embedding
        attended, _ = self.attention(image, spectral, spectral, need_weights=False)
        original, fused = image.mean(1), attended.mean(1)
        gate = self.gate(torch.cat((original, fused), dim=-1))
        return self.output(gate * fused + (1 - gate) * original)
