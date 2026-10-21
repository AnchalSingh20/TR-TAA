"""
Default hyperparameters and class lists.

The PGD settings follow the Task-Agnostic Attack of Pulfer et al.
(CVPRW 2025) and are shared by TAA and TR-TAA. Every script also exposes
them as command-line arguments.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class AttackConfig:
    """PGD hyperparameters (Section 3.2 of the TAA paper)."""
    # PGD parameters (from the TAA paper: "We set the step size α = 0.0004,
    # the total number of optimization steps to 50, and ε∞ = 8/255.")
    step_size: float = 0.0004          # α in Algorithm 1
    num_steps: int = 50                 # N in Algorithm 1
    eps_inf: float = 8.0 / 255.0       # ε∞ - maximum per-pixel L∞ perturbation
    target_psnr: float = 40.0          # Target PSNR in dB for fair comparison


@dataclass
class DataConfig:
    """Dataset paths (scripts take them as command-line arguments)."""
    pascalvoc_root: str = "/path/to/VOCdevkit/VOC2012"
    imagenette_root: str = "/path/to/imagenette2"

    # Number of training samples for computing mean vector (μ)
    num_mean_samples: int = 5000

    # Number of workers for data loading
    num_workers: int = 4
    batch_size: int = 32


@dataclass
class ModelConfig:
    """Model configurations matching Table 1 of the paper."""
    # Backbones used in the experiments
    table4_models: List[str] = field(default_factory=lambda: [
        "dinov2_vits14",
        "dinov2_vitb14",
    ])

    # Feature dimensions
    feature_dims: dict = field(default_factory=lambda: {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    })

    # Patch sizes
    patch_sizes: dict = field(default_factory=lambda: {
        "dinov2_vits14": 14,
        "dinov2_vitb14": 14,
        "dinov2_vitl14": 14,
        "dinov2_vitg14": 14,
    })


@dataclass
class ExperimentConfig:
    attack: AttackConfig = field(default_factory=AttackConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs"


# ============================================================
# PascalVOC class mappings
# ============================================================
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]
NUM_VOC_CLASSES = 21  # 20 classes + background

# ImageNette class mappings (subset of ImageNet)
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute"
]
NUM_IMAGENETTE_CLASSES = 10
