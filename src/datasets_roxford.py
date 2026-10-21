"""
R-Oxford (Revisited Oxford5k) dataset for image retrieval.

Reference: Radenovic et al., "Revisiting Oxford and Paris: Large-Scale
Image Retrieval Benchmarking", CVPR 2018.

Expected directory layout:
    <root>/roxford5k/
        gnd_roxford5k.pkl
        images/
            all_souls_000000.jpg
            ...

The gnd pickle (official format) contains:
    cfg['gnd']     : list of 70 dicts; each has
                     'bbx'  -> [x1, y1, x2, y2] query crop box (floats)
                     'easy' -> list of int db indices (easy positives)
                     'hard' -> list of int db indices (hard positives)
                     'junk' -> list of int db indices (to ignore)
    cfg['imlist']  : list of 4993 database image names (no extension)
    cfg['qimlist'] : list of 70 query image names (no extension)

Difficulty protocols used for evaluation:
    Easy:   pos = easy        ; junk = hard + junk
    Medium: pos = easy + hard ; junk = junk
    Hard:   pos = hard        ; junk = easy + junk

This module returns image tensors in [0,1], shape (3, H, W), resized to a
fixed square so they can be batched and so they match the resolution used
elsewhere in the pipeline (default 518 for DINOv2 ViT/14).
"""

import os
import pickle
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from src.retrieval_preprocess import load_image_aspect


def load_roxford_gnd(root):
    """
    Load the R-Oxford ground-truth config.

    Args:
        root: path to the roxford5k directory (the one containing
              gnd_roxford5k.pkl and images/)

    Returns:
        cfg: dict with keys 'gnd', 'imlist', 'qimlist', plus convenience
             keys 'image_dir' and 'n_db', 'n_q'
    """
    gnd_path = os.path.join(root, "gnd_roxford5k.pkl")
    if not os.path.exists(gnd_path):
        raise FileNotFoundError(
            f"gnd_roxford5k.pkl not found at {gnd_path}. "
            f"Expected layout: {root}/gnd_roxford5k.pkl and {root}/images/"
        )
    with open(gnd_path, "rb") as f:
        cfg = pickle.load(f)

    cfg["image_dir"] = os.path.join(root, "images")
    cfg["n_db"] = len(cfg["imlist"])
    cfg["n_q"]  = len(cfg["qimlist"])

    if not os.path.isdir(cfg["image_dir"]):
        raise FileNotFoundError(
            f"images/ directory not found at {cfg['image_dir']}"
        )
    return cfg


def _load_image(path, image_size, bbox=None, pad="square"):
    """
    Load an image, optionally crop to a bounding box (used for queries),
    then resize to a fixed square.

    Args:
        path: image file path
        image_size: target square size (int)
        bbox: optional [x1, y1, x2, y2] in original pixel coords

    Returns:
        tensor (3, image_size, image_size) in [0, 1]
    """
    # Delegate to the shared aspect-preserving loader so the clean baseline
    # and the dataset use identical preprocessing.
    return load_image_aspect(path, image_size, bbox=bbox, pad=pad)


class ROxfordDatabase(Dataset):
    """
    The 4993 database images (in cfg['imlist'] order).

    __getitem__ returns (image_tensor, index, image_name).
    The order is canonical: index i corresponds to cfg['imlist'][i],
    which is what the gnd positive/junk index lists refer to.
    """

    def __init__(self, cfg, image_size=518, pad="square"):
        self.cfg = cfg
        self.image_size = image_size
        self.image_dir = cfg["image_dir"]
        self.names = list(cfg["imlist"])
        self.pad = pad

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = os.path.join(self.image_dir, name + ".jpg")
        img = _load_image(path, self.image_size, bbox=None, pad=self.pad)
        return img, idx, name


class ROxfordQueries(Dataset):
    """
    The 70 query images, each CROPPED to its ground-truth bounding box
    (standard R-Oxford protocol), then resized.

    __getitem__ returns (image_tensor, query_index, query_name).
    """

    def __init__(self, cfg, image_size=518, pad="square"):
        self.cfg = cfg
        self.image_size = image_size
        self.image_dir = cfg["image_dir"]
        self.qnames = list(cfg["qimlist"])
        self.gnd = cfg["gnd"]
        self.pad = pad

    def __len__(self):
        return len(self.qnames)

    def __getitem__(self, idx):
        name = self.qnames[idx]
        path = os.path.join(self.image_dir, name + ".jpg")
        bbox = self.gnd[idx]["bbx"]
        img = _load_image(path, self.image_size, bbox=bbox, pad=self.pad)
        return img, idx, name


def build_difficulty_gnd(cfg):
    """
    Pre-compute, for each query, the (positives, junk) index sets for
    each of the three difficulty protocols.

    Returns:
        dict protocol -> list (len n_q) of dicts {'ok': np.array, 'junk': np.array}
        protocol in {'easy', 'medium', 'hard'}
    """
    gnd = cfg["gnd"]
    out = {"easy": [], "medium": [], "hard": []}

    for g in gnd:
        easy = np.array(g.get("easy", []), dtype=int)
        hard = np.array(g.get("hard", []), dtype=int)
        junk = np.array(g.get("junk", []), dtype=int)

        # Easy: positives = easy ; ignore = hard + junk
        out["easy"].append({
            "ok":   easy,
            "junk": np.concatenate([hard, junk]) if len(hard) + len(junk) else np.array([], dtype=int),
        })
        # Medium: positives = easy + hard ; ignore = junk
        out["medium"].append({
            "ok":   np.concatenate([easy, hard]) if len(easy) + len(hard) else np.array([], dtype=int),
            "junk": junk,
        })
        # Hard: positives = hard ; ignore = easy + junk
        out["hard"].append({
            "ok":   hard,
            "junk": np.concatenate([easy, junk]) if len(easy) + len(junk) else np.array([], dtype=int),
        })

    return out
