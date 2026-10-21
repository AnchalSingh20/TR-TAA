"""
VQAv2 dataset loader.

Reference: Goyal et al., "Making the V in VQA Matter", CVPR 2017.

Expected directory layout:
    <root>/
        val2014/                                       40,504 COCO val2014 JPGs
            COCO_val2014_000000000042.jpg
            ...
        v2_OpenEnded_mscoco_val2014_questions.json
        v2_mscoco_val2014_annotations.json

JSON schema (standard VQAv2 format):
  questions.json:
    {
      "questions": [
        {"question_id": 458752000, "image_id": 458752,
         "question": "What is this photo taken looking through?"},
        ...
      ]
    }
  annotations.json:
    {
      "annotations": [
        {"question_id": 458752000, "image_id": 458752,
         "question_type": "what is this",
         "answer_type": "other",                # one of {"yes/no","number","other"}
         "multiple_choice_answer": "net",        # the consensus answer
         "answers": [                            # 10 human-annotated answers
           {"answer": "net", "answer_id": 1, "answer_confidence": "maybe"},
           ...
         ]},
        ...
      ]
    }

For the VQA attack experiment we:
  1. Subsample N images (default 5000) from val2014 with a fixed seed
  2. For each sampled image, expose all of its questions (typically 3)
  3. Provide ground-truth answer lists for VQA scoring

Total dataset: 40,504 images x ~3 questions = ~120K questions in val.
At 5000 image subsample x ~3 questions = ~15K questions.
"""

import os
import json
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# Filenames used by the VQAv2 release
QUESTIONS_FN = "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_FN = "v2_mscoco_val2014_annotations.json"
IMAGE_DIR = "val2014"

# COCO image filename pattern: COCO_val2014_<12-digit zero-padded ID>.jpg
def coco_image_filename(image_id):
    return f"COCO_val2014_{int(image_id):012d}.jpg"


def load_questions_and_annotations(root):
    """
    Load the two JSON files, return:
        questions_by_id: dict {question_id: dict}
        annotations_by_id: dict {question_id: dict}
        image_to_qids: dict {image_id: list of question_ids}
    """
    qpath = os.path.join(root, QUESTIONS_FN)
    apath = os.path.join(root, ANNOTATIONS_FN)
    for p in (qpath, apath):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"VQAv2 file missing: {p}\n"
                f"Expected layout: {root}/{QUESTIONS_FN}, "
                f"{root}/{ANNOTATIONS_FN}, {root}/{IMAGE_DIR}/*.jpg"
            )

    with open(qpath, "r") as f:
        qjson = json.load(f)
    with open(apath, "r") as f:
        ajson = json.load(f)

    questions_by_id = {q["question_id"]: q for q in qjson["questions"]}
    annotations_by_id = {a["question_id"]: a for a in ajson["annotations"]}

    image_to_qids = defaultdict(list)
    for q in qjson["questions"]:
        image_to_qids[q["image_id"]].append(q["question_id"])

    return questions_by_id, annotations_by_id, dict(image_to_qids)


def subsample_images(image_to_qids, n_images, seed=42):
    """
    Randomly subsample n_images IMAGE IDs (with replacement=False).

    Returns:
        list of selected image_ids, sorted for determinism
    """
    all_image_ids = sorted(image_to_qids.keys())
    if n_images >= len(all_image_ids):
        return all_image_ids
    rng = random.Random(seed)
    chosen = rng.sample(all_image_ids, n_images)
    return sorted(chosen)


class VQAv2Subsample(Dataset):
    """
    Image-level VQAv2 subsample. One Dataset item = one image, with its
    list of associated questions and ground-truth answers.

    __getitem__ returns:
        image: (3, image_size, image_size) float tensor in [0,1]
        image_id: int
        question_ids: list of int
        questions: list of str (question texts)
        gt_answers: list of list of str (10 GT answers per question)
        answer_types: list of str ('yes/no'|'number'|'other'); for scoring

    Args:
        root: directory containing val2014/, questions.json, annotations.json
        n_images: subsample size (default 5000)
        seed: subsample seed (default 42)
        image_size: target square size for the input image (default 384,
                    matching SigLip's expected input)
    """

    def __init__(self, root, n_images=5000, seed=42, image_size=384):
        self.root = root
        self.image_size = image_size
        self.image_dir = os.path.join(root, IMAGE_DIR)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"COCO image directory not found: {self.image_dir}. "
                f"Did you unzip val2014.zip?"
            )

        questions_by_id, annotations_by_id, image_to_qids = (
            load_questions_and_annotations(root)
        )

        # Subsample images by ID
        chosen_image_ids = subsample_images(image_to_qids, n_images, seed)

        # Build the per-image bundle list
        self.items = []
        for img_id in chosen_image_ids:
            qids = sorted(image_to_qids[img_id])
            entries = []
            for qid in qids:
                q = questions_by_id[qid]
                a = annotations_by_id[qid]
                entries.append({
                    "question_id": qid,
                    "question": q["question"],
                    "answers": [ans["answer"] for ans in a["answers"]],
                    "answer_type": a["answer_type"],
                    "multiple_choice_answer": a["multiple_choice_answer"],
                })
            self.items.append({"image_id": img_id, "entries": entries})

        # Diagnostics
        n_questions = sum(len(it["entries"]) for it in self.items)
        type_counts = defaultdict(int)
        for it in self.items:
            for e in it["entries"]:
                type_counts[e["answer_type"]] += 1
        print(f"[VQAv2] subsample={n_images} (seed={seed}): "
              f"{len(self.items)} images, {n_questions} questions")
        print(f"        answer types: "
              f"yes/no={type_counts['yes/no']}, "
              f"number={type_counts['number']}, "
              f"other={type_counts['other']}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_id = it["image_id"]
        path = os.path.join(self.image_dir, coco_image_filename(img_id))
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)              # (3, H, W) in [0,1]

        question_ids   = [e["question_id"]           for e in it["entries"]]
        questions      = [e["question"]              for e in it["entries"]]
        gt_answers     = [e["answers"]               for e in it["entries"]]
        answer_types   = [e["answer_type"]           for e in it["entries"]]

        return img_tensor, img_id, question_ids, questions, gt_answers, answer_types


def collate_one_image(batch):
    """
    DataLoader collate_fn for batch_size=1 (each image has variable #questions
    so we can't stack questions across images naively). For attack generation
    you want batch_size>1 over IMAGES; this collate stacks images but keeps
    questions as nested lists.
    """
    images       = torch.stack([b[0] for b in batch], dim=0)
    image_ids    = [b[1] for b in batch]
    question_ids = [b[2] for b in batch]
    questions    = [b[3] for b in batch]
    gt_answers   = [b[4] for b in batch]
    answer_types = [b[5] for b in batch]
    return images, image_ids, question_ids, questions, gt_answers, answer_types
