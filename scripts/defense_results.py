"""
Input-preprocessing defense numbers used by the two defense tables,
Figure 5 and Figure 6. Everything is read from the result files and the
degradations are computed here, so figures and tables share one source.

Degradation = metric(defended clean) - metric(defended adversarial), in pp.
Retained strength = degradation under a defense / undefended degradation.

SOURCES (ViT-B/14; directory: $TR_TAA_RESULTS, default ./results)
  Classification  multi-label mAP, full VOC val (n=5823)
                  rescore_multilabel_table16.pt
                  rescore_multilabel_jpegsweep_cls.pt   (q=90/30, Figure 5 only)
                  (the 'cls' field of defenses_voc_*.pt is single-label
                  top-1 accuracy and is not used)
  Segmentation    mIoU, full VOC val (n=1449)
                  defenses_voc_dinov2_vitb14_jpegsweep.pt
                  defenses_voc_dinov2_vitb14_defenses4_full.pt
                  defenses_voc_dinov2_vitb14_dither_full.pt
  Depth           delta1 (pp, as in the main depth table)
                  depth_nyuv2_dinov2_vitb14_defenses.pt
                  depth_nyuv2_dinov2_vitb14_dither.pt
  Retrieval       Medium-protocol mAP, single scale
                  retrieval_transforms_dinov2_vitb14_defenses.json
  VQA             PaliGemma 2, 500-image VQAv2 subset, overall accuracy
                  (VQA_ACC below; see vqa_paligemma_defenses_n500.txt)

USAGE
    python scripts/defense_results.py      # prints every defense number
"""

import json
import os

import torch

RES = os.environ.get("TR_TAA_RESULTS", "./results")

# canonical defense name -> per-source key
DEFENSES = ["JPEG q=50", "Bit-depth", "Resize+pad", "Median", "Dithering"]
_VOC_KEY = {
    "Bit-depth": "BitDepth b=4",
    "Resize+pad": "ResizePad s=[0.9,1.0]",
    "Median": "Median k=3",
    "Dithering": "DitherFS L=4",
}
_DEPTH_KEY = {"Bit-depth": "BitDepth b=4", "Resize+pad": "ResizePad",
              "Median": "Median k=3", "Dithering": "DitherFS L=4"}
_RET_KEY = {"None": "None", "Bit-depth": "BitDepth b=4", "Resize+pad": "ResizePad",
            "Median": "Median k=3", "Dithering": "DitherFS L=4"}

# PaliGemma 2 overall VQA accuracy (%) -- (clean, TAA, TR-TAA K=8)
VQA_ACC = {
    "None":       (87.10, 37.21, 39.57),
    "JPEG q=50":  (85.62, 80.94, 49.32),
    "Bit-depth":  (86.85, 42.72, 40.91),
    "Resize+pad": (86.15, 82.63, 58.80),
    "Median":     (84.66, 74.75, 38.68),
    "Dithering":  (84.64, 78.57, 47.59),
}


def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing result file: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _pair(clean, taa, tr):
    return (clean - taa, clean - tr)


# ------------------------------------------------------------------
# per-task degradation tables: {defense: (taa_drop, tr_drop)}
# ------------------------------------------------------------------

def classification(qualities=(75, 50, 10)):
    """Multi-label mAP drop. 'None', JPEG q in `qualities`, and the four
    non-JPEG defenses. q=90/30 come from the jpegsweep_cls rescore."""
    t16 = _load("rescore_multilabel_table16.pt")
    sweep = None
    if any(q in (90, 30) for q in qualities):
        sweep = _load("rescore_multilabel_jpegsweep_cls.pt")

    def get(key):
        src = t16 if f"{key}__Clean" in t16 else sweep
        if src is None or f"{key}__Clean" not in src:
            raise KeyError(f"classification: no multi-label result for '{key}'")
        return _pair(src[f"{key}__Clean"], src[f"{key}__TAA"], src[f"{key}__EoT-TAA K=8"])

    out = {"None": get("No defense")}
    for q in qualities:
        out[f"JPEG q={q}"] = get(f"JPEG q={q}")
    for d, k in _VOC_KEY.items():
        out[d] = get(k)
    return out


def segmentation(qualities=(75, 50, 10)):
    sweep = _load("defenses_voc_dinov2_vitb14_jpegsweep.pt")
    d4 = _load("defenses_voc_dinov2_vitb14_defenses4_full.pt")
    dith = _load("defenses_voc_dinov2_vitb14_dither_full.pt")

    def get(src, key):
        r = src[key]
        return _pair(r["Clean"]["miou"], r["TAA"]["miou"], r["EoT-TAA K=8"]["miou"])

    out = {"None": get(sweep, "No defense")}
    for q in qualities:
        out[f"JPEG q={q}"] = get(sweep, f"JPEG q={q}")
    for d in ["Bit-depth", "Resize+pad", "Median"]:
        out[d] = get(d4, _VOC_KEY[d])
    out["Dithering"] = get(dith, _VOC_KEY["Dithering"])
    return out


def depth():
    """delta1 drop in pp."""
    a = _load("depth_nyuv2_dinov2_vitb14_defenses.pt")
    b = _load("depth_nyuv2_dinov2_vitb14_dither.pt")

    def get(src, key):
        return _pair(100 * src["Clean"][key]["delta1"],
                     100 * src["TAA"][key]["delta1"],
                     100 * src["EoT-TAA K=8"][key]["delta1"])

    out = {"None": get(a, "No defense"), "JPEG q=50": get(a, "JPEG q=50")}
    for d in ["Bit-depth", "Resize+pad", "Median"]:
        out[d] = get(a, _DEPTH_KEY[d])
    out["Dithering"] = get(b, _DEPTH_KEY["Dithering"])
    return out


def retrieval():
    with open(os.path.join(RES, "retrieval_transforms_dinov2_vitb14_defenses.json")) as f:
        r = json.load(f)

    def get(key):
        return _pair(r["clean"]["results"][key]["medium"],
                     r["taa"]["results"][key]["medium"],
                     r["k8"]["results"][key]["medium"])

    out = {"None": get("None"), "JPEG q=50": get("JPEG q=50")}
    for d in ["Bit-depth", "Resize+pad", "Median", "Dithering"]:
        out[d] = get(_RET_KEY[d])
    return out


def vqa():
    return {d: _pair(*v) for d, v in VQA_ACC.items()}


def retained(table, defenses=DEFENSES):
    """{defense: (taa_%, tr_%)} relative to the 'None' row."""
    n_taa, n_tr = table["None"]
    return {d: (100 * table[d][0] / n_taa, 100 * table[d][1] / n_tr) for d in defenses}


def all_tasks():
    return {
        "Classification": classification(),
        "Segmentation": segmentation(),
        "Depth": depth(),
        "VQA": vqa(),
        "Retrieval": retrieval(),
    }


if __name__ == "__main__":
    # Prints every number of the two defense tables and Figure 6.
    for task, tab in all_tasks().items():
        print(f"\n=== {task}  (degradation, pp | retained %)")
        ret = retained(tab)
        for d, (a, b) in tab.items():
            extra = ""
            if d in ret:
                extra = f"   | retained  TAA {ret[d][0]:5.1f}%   TR-TAA {ret[d][1]:5.1f}%"
            print(f"  {d:<11} TAA {a:7.2f}   TR-TAA {b:7.2f}{extra}")
