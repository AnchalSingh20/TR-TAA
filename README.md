# TR-TAA: Transformation-Robust Task-Agnostic Attack


**"TR-TAA: A Transformation-Robust Task-Agnostic Attack on Vision Foundation Models"**.

![overview](figures/teaser.png)

TR-TAA perturbs the shared, mean-centred patch-token representation of a
frozen vision encoder, without access to any downstream task, label or
head. The perturbation is optimised in expectation over randomly sampled,
differentiable image transformations, against a fixed clean reference.
The same perturbation then degrades five downstream tasks (multi-label
classification, semantic segmentation, monocular depth estimation, image
retrieval, visual question answering) and remains effective after common
transformations such as JPEG compression, resizing, blur, rotation, noise
and print-scan.

The core attack is in [`src/eot_attack.py`](src/eot_attack.py)
(`eot_taa_attack`, Algorithm 1; transformation distribution of Table 1 in
`sample_random_transform`). The TAA baseline is in
[`src/attacks.py`](src/attacks.py) (`task_agnostic_attack`).


## Repository structure

```
.
├── src/            attacks, models, datasets, heads, metrics, defenses
├── scripts/        attack generation, evaluation, ablations, statistics, figures
├── configs/        default hyperparameters
├── checkpoints/    trained downstream heads (DINOv2 ViT-S/14 and ViT-B/14)
├── results/        result files behind every table and figure of the paper
├── figures/        figures of the paper
└── requirements.txt
```

| `src/` file | Content |
|---|---|
| `eot_attack.py` | **TR-TAA** attack and the 8-operator differentiable transformation distribution |
| `attacks.py` | **TAA** baseline, mean-feature estimation, PSNR |
| `eot_transforms_frozen.py` | same transformation distribution with parameters frozen at sampling time (VQA attack) |
| `models.py` | DINOv2 loading, normalisation, token extraction |
| `datasets*.py` | PASCAL VOC (single/multi-label, segmentation), NYU Depth v2, ROxford5k, VQAv2 |
| `heads*.py`, `losses_depth.py` | linear classification, segmentation and depth heads |
| `metrics*.py` | mAP, mIoU, depth metrics, retrieval mAP (revisitop protocol), VQA accuracy |
| `defenses.py` | input-preprocessing defenses (JPEG, bit-depth, resize-and-pad, median, dithering) |
| `retrieval_preprocess.py`, `reproducibility.py` | retrieval descriptors, seeding |

## Setup

```bash
conda create -n tr_taa python=3.10 && conda activate tr_taa
pip install -r requirements.txt
```

DINOv2 weights are downloaded through `torch.hub` on first use. The VQA
experiments use `google/paligemma2-3b-mix-448` from the Hugging Face Hub
(its licence must be accepted on the Hub and `huggingface-cli login` run
once). All commands below are run from the repository root on a CUDA GPU.

### Datasets (not redistributed)

| Task | Dataset | Argument | Expected layout |
|---|---|---|---|
| Classification, segmentation | PASCAL VOC 2012 | `--pascalvoc_root` | `VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets}` |
| Depth | NYU Depth v2 (BTS release, Eigen test split) | `--nyuv2_root` | `<root>/nyu_data/data/{nyu2_test,nyu2_train,nyu2_test.csv,nyu2_train.csv}` |
| Retrieval | Revisited Oxford (ROxford5k) | `--roxford_root` | `<root>/{gnd_roxford5k.pkl,images/}` |
| VQA | VQAv2 (val2014) | `--vqav2_root` | `<root>/{val2014/,v2_OpenEnded_mscoco_val2014_questions.json,v2_mscoco_val2014_annotations.json}` |

## Checkpoints

`checkpoints/` contains the heads trained on frozen backbone features that
produce every number in the paper. All scripts read them from
`./checkpoints` by default (`--heads_dir`).

| File | Head |
|---|---|
| `<model>_classification_head_multilabel.pth` | multi-label classification (all reported classification results) |
| `<model>_segmentation_head.pth` | semantic segmentation |
| `<model>_depth_head_linear.pth` (+ `_history.pt`) | linear depth head |
| `<model>_classification_head.pth` | single-label head, loaded by some scripts but not used for reported results |

Retrieval and VQA need no trained head. To retrain the heads:

```bash
python scripts/train_heads_multilabel.py --pascalvoc_root <VOC2012> --model_name dinov2_vits14
python scripts/train_heads.py            --pascalvoc_root <VOC2012> --model_name dinov2_vits14 --task both
python scripts/train_heads_depth.py      --nyuv2_root <NYUv2>       --model_name dinov2_vits14
```

## Attack settings

Shared by TAA and TR-TAA: T = 50 PGD steps, step size 4e-4, eps = 8/255,
patch-token loss on mean-centred features (mu from 5,000 VOC training
images), 8-bit quantisation of the output. TR-TAA uses K = 8 sampled
transformations per step (K = 4 for the K ablation) and the fixed clean
reference. VQA uses the same settings on the PaliGemma 2 vision tower at
448x448 with step size 0.4/255.

`eot_taa_attack` uses the fixed clean reference by default. The
transform-both variant of the reference-construction ablation is selected
with `transform_reference=True` (or `EOT_TRANSFORM_REFERENCE=1`).

## Reproducing the results

Adversarial images are cached under `outputs/` (several GB per task and
backbone) and are not included; each step below writes the cache that the
next step reads. Replace `<model>` by `dinov2_vits14` and `dinov2_vitb14`.

**1. PASCAL VOC (classification + segmentation).** Run first: it also
writes the mean vector `mu` reused by the depth and retrieval experiments.

```bash
python scripts/generate_taa_baseline.py  --pascalvoc_root <VOC2012> --model_name <model>
python scripts/generate_eot_taa_clean.py --pascalvoc_root <VOC2012> --model_name <model> --K 8
python scripts/generate_eot_taa_clean.py --pascalvoc_root <VOC2012> --model_name <model> --K 4
python scripts/evaluate_eot_taa_multilabel.py --pascalvoc_root <VOC2012> --model_name <model>
```

**2. Depth (NYU Depth v2).**

```bash
python scripts/generate_eot_taa_nyuv2.py --nyuv2_root <NYUv2> --model_name <model> --taa_baseline
python scripts/generate_eot_taa_nyuv2.py --nyuv2_root <NYUv2> --model_name <model> --K 8
python scripts/generate_eot_taa_nyuv2.py --nyuv2_root <NYUv2> --model_name <model> --K 4
python scripts/evaluate_depth_nyuv2.py   --nyuv2_root <NYUv2> --model_name <model> --transforms
```

**3. Retrieval (ROxford5k).** Generation and evaluation in one script; run `clean` first.

```bash
for c in clean taa k8 k4; do
  python scripts/evaluate_roxford_transforms.py --roxford_root <ROxford> --model_name <model> --condition $c
done
```

**4. VQA (PaliGemma 2, 500 VQAv2 images).**

```bash
python scripts/attack_paligemma_vision_tower.py --vqav2_root <VQAv2> --attack taa
python scripts/attack_paligemma_vision_tower.py --vqav2_root <VQAv2> --attack eot --K 8
python scripts/evaluate_vqa_paligemma.py        --vqav2_root <VQAv2> --transforms
```

**5. Cross-encoder transfer** (uses the VOC caches of both backbones).

```bash
python scripts/evaluate_cross_encoder_transfer.py
python scripts/rescore_multilabel_all.py --target cross_encoder --pascalvoc_root <VOC2012>
```

**6. Ablations** (first 500 VOC val images per task, K = 8).

```bash
python scripts/ablation_transform_both.py        --pascalvoc_root <VOC2012> --model_name <model>          # reference construction
python scripts/ablation_transform_families.py    --pascalvoc_root <VOC2012> --model_name dinov2_vits14   # transformation families
python scripts/ablation_transform_probability.py --pascalvoc_root <VOC2012> --model_name dinov2_vits14   # application probabilities
python scripts/ablation_transform_range.py       --pascalvoc_root <VOC2012> --model_name dinov2_vits14   # parameter ranges
python scripts/rescore_multilabel_all.py --target table13 --pascalvoc_root <VOC2012>   # multi-label mAP, reference construction
python scripts/rescore_multilabel_all.py --target table14 --pascalvoc_root <VOC2012>   # multi-label mAP, families
python scripts/rescore_multilabel_all.py --target table15 --pascalvoc_root <VOC2012>   # multi-label mAP, probability + range
```

The ablation and cross-encoder scripts score classification with the
single-label head; the multi-label mAP reported in the paper comes from
`rescore_multilabel_all.py`, which re-scores the same cached images.
Segmentation numbers are taken directly from the ablation scripts.

**7. Input-preprocessing defenses (ViT-B/14).**

```bash
C="--model_name dinov2_vitb14 --cache_dir ./outputs/cached_images_clean/dinov2_vitb14 \
   --taa_cache_dir ./outputs/cached_images/dinov2_vitb14 --no_diffpure"
python scripts/evaluate_defenses_voc.py $C --jpeg_qualities 90 75 50 30 10 --no_bitdepth --no_resize_pad --no_median --no_dither --tag jpegsweep
python scripts/evaluate_defenses_voc.py $C --jpeg_qualities 75 --no_dither --tag defenses4_full
python scripts/evaluate_defenses_voc.py $C --jpeg_qualities 75 --no_bitdepth --no_resize_pad --no_median --tag dither_full
python scripts/rescore_multilabel_all.py --target table16       --pascalvoc_root <VOC2012>
python scripts/rescore_multilabel_all.py --target jpegsweep_cls --pascalvoc_root <VOC2012>
python scripts/evaluate_depth_nyuv2.py --nyuv2_root <NYUv2> --model_name dinov2_vitb14 --defenses
python scripts/evaluate_depth_nyuv2.py --nyuv2_root <NYUv2> --model_name dinov2_vitb14 --dither
for c in clean taa k8; do
  python scripts/evaluate_roxford_transforms.py --roxford_root <ROxford> --model_name dinov2_vitb14 --condition $c --scales 1.0 --tag defenses
done
python scripts/evaluate_vqa_paligemma.py --vqav2_root <VQAv2> --defenses
python scripts/evaluate_vqa_paligemma.py --vqav2_root <VQAv2> --dither
python scripts/defense_results.py          # prints every defense number
```

**8. Seed stability and significance tests (ViT-B/14, seeds 0/1/2).**

```bash
for s in 0 1 2; do
  python scripts/generate_eot_taa_clean.py --pascalvoc_root <VOC2012> --model_name dinov2_vitb14 --K 8 --seed $s
  python scripts/statistical_analysis.py collect --pascalvoc_root <VOC2012> --model_name dinov2_vitb14 \
      --new_cache_dir ./outputs/cached_images_clean_seed$s --seed $s \
      --out ./outputs/results/stats_perimage_vitb14_seed$s.npz
done
python scripts/statistical_analysis.py analyze --npz ./outputs/results/stats_perimage_vitb14_seed{0,1,2}.npz \
    --latex_out ./outputs/results/table_stats_vitb14.tex --json_out ./outputs/results/table_stats_vitb14.json
```


