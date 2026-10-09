# Water Segmentation with a Pretrained ResNet U-Net

Binary water segmentation on 128×128 multi-band satellite tiles. A U-Net with an ImageNet-pretrained ResNet encoder (`segmentation-models-pytorch`) is fine-tuned on N input channels and compared with a U-Net trained from scratch.

This repository contains two notebooks that share the same data pipeline, features and seed. They differ only in how the data is split and how the final model is selected.

| Notebook | Split | Weights used | Scratch baseline | Final model |
|---|---|---|---|---|
| `watersegementationpretrained.ipynb` | 70 / 15 / 15 (214 train, 45 val, 47 test) | Best epoch by val IoU | Hard-coded Week 1 results | ResNet50, 16-ch zero-init, BCE+Tversky, 2x upscaling |
| `twosplitswsp.ipynb` | 80 / 20 (245 train, 61 val) | Final epoch | Retrained in the notebook on the same split | ResNet50, 12-ch average init, BCE+Dice, 2x upscaling |

## Results

**Three-way split** (`watersegementationpretrained.ipynb`)

| Model | Val IoU | Val F1 | Test IoU | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|---|
| Scratch U-Net | 0.7646 | 0.8666 | 0.8004 | 0.8891 | 0.8684 | 0.9109 |
| Pretrained U-Net | 0.8269 | 0.9053 | 0.8466 | 0.9169 | 0.9167 | 0.9171 |
| Pretrained U-Net + TTA | 0.8420 | 0.9142 | 0.8571 | 0.9230 | 0.9284 | 0.9177 |

**Two-way split** (`twosplitswsp.ipynb`, validation only)

| Model | Val IoU | Val F1 | Val Precision | Val Recall |
|---|---|---|---|---|
| Scratch U-Net | 0.7674 | 0.8684 | 0.8588 | 0.8781 |
| Pretrained U-Net | 0.8454 | 0.9163 | 0.9304 | 0.9025 |
| Pretrained U-Net + TTA | 0.8548 | 0.9217 | 0.9402 | 0.9040 |

**Seed stability** (seeds 42, 43, 44; the split stays fixed): val IoU 0.8300 ± 0.0023 and test IoU 0.8455 ± 0.0012 (three-way); val IoU 0.8402 ± 0.0047 (two-way).

## Data

- 306 GeoTIFF tiles of shape 128×128×12 in `images/*.tif`, with binary masks in `labels/*.png`.
- 12 raw channels: Coastal aerosol, Blue, Green, Red, NIR, SWIR1, SWIR2, QA band, MERIT DEM, Copernicus DEM, ESA WorldCover, Water occurrence.
- Water is 25.98% of pixels, and 14.71% of images contain no water.
- The notebooks read the data from `DATA_DIR` (set in the setup cell). Change it to your own path.

## Pipeline

**Feature engineering**
- Indices: NDWI, MNDWI, NDVI and AWEI.
- Terrain: Slope and Relief (DEM minus its 15×15 local mean).
- Land cover: ESA WorldCover converted to binary `WC water` (class 80) and `WC wetland` (classes 90, 95) channels.
- Normalization: per-channel clipping at the 1st and 99th percentiles of the training set, then z-scoring. Binary channels are left unchanged.
- Two feature sets are compared: `all_12` (raw bands) and `plus_terrain_landcover` (16 channels).

**Model**
- `smp.Unet` with a ResNet34 or ResNet50 encoder and ImageNet weights.
- The first convolution is adapted from 3 to N channels in one of two ways:
  - `average`: the RGB filters are averaged, repeated across N channels and scaled by 3/N.
  - `zero`: the RGB filters go to the Red, Green and Blue slots and all other channels start at zero, so training starts from the pretrained RGB model.
- Optional components: input channel attention (identity-initialised), and 2x bilinear input upscaling with the logits average-pooled back to 128×128.

**Training**
- Loss: BCE + Dice, or BCE + Tversky (α=0.3 on false positives, β=0.7 on false negatives).
- Optimizer: AdamW (weight decay 1e-4) with LR 3e-4 for the encoder and 1e-3 for the rest.
- Schedule: 3 warm-up epochs, then cosine decay to 1% of the base LR. 150 epochs, batch size 32.
- Augmentation: random horizontal and vertical flips and 90° rotations.
- Optional EMA of the weights (decay 0.99).
- Test-time augmentation: 8 flip and rotation transforms, with probabilities averaged.

**Experiment design**
- Stage 1 compares input channels and first-layer adaptation using a lean recipe (ResNet34, BCE+Dice).
- Stage 2 adds one technique at a time to the best model and keeps it only if validation IoU improves.

| Stage 2 step | Three-way val IoU | Two-way val IoU |
|---|---|---|
| Baseline | 0.8088 (16ch zero-init) | 0.8218 (12ch average) |
| + Tversky loss | 0.8093 (kept) | 0.8185 (dropped) |
| + attention + EMA | 0.7960 (dropped) | 0.8191 (dropped) |
| + 2x input upscaling | 0.8258 (kept) | 0.8397 (kept) |
| + ResNet50 encoder | 0.8269 (kept) | 0.8454 (kept) |

## Analysis included in the notebooks

- Permutation importance per channel.
- A test with all ancillary channels (DEM, slope, relief, water occurrence, land cover) zeroed. Test IoU drops from 0.8466 to 0.8100 (three-way), and validation IoU drops from 0.8454 to 0.8209 (two-way).
- Recall by connected water-body size, compared with the scratch U-Net.
- Qualitative RGB / ground truth / prediction panels.

## Key findings

- The pretrained encoder improves IoU by about 0.06 to 0.08 over the scratch U-Net.
- 2x input upscaling is the only technique that gives a clear gain in both notebooks (about +0.017).
- Tversky loss, attention and EMA do not help consistently.
- Gains are concentrated on medium and large water bodies. Recall on very small bodies (up to about 50 pixels) is below the scratch U-Net in both notebooks.

## Limitations

- The dataset is small (306 tiles), so validation and test sets contain 45 to 61 images and small IoU differences are within noise. The three-seed standard deviation is 0.002 to 0.005.
- Technique selection is greedy and uses a single seed.
- In the two-way notebook the validation set is used both to choose techniques and to report the score, so its numbers are optimistic. In the three-way notebook only the test score is untouched by selection.
- The scratch comparison is not perfectly controlled. The pretrained model trains for 150 epochs with rotation augmentation, and the scratch model for 100 epochs without it. The three-way notebook also uses hard-coded scratch numbers.

## Running the notebooks

The notebooks were run on Kaggle with a GPU and internet enabled (needed to download ImageNet weights).

```bash
pip install torch segmentation-models-pytorch numpy pandas matplotlib tifffile scipy pillow tqdm scikit-learn
```

1. Set `DATA_DIR` in the setup cell to a folder containing `images/` and `labels/`.
2. Run the notebook top to bottom. Each full run trains many 150-epoch models, so a GPU is strongly recommended.

## Outputs

| File | Produced by |
|---|---|
| `best_pretrained_model.pt` | three-way notebook |
| `training_histories.csv` | three-way notebook |
| `best_pretrained_model_two_way.pt` | two-way notebook |
| `training_histories_two_way.csv` | two-way notebook |
| `normalization_stats.csv` | both notebooks (the second run overwrites the first) |

## Repository layout

```
.
├── watersegementationpretrained.ipynb   # 70/15/15 split, best-epoch selection, clean test set
├── twosplitswsp.ipynb                   # 80/20 split, final-epoch weights, scratch retrained
└── README.md
```
