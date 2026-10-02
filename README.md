# LandSentry

Bitemporal Siamese CNN and transformer-based (BIT/ChangeFormer-style) landslide
change detection over Kodagu (2018) and Wayanad (2024), with explicit
cross-event generalization testing and an interactive Google Earth Engine
dashboard.

## What's actually in this repo

This is a **complete, runnable pipeline** — architectures, data loader,
training loop, evaluation/cross-event script, and a live dashboard — written
from scratch and ready to train on your own downloaded imagery. It does
**not** ship pretrained weights or benchmark numbers: nobody has run the
training here (no GPU/dataset in this environment), so any F1/IoU figures
you see are only meaningful once you train on your own data. Don't put
fabricated numbers in your synopsis defense — run `train.py` / `evaluate.py`
yourself and report what you actually get.

## Project structure

```
landsentry/
├── config.py                 # shared constants (bands, patch size, pixel size)
├── data/
│   └── dataset.py             # PyTorch Dataset for pre/post/mask triplets
├── models/
│   ├── siamese_cnn.py         # Siamese CNN (shared-weight encoder + U-Net decoder)
│   ├── transformer_cd.py      # BIT/ChangeFormer-style transformer change detector
│   └── losses.py               # BCE + Dice combined loss
├── utils/
│   ├── metrics.py               # Precision, Recall, F1, IoU
│   ├── preprocessing.py         # cloud masking, normalization, co-registration
│   └── gee_utils.py             # Google Earth Engine fetch (dashboard only)
├── scripts/
│   └── prepare_hrgldd.py        # reorganizes a raw download into pre/post/mask/
├── train.py                     # trains either model on one region
├── evaluate.py                  # evaluates a checkpoint, incl. cross-event
└── dashboard/
    └── app.py                    # Streamlit AOI-drawing + live inference dashboard
```

## How it works

**Siamese CNN** (`models/siamese_cnn.py`): the pre- and post-event images pass
through the *same* encoder instance. At each of the encoder's 4 scales, the
absolute difference between the pre- and post-event feature maps becomes a
skip connection into a U-Net-style decoder, which upsamples back to a
pixel-wise landslide probability map. Comparing features at matching
locations, instead of segmenting one image alone, is what lets it reject
look-alikes such as bare farmland, quarries, or seasonal vegetation change.

**Transformer detector** (`models/transformer_cd.py`, BIT/ChangeFormer-style):
a shared CNN backbone extracts a feature map per image, each map is pooled
into a handful of "semantic tokens", a transformer encoder lets pre- and
post-event tokens attend over each other (this is the part the Siamese CNN
can't do — it never lets one time step attend to spatial context in the
other), context is projected back onto the pixel grid via cross-attention,
and the difference of the two enriched maps is decoded into a change mask.

**Cross-event generalization**: train once on Kodagu, once on Wayanad
(`train.py --run_name ...`), then run `evaluate.py` with each checkpoint
against *both* regions' data to fill out the 2×2 matrix (same-event and
cross-event) for both architectures — this is the comparison the synopsis
identifies as missing from prior work.

## Setup

```bash
cd landsentry
python -m venv venv && source venv/bin/activate     # or conda
pip install -r requirements.txt

# One-time Earth Engine auth (only needed for the dashboard):
earthengine authenticate
```

## 1. Get and organize the data

Download the HR-GLDD dataset from Zenodo for Kodagu, and pull a Wayanad
Sentinel-2 pre/post pair + mask via Google Earth Engine (or Copernicus
Browser) for the July 2024 event. Folder layouts of raw downloads vary, so
`scripts/prepare_hrgldd.py` is a small template — edit the three glob
patterns at the top of the file to match your actual filenames, then run:

```bash
python scripts/prepare_hrgldd.py --raw_dir downloads/kodagu_raw  --out_dir data/kodagu
python scripts/prepare_hrgldd.py --raw_dir downloads/wayanad_raw --out_dir data/wayanad
```

Each `data/<region>/` folder ends up with `pre/`, `post/`, `mask/`
subfolders containing matching-filename GeoTIFF patches.

## 2. Train both models on both regions

```bash
python train.py --model siamese     --data_root data/kodagu  --run_name siamese_kodagu
python train.py --model transformer --data_root data/kodagu  --run_name transformer_kodagu
python train.py --model siamese     --data_root data/wayanad --run_name siamese_wayanad
python train.py --model transformer --data_root data/wayanad --run_name transformer_wayanad
```

Each run saves its best checkpoint (by validation F1) to
`checkpoints/<run_name>_best.pt`. No dedicated GPU is required for
preprocessing; for actual training, use a free-tier Colab/Kaggle T4/P100 GPU
— just `git clone`/upload this repo there and run the same commands.

## 3. Build the cross-event generalization matrix

```bash
python evaluate.py --checkpoint checkpoints/siamese_kodagu_best.pt      --data_root data/kodagu    # same-event
python evaluate.py --checkpoint checkpoints/siamese_kodagu_best.pt      --data_root data/wayanad   # cross-event
python evaluate.py --checkpoint checkpoints/siamese_wayanad_best.pt     --data_root data/wayanad   # same-event
python evaluate.py --checkpoint checkpoints/siamese_wayanad_best.pt     --data_root data/kodagu    # cross-event
# repeat the four commands with the transformer_* checkpoints
```

This gives you 8 rows total (2 architectures × 2 train regions × 2 test
regions), which is exactly the comparison table your synopsis's Objectives
section calls for.

## 4. Run the interactive dashboard

```bash
streamlit run dashboard/app.py
```

Draw an AOI on the map, pick Before/After/Present dates, and click "Fetch imagery &
detect changes". The app pulls cloud-masked Sentinel-2 median composites for
all three date windows via Earth Engine, runs your chosen trained checkpoint on
the before/post pair, and compares the post-event baseline with the present
composite. It shows the pre-event, post-event, detected-change overlay,
present image, and present-change screening overlay, with geotagged region
centroids and affected areas in km².

The present result is a remote-sensing screening signal, not a guarantee that
an area is physically safe. "No substantial further change detected" means the
current composite is broadly consistent with the post-event baseline; field
verification is still required for safety decisions.

## Notes and honest limitations

- HR-GLDD's exact raw folder/file naming isn't something this script assumes
  with certainty — `prepare_hrgldd.py` is deliberately a template you adapt,
  not a hardcoded parser.
- `utils/preprocessing.coregister()` is a light ECC-based refinement, not a
  full ortho-rectification pipeline; Sentinel-2 L2A products are already
  geo-referenced so this is usually a small correction.
- The transformer's `feat_dim` (64) must stay divisible by `heads` (4); patch
  size must stay divisible by 4 so the transformer's decoder reconstructs
  the exact input resolution.
