# ddPCR Droplet Labeling

This repository contains Python scripts for automatic labeling of droplet digital PCR (ddPCR) chip images.

The project includes two labeling pipelines:

- **Bright-field labeling**: detects and labels wells/droplets in bright-field images.
- **Fluorescence labeling**: detects positive, negative, and bubble droplets in fluorescence images.

The scripts are designed for 20 x 20 ddPCR chip images and produce CSV label files plus visual inspection overlays.

## Features

### Bright-Field Labeling

`ddpcr_label.py`

The bright-field script detects the 20 x 20 well grid and labels each well as:

| Label ID | Label Name | Meaning |
|---|---|---|
| 0 | Droplet | Normal droplet |
| 1 | Artifact | Merged or abnormal droplet |
| 2 | Empty | Empty or invalid well |
| 3 | Bubble | Bubble |

Main processing steps:

- TIF image reading and normalization
- Contrast enhancement with CLAHE
- Hough circle detection for well localization
- Projection-based fallback grid detection when Hough fails
- Perspective/affine alignment of the chip grid
- Per-well feature extraction
- Empty, bubble, merged-droplet, and normal-droplet classification
- CSV outputs and overlay visualization

### Fluorescence Labeling

`ddpcr_fluor_label.py`

The fluorescence script labels each well as:

| Label ID | Label Name | Meaning |
|---|---|---|
| 0 | Negative | Negative fluorescence droplet |
| 1 | Positive | Positive fluorescence droplet |
| 2 | Bubble | Nearly black bubble/empty droplet |

Main processing steps:

- 20 x 20 fluorescence grid detection
- Local background estimation for each well
- Robust median/percentile fluorescence features
- Weak-positive rescue based on droplet-scale fluorescence regions
- Speckle/noise cleanup to reject tiny irregular bright artifacts
- Extra retention of very bright full-size positive droplets
- CSV outputs and visualization overlays

## Repository Structure

Recommended structure:

```text
ddpcr-droplet-labeling/
├── ddpcr_label.py
├── ddpcr_fluor_label.py
├── README.md
├── requirements.txt
└── .gitignore
```

Example dataset layout:

```text
dataset_bright/
├── images/
│   ├── 00001.tif
│   ├── 00002.tif
│   └── ...
└── labels/
    ├── geom/
    ├── state/
    ├── vis/
    ├── debug/
    └── AE_patches/

dataset_dark/
├── images_fluor/
│   ├── 00001.tif
│   ├── 00002.tif
│   └── ...
└── labels/
    ├── state/
    ├── vis/
    └── fluor_batch_summary.csv
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.9+
```

## Usage

### 1. Bright-field labeling

Open `ddpcr_label.py` and modify the path configuration:

```python
dataset_root = r"path/to/dataset_bright"
images_dir = os.path.join(dataset_root, "images")
```

Then run:

```bash
python ddpcr_label.py
```

Outputs are saved under:

```text
dataset_bright/labels/
```

Main outputs:

- `geom/`: well geometry and grid information
- `state/`: label CSV files
- `debug/`: per-well diagnostic features
- `vis/`: overlay visualizations
- `AE_patches/`: patches for artifact/empty/bubble review
- `batch_summary.csv`: batch-level processing summary

### 2. Fluorescence labeling

Open `ddpcr_fluor_label.py` and modify the path configuration:

```python
dataset_root = r"path/to/dataset_dark"
fluor_dir = os.path.join(dataset_root, "images_fluor")
output_dir = os.path.join(dataset_root, "labels")
```

Then run:

```bash
python ddpcr_fluor_label.py
```

Outputs are saved under:

```text
dataset_dark/labels/
```

Main outputs:

- `state/`: per-image, per-well fluorescence labels and features
- `vis/`: visual overlay images for manual inspection
- `fluor_batch_summary.csv`: batch-level positive/negative/bubble counts

## Output CSV Format

### Bright-field state CSV

Each row corresponds to one well.

Typical fields:

- `well_id`
- `label_id`
- `label_name`
- `review_flag`
- `note`

### Fluorescence state CSV

Each row corresponds to one well.

Typical fields:

- `well_idx`
- `row`
- `col`
- `cx`, `cy`, `cr`
- fluorescence features such as `blob_score`, `gmax`, `area_ratio`, `core_area_ratio`
- `rescued`
- `noise_like`
- `label`

For fluorescence labels:

```text
0 = Negative
1 = Positive
2 = Bubble
```

## Notes

- The scripts currently assume a 20 x 20 ddPCR chip layout.
- Input images are expected to be TIF files.
- If your images use `.TIF` or `.tiff`, update the `glob` pattern in the scripts.
- For very large datasets, visualizations may take additional storage and time.
- You can reduce visualization output by increasing `VIS_INTERVAL` in `ddpcr_fluor_label.py`.
- Fixed local paths in the scripts should be changed before running on a new machine or dataset.

## Suggested Citation / Acknowledgement

If this code is used in research or development, please cite or acknowledge the ddPCR droplet labeling workflow developed for automated bright-field and fluorescence image annotation.

## License

Add a license file before public release if needed. MIT License is a common choice for open-source research code.
