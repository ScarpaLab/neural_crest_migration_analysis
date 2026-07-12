# Neural Crest Migration Analysis

Image analysis pipeline for studying trunk neural crest cell migration from 2D time-lapse fluorescence microscopy data.

## Pipeline overview & Folder description

```
01-preprocess/          Cell segmentation (Cellpose) + optional manual editing + drift correction
        ↓
02-extraction_scripts/  Per-cell feature & track extraction, kymograph generation
        ↓
03-extracted_params/    Folder for output data (CSVs, pickles, TIFFs)
        ↓
04-plot-script/         Statistical analysis and figure generation
        ↓
05-plots/               Folder for generated figures

utility/                Folder for utility functions
```

Stages must be run in order.

## Prerequisites

- **Analysis environment:** install from `cellMigEnv.yml` (~5–10 min). See [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for instructions. Includes `tifffile`, `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-image`, `tqdm`, and all other dependencies at pinned versions.
- **Cellpose:** install in a **separate** conda environment (~5 min) — see the [Cellpose installation guide](https://github.com/MouseLand/cellpose) for details. A CUDA-capable GPU is recommended for segmentation; all other pipeline steps run fine on CPU.
- **Napari:** included in the analysis environment for interactive label editing.

## Setup

The `utility/` package (included in this repo) must be on `PYTHONPATH` before running `02-extraction_scripts` or the preprocessing notebooks.

```bash
# Option A: add the repo root to PYTHONPATH
export PYTHONPATH="/path/to/neural_crest_migration_analysis:$PYTHONPATH"

# Option B: install as an editable package (if setup.py/pyproject.toml is present)
pip install -e .
```

Edit `utility/config.py` to point to your local data directory before running anything:

```python
data_dir = "D:\\Kou - Trunk Neural Crest 2D\\project-01"
metadata_path = "D:\\Kou - Trunk Neural Crest 2D\\project-01\\metadata.csv"
```

## Quick start

```bash
# 1. Segment cells
#    Open and run: 01-preprocess/cyh_01_automatic_segmentation.ipynb
#    (Optional) Open and run: 01-preprocess/cyh_02_manuel_labellings.ipynb
#    Run drift correction: python 01-preprocess/cyh_03_image_registration.py

# 2. Extract cell properties
python 02-extraction_scripts/cyh_extract_cell_props.py
python 02-extraction_scripts/cyh_combine_cell_props.py
python 02-extraction_scripts/cyh_create_kymo_tiff.py

# 3. Analyse and plot
#    Open and run: 04-plot-script/analysis.ipynb
```

## Expected run time

- **Cellpose segmentation:** 10 min to 1 hr depending on image size — see the [Cellpose README](https://github.com/MouseLand/cellpose) for details.
- **Manual annotation:** the most time-intensive step, as it requires active review and editing of segmentation masks.
- **All other steps:** typically < 1 min per sample.

## Data

- **Raw data root:** config.data_dir
- **Subject naming:** `sub-XXX` (currently subjects 17–26)
- **Channel convention:** Channel 0 = membrane, Channel 1 = nucleus
- **Image format:** Multi-frame TIFF stacks; masks saved as uint16 TIFFs
- Generated data files (`.tif`, `.csv`, `.pkl`) are git-ignored

## Utility package

`utility/` is a shared package imported by both the preprocessing notebooks and extraction scripts.

| Module | Class / functions | Purpose |
|--------|-------------------|---------|
| `config.py` | — | Central config: `data_dir`, `metadata_path` |
| `util.py` | `load_metadata`, `load_pkl`, `save_pkl` | CSV/pickle I/O helpers |
| `data_loader.py` | `DataLoader` | Resolves subject file paths and loads TIFF/pickle data |
| `registration.py` | `DriftCorrector` | Phase-correlation drift estimation and frame correction |
| `track_analysis.py` | `CellTrackAnalyzer` | Morphology extraction, nearest-neighbour track linking, spline fitting, kymograph generation |
| `plotter.py` | `CellPlotter` | Visualisation helpers for cell properties and kymographs |
