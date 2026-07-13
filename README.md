# Neural Crest Migration Analysis

Image analysis pipeline for studying trunk neural crest cell migration from 2D time-lapse fluorescence microscopy data.

Test data can be found [HERE](https://zenodo.org/records/21338711?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhiZjkzMThmLWRjYWUtNDllOS05M2RmLTAzOGIzOGYzZGM4NCIsImRhdGEiOnt9LCJyYW5kb20iOiJmYzcwOTQ1OGEwYzFmYzE1ZTIyYWViYWE5ZDRkODY4YyJ9.Osc1MuT-vMuDbYTnUIYi-ctS9fLVLULRvwCuQX-Phb7ZIK-EuUakiGEM0-YWK4tcgv6L9HiDd1QC9ED_0LrJPg)

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

- **Cellpose segmentation:** 10 min to 1 hr depending on image size and your computer spec — see the [Cellpose README](https://github.com/MouseLand/cellpose) for details.
- **Manual annotation:** the most time-intensive step, as it requires active review and editing of segmentation masks.
- **All other steps:** typically < 1 min per sample.

## Data

- **Raw data root:** `config.data_dir` (edit `utility/config.py` to match your machine)
- **Subject naming:** `sub-XXX` (currently subjects 17–26)
- **Channel convention:** Channel 0 = membrane, Channel 1 = nucleus
- **Image format:** Multi-frame TIFF stacks; masks saved as uint16 TIFFs
- Generated data files (`.tif`, `.csv`, `.pkl`) are git-ignored

### Data directory structure

Files are organised under `data_dir` following a strict naming convention enforced by `DataLoader.generate_file_path`. **Do not rename files or flatten the folder structure** — path construction will break.

P.S. the file structure largely follow [NeuroBlueprint](https://neuroblueprint.neuroinformatics.dev/latest/index.html) architecture

```
data_dir/
├── metadata.csv                            ← subject-level metadata (see below)
└── sub-XXX/                                ← one folder per subject; XXX = zero-padded integer
    ├── sub-XXX_data-xyProjection.tif       ← raw input; shape (T, 2, H, W), uint16
    │                                            channel 0 = membrane, channel 1 = nucleus
    ├── sub-XXX_data-memMask.tif            ← membrane auto-segmentation mask, uint16
    ├── sub-XXX_data-nucMask.tif            ← nucleus auto-segmentation mask, uint16
    ├── sub-XXX_data-memManualLabel.tif     ← (optional) manually corrected membrane labels, uint16
    ├── sub-XXX_data-nucManualLabel.tif     ← (optional) manually corrected nucleus labels, uint16
    ├── sub-XXX_data-drift.pkl              ← cumulative drift array; shape (T, 2), int16
    └── sub-XXX_data-driftPlot.png          ← drift QC figure (not consumed downstream)
```

All files follow the pattern: `sub-<XXX>/sub-<XXX>_data-<name>.<ext>`

### metadata.csv

Located at `config.metadata_path`. Loaded with `util.load_metadata()` and passed row-by-row to `DataLoader`.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `sub` | integer | `17` | Subject ID; must match the `XXX` in each `sub-XXX/` directory |
| `experimenterName` | string | `soraya` | Name of the experimenter who acquired the data |
| `date` | string (DD-MM-YYYY) | `24-07-2024` | Acquisition date |
| `treatment` | string | `15uM CBD`, `control` | Experimental condition / drug treatment |
| `comments` | string | `MAX_XY4_CiliobrevinD15uMregistered` | Free-text notes, typically the original filename |
| `bodyPart` | string | `trunk`, `cranial` | Region of the embryo imaged |

All columns are loaded into `DataLoader` as instance attributes and can be used for grouping or filtering in analysis scripts. The CSV must be encoded as `unicode_escape` (handled automatically by `load_metadata()`).

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
