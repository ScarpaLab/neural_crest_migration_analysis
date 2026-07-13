# 02-extraction_scripts

Extracts per-cell features, links cell tracks across frames, and generates kymographs from the segmentation masks produced by `01-preprocess`.

## Run order

Run the three scripts in sequence:

```bash
python cyh_01_extract_cell_props.py
python cyh_02_combine_cell_props.py
python cyh_03_create_kymo_tiff.py
```

## Script details

### 1. `cyh_01_extract_cell_props.py`

- Loads segmentation masks
- Applies drift correction offsets
- Runs `CellTrackAnalyzer` to extract cell features and link tracks across frames
- Generates line-intensity kymographs
- **Outputs** (to `03-extracted_params/`):
  - `tracklet-stats/cell_props_sub-XXX.csv`
  - `kymo-pkl/kymo_data_sub-XXX.pkl`

### 2. `cyh_02_combine_cell_props.py`

- Reads all `cell_props_sub-XXX.csv` files from `tracklet-stats/`
- Adds a subject ID column and merges into a single file
- **Output:** `03-extracted_params/combined_cell_profiles.csv`

### 3. `cyh_03_create_kymo_tiff.py`

- Reads pickled kymograph data from `kymo-pkl/`
- Converts two-channel kymograph arrays to OME-TIFF format
- **Output:** `03-extracted_params/kymo-tiff/sub-XXX_cell-YY_kymo.tif`

## Reference

`cyh_line_intensity_extraction_demo.ipynb` — step-by-step walkthrough of the kymograph extraction logic. Not required for the pipeline.

## Next step

Once all outputs are generated, proceed to `04-plot-script/`.
