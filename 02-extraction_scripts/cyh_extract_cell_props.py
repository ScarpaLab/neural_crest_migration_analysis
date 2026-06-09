import sys
sys.path.append("..")
from utility import util
from utility import data_loader
from utility.registration import DriftCorrector
from utility.track_analysis import CellTrackAnalyzer
import numpy as np
import pickle

metadata = util.load_metadata()

for k in range(len(metadata)):
    if k <30:
        continue
    dl = data_loader.DataLoader(metadata.iloc[k])
    print(f'Processing subject {dl.sub}')
    drift = dl.load_data('drift')
    xyProj = dl.load_data('xyProjection')
    dc = DriftCorrector(xyProj[:,1,:,:]) # Use Nuclei for drift calculation
    dc.drifts = drift

    analyzer = CellTrackAnalyzer(
        nuc_labels=dc.apply_correction(dl.load_data('nucManualLabel')), 
        mem_labels=dc.apply_correction(dl.load_data('memManualLabel')),
        nuc_img=dc.apply_correction(xyProj[:,1,:,:]),
        mem_img=dc.apply_correction(xyProj[:,0,:,:]),
        s=100
    )

    analyzer.extract_cell_features()
    analyzer.run_track_analysis()
    analyzer.extract_line_profile()

    tracked_cells = np.array(list(analyzer.kymo_data.keys()))
    cells_df = analyzer.df
    cells_df = cells_df[cells_df['cell_idx'].isin(tracked_cells)]
    kymo_data = analyzer.kymo_data

    # Save cells_df as a csv file
    cells_df.to_csv(f'../03-extracted_params/tracklet-stats/cell_props_sub-{dl.sub:03d}.csv', index=False)
    # Save kymo_data as a pickle file

    with open(f'../03-extracted_params/kymo-pkl/kymo_data_sub-{dl.sub:03d}.pkl', 'wb') as f:
        pickle.dump(kymo_data, f)