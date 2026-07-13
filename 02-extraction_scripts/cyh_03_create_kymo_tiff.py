import tifffile
import numpy as np
import pandas as pd

def save_channels_to_tiff(channel_dict, output_path):
    """
    Saves a dictionary of 2D matrices as a multi-channel OME-TIFF.
    
    Args:
        channel_dict: Dictionary with keys (e.g., 'nuc', 'mem') 
                      containing 2D or 3D numpy arrays.
        output_path: String path ending in .tif or .tiff
    """
    # Extract the channels
    keys = ['nuc', 'mem']
    data_list = [channel_dict[k] for k in keys if k in channel_dict]
    
    # Stack into a single array
    # If arrays are 2D (Y, X) -> Result is (C, Y, X)
    stacked_data = np.stack(data_list, axis=0)

    # Save with Metadata
    # metadata dictionary helps ImageJ/Fiji label the channels correctly
    metadata = {
        'axes': 'CYX',
        'Channel': {'Name': keys},
    }

    tifffile.imwrite(
        output_path, 
        stacked_data.astype(np.uint16), # Ensure consistent bit-depth
        imagej=True, 
        metadata=metadata
    )
    print(f"Successfully saved {metadata['axes']} stack to {output_path}")

subs = np.arange(31,35)

for sub in subs:
    kymo_path = f'..\\03-extracted_params\\kymo-pkl\\kymo_data_sub-{sub:03d}.pkl'

    kymo_df = pd.read_pickle(kymo_path)

    for cell_id in kymo_df.keys():
        output_path = f'..\\03-extracted_params\\kymo-tiff\\sub-{sub:03d}_cell-{cell_id:02d}_kymo.tif'
        save_channels_to_tiff(kymo_df[cell_id], output_path)