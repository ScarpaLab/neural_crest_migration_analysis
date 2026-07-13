import pandas as pd
import numpy as np
import glob
import os
import re

path = r'..\03-extracted_params\tracklet-stats'

# Get a list of all CSV files in that folder
all_files = glob.glob(os.path.join(path, "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename)
    # Extract the 'sub' number from the filename
    # We look for 'sub-' followed by digits (\d+)
    match = re.search(r'sub-(\d+)', os.path.basename(filename))
    if match:
        sub_number = int(match.group(1)) # Convert '002' to integer 2
        df['sub'] = sub_number
    else:
        df['sub'] = np.nan # Or handle files that don't match the pattern
        
    li.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(li, axis=0, ignore_index=True)
print(f"Successfully combined {len(all_files)} files.")

# Save the combined dataframe to a new CSV file
output_path = r'..\03-extracted_params\combined_cell_profiles.csv'
combined_df.to_csv(output_path, index=False)