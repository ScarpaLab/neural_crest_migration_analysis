import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import make_splprep
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from skimage.measure import profile_line

class CellTrackAnalyzer:
    def __init__(self, nuc_labels, mem_labels, nuc_img, mem_img, s=10, min_frames=30):
        '''
        Initializes the analyzer, extracting basic morphology stats immediately.
        '''
        self.nuc_labels = nuc_labels
        self.mem_labels = mem_labels
        self.nuc_img = nuc_img
        self.mem_img = mem_img
        self.s = s # smoothing condition for spline fit
        self.min_frames = min_frames # minimum number of frames that must exist for a cell to be analyzed
        self.track_pos_smooth = 5 # rolling window for smoothing the position on track (in frames)

    def extract_cell_features(self):
        '''
        Extract morphological features
        '''
        self.nuc_df = self._extract_morph(self.nuc_labels)
        self.mem_df = self._extract_morph(self.mem_labels)
        self.nuc_df = self._calculate_speeds(self.nuc_df)
        self.mem_df = self._calculate_speeds(self.mem_df)

        # Combine nuc and mem measurements
        self.df = pd.merge(self.nuc_df, self.mem_df, on=['frame', 'cell_idx'], how='left', suffixes=('_nuc', '_mem'))

        # Counting number of cells
        self.num_cells = self.df['cell_idx'].nunique()

    def run_track_analysis(self):
        '''
        Extract the cell track and define a spline, and project the nuclei dynamics on it
        min_frames: minimum number of frames a cell must have to run this analysis
        '''
        new_cols = ['pos_on_track']
        for col in new_cols:
            self.df[col] = np.nan

        for cell_id, cell_df in tqdm(self.df.groupby('cell_idx'), desc="Analysing cell tracks..."):
            # If cell was tracked for small number of frames, its not interpolation-worthy
            if len(cell_df) < self.min_frames:
                print(f'Too few frames: only {len(cell_df)} frames found for cell {cell_id}')
                continue
            
            # Spline interpret the cell track
            try:
                tx, ty = self._get_spline_track(cell_df['centroid-x_nuc'], cell_df['centroid-y_nuc'])
            except Exception:
                print('Spline interpolation failed')
                continue

            # Project nuclei to interpolated track
            tree = cKDTree(np.column_stack((tx,ty)))
            max_idx = len(tx) - 1
            raw_pts = np.column_stack((cell_df['centroid-x_nuc'], cell_df['centroid-y_nuc']))
            _, nuc_indices = tree.query(raw_pts)

            # Apply rolling average to smooth the nuc_indices
            nuc_indices = pd.Series(nuc_indices).rolling(window=self.track_pos_smooth, min_periods=1, center=True).mean().values
            self.df.loc[cell_df.index, 'pos_on_track'] = nuc_indices

    def extract_line_profile(self, linewidth=50, linelength=300, moving_window=16):
        '''
        Extract line profile of the image on each frame
        linewidth: Width of the scan, perpendicular to the line
        linelength: Length of the scan, centre at the nuclei location
        moving_window: the vector direction of the scan was defined as 
            v[t] = pos[t+moving_window//2] - pos[t-moving_window//2]
            unit: frames
        '''
        new_cols = ['mem_front', 'mem_back']
        for col in new_cols:
            self.df[col] = np.nan

        self.kymo_data = {}

        for cell_id, cell_df in tqdm(self.df.groupby('cell_idx'), desc="Extracting line profiles..."):
            cell_df = cell_df.sort_values('frame')
            n_frames = len(cell_df)
            if  n_frames < self.min_frames:
                print(f'Too few frames: only {len(cell_df)} frames found for cell {cell_id}')
                continue

            x_pos, y_pos = cell_df['centroid-x_nuc'].values, cell_df['centroid-y_nuc'].values
            x_vec, y_vec = self._calculate_motion_vector(x_pos, y_pos, moving_window)
            nuc_lineProfile = np.zeros((linelength, n_frames))
            mem_lineProfile = np.copy(nuc_lineProfile)
            for t in range(n_frames):
                frame_id = cell_df['frame'].values[t]
                vec = (x_vec[t], y_vec[t])
                nuc_centroid = (x_pos[t], y_pos[t])

                nuc_lineProfile[:,t] = self._compute_line_profile(
                    self.nuc_img[frame_id]*(self.nuc_labels[frame_id] == cell_id), 
                    nuc_centroid, vec, linewidth, linelength)[0:linelength]
                mem_lineProfile[:,t] = self._compute_line_profile(
                    self.mem_img[frame_id]*(self.mem_labels[frame_id] == cell_id), 
                    nuc_centroid, vec, linewidth, linelength)[0:linelength]
                
            mem_back, mem_front = self.find_mem_start_end(mem_lineProfile > 0)
            self.df.loc[cell_df.index, 'mem_front'] = mem_front + cell_df['pos_on_track'] - linelength//2
            self.df.loc[cell_df.index, 'mem_back'] = mem_back + cell_df['pos_on_track'] - linelength//2

            self.kymo_data[cell_id] = {
                'nuc': self.offset_kymo_by_pos(nuc_lineProfile, cell_df['pos_on_track'].values),
                'mem': self.offset_kymo_by_pos(mem_lineProfile, cell_df['pos_on_track'].values),
            }

    def _extract_morph(self, labels):
        '''
        Extracts morphological features from the label stack.
        '''
        props_list = ['label', 'area', 'centroid', 'solidity', 'perimeter', 'eccentricity','axis_major_length', 'axis_minor_length']
        all_frames = []
        
        for t in tqdm(range(labels.shape[0]), desc="Extracting features from frames..."):
            if np.max(labels[t]) == 0: continue
            
            df_f = pd.DataFrame(regionprops_table(labels[t], properties=props_list))
            df_f['frame'] = t
            all_frames.append(df_f)

        df = pd.concat(all_frames, ignore_index=True)
        df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
        df['aspect_ratio'] = df['axis_major_length'] / df['axis_minor_length']
        return df.rename(columns={'label': 'cell_idx', 'centroid-0': 'centroid-y', 'centroid-1': 'centroid-x'})
    
    def _calculate_speeds(self, df):
        """Vectorized instantaneous speed calculation."""
        df = df.sort_values(['cell_idx', 'frame']).reset_index(drop=True)
        dx = df.groupby('cell_idx')['centroid-x'].diff()
        dy = df.groupby('cell_idx')['centroid-y'].diff()
        df['speed'] = np.sqrt(dx**2 + dy**2)
        return df

    def _get_spline_track(self, x_vals, y_vals):
        '''
        Internal helper to get a 1-pixel spaced spline.
        '''
        spl_n, _ = make_splprep([x_vals, y_vals], s=self.s)
        pts_fine = spl_n(np.linspace(0, 1, 1000))
        dist = np.sum(np.sqrt(np.diff(pts_fine[0])**2 + np.diff(pts_fine[1])**2))
        return spl_n(np.linspace(0, 1, int(np.round(dist))))

    @staticmethod
    def _calculate_motion_vector(x_pos, y_pos, moving_window=16):
        '''
        Calculate velosity vectors v[t] = pos[t+w/2] - pos[t-w/2]
        Pads the beginning and the end with the first and last valid vectors.
        '''
        n_frames = len(x_pos)
        half_w = moving_window//2

        dx = np.zeros(n_frames)
        dy = np.zeros(n_frames)

        valid_range = slice(half_w, n_frames - half_w)
        dx[valid_range] = x_pos[moving_window:] - x_pos[:n_frames-moving_window]
        dy[valid_range] = y_pos[moving_window:] - y_pos[:n_frames-moving_window]

        dx[:half_w] = dx[half_w]
        dy[:half_w] = dy[half_w]
        dx[n_frames - half_w:] = dx[n_frames - half_w - 1]
        dy[n_frames - half_w:] = dy[n_frames - half_w - 1]

        return dx, dy
    
    def _compute_line_profile(self, image, centroid, vector, linewidth=1, linelength=200):
        '''
        Calculates the intensity profile and total sum along a vector starting from a centroid.
        '''
        x0, y0 = centroid
        dx, dy = vector
        # Normalize the vector to 1 and then scale to desired length
        norm = np.sqrt(dx**2 + dy**2)
        dx = dx/norm
        dy = dy/norm

        # 1. Calculate the end point of the vector
        x1, y1 = x0 + dx*linelength/2, y0 + dy*linelength/2
        x0, y0 = x0 - dx*linelength/2, y0 - dy*linelength/2  # Start point is also shifted back to keep the centroid in the middle

        # 2. Extract the intensity profile
        # WARNING: profile_line expects coordinates as (row, col), which corresponds to (y, x)!
        profile = profile_line(
            image, (y0, x0), (y1, x1), linewidth=linewidth, mode="constant", cval=0
        )

        # 3. Calculate the sum
        total_sum = np.sum(profile)

        return profile
    
    @staticmethod
    def find_mem_start_end(binary_array):
        '''
        Column by column, find where the membrane start and finish
        '''
        binary_array = binary_array.T
        row_has_ones = np.any(binary_array == 1, axis=1)
        first_one_idx = np.argmax(binary_array == 1, axis=1)
        width = binary_array.shape[1]
        last_one_idx = width - 1 - np.argmax(binary_array[:, ::-1] == 1, axis=1)
        turns_to_zero_idx = last_one_idx + 1
        first_one_idx[~row_has_ones] = -1
        turns_to_zero_idx[~row_has_ones] = -1
        return first_one_idx, turns_to_zero_idx

    @staticmethod
    def offset_kymo_by_pos(kymo, nuc_pos, fill_value = 0):
        '''
        Offsets each column of the kymograph by a specific value.
        '''
        d, n_frames = kymo.shape
        offset = np.round(nuc_pos).astype(int)
        # Define a new d, the number of rows in the offset_kymograph
        new_d = d + np.max(offset)

        # Create a grid of coordinates for the whole matrix
        rows, cols = np.indices((new_d, n_frames))

        # Calculate the 'source' index for every pixel
        source_rows = rows - offset

        valid_mask = (source_rows >= 0) & (source_rows < d)
        res = np.full_like(rows, fill_value, dtype=float)
        res[rows[valid_mask], cols[valid_mask]] = kymo[source_rows[valid_mask], cols[valid_mask]]
        return res
    

    # def plot_spatial_track(self, label_idx, channel_to_plot = 'nuc', ax=None):
    #     """
    #     Plots the cell centroids and the calculated spline over the image with local contrast.
    #     """
    #     df_cell = self.df[self.df['label_idx'] == label_idx]

    #     # 1. Extract the specific frame and spline
    #     if channel_to_plot == 'nuc':
    #         img = self.nuc_img[int(df_cell['frame'].iloc[0])]
    #     elif channel_to_plot == 'mem':
    #         img = self.mem_img[int(df_cell['frame'].iloc[0])]
    #     tx, ty = self._get_spline_track(df_cell['centroid-x_nuc'], df_cell['centroid-y_nuc'])
        
    #     # 2. Define the exact padding and bounding box we plan to display
    #     d = 20
    #     x_min, x_max = tx.min() - d, tx.max() + d
    #     y_min, y_max = ty.min() - d, ty.max() + d
        
    #     # 3. Safely clip the bounding box to the actual image dimensions 
    #     # (in case the cell is right on the edge of the image)
    #     y_start = max(0, int(y_min))
    #     y_end = min(img.shape[0], int(y_max))
    #     x_start = max(0, int(x_min))
    #     x_end = min(img.shape[1], int(x_max))
        
    #     # 4. Extract JUST the local region being displayed
    #     local_region = img[y_start:y_end, x_start:x_end]
        
    #     # 5. Calculate contrast percentiles ONLY on this local area
    #     if local_region.size > 0:
    #         vmin, vmax = np.percentile(local_region, (2, 98))
    #     else:
    #         vmin, vmax = img.min(), img.max() # Fallback

    #     # 6. Plot the full image, but locked to the local contrast limits
    #     ax.imshow(img, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
        
    #     ax.scatter(df_cell['centroid-x_nuc'], df_cell['centroid-y_nuc'], color='red', label='Centroids', s=20)
    #     ax.plot(tx, ty, 'g-', label='BSpline')
        
    #     # 7. Zoom into the specific region
    #     ax.set_xlim(x_min, x_max)
    #     ax.set_ylim(y_max, y_min) # ty.max() at bottom, ty.min() at top
    #     ax.legend()
        
    #     return ax

    # def plot_kymograph(self, label_idx, ch = 'nuc',ax=None):
    #     """Plots image intensity along the 1D track."""
    #     df_cell = self.df[self.df['label_idx'] == label_idx].sort_values('frame')
    #     if len(df_cell) < 4: 
    #         return ax
        
    #     raw_img = self.nuc_img if ch == 'nuc' else self.mem_img


    #     tx, ty = self._get_spline_track(df_cell['centroid-x_nuc'], df_cell['centroid-y_nuc'])
    #     frames = df_cell['frame'].astype(int).values
    #     kymo = np.array([map_coordinates(raw_img[f], [ty, tx], order=1) for f in frames]).T
        
    #     ax.imshow(kymo, aspect='auto', cmap='gray', extent=[frames[0], frames[-1], len(tx)-1, 0], interpolation='none')
    #     ax.set(title=f'Kymo: Cell {label_idx}', xlabel='Frame', ylabel='Track Pos (px)')
    #     return ax

    # def plot_1d_progress(self, label_idx, ax=None):
    #     """Plots the 1D progress of nucleus, front, and back."""
    #     if ax is None: _, ax = plt.subplots(figsize=(6, 4))
    #     df_c = self.df[self.df['label_idx'] == label_idx]
    #     if df_c.empty: return ax

    #     vf, vb = df_c['valid_mem_on_track_front'] == True, df_c['valid_mem_on_track_back'] == True
    #     x = df_c['frame']

    #     ax.plot(x, df_c['pos_on_track'], 'k-', label='Nucleus')
    #     ax.plot(x[vf], df_c['mem_on_track_front'][vf], 'o', color='tab:blue', label='Front')
    #     ax.plot(x[~vf], df_c['mem_on_track_front'][~vf], 'o', color='red', label='Invalid Boundary')
    #     ax.plot(x[vb], df_c['mem_on_track_back'][vb], 's', color='tab:orange', label='Back')
    #     ax.plot(x[~vb], df_c['mem_on_track_back'][~vb], 's', color='red', label='_nolegend_')

    #     ax.invert_yaxis()
    #     ax.set(title=f'1D Dynamics: Cell {label_idx}', xlabel='Frame', ylabel='Track Pos (px)')
        
    #     # Deduplicate legend
    #     handles, labels = ax.get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
    #     return ax

    # def plot_vals(self, label_idx, val ='circularity', color='k', ax=None):
    #     """Plots the 1D progress of nucleus, front, and back."""
    #     if ax is None: _, ax = plt.subplots(figsize=(6, 4))
    #     df_c = self.df[self.df['label_idx'] == label_idx]
    #     if df_c.empty: return ax

    #     ax.plot(df_c['frame'], df_c[val], color=color, label=val)
    #     return ax
    
    # def plot_movement_arrow(self, label_idx, step=15, ch='nuc', ax=None):
    #     df_cell = self.df[self.df['label_idx'] == label_idx].sort_values('frame')
    #     x_pos, y_pos = df_cell['centroid-x_nuc'].values, df_cell['centroid-y_nuc'].values

    #     img = self.nuc_img if ch == 'nuc' else self.mem_img
    #     num_df_frames = len(df_cell)
    #     show_img = img[int(df_cell['frame'].iloc[num_df_frames//2])]
    #     # # Show the middle frame of the cell track as background for context
    #     # x_range = np.arange(int(x_pos.min())-20, int(x_pos.max())+20)
    #     # y_range = np.arange(int(y_pos.min())-20, int(y_pos.max())+20)
        
    #     # show_img_crop = show_img[x_range[0]:x_range[-1], y_range[0]:y_range[-1]]
    #     # vmin = np.percentile(show_img_crop, 1)
    #     # vmax = np.percentile(show_img_crop, 99)
    #     ax.imshow(show_img, cmap='gray', interpolation='none')

    #     for t in range(0, len(x_pos)-step, step):
    #         vec = [(x_pos[t+step]-x_pos[t]), (y_pos[t+step]-y_pos[t])]
    #         ax.arrow(x_pos[t], y_pos[t], vec[0], vec[1], head_width=5, head_length=5, color='red', alpha=0.6)
    #         ax.scatter(x_pos[t], y_pos[t], color='blue', s=10)
        
    #     ax.set_xlim(x_pos.min() - 20, x_pos.max() + 20)
    #     ax.set_ylim(y_pos.max() + 20, y_pos.min() - 20)

    # def calculate_vector_intensity(self, image, centroid, vector, linewidth=1, linelength=200):
    #     """Calculates the intensity profile and total sum along a vector starting from a centroid.

    #     Parameters:
    #     -----------
    #     image : 2D numpy array
    #         The image (e.g., your membrane frame).
    #     centroid : tuple or list (x, y)
    #         The starting point of your measurement.
    #     vector : tuple or list (dx, dy)
    #         The direction and length you want to measure.
    #     linewidth : int
    #         Width of the line to sample. If > 1, it averages pixels
    #         perpendicular to the line (great for reducing noise).
    #     linelength : int
    #         The length of the line to sample.

    #     Returns:
    #     --------
    #     total_sum : float
    #         The sum of intensities along the vector.
    #     profile : 1D numpy array
    #         The raw intensity values along the line from start to finish.
    #     """
    #     x0, y0 = centroid
    #     dx, dy = vector
    #     # Normalize the vector to 1 and then scale to desired length
    #     norm = np.sqrt(dx**2 + dy**2)
    #     dx = dx/norm
    #     dy = dy/norm

    #     # 1. Calculate the end point of the vector
    #     x1, y1 = x0 + dx*linelength/2, y0 + dy*linelength/2
    #     x0, y0 = x0 - dx*linelength/2, y0 - dy*linelength/2  # Start point is also shifted back to keep the centroid in the middle

    #     # 2. Extract the intensity profile
    #     # WARNING: profile_line expects coordinates as (row, col), which corresponds to (y, x)!
    #     profile = profile_line(
    #         image, (y0, x0), (y1, x1), linewidth=linewidth, mode="constant", cval=0
    #     )

    #     # 3. Calculate the sum
    #     total_sum = np.sum(profile)

    #     return total_sum, profile
    
    # def extract_line_profile(self, cell_idx, linewidth=50, linelength=300, step=15):

    #     def find_row_transitions(binary_array):
    #         # 1. Identify which rows actually have a cell in them
    #         # (This prevents rows with all 0s from giving false results)
    #         row_has_ones = np.any(binary_array == 1, axis=1)
            
    #         # 2. Find where 0 first turns to 1
    #         # np.argmax finds the first index where the condition is True
    #         first_one_idx = np.argmax(binary_array == 1, axis=1)
            
    #         # 3. Find where the 1 last turns to 0
    #         # We flip the columns backwards [:, ::-1] and find the first 1 from the back
    #         width = binary_array.shape[1]
    #         last_one_idx = width - 1 - np.argmax(binary_array[:, ::-1] == 1, axis=1)
            
    #         # Since you asked where it *turns to 0*, that happens on the pixel right AFTER the last 1
    #         turns_to_zero_idx = last_one_idx + 1
            
    #         # 4. Cleanup: Set rows with no cell to -1 (so they don't give fake 0 indices)
    #         first_one_idx[~row_has_ones] = -1
    #         turns_to_zero_idx[~row_has_ones] = -1
            
    #         return first_one_idx, turns_to_zero_idx
        
    #     df_cell = self.df[self.df['label_idx'] == cell_idx].sort_values('frame')
    #     x_pos, y_pos = df_cell['centroid-x_nuc'].values, df_cell['centroid-y_nuc'].values

    #     t_range = np.arange(0, len(x_pos), 1)
    #     profile_nuc_all = np.zeros((len(t_range), linelength))  # Assuming linelength=300
    #     profile_mem_all = np.zeros((len(t_range), linelength))  # Assuming linelength=300
    #     for i, t in enumerate(t_range):
    #         frame_idx = df_cell['frame'].values[t]
    #         # img = dc_nuc_img[frame_idx] 
    #         centroid = (x_pos[t], y_pos[t])
    #         if t < len(x_pos)-step:
    #             vec = (x_pos[t+step]-x_pos[t], y_pos[t+step]-y_pos[t])
    #         total_sum, profile_nuc = self.calculate_vector_intensity(self.nuc_img[frame_idx]*(self.nuc_labels[frame_idx] == cell_idx), centroid, vec, linewidth=linewidth, linelength=linelength)
    #         total_sum, profile_mem = self.calculate_vector_intensity(self.mem_img[frame_idx]*(self.mem_labels[frame_idx] == cell_idx), centroid, vec, linewidth=linewidth, linelength=linelength)
    #         profile_nuc_all[i] = profile_nuc[0:linelength]  # Ensure it fits the preallocated array
    #         profile_mem_all[i] = profile_mem[0:linelength]  # Ensure it fits the preallocated array

    #     profile_mem_all = np.array(profile_mem_all)
    #     profile_nuc_all = np.array(profile_nuc_all)
    #     mem_end, mem_start = find_row_transitions(profile_mem_all > 0)

    #     # Add mem_start and mem_end to the dataframe for this cell
    #     df_cell = df_cell.iloc[t_range].copy()  # Only keep the frames we analyzed
    #     df_cell['mem_start'] = mem_start + df_cell['pos_on_track'] - linelength//2  # Adjust mem_start to be in the same coordinate system as pos_on_track
    #     df_cell['mem_end'] = mem_end + df_cell['pos_on_track'] - linelength//2  # Adjust mem_end to be in the same coordinate system as pos_on_track

    #     return profile_nuc_all, profile_mem_all, mem_start, mem_end, df_cell