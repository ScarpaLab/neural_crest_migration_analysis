import numpy as np
import matplotlib.pyplot as plt


class CellPlotter():
    def __init__(self, df, kymo_data, mem_img, nuc_img):
        self.df = df
        self.kymo_data = kymo_data
        self.mem_img = mem_img
        self.nuc_img = nuc_img

    def plot_cell_props(self, cell_idx, prop = 'pos_on_track', ax=None, **kwargs):
        '''
        Plots the cell properties for a given cell id and property name
        '''
        cell_df = self.df[self.df['cell_idx'] == cell_idx]
        
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5))
        
        ax.plot(cell_df['frame'], cell_df[prop], label=prop, **kwargs)
        ax.set_xlabel('Frame')

    def plot_cell_props_scatter(self, cell_idx, prop = 'mem_front', ax=None, **kwargs):
        '''
        Plots the cell properties for a given cell id and property name
        '''
        cell_df = self.df[self.df['cell_idx'] == cell_idx]
        
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5))
        
        ax.scatter(cell_df['frame'], cell_df[prop], label=prop, **kwargs)
        ax.set_xlabel('Frame')

    def plot_kymograph(self, cell_idx, ax=None):
        '''
        Plots the kymograph for a given cell id
        '''
        kymo = self.kymo_data[cell_idx]
        
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5))
        
        self._plot_additive_composite(kymo['mem'], kymo['nuc'], ax=ax, aspect='auto')

    def plot_movement_arrow(self, cell_idx, step=16, ch='nuc', ax=None, frame_id = None):
        cell_df = self.df[self.df['cell_idx'] == cell_idx].sort_values('frame')
        x_pos, y_pos = cell_df['centroid-x_nuc'].values, cell_df['centroid-y_nuc'].values
        x_pos_int, y_pos_int = x_pos.astype(int), y_pos.astype(int)

        img = self.nuc_img if ch == 'nuc' else self.mem_img
        num_df_frames = len(cell_df)
        if frame_id is None:
            frame_id = num_df_frames // 2
        nuc_frame = self.nuc_img[int(cell_df['frame'].iloc[frame_id])]
        mem_frame = self.mem_img[int(cell_df['frame'].iloc[frame_id])]
        # ax.imshow(frame, cmap='gray', interpolation='none')

        mem_frame_show = mem_frame[y_pos_int.min()-20:y_pos_int.max()+20, x_pos_int.min()-20:x_pos_int.max()+20]
        nuc_frame_show = nuc_frame[y_pos_int.min()-20:y_pos_int.max()+20, x_pos_int.min()-20:x_pos_int.max()+20]
        #ax.imshow(mem_frame_show, cmap='gray', interpolation='none')
        self._plot_additive_composite(mem_frame_show, nuc_frame_show, ax=ax)

        offset_x, offset_y = x_pos_int.min()-20, y_pos_int.min()-20
        x_pos_offset, y_pos_offset = x_pos - offset_x, y_pos - offset_y

        vx, vy = self._calculate_motion_vector(x_pos, y_pos, moving_window=step)
        ax.plot(x_pos_offset, y_pos_offset, color='blue', alpha=0.5, linestyle='--')
        for t in range(0, len(x_pos)-step//2, 1):
            vec = [(vx[t]), (vy[t])]
            ax.arrow(x_pos_offset[t], y_pos_offset[t], vec[0]/5, vec[1]/5, head_width=5, head_length=5, color='red', alpha=0.6)
            ax.scatter(x_pos_offset[t], y_pos_offset[t], color='blue', s=10, alpha=0.6)

    @staticmethod
    def _plot_additive_composite(chan_lime, chan_magenta, ax=None, **kwargs):
        '''
           Creates an additive RGB overlay of two 2D arrays.
        '''
        def normalize(img):
            denom = img.max() - img.min()
            return (img - img.min()) / denom if denom > 0 else img

        img_l = normalize(chan_lime)
        img_m = normalize(chan_magenta)

        color_lime = np.array([0, 1, 0])
        color_magenta = np.array([1, 0, 1])

        rgb_lime = img_l[..., None] * color_lime
        rgb_magenta = img_m[..., None] * color_magenta

        composite = np.clip(rgb_lime + rgb_magenta, 0, 1)

        ax.imshow(composite, interpolation='none', **kwargs)

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