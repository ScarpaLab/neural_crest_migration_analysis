import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt
from tqdm import tqdm

class DriftCorrector:
    def __init__(self, reference_stack):
        """Initialize with the stack used to calculate drift (e.g., Nuclei)."""
        self.reference_stack = reference_stack.astype(np.float32)
        self.drifts = None
        self.dims = reference_stack.shape # (t, y, x)

    def _get_single_drift(self, img1, img2):
        """Internal helper: cross-correlation logic."""
        f1, f2 = fftn(img1), fftn(img2)
        cc = ifftn(f1 * np.conj(f2))
        drift = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
        drift = np.array(drift, dtype=float)
        for i in range(len(drift)):
            if drift[i] > self.dims[i+1] // 2:
                drift[i] -= self.dims[i+1]
        return drift

    def calculate_all_drifts(self):
        """Computes and stores the cumulative drift."""
        raw_drifts = [[0, 0]]
        for i in tqdm(range(self.dims[0] - 1), desc="Calculating Drift"):
            raw_drifts.append(self._get_single_drift(self.reference_stack[i], self.reference_stack[i+1]))
        
        cum_drift = np.cumsum(raw_drifts, axis=0)
        # Normalize so all drifts are positive (to avoid indexing errors)
        cum_drift[:, 0] -= np.min(cum_drift[:, 0])
        cum_drift[:, 1] -= np.min(cum_drift[:, 1])
        self.drifts = cum_drift.astype(np.int16)
        print("Drift calculation complete.")

    def apply_correction(self, stack_to_fix):
        """Applies the stored drifts to any stack (Nuclei or Membrane)."""
        if self.drifts is None:
            raise ValueError("drifts haven't been calculated yet! Run calculate_all_drifts() first.")

        new_shape = (
            stack_to_fix.shape[0],
            stack_to_fix.shape[1] + np.max(self.drifts[:, 0]),
            stack_to_fix.shape[2] + np.max(self.drifts[:, 1])
        )
        corrected = np.zeros(new_shape, dtype=stack_to_fix.dtype)

        for t in range(stack_to_fix.shape[0]):
            sx, sy = self.drifts[t]
            h, w = stack_to_fix.shape[1], stack_to_fix.shape[2]
            corrected[t, sx : sx + h, sy : sy + w] = stack_to_fix[t]
        return corrected

    def plot_drifts(self):
        """Visual check of the drift over time."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.drifts[:, 0], label='Y Drift') # Assuming index 0 is Y
        plt.plot(self.drifts[:, 1], label='X Drift')
        plt.legend(); plt.show()