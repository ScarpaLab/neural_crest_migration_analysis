import sys
sys.path.append("..")
from utility import util
from utility import data_loader
from utility.registration import DriftCorrector
from matplotlib import pyplot as plt

metadata = util.load_metadata()

for i in range(len(metadata)):
    dl = data_loader.DataLoader(metadata.iloc[i])
    print(f"Processing dataset: {dl.sub}")
    img = dl.load_data('xyProjection')
    img_ch0 = img[:,0,:,:] # mem
    img_ch1 = img[:,1,:,:] # nuc

    dc = DriftCorrector(img_ch1)
    dc.calculate_all_drifts()

    plt.plot(dc.drifts[:,0], label = 'Y drift')
    plt.plot(dc.drifts[:,1], label = 'X drift')
    plt.legend()

    # Save figure as PNG
    path = dl.generate_file_path('driftPlot', ext='png')
    plt.savefig(path)
    # Close the plot to free memory
    plt.close()

    # Save drift data as pkl
    path = dl.generate_file_path('drift', ext='pkl')
    util.save_pkl(dc.drifts, path)