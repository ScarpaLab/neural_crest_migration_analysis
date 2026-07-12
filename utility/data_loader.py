from utility import config
import os
import tifffile as tiff
import pickle

class DataLoader:
    '''
    DataLoader class to load data from files
    '''
    def __init__(self, metadata_sub):
        self.data_dir = config.data_dir
        if hasattr(metadata_sub, "to_dict"):
            self.__dict__.update(metadata_sub.to_dict())
        self.sub = int(self.sub)
        self.sub_dir = f"sub-{self.sub:03d}"

    def generate_file_path(self, file_name, ext = 'tif'):
        '''
        Generate the file path for a given file name
        '''
        return os.path.join(self.data_dir, self.sub_dir, f'{self.sub_dir}_data-{file_name}.{ext}')
    
    def load_data(self, file_name, ext = 'tif'):
        '''
        Load data from a given file name
        file_name can be one of the following:
        - 'xyProjection': the xy projection over time
        - 'memManualLabel': the manual label of the membrane
        - 'memMask': the membrane mask from auto-segmentation
        - 'nucManualLabel': the manual label of the nucleus
        - 'nucMask': the nucleus mask from auto-segmentation
        - 'drift': the drift correction data
        '''
        if file_name in ['drift']:
            ext = 'pkl'
            
        file_path = self.generate_file_path(file_name, ext)
        if ext == 'tif':
            print(f'Loading {file_name} ...')
            return tiff.imread(file_path)
        elif ext == 'pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")