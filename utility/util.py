from utility import config
import pandas as pd
import numpy as np
import pickle

def load_metadata():
    '''
    Load metadata from the specified path in the config file.
    '''
    metadata = pd.read_csv(config.metadata_path, encoding='unicode_escape')
    return metadata

def load_pkl(path):
    with open(path, "rb") as f:
        content = pickle.load(f)
    return content

def save_pkl(content, path):
    with open(path, 'wb') as file: 
        # A new file will be created 
        pickle.dump(content, file)