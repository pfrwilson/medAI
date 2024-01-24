"""Handles data access for the NCT2013 dataset.
"""
from abc import ABC, abstractmethod
import pandas as pd
import os
import numpy as np
from PIL import Image
import skimage.transform
import json 
from pathlib import Path
    

class ExactNCT2013DataAccessorBase:
    @abstractmethod
    def get_metadata_table(self): 
        """Returns the metadata table."""

    @abstractmethod
    def get_rf_image(self, core_id, frame_idx=0):
        """Returns the RF image for the given core id and frame idx."""
        pass
    
    @abstractmethod
    def get_bmode_image(self, core_id, frame_idx=0):
        """Returns the B-mode image for the given core id and frame idx."""
        pass
    
    @abstractmethod
    def get_num_frames(self, core_id):
        """Returns the number of frames for the given core id."""
        pass

    @abstractmethod
    def get_prostate_mask(self, core_id):
        """Returns the prostate mask for the given core id."""
        pass

    @abstractmethod
    def get_needle_mask(self, core_id):
        """Returns the needle mask for the given core id."""
        pass

    
class VectorClusterExactNCT2013DataAccessor(ExactNCT2013DataAccessorBase):
    DATA_ROOT = "/ssd005/projects/exactvu_pca/nct2013"
    METADATA = pd.read_csv("/ssd005/projects/exactvu_pca/nct2013/metadata.csv", index_col=0)
    METADATA = METADATA.sort_values(by=['core_id']).reset_index(drop=True)
    needle_mask = np.load(
        "/ssd005/projects/exactvu_pca/nct2013/needle_mask.npy"
    )
    needle_mask = skimage.transform.resize(
        needle_mask, (512, 512), order=0, preserve_range=True, anti_aliasing=False
    )

    def __init__(self, prostate_segmentation_dir=None):
        self.prostate_segmentation_dir = prostate_segmentation_dir
        self._bmode_data = None 
        self._bmode_tag2idx = None 

    def get_metadata_table(self):
        return self.METADATA
    
    def get_rf_image(self, core_id, frame_idx=0):
        fpath = (Path(self.DATA_ROOT) / 'rf').glob(f'{core_id}*.npy')
        arr = np.load(list(fpath)[0])
        return arr[..., frame_idx]

    def get_bmode_image(self, core_id, frame_idx=0):        
        fpath = (Path(self.DATA_ROOT) / 'bmode').glob(f'{core_id}*.npy')
        arr = np.load(list(fpath)[0], mmap_mode='r')
        return arr[..., frame_idx]
        
        return arr

    def get_num_frames(self, core_id):
        fpath = (Path(self.DATA_ROOT) / 'bmode').glob(f'{core_id}*.npy')
        arr = np.load(list(fpath)[0], mmap_mode='r')
        return arr.shape[-1]

    def get_prostate_mask(self, core_id):
        image = Image.open(os.path.join(self.prostate_segmentation_dir, f"{core_id}.png"))
        image = (np.array(image) / 255) > 0.5
        image = image.astype(np.uint8)
        image = np.flip(image, axis=0).copy()
        return image

    def get_needle_mask(self, core_id):
        return self.needle_mask


def setup_data_accessor(): 
    """Sets up the data accessor for the NCT2013 dataset.

    This function should parse system configuration and return the appropriate
    data accessor. class instance
    """

    return VectorClusterExactNCT2013DataAccessor(prostate_segmentation_dir='/ssd005/projects/exactvu_pca/nct_segmentations_medsam_finetuned_2023-11-10')


data_accessor = setup_data_accessor()
