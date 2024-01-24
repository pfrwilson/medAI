from torch.utils.data import Dataset 
from .data_access import data_accessor
import typing as tp
from tqdm import tqdm


class BModeDatasetV1(Dataset): 
    """Dataset for B-mode images.

    Samples are dictionaries with the following keys:
        bmode_frames (np.ndarray): Shape (H, W) uint8
        prostate_mask (np.ndarray): Shape (H, W)
        needle_mask (np.ndarray): Shape (H, W)
        core_id (str): Core id of the sample.
        frame_idx (int): Frame index of the sample.
        psa (float): PSA value of the sample.
        primary_grade (int): Primary grade of the sample.
        secondary_grade (int): Secondary grade of the sample.
        age (int): Age of the sample.
        family_history (int): Family history of the sample.
        ... etc (other metadata)

    Args:
        core_ids (list): List of core ids to include in the dataset.
        transform (callable): Transform to apply to each sample.
        frames (str): If 'first', only the first frame is returned. If 'all', all frames are returned.
    """

    def __init__(self, core_ids, transform=None, frames: tp.Literal['first', 'all']='first'): 
        self.metadata = data_accessor.get_metadata_table().copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]

        self._indices = []
        self.core_ids = core_ids
        for i, core_id in enumerate(tqdm(self.core_ids, desc='Loading dataset')): 
            if frames == 'first': 
                self._indices.append((i, 0))
            elif frames == 'all': 
                n = data_accessor.get_num_frames(core_id)
                self._indices.extend([(i, j) for j in range(n)])

        self.transform = transform

    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, idx):
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]
        
        bmode = data_accessor.get_bmode_image(core_id, frame_idx)
        prostate_mask = data_accessor.get_prostate_mask(core_id)
        needle_mask = data_accessor.get_needle_mask(core_id)

        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        metadata['frame_idx'] = frame_idx

        output = {
            'bmode': bmode,
            'prostate_mask': prostate_mask,
            'needle_mask': needle_mask,
            **metadata
        }

        if self.transform:
            output = self.transform(output)

        return output
        