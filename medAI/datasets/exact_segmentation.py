import os 
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


DATA_ROOT=os.environ.get("DATA_ROOT")
if DATA_ROOT is None:
    raise ValueError("Environment variable DATA_ROOT must be set")


class AlignedFilesDataset(Dataset):
    PIXEL_MEAN, PIXEL_STD = (
        [0.5343046716935111],
        [0.13430083029000342],
    )  # mean and std of the dataset calculated manually

    def __init__(self, root=None, split="train", transform=None):
        """
        root: path to the root directory for all datasets. Dataset should contained a directory
            called "aligned_files_v1_data" organized as follows:
            aligned_files_v1_data/
            ├── Testing Images
            ├── Testing Masks
            ├── Training Images
            └── Training Masks
            if root is None, will attempt to read from the environment variable "DATA_ROOT"

        transform (optional) : a callable object that takes in an image and mask and returns a transformed version of both
        """
        super().__init__()

        if root is None:
            root = Path(DATA_ROOT)
        self.root = root / "aligned_files_v1_data"

        self.split = split

        if self.split == "train":
            self.img_dir = self.root / "Training Images"
            self.mask_dir = self.root / "Training Masks"
        elif self.split == "test":
            self.img_dir = self.root / "Testing Images"
            self.mask_dir = self.root / "Testing Masks"
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        self.mask_paths = sorted(self.mask_dir.glob("*.jpg"))

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        out = {}
        out["image"] = np.array(Image.open(self.img_paths[idx]))
        out["mask"] = np.array(Image.open(self.mask_paths[idx]))
        out["depth"] = 48

        if self.transform is not None:
            out = self.transform(out)

        return out

    def show_item(self, idx, ax=None):
        ax = ax or plt.gca()

        img = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        img = np.array(img)
        mask = np.array(mask)
        img = resize(img, (512, 512))
        mask = resize(mask, (512, 512))

        ax.imshow(img, cmap="gray")
        ax.imshow(mask, alpha=0.5)
        ax.axis("off")

        ax.set_title(f"Image {idx}")


from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
root = Path('/ssd005/projects/exactvu_pca')


class AnnotatedQASegmentationDataset1(Dataset): 
    def __init__(self, root=None, split='train', transform=None):
        self.root = Path(root or os.environ['DATA_ROOT'])
        self.split = split
        self.transform = transform

        self.table = pd.read_csv(self.root / "annotated_qa_data_v1" / "Info.csv")

        TRAIN_SUBJECTS = [1, 2, 3]
        TEST_SUBJECTS = [4]
        match self.split: 
            case 'train': 
                self.table = self.table[self.table['Subject'].isin(TRAIN_SUBJECTS)]
            case 'test': 
                self.table = self.table[self.table['Subject'].isin(TEST_SUBJECTS)]
            case 'all': 
                pass
            case _: 
                raise ValueError(f"Unknown split {self.split}")  
            
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        image_num = int(row['Image'])
        image = Image.open(self.root / "annotated_qa_data_v1" / "Images" / f"Image{str(image_num).zfill(5)}.jpg")
        mask = Image.open(self.root / "annotated_qa_data_v1" / "Masks" / f"Mask{str(image_num).zfill(5)}.jpg")

        item = {
            'image': image,
            'mask': mask,
            **row.to_dict()
        }

        if self.transform is not None:
            item = self.transform(item)

        return item
    
    @staticmethod
    def default_to_tensor(self, item):
        item['image'] = np.array(item['image'])
        item['mask'] = np.array(item['mask'])
        from torchvision import transforms as T 
        item['image'] = T.ToTensor()(item['image'])
        item['mask'] = T.ToTensor()(item['mask'])

        return item['image'], item['mask']


