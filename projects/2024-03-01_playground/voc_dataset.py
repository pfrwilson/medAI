import torch 
from torch.utils.data import Dataset
from PIL import Image
import os 


class VOCDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, image_ids_file, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_ids = image_ids_file
        self.transform = transform

        with open(image_ids_file, "r") as f:
            self.image_ids = f.read().splitlines()

        self.images = []
        for image_id in self.image_ids:
            self.images.append(os.path.join(self.images_dir, image_id + ".jpg"))
        self.masks = []
        for image_id in self.image_ids:
            self.masks.append(os.path.join(self.annotations_dir, image_id + ".png"))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self): 
        return len(self.images)


def build_dataset_vector_cluster(set='train', transform=None):
    images_dir = '/scratch/ssd002/datasets/VOC2012/JPEGImages'
    annotations_dir = '/ssd005/projects/exactvu_pca/voc/SegmentationClassAug/SegmentationClassAug'

    if set == 'train':
        image_ids_file = "/ssd005/projects/exactvu_pca/voc/train_aug.txt"
    elif set == 'val':
        image_ids_file = "/ssd005/projects/exactvu_pca/voc/val.txt"

    return VOCDataset(images_dir, annotations_dir, image_ids_file, transform=transform)