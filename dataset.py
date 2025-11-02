# dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None, mask_transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            mask_paths (list): List of corresponding mask file paths (optional).
            transform (callable, optional): Transform to be applied on the images.
            mask_transform (callable, optional): Transform to be applied on the masks.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform the image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # If mask paths are provided, load and transform the mask too
        if self.mask_paths:
            mask = Image.open(self.mask_paths[idx])
            if self.mask_transform:
                mask = self.mask_transform(mask)
            return image, mask, os.path.basename(self.image_paths[idx])
        else:
            # Only image is returned during prediction
            return image, os.path.basename(self.image_paths[idx])
