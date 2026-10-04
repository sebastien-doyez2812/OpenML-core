# author: Sebastien Doyez
# data is the file where we create custom Dataset class
# SegmentationDataset is the dataset class for 1 hot encoding dataset

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

class SegmentationDataset(Dataset):
    def __init__(self, root_dir="data/train", transform=None, num_classes = 8):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "imgs")
        self.masks_dir = os.path.join(root_dir, "labels")
        self.transform = transform
        self.num_classes = num_classes

        self.img_names = sorted(os.listdir(self.images_dir))
        self.mask_names = sorted(os.listdir(self.masks_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.img_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_np = np.array(mask, dtype = np.uint8)
        masks = [((mask_np >> i ) & 1) for i in range (self.num_classes)]

        mask_stacked = np.stack(masks, axis = 0).astype(np.float32)
        mask_tensor = tv_tensors.Mask(mask_stacked)

        if self.transform is not None:
            image, mask_tensor = self.transform(image, mask_tensor)
        else:
            image = v2.functional.to_image(image)
            image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)

        return image, mask_tensor


