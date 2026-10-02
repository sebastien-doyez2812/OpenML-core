import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask, Image as TVImage

class SegmentationDataset(Dataset):
    def __init__(self, map_label=None, root_dir="data/train", transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "img")
        self.masks_dir = os.path.join(root_dir, "label")
        self.transform = transform
        self.mapping = map_label

        self.img_names = sorted(os.listdir(self.images_dir))
        self.mask_names = sorted(os.listdir(self.masks_dir))

    def __len__(self):
        return len(self.img_names)

    def _apply_mapping(self, mask_np):
        mapped_mask = np.full_like(mask_np, fill_value=255, dtype=np.int64)
        for original_value, new_value in self.mapping.items():
            mapped_mask[mask_np == original_value] = new_value
        return mapped_mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.img_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 
        mask_np = np.array(mask)

        if self.mapping is not None:
            mask_np = self._apply_mapping(mask_np)
        else:
            mask_np = mask_np.astype(np.int64)

        image_tensor = v2.functional.to_image(image) # Convertit PIL -> TVImage / Tensor
        mask_tensor = Mask(torch.tensor(mask_np, dtype=torch.long)) # Marque le tensor comme masque de segmentation
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        else:
            image_tensor = v2.functional.to_dtype(image_tensor, dtype=torch.float32, scale=True)

        return image_tensor, mask_tensor