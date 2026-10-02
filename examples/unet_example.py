import sys
from pathlib import Path

import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2, InterpolationMode
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from openmlcore.loss.loss import CrossEntropyLoss, CustomLoss, DiceLoss, MeanSquaredErrorLoss
from torch.utils.data import DataLoader, Subset
from src.openmlcore.models.UNets import UNet
from src.openmlcore.base.data import SegmentationDataset
from datasets import load_dataset

import matplotlib.pyplot as plt

# CITYSCAPES_MAPPING = {
#     0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
#     7: 0,   # Road -> Class 0
#     8: 1,   # Sidewalk -> Class 1
#     9: 255, 10: 255,
#     11: 2,  # Building -> Class 2
#     12: 3,  # Wall -> Class 3
#     13: 4,  # Fence -> Class 4
#     14: 255, 15: 255, 16: 255,
#     17: 5,  # Pole -> Class 5
#     18: 255,
#     19: 6,  # Traffic light -> Class 6
#     20: 7,  # Traffic sign -> Class 7
#     21: 8,  # Vegetation -> Class 8
#     22: 9,  # Terrain -> Class 9
#     23: 10, # Sky -> Class 10
#     24: 11, # Person -> Class 11
#     25: 12, # Rider -> Class 12
#     26: 13, # Car -> Class 13
#     27: 14, # Truck -> Class 14
#     28: 15, # Bus -> Class 15
#     29: 255, 30: 255,
#     31: 16, # Train -> Class 16
#     32: 17, # Motorcycle -> Class 17
#     33: 18, # Bicycle -> Class 18
#     -1: 255
# }

CITYSCAPES_MAPPING = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   # Road -> Class 0
    8: 1,   # Sidewalk -> Class 1
    9: 255, 10: 255,
    11: 2,  # Building -> Class 2
    12: 3,  # Wall -> Class 3
    13: 255,  # Fence -> Class 4
    14: 255, 15: 255, 16: 255,
    17: 255,  # Pole -> Class 5
    18: 255,
    19: 255,  # Traffic light -> Class 6
    20: 255,  # Traffic sign -> Class 7
    21: 255,  # Vegetation -> Class 8
    22: 255,  # Terrain -> Class 9
    23: 255, # Sky -> Class 10
    24: 255, # Person -> Class 11
    25: 255, # Rider -> Class 12
    26: 4, # Car -> Class 13
    27: 4, # Truck -> Class 14
    28: 4, # Bus -> Class 15
    29: 255, 30: 255,
    31: 4, # Train -> Class 16
    32: 4, # Motorcycle -> Class 17
    33: 4, # Bicycle -> Class 18
    -1: 255
}
transforms = v2.Compose([
    v2.ToDtype(torch.float32, scale=True), # passe en [0.0, 1.0]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SegmentationDataset(root_dir="data/train", transform=transforms, map_label=CITYSCAPES_MAPPING)
val_dataset = SegmentationDataset(root_dir="data/val", transform=transforms, map_label=CITYSCAPES_MAPPING)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=True)

# Create the model:
# TODO: put this in a specific metric file
def accuracy(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1)
    mask = (y_true != 255)
    return (preds[mask] == y_true[mask]).float().mean()
metrics = {
    "Accuracy": accuracy
}
myLoss = CustomLoss(loss_fcns=[ DiceLoss(index_ignore=255), CrossEntropyLoss(index_ignore=255)], name="CustomLoss", coefficients=[0.2, 0.8])
myUnet = UNet(input_channels=3, output_channels=5, depth=4, initial_filters=32, loss_fn= myLoss, metrics=metrics)
myUnet.create_model()
myUnet.optimizer = torch.optim.Adam(params=myUnet.parameters(), lr=3e-4)

myUnet.get_model_info()

# Train the model:
myUnet.train(train_loader,val_loader, epochs=50)

#Test the model:
data_iter = iter(train_loader)
given_input, GT = next(data_iter)
predictions = myUnet.predict(given_input)

img=given_input[0].permute(1,2,0).cpu().numpy()
pred_class = torch.argmax(predictions[0], dim=0).cpu().numpy()

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.imshow(pred_class, cmap='jet', alpha=0.5) # alpha gère la transparence (0 = invisible, 1 = opaque)
plt.title("Image avec Overlay Masque")
plt.axis("off")

plt.show()

# Save the model:
myUnet.save_model("unet_model.pth")
myUnet.save_model_in_onnx("unet_model.onnx", input_sample= torch.randn(1, 3, 64, 64)) #Inputsize = (batch_size, channels, height, width)