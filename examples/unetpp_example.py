###     Example for a Basic UNet      ###
# Author: Sebastien Doyez
# This python script explained how to train a UNet++ modele using my Framework

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from openmlcore.loss.loss import BCEWithLogitsLoss, CustomLoss, DiceLoss
from torch.utils.data import DataLoader, Subset
from src.openmlcore.models.unetpp import UNetPP
from src.openmlcore.base.data import SegmentationDataset
from src.openmlcore.metrics.metrics import *

from torchvision.transforms import v2

# Normalization:
transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256), interpolation=v2.InterpolationMode.NEAREST),
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

NB_CLASSES = 5

# DataPreparation:
train_dataset = SegmentationDataset(root_dir="data/train", num_classes=NB_CLASSES, transform=transforms)
val_dataset   = SegmentationDataset(root_dir="data/val"  , num_classes=NB_CLASSES, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=True)

# Metrics
metrics = {
    "Accuracy": accuracy,
    "IoU": iou,
    "Precision": precision,
    "Recall": recall,
    "F1": F1
}

weights_for_class = torch.tensor([4.0, 1.0, 4.0, 3.5, 2.0])
assert len(weights_for_class) == NB_CLASSES
weights_type_loss = [0.7, 0.3]

myLoss = CustomLoss(loss_fcns=[ DiceLoss(weights=weights_for_class), BCEWithLogitsLoss(weights=weights_for_class)], name="CustomLoss", coefficients=weights_type_loss)
myUnetpp = UNetPP(input_channels=3, output_channels=NB_CLASSES, depth=4, initial_filters=32, loss_fn= myLoss, metrics=metrics)
myUnetpp.create_model()
myUnetpp.optimizer = torch.optim.Adam(params=myUnetpp.parameters(), lr=8e-4)
myUnetpp.load_model("checkpoint_UNet++.pt")
myUnetpp.get_model_info()

# Train the model:
myUnetpp.train(train_loader,val_loader, epochs=0)

# Save the model:
myUnetpp.save_model("unetpp_model.pt")
myUnetpp.save_model_in_onnx("unetpp_model.onnx", input_sample= torch.randn(1, 3, 64, 64)) #Inputsize = (batch_size, channels, height, width)

# Final evaluation:
metrics = myUnetpp.evaluate(val_loader)
print(metrics)

# Show the result:
# Define the name of classes:
# class_names = ["Water", "Land", "Road", "Building", "Vegetation", "Unlabeled"]
class_names = ["Water", "Land", "Road", "Building", "Vegetation"]

assert len(class_names) == NB_CLASSES
cmap_6 = mcolors.ListedColormap(plt.cm.tab10.colors[:len(class_names)])

# Run on validation...
data_iter = iter(val_loader)
for i in range(10):    
    given_input, GT = next(data_iter)
    predictions = myUnetpp.predict(given_input)

    img = given_input[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_unnormalized = np.clip(img * std + mean, 0, 1)
    
    img_rgb = img_unnormalized
    pred_class = torch.argmax(predictions[0], dim=0).cpu().numpy()

    if GT[0].ndim == 3:
        gt_class = torch.argmax(GT[0], dim=0).cpu().numpy()
    else:
        gt_class = GT[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, NB_CLASSES))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img_rgb)
    im_gt = axes[1].imshow(gt_class, cmap=cmap_6, alpha=0.5, vmin=0, vmax=5)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(img_rgb)
    im_pred = axes[2].imshow(pred_class, cmap=cmap_6, alpha=0.5, vmin=0, vmax=5)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), shrink=0.7, ticks=range(NB_CLASSES))
    cbar.ax.set_yticklabels([f"{i}: {name}" for i, name in enumerate(class_names)])

    plt.tight_layout()
    plt.show()
