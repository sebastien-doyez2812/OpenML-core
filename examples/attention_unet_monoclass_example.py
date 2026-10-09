###     Example for a Basic UNet      ###
# Author: Sebastien Doyez
# This python script explained how to train a Attention UNet modele using my Framework

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
from src.openmlcore.models.attUnets import AttentionUNet
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

NB_CLASSES = 1

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

weights_for_class = torch.tensor([1.0])
assert len(weights_for_class) == NB_CLASSES
weights_type_loss = [0.7, 0.3]

myLoss = CustomLoss(loss_fcns=[ DiceLoss(weights=weights_for_class), BCEWithLogitsLoss(weights=weights_for_class)], name="CustomLoss", coefficients=weights_type_loss)
myAttUnet = AttentionUNet(input_channels=3, output_channels=NB_CLASSES, depth=4, initial_filters=32, loss_fn= myLoss, metrics=metrics)
myAttUnet.create_model()
myAttUnet.load_model("attention_unet_model.pt")
myAttUnet.optimizer = torch.optim.Adam(params=myAttUnet.parameters(), lr=8e-4)

myAttUnet.get_model_info()

# Train the model:
myAttUnet.train(train_loader,val_loader, epochs=600)

# Save the model:
myAttUnet.save_model("attention_unet_model.pt")
myAttUnet.save_model_in_onnx("attention_unet_model.onnx", input_sample= torch.randn(1, 3, 64, 64)) #Inputsize = (batch_size, channels, height, width)

# Final evaluation:
metrics = myAttUnet.evaluate(val_loader)
print(metrics)

# Show the result:
# Define the name of classes:
class_names = ["Cells"] 

assert len(class_names) == NB_CLASSES
cmap_6 = mcolors.ListedColormap(plt.cm.tab10.colors[:len(class_names)])

# Run on validation...
data_iter = iter(val_loader)
for i in range(10):    
    given_input, GT = next(data_iter)
    
    with torch.no_grad():
        predictions = myAttUnet.predict(given_input)

    img = given_input[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = np.clip(img * std + mean, 0, 1)
    
    prob_map = torch.sigmoid(predictions[0][0])
    pred_class = (prob_map > 0.5).float().cpu().numpy()
    
    print(f"Prob min={prob_map.min():.3f}, max={prob_map.max():.3f} | Pixels cellules={int(pred_class.sum())}")

    if GT[0].ndim == 3:
        gt_class = GT[0][0].cpu().numpy()
    else:
        gt_class = GT[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img_rgb)
    axes[1].imshow(gt_class, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(img_rgb)
    axes[2].imshow(pred_class, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Predictions", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
