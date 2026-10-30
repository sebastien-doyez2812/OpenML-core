###     Example for a PointNet     ###
# Author: Sebastien Doyez
# This python script explained how to train a PointNet model using my Framework

import sys
import torch
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from openmlcore.loss.loss import BCEWithLogitsLoss, CustomLoss, DiceLoss
from torch.utils.data import DataLoader, Subset
from src.openmlcore.models.pointnet import PointNet
from src.openmlcore.base.data3d import Segmentation3dDataset
from src.openmlcore.metrics.metrics3d import *
from torchvision.transforms import v2

import torch.nn as nn #todo to remove
NB_CLASSES = 4
NAME_MODEL = "PointNet"

# DataPreparation:
train_dataset = Segmentation3dDataset(dataset_path="data\Dataset\Airplane", split="train")
val_dataset   = Segmentation3dDataset(dataset_path="data\Dataset\Airplane", split="val")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=True)

#todo : improve metrics
# Metrics
metrics = {
    # "Accuracy": accuracy3d,
    "IoU": iou3d,
    # "Precision": Precision3d,
    # "Recall": Recall3d,
    # "F1": F13d
}

#TODO
# weights_for_class = torch.tensor([1.0])
# assert len(weights_for_class) == NB_CLASSES
# weights_type_loss = [0.7, 0.3]

# myLoss = CustomLoss(loss_fcns=[ DiceLoss(weights=weights_for_class), BCEWithLogitsLoss(weights=weights_for_class)], name="CustomLoss", coefficients=weights_type_loss)
myLoss = nn.CrossEntropyLoss()

with mlflow.start_run(run_name=NAME_MODEL):
    myPoinNet = PointNet(input_channels=3, output_channels=NB_CLASSES, loss_fn= myLoss, metrics=metrics)
    myPoinNet.create_model()
    
    myPoinNet.optimizer = torch.optim.Adam(params=myPoinNet.parameters(), lr=8e-4)

    myPoinNet.get_model_info()

    # Train the model:
    myPoinNet.train(train_loader,val_loader, epochs=600)

    # Save the model:
    myPoinNet.save_model("attention_unet_model.pt")
    myPoinNet.save_model_in_onnx("attention_unet_model.onnx", input_sample= torch.randn(1, 3, 64, 64)) #Inputsize = (batch_size, channels, height, width)

# Final evaluation:
metrics = myPoinNet.evaluate(val_loader)
print(metrics)

# # Show the result:
# # Define the name of classes:
# class_names = ["Cells"] 

# assert len(class_names) == NB_CLASSES
# cmap_6 = mcolors.ListedColormap(plt.cm.tab10.colors[:len(class_names)])

# # Run on validation...
# data_iter = iter(val_loader)
# for i in range(10):    
#     given_input, GT = next(data_iter)
    
#     with torch.no_grad():
#         predictions = myAttUnet.predict(given_input)

#     img = given_input[0].permute(1, 2, 0).cpu().numpy()
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img_rgb = np.clip(img * std + mean, 0, 1)
    
#     prob_map = torch.sigmoid(predictions[0][0])
#     pred_class = (prob_map > 0.5).float().cpu().numpy()
    
#     print(f"Prob min={prob_map.min():.3f}, max={prob_map.max():.3f} | Pixels cellules={int(pred_class.sum())}")

#     if GT[0].ndim == 3:
#         gt_class = GT[0][0].cpu().numpy()
#     else:
#         gt_class = GT[0].cpu().numpy()

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     axes[0].imshow(img_rgb)
#     axes[0].set_title("Image", fontsize=12)
#     axes[0].axis("off")

#     axes[1].imshow(img_rgb)
#     axes[1].imshow(gt_class, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
#     axes[1].set_title("Ground Truth", fontsize=12)
#     axes[1].axis("off")

#     axes[2].imshow(img_rgb)
#     axes[2].imshow(pred_class, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
#     axes[1].set_title("Predictions", fontsize=12)
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.show()
