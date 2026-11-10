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

from openmlcore.loss.loss3d import CrossEntropyLoss, CustomLoss3d, DiceLoss3d
from torch.utils.data import DataLoader
from src.openmlcore.models.pointnet import PointNet
from src.openmlcore.base.data3d import Segmentation3dDataset
from src.openmlcore.metrics.metrics3d import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch.nn as nn #todo to remove
NB_CLASSES = 5 # 4 + 1 (noise)
NAME_MODEL = "PointNet"

# DataPreparation:
train_dataset = Segmentation3dDataset(dataset_path="data\Dataset\Airplane", split="train")
val_dataset   = Segmentation3dDataset(dataset_path="data\Dataset\Airplane", split="val")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=True)

# Metrics
metrics = {
    "Accuracy": accuracy3d,
    "IoU": iou3d
}

weights_for_class = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
assert len(weights_for_class) == NB_CLASSES
weights_type_loss = [0.7, 0.3]

myLoss = CustomLoss3d(loss_fcns=[ DiceLoss3d(weights=weights_for_class), CrossEntropyLoss(weights=weights_for_class)], name="CustomLoss", coefficients=weights_type_loss)

with mlflow.start_run(run_name=NAME_MODEL):
    myPoinNet = PointNet(input_channels=3, output_channels=NB_CLASSES, loss_fn= myLoss, metrics=metrics)
    myPoinNet.create_model()
    # myPoinNet.load_model("pointnet_model.pt")
    myPoinNet.optimizer = torch.optim.Adam(params=myPoinNet.parameters(), lr=0.0005)

    myPoinNet.get_model_info()

    # Train the model:
    myPoinNet.train(train_loader,val_loader, epochs=5)

    # Save the model:
    myPoinNet.save_model("pointnet_model.pt")
    myPoinNet.save_model_in_onnx("pointnet_model.onnx", input_sample= torch.randn(1, 3, 2048)) #Inputsize = (batchsize, 3 (for 3 dimensions), nb 3D points)

# Final evaluation:
metrics = myPoinNet.evaluate(val_loader)
print(metrics)
