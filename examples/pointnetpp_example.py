###     Example for a PointNet++     ###
# Author: Sebastien Doyez
# This python script explained how to train a PointNet++ model using my Framework

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
from src.openmlcore.models.pointNetPP import PointNetPP
from src.openmlcore.base.data3d import Segmentation3dDataset
from src.openmlcore.metrics.metrics3d import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch.nn as nn #todo to remove
NB_CLASSES = 4
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

weights_for_class = torch.tensor([1.0, 1.0, 1.0, 1.0])
assert len(weights_for_class) == NB_CLASSES
weights_type_loss = [0.7, 0.3]

myLoss = CustomLoss3d(loss_fcns=[ DiceLoss3d(weights=weights_for_class), CrossEntropyLoss(weights=weights_for_class)], name="CustomLoss", coefficients=weights_type_loss)

with mlflow.start_run(run_name=NAME_MODEL):
    myPoinNetPP = PointNetPP(output_channels=NB_CLASSES, loss_fn= myLoss, metrics=metrics)
    myPoinNetPP.create_model()
    # myPoinNetPP.load_model("pointnet_PP_model.pt")
    myPoinNetPP.optimizer = torch.optim.Adam(params=myPoinNetPP.parameters(), lr=0.0005)

    myPoinNetPP.get_model_info()

    # Train the model:
    myPoinNetPP.train(train_loader,val_loader, epochs=2)

    # Save the model:
    myPoinNetPP.save_model("pointnet_PP_model.pt")
    myPoinNetPP.save_model_in_onnx("pointnet_PP_model.onnx", input_sample= torch.randn(1, 3, 2048)) #Inputsize = (batchsize, 3 (for 3 dimensions), nb 3D points)

# Final evaluation:
metrics = myPoinNetPP.evaluate(val_loader)
print(metrics)
