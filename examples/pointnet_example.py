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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    "Accuracy": accuracy3d,
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
    
    myPoinNet.optimizer = torch.optim.Adam(params=myPoinNet.parameters(), lr=0.0005)

    myPoinNet.get_model_info()

    # Train the model:
    myPoinNet.train(train_loader,val_loader, epochs=5)

    # Save the model:
    myPoinNet.save_model("pointnet_model.pt")
    # myPoinNet.save_model_in_onnx("pointnet_model.onnx", input_sample= torch.randn(1, 3, 64, 64)) #Inputsize = (batch_size, channels, height, width)

# Final evaluation:
# metrics = myPoinNet.evaluate(val_loader)
# print(metrics)


#################
# Visualisation #
#################
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def predict_full_scene(model, dataset, scene_base_path, num_points=2048):
    model.model.eval()
    
    # Charger la scène complète via la méthode du dataset
    points, ground_truth = dataset._load_scene_data(scene_base_path)
    total_points = points.shape[0]
    
    all_preds = np.zeros(total_points, dtype=np.int64)
    
    with torch.no_grad():
        for start_idx in range(0, total_points, num_points):
            end_idx = min(start_idx + num_points, total_points)
            block_points = points[start_idx:end_idx]
            
            current_count = block_points.shape[0]
            
            # Padding si le bloc est plus petit que num_points
            if current_count < num_points:
                pad_indices = np.random.choice(current_count, num_points - current_count, replace=True)
                block_points_padded = np.vstack([block_points, block_points[pad_indices]])
            else:
                block_points_padded = block_points
            
            # Normalisation identique à celle du Dataset
            centroid = np.mean(block_points_padded, axis=0)
            block_centered = block_points_padded - centroid
            max_dist = np.max(np.sqrt(np.sum(block_centered ** 2, axis=1)))
            block_normalized = block_centered / max_dist if max_dist > 0 else block_centered
                
            # Passage au format tenseur (1, 3, N) et envoi sur le bon device
            points_tensor = torch.tensor(block_normalized, dtype=torch.float32).transpose(1, 0).unsqueeze(0).to(model.device)
            
            outputs = model.model(points_tensor)  # Shape: (1, C, N)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
            all_preds[start_idx:end_idx] = preds[:current_count]
            
    # Ajuster ground_truth de [1..C] à [0..C-1] pour matcher les prédictions du modèle
    ground_truth_adjusted = ground_truth - 1
    
    return points, ground_truth_adjusted, all_preds

# --- Exécution de la visualisation ---
# Sélectionner la première scène de validation
sample_scene_dir = val_dataset.scene_dirs[0] 

points_np, ground_truth, preds = predict_full_scene(myPoinNet, val_dataset, sample_scene_dir, num_points=2048)
print(f"Scène complète chargée : {points_np.shape[0]} points.")

# Création de la figure Plotly 1x2
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=("<b>Vérité Terrain (Ground Truth)</b>", "<b>Prédictions PointNet</b>")
)

# Trace 1 : Vérité terrain
fig.add_trace(
    go.Scatter3d(
        x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2],
        mode='markers',
        marker=dict(
            size=2,  
            color=ground_truth,
            colorscale='Viridis',
            opacity=0.9,
            colorbar=dict(title="Classes GT", x=0.45)
        ),
        name='Ground Truth'
    ),
    row=1, col=1
)

# Trace 2 : Prédictions du modèle
fig.add_trace(
    go.Scatter3d(
        x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=preds,
            colorscale='Viridis',
            opacity=0.9,
            colorbar=dict(title="Classes Pred", x=1.0)
        ),
        name='Prediction'
    ),
    row=1, col=2
)

# Mise en page
fig.update_layout(
    title_text="<b>Comparaison 3D - Validation PointNet</b>",
    width=1300,
    height=750,
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data')
)

fig.show()