###     Example for a MobileUnetR      ###
# Author: Sebastien Doyez
# This python script explained how to train a MobileUnetR modele using my Framework

import sys
import torch
import mlflow # Optionnal...
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from openmlcore.loss.loss import BCEWithLogitsLoss, CustomLoss, DiceLoss
from torch.utils.data import DataLoader, Subset
from src.openmlcore.models.MobileUnetR import MobileUnetR, MobileUnetR_v2
from src.openmlcore.base.data import SegmentationDataset
from src.openmlcore.metrics.metrics import *

from torchvision.transforms import v2


NB_CLASSES = 1
SIZE = 256
NAME_MODEL = "MobileUnetR"

# Normalization:
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((SIZE, SIZE), interpolation=v2.InterpolationMode.NEAREST),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((SIZE, SIZE), interpolation=v2.InterpolationMode.NEAREST),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# DataPreparation:
train_dataset = SegmentationDataset(root_dir="data/train", num_classes=NB_CLASSES, transform=train_transforms)
val_dataset   = SegmentationDataset(root_dir="data/val"  , num_classes=NB_CLASSES, transform=val_transforms)

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
weights_type_loss = [0.8, 0.2]

myLoss = CustomLoss(loss_fcns=[ DiceLoss(weights=weights_for_class), BCEWithLogitsLoss(weights=weights_for_class)], name="CustomLoss", coefficients=weights_type_loss)

with mlflow.start_run(run_name=NAME_MODEL):
    # myMobileUnetR = MobileUnetR_v2(input_channels=3, output_channels=NB_CLASSES, loss_fn= myLoss, metrics=metrics)
    myMobileUnetR = MobileUnetR(input_channels=3, output_channels=NB_CLASSES, loss_fn= myLoss, metrics=metrics, use_mlflow= True)
    myMobileUnetR.create_model()
    myMobileUnetR.load_model("mobileUnetR.pt")

    myMobileUnetR.optimizer = torch.optim.AdamW(params=myMobileUnetR.parameters(), lr=1e-4, weight_decay= 0.01, betas=(0.9, 0.999))
    myMobileUnetR.get_model_info()

    # Train the model:
    myMobileUnetR.train(train_loader,val_loader, epochs=3)

    # Save the model:
    myMobileUnetR.save_model("mobileUnetR.pt")
    myMobileUnetR.save_model_in_onnx("mobileUnetR.onnx", input_sample= torch.randn(1, 3, SIZE, SIZE)) #Inputsize = (batch_size, channels, height, width)


# Show the result:
# Define the name of classes:
class_names = ["Cell"]

assert len(class_names) == NB_CLASSES
cmap_6 = mcolors.ListedColormap(plt.cm.tab10.colors[:len(class_names)])

# Run on validation...
data_iter = iter(val_loader)
for i in range(10):    
    given_input, GT = next(data_iter)
    
    with torch.no_grad():
        predictions = myMobileUnetR.predict(given_input)

    img = given_input[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = np.clip(img * std + mean, 0, 1)
    
    prob_map = torch.sigmoid(predictions[0][0])
    pred_class = (prob_map > 0.5).float().cpu().numpy()

    if GT[0].ndim == 3:
        gt_class = GT[0][0].cpu().numpy()
    else:
        gt_class = GT[0].cpu().numpy()

    current_iou = iou(torch.Tensor(prob_map).to(myMobileUnetR.device), torch.Tensor(gt_class).to(myMobileUnetR.device))
    print(np.mean(current_iou, axis=0))


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
