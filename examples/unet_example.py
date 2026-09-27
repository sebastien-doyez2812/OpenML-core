import sys
from pathlib import Path

import torch
import numpy as np
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from openmlcore.loss.loss import DiceLoss
from torch.utils.data import DataLoader, Subset
from src.openmlcore.models.UNets import UNet
from datasets import load_dataset

import matplotlib.pyplot as plt

# Load the dataset:
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
    transforms.Lambda(
        lambda mask: torch.from_numpy(
            np.array(mask, dtype=np.uint8).copy()
        ).long() - 1
    ),
])

dataset = datasets.OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="segmentation",
    transform=image_transform,
    target_transform=mask_transform,
    download=True,
)

train_dataset = Subset(dataset, range(32))
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# Create the model:
myUnet = UNet(input_channels=3, output_channels=3, depth=4, initial_filters=16, loss_fn= DiceLoss(), metrics=[("MSE", "pixel_acc", lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2))])
myUnet.create_model()
myUnet.optimizer = torch.optim.Adam(params=myUnet.parameters(), lr=0.001)

myUnet.get_model_info()

# Train the model:
myUnet.train(train_loader, epochs=50)

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