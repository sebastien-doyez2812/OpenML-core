import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.openmlcore.models.UNets import UNet

myUnet = UNet(input_channels=1, output_channels=1, depth=4, initial_filters=64)
myUnet.create_model()
myUnet.get_model_info()
