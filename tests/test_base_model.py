import sys
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchgen import model
from torchgen import model

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.openmlcore.models.base_model import *
from src.openmlcore.metrics.metrics import *

# class DummyModel(BaseModel):
#     def __init__(self):
#         super().__init__(name = "DummyLinearModel")
#         self.create_model()
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
#         self.metrics = [("MSE", lambda pred, target: nn.functional.mse_loss(pred, target))]

#     def help(self):
#         print("This is a dummy linear model for testing purposes.")

#     def create_model(self):
#         self.fc = nn.Linear(10, 2)
#         self.model = self.fc

class DummyModel(BaseModel):
    def __init__(self):
        super().__init__(name="DummyLinearModel")
        # On appelle super().__init__() avant d'assigner self.model
        self.create_model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.metrics = {
            "Accuracy": accuracy,
            "IoU": iou
        }
        
    def help(self):
        print("This is a dummy linear model for testing purposes.")

    def create_model(self):
        # Utiliser nn.Sequential garantit que tous les sous-modules 
        # basculent sur le bon device en même temps que self.model
        self.model = nn.Sequential(
            nn.Linear(10, 2)
        ).to(self.device)

@pytest.fixture
def dummy_dataloader():
    X = torch.randn(64, 10)
    Y = torch.randn(64, 2)
    dataset = torch.utils.data.TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=8, shuffle=True)


def test_abstract_base_modeel():
    base_model = BaseModel()
    with pytest.raises(NotImplementedError):
        base_model.create_model()

    with pytest.raises(NotImplementedError):    
        base_model.help()

def test_training_exceptions():
    base_model = BaseModel()
    loader = torch.utils.data.DataLoader(torch.randn(64, 10), batch_size=8)

    with pytest.raises(NotImplementedError):
        base_model.train(loader, epochs=1)

def test_model_training(dummy_dataloader):
    model = DummyModel()
    model.train(dummy_dataloader, epochs=2)

def test_model_prediction(dummy_dataloader):
    model = DummyModel()
    print(model.device)
    X_sample = torch.randn(5, 10).to(model.device)
    preds = model.predict(X_sample)
    assert preds.shape == (5, 2)

def test_model_save_and_load(tmp_path, dummy_dataloader):
    model = DummyModel()
    file_path = tmp_path / "model.pt"
    
    model.save_model(str(file_path))
    assert file_path.exists()

    new_model = DummyModel()
    new_model.load_model(str(file_path), is_train_mode=False)
    
    for p1, p2 in zip(model.model.parameters(), new_model.model.parameters()):
        assert torch.equal(p1, p2)


def test_onnx_export(tmp_path):
    model = DummyModel()
    file_path = tmp_path / "model.onnx"
    dummy_input = torch.randn(1, 10).to(model.device)
    
    model.save_model_in_onnx(str(file_path), dummy_input)
    assert file_path.exists()    