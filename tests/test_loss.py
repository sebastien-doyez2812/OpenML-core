import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.openmlcore.loss.loss import *

def test_BCEWithLogits_loss():
    loss = BCEWithLogitsLoss(weights = [1.0])
    assert loss is not None

def test_mean_squared_error_loss():
    loss = MeanSquaredErrorLoss(weights = [1.0])
    assert loss is not None

def test_dice_loss():
    loss = DiceLoss(weights = [1.0])
    assert loss is not None

def test_focal_loss():
    loss = FocalLoss(weights = [1.0])
    assert loss is not None

def test_huber_loss():
    loss = HuberLoss(weights = [1.0])
    assert loss is not None    