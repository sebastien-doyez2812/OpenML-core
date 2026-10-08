import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.openmlcore.loss.loss import *

def test_BCEWithLogits_loss():
    loss = BCEWithLogitsLoss()
    assert loss is not None

def test_mean_squared_error_loss():
    loss = MeanSquaredErrorLoss()
    assert loss is not None

def test_dice_loss():
    loss = DiceLoss()
    assert loss is not None

def test_focal_loss():
    loss = FocalLoss()
    assert loss is not None

def test_huber_loss():
    loss = HuberLoss()
    assert loss is not None

def test_kl_divergence_loss():
    loss = KLDivergenceLoss()
    assert loss is not None

    