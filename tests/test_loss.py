from openml.computer_vision.loss.loss import *

def test_cross_entropy_loss():
    loss = CrossEntropyLoss()
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

    